"""Stronger candidate for matrix multiplication 3x3 * 3x7 tensor decomposition.

Strategy:
  1. Try to construct a rank-59 tensor decomposition (known result by Strassen &
     for 3x3x7 it is conjectured achievable at 59 by a 2x2x2 Strassen-based
     recursion combined with schoolbook, but we do not have an explicit
     closed-form decomposition here.
     Therefore we implement a generic iterative tensor-decomposition search
     (tensor-train / ALS on a rank-R truncated tensor network) with integer
     rounding, and verify the resulting factors exactly against the
     multiplication tensor.
  2. Start from a random rank-R initialization (R from a small list of
     candidate ranks: 55, 56, 57, 58, 59, then 60, 61, 62, 63).
  3. Use alternating least squares (ALS) to minimize the Frobenius error of the
     decomposition C = sum_r U[:,r] outer V[:,r] outer W[:,r] with the
     multiplication tensor (reshaped).
  4. Round to nearest integer; if it exactly reproduces the tensor, return.
     Otherwise, try to repair with a small local search (perturb a few
     entries, re-check).
  5. Fall back to schoolbook rank 63 if no lower rank can be certified within
     the 150 s deadline.

Time budget: 150 s hard kill, but we target ~135 s to be safe.
"""

import time
import numpy as np
import itertools

DIMENSIONS = [3, 3, 7]

# Multiplication tensor T of shape (n, m, p) with T[i, j, k] = 1 if i,j,k
# appear in A*B: C[i,k] = sum_j A[i,j]*B[j,k].
# So the tensor has a 1 at (i, j, k) for each (i, j, k) in [0,n)x[0,m)x[0,p)
# such that the product contributes to C[i,k] via B[j,k] and A[i,j].
# That means every (i, j, k) triple contributes -> the tensor is the full outer
# product of three ones vectors of sizes n, m, p.
# So T = ones(n) outer ones(m) outer ones(p).  Its entries are all 1.
# The decomposition C[i,k] = sum_j A[i,j] B[j,k] corresponds to the tensor
# T[i, j, k] = 1 for all i, j, k.  This is a rank-1 tensor.
# Wait -- that cannot be right.  Let me re-derive.

# In matrix multiplication C = A * B, A is n x m, B is m x p.
# C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k].
# The tensor T of shape (n, p, m) or (n, m, p) has T[i, k, j] = 1,
# because the j-th term in the sum uses A[i,j] and B[j,k].
# If we reshape C to a vector of length n*p, we can write:
#   C_flat = sum_j vec(A[:,j]) * vec(B[j,:])  (outer product in 3D sense)
# So T[i, k, j] = 1 for all i in [0,n), k in [0,p), j in [0,m).
# This tensor is the outer product of three vectors: ones(n) x ones(p) x ones(m),
# hence rank 1.  That's not useful.
# 
# The correct tensor formulation for tensor rank: the tensor T of shape (n, m, p)
# with T[i,j,k] = 1 if and only if the monomial A[i,j] * B[j,k] appears in
# C[i,k].  Since every A[i,j] * B[j,k] appears exactly once in C[i,k],
# T[i,j,k] = 1 for ALL (i,j,k) in the box [0,n) x [0,m) x [0,p).
# So T = ones(n,m,p), rank 1.  This is the bilinear form that maps
# (A, B) -> C where C is a flat n*p array? No, C is n x p, and the j dimension
# is summed.  The tensor that we decompose is of shape (n, m, p) with
# T[i, j, k] = 1, and the decomposition is
#   C[i,k] = sum_r U[i, r] V[k, r] ... no, that's 2-mode.

# Let me use the standard formulation:
# We seek matrices U (n*m x R), V (m*p x R), W (n*p x R) such that
#   sum_{r=1}^{R} U[i*m + j, r] * V[j*p + k, r] = W[i*n + k, r]  ... no.
# 
# Actually the standard tensor decomposition for matmul (n x m) * (m x p) -> (n x p):
# The tensor T of shape (n*p, n*m, m*p) with
#   T[i*p + k, i*m + j, j*p + k] = 1  for all valid (i,j,k).
# The decomposition is:  T = sum_r W[:,r] outer U[:,r] outer V[:,r]
# i.e.,  T[a, b, c] = sum_r W[a, r] * U[b, r] * V[c, r]
# where a indexes C (size n*p), b indexes A (size n*m), c indexes B (size m*p).
# The trace convention: W[k*n + i, r].  So W is indexed by output position
# (i, k) mapped to k*n + i (column-major with i fastest?  No, k*n+i means i is
# the fast index within each k-block, so W row index = k*n + i, i.e. output
# is stored column-major by output column k, then row i.  Actually the task says
# "W[k*n+i, r]" which fixes the row order of W.
# U is indexed by input A position (i, j) -> i*m + j.
# V is indexed by input B position (j, k) -> j*p + k.
# T[a, b, c] = 1  iff  a = k*n + i,  b = i*m + j,  c = j*p + k
# for some (i, j, k).  This holds for all (i,j,k) in [0,n)x[0,m)x[0,p).
# There are n*m*p such entries, each with value 1.  The rest are 0.

def _build_T(n, m, p):
    """Build the (n*p, n*m, m*p) multiplication tensor (sparse representation).
    Returns list of (a, b, c) triples with T[a,b,c]=1."""
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = k * n + i  # output index (W row)
                b = i * m + j  # input A index (U row)
                c = j * p + k  # input B index (V row)
                triples.append((a, b, c))
    return triples

def _check_decomposition(U, V, W, n, m, p, tol=0):
    """Check if the decomposition is exact: for all (i,j,k),
    sum_r U[i*m+j, r] * V[j*p+k, r] == W[k*n+i, r] ... no, the check is:
    T[a,b,c] = sum_r W[a,r] * U[b,r] * V[c,r]  should equal 1 if (a,b,c)
    corresponds to a valid (i,j,k), else 0.
    
    More precisely, the tensor reconstructed is:
      T_recon[a, b, c] = sum_r W[a, r] * U[b, r] * V[c, r]
    We need T_recon == T (the multiplication tensor) exactly."""
    # Build T as a dense array (small enough: 21*9*21 = 3969 elements)
    a_dim = n * p
    b_dim = n * m
    c_dim = m * p
    T_recon = np.zeros((a_dim, b_dim, c_dim), dtype=np.int64)
    for r in range(len(W[0])):
        T_recon += np.outer(W[:, r], np.outer(U[:, r], V[:, r])).reshape(a_dim, b_dim, c_dim)
    # Build T
    T = np.zeros_like(T_recon)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = k * n + i
                b = i * m + j
                c = j * p + k
                T[a, b, c] = 1
    return np.array_equal(T_recon, T)

def _als_decompose(T_dense, n, m, p, R, max_iter=200, tol=1e-10, seed=42):
    """ALS tensor decomposition.  T_dense shape (a_dim, b_dim, c_dim).
    Returns U (b_dim x R), V (c_dim x R), W (a_dim x R) as float arrays."""
    a_dim, b_dim, c_dim = T_dense.shape
    rng = np.random.default_rng(seed)
    # Sparse initialization might be better for integer solutions
    U = rng.standard_normal((b_dim, R))
    V = rng.standard_normal((c_dim, R))
    W = rng.standard_normal((a_dim, R))
    
    # Threshold to encourage sparsity
    sparsity_threshold = 0.5
    U[np.abs(U) < sparsity_threshold] = 0
    V[np.abs(V) < sparsity_threshold] = 0
    W[np.abs(W) < sparsity_threshold] = 0
    
    # Normalize to avoid scale drift
    for r in range(R):
        normW = np.linalg.norm(W[:, r])
        normU = np.linalg.norm(U[:, r])
        normV = np.linalg.norm(V[:, r])
        if normW > 1e-15: W[:, r] /= normW
        if normU > 1e-15: U[:, r] /= normU
        if normV > 1e-15: V[:, r] /= normV
        
    prev_err = float('inf')
    for it in range(max_iter):
        # Update U
        VW = np.einsum('ar,cr->acr', W, V)
        U_new = np.einsum('abc,acr->br', T_dense, VW)
        U = U_new
        # Update V
        WU = np.einsum('ar,br->abr', W, U)
        V_new = np.einsum('abc,abr->cr', T_dense, WU)
        V = V_new
        # Update W
        UV = np.einsum('br,cr->bcr', U, V)
        W_new = np.einsum('abc,bcr->ar', T_dense, UV)
        W = W_new
        
        # Enforce sparsity again? No, let it converge first.
        
        # Compute error
        T_recon = np.einsum('ar,br,cr->abc', W, U, V)
        err = np.linalg.norm(T_recon - T_dense)
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    return U, V, W

def _round_and_check(U, V, W, n, m, p, allow_half_int=True):
    """Round U, V, W to nearest integers (or half-integers) and check exactness."""
    if allow_half_int:
        # Scale by 2, round to int, then divide by 2
        U_scaled = np.round(U * 2).astype(np.int64)
        V_scaled = np.round(V * 2).astype(np.int64)
        W_scaled = np.round(W * 2).astype(np.int64)
        # Check with scaled values
        if _check_decomposition_scaled(U_scaled, V_scaled, W_scaled, n, m, p):
             # Return as lists of floats (x.0 or x.5) or keep as scaled ints?
             # Task says "integer/half-integer entries". 
             # Returning floats is safer for the evaluator if it expects numbers.
             return U_scaled / 2.0, V_scaled / 2.0, W_scaled / 2.0
    else:
        U_r = np.round(U).astype(np.int64)
        V_r = np.round(V).astype(np.int64)
        W_r = np.round(W).astype(np.int64)
        if _check_decomposition(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
            return U_r, V_r, W_r
    return None

def _check_decomposition_scaled(U, V, W, n, m, p):
    """Check if decomposition with scaled integers is exact.
    T[a,b,c] = sum_r W[a,r] U[b,r] V[c,r]
    If U,V,W are scaled by 2, then sum_r (2W)(2U)(2V) = 8 * T.
    So we check if sum_r W_s[a,r] U_s[b,r] V_s[c,r] == 8 * T[a,b,c].
    """
    a_dim = n * p
    b_dim = n * m
    c_dim = m * p
    T_recon = np.zeros((a_dim, b_dim, c_dim), dtype=np.int64)
    for r in range(len(W[0])):
        T_recon += np.outer(W[:, r], np.outer(U[:, r], V[:, r])).reshape(a_dim, b_dim, c_dim)
    
    # Target is 8 * T
    T = np.zeros((a_dim, b_dim, c_dim), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = k * n + i
                b = i * m + j
                c = j * p + k
                T[a, b, c] = 8 # 1 * 8
    return np.array_equal(T_recon, T)

def _local_search(U, V, W, n, m, p, max_steps=200):
    """Small local search: perturb a few entries, re-round, check."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    for _ in range(max_steps):
        # Pick a random matrix, random row, random col, add small perturbation
        mat = np.random.choice([0, 1, 2])
        if mat == 0:
            rows, cols = U.shape
        elif mat == 1:
            rows, cols = V.shape
        else:
            rows, cols = W.shape
        r = np.random.randint(rows)
        c = np.random.randint(cols)
        old = None
        if mat == 0:
            old = U[r, c]
            U[r, c] += np.random.choice([-1, 1]) * 0.5
        elif mat == 1:
            old = V[r, c]
            V[r, c] += np.random.choice([-1, 1]) * 0.5
        else:
            old = W[r, c]
            W[r, c] += np.random.choice([-1, 1]) * 0.5
        res = _round_and_check(U, V, W, n, m, p)
        if res is not None:
            return res
        # revert
        if mat == 0:
            U[r, c] = old
        elif mat == 1:
            V[r, c] = old
        else:
            W[r, c] = old
    return None

def _schoolbook(n, m, p):
    """Rank n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 135  # 135 s hard limit inside 150 s evaluation
    
    # Build the dense tensor
    a_dim = n * p
    b_dim = n * m
    c_dim = m * p
    T = np.zeros((a_dim, b_dim, c_dim), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = k * n + i
                b = i * m + j
                c = j * p + k
                T[a, b, c] = 1.0
    
    schoolbook = _schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try ranks from low to high. Focus on 59 (known bound) and nearby.
    # We allow half-integers, so we search in the scaled integer space.
    candidate_ranks = [59, 60, 61, 62]
    
    for R in candidate_ranks:
        if time.time() > deadline - 10:
            break
        # More seeds for lower ranks
        seeds = [42, 123, 7, 256, 999, 1, 2, 3, 4, 5, 6, 8, 10, 20, 50, 100]
        for seed in seeds:
            if time.time() > deadline - 10:
                break
            try:
                # Increase iterations for lower ranks
                max_iter = 200 if R <= 59 else 100
                U, V, W = _als_decompose(T, n, m, p, R, max_iter=max_iter, seed=seed)
                if time.time() > deadline - 10:
                    break
                res = _round_and_check(U, V, W, n, m, p, allow_half_int=True)
                if res is not None:
                    U_r, V_r, W_r = res
                    if R < best_rank:
                        best_rank = R
                        best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                        break  # found best for this R, move to next R
                # Try local search on the float values
                if time.time() < deadline - 20:
                    res2 = _local_search(U, V, W, n, m, p, max_steps=100)
                    if res2 is not None:
                        U_r, V_r, W_r = res2
                        if R < best_rank:
                            best_rank = R
                            best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                            break
            except Exception:
                pass
    
    return best_result
# EVOLVE-BLOCK-END
