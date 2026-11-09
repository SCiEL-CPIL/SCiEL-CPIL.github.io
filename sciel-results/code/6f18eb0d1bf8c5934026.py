"""
Rank-minimal tensor decomposition for 2x3x5 matrix multiplication.

Strategy:
  The known lower bound for 2x3x5 matrix multiplication is 18 (from Strassen-like
  constructions and known results). We attempt to find a rank-18 decomposition with
  integer/half-integer entries.

  Approach:
  1. Build the 2x3x5 multiplication tensor T[i,j,k] = 1 if the (i,k)-th entry of the
     product depends on A[i,j]*B[j,k]. Actually, the tensor for matrix multiplication
     C = A*B where A is n x m and B is m x p has entries T[i,j,k] = 1 for all valid
     (i,j,k), representing C[i,k] += A[i,j]*B[j,k].
     
     More precisely, the multiplication tensor M_{n,m,p} has entries:
     M[i, j, k] = 1 for i in [0,n), j in [0,m), k in [0,p)
     and the decomposition is:
     M = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
     where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
     
     The convention: U[i*m + j, r], V[j*p + k, r], W[k*n + i, r].
     So the tensor entry at (i, j, k) is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
     This must equal 1 for all i,j,k (since every triple (i,j,k) contributes to the product).

  2. We use a structured search: start from known small-rank constructions or use
     a heuristic/optimization approach. Given the small dimensions (2,3,5), rank 18
     is the target.

  3. Use ALS (Alternating Least Squares) or a direct linear algebra approach to find
     a rank-18 decomposition, then round to integer/half-integer values and verify.

  4. If exact integer/half-integer solution isn't found, fall back to schoolbook (rank 30).

  Given the 150s deadline, we need to be efficient. Let's try:
  - First, attempt a direct construction based on known 2x3x3 and 3x3x3 results.
  - Use scipy/numpy for numerical optimization.
  - Verify with exact rational arithmetic (fractions) or by checking with integers.

  Actually, for a clean approach: use numpy to solve the system. The tensor has 2*3*5=30
  entries, each must equal 1. We have rank * (n*m + m*p + p*n) = rank * (6 + 15 + 10) = 31*rank
  unknowns. For rank=18, that's 558 unknowns and 30 equations. This is highly underdetermined,
  so solutions should exist. The challenge is finding one with small integer/half-integer entries.

  Let's use a randomized approach: initialize U, V, W randomly, then use ALS to minimize
  ||T - sum_r U_r ⊗ V_r ⊗ W_r||^2, then round and verify.

  Better: use the fact that we can fix one factor (e.g., set W to a basis-like structure)
  and solve for the others. But with rank 18 > 30 entries, we have freedom.

  Let's try: 
  - Set up the linear system for each tensor entry.
  - Use a greedy/structured initialization.
  - Use numpy lstsq to solve.
  - Round to nearest half-integer and verify.
"""

import numpy as np
import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the multiplication tensor as a 3D array T[i,j,k] = 1."""
    T = np.ones((n, m, p), dtype=np.float64)
    return T

def _tensor_to_vec(T, n, m, p):
    """Flatten tensor to vector of length n*m*p in order (i,j,k)."""
    return T.reshape(-1)

def _vec_to_tensor(v, n, m, p):
    """Reshape vector back to tensor."""
    return v.reshape((n, m, p))

def _compute_decomposition_tensor(U, V, W, n, m, p):
    """
    Compute the tensor from U (n*m x rank), V (m*p x rank), W (p*n x rank).
    Returns T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = U[:, r]  # length n*m
        v = V[:, r]  # length m*p
        w = W[:, r]  # length p*n
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u[i * m + j] * v[j * p + k] * w[k * n + i]
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Check if the decomposition exactly represents the multiplication tensor."""
    T = _compute_decomposition_tensor(U, V, W, n, m, p)
    T_expected = np.ones((n, m, p), dtype=np.float64)
    return np.allclose(T, T_expected, atol=tol, rtol=0)

def _round_to_half_integer(x):
    """Round a number to the nearest half-integer (multiple of 0.5)."""
    return round(x * 2) / 2.0

def _round_matrix_to_half_int(M):
    """Round all entries of a matrix to the nearest half-integer."""
    M_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in M])
    return M_rounded

def _try_rank(n, m, p, rank, max_iters=200, seed=42):
    """
    Attempt to find a rank-r decomposition with half-integer entries.
    Uses ALS with random initialization and rounding.
    Returns (U, V, W) if found, else None.
    """
    np.random.seed(seed)
    
    T_target = np.ones((n, m, p), dtype=np.float64)
    
    # Initialize U, V, W randomly
    U = np.random.randn(n * m, rank) * 0.5
    V = np.random.randn(m * p, rank) * 0.5
    W = np.random.randn(p * n, rank) * 0.5
    
    for it in range(max_iters):
        # ALS: update W, then V, then U
        
        # Update W: for each k*n+i, the equation involves summing over j
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For fixed i,k, we have: 1 = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This is a linear system in W[k*n+i, :]
        
        # Let's do proper ALS
        # For each (i,k), define:
        #   a_{i,k,j,r} = U[i*m+j, r] * V[j*p+k, r]
        #   Then 1 = sum_j sum_r a_{i,k,j,r} * W[k*n+i, r]
        
        # Build the linear system for W
        # Number of equations: n * p
        # Number of unknowns: p * n (each W[k*n+i, r] for r in range(rank))
        # Actually, W has p*n rows, each of length rank.
        
        # Let's reshape: for each (i,k), the unknowns are W[k*n+i, 0:rank]
        # The equation is: sum_{j=0}^{m-1} sum_{r=0}^{rank-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # This is: sum_r (sum_j U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
        
        # So for each (i,k), we have a linear equation in the vector W[k*n+i, :]
        # Coefficient for W[k*n+i, r] is: sum_j U[i*m+j, r] * V[j*p+k, r]
        
        A_W = np.zeros((n * p, rank))
        b_W = np.ones(n * p)
        idx = 0
        for i in range(n):
            for k in range(p):
                for r in range(rank):
                    coeff = 0.0
                    for j in range(m):
                        coeff += U[i * m + j, r] * V[j * p + k, r]
                    A_W[idx, r] = coeff
                idx += 1
        
        # Solve A_W * w_vec = b_W, where w_vec is the flattened W (p*n*rank)
        # But A_W is (n*p) x rank, and we want to solve for each (i,k) row of W
        # This is a least squares problem: minimize ||A_W * w - b_W||^2
        # where w has length rank, but we have n*p independent equations... 
        # Wait, each equation involves a different row of W. So we can't combine them.
        
        # Let me redo: for each (i,k), we have:
        #   A_{i,k} * W[k*n+i, :] = 1
        # where A_{i,k} is a 1 x rank vector.
        # This is underdetermined (rank unknowns, 1 equation). 
        # So we need a different approach.
        
        # Let's use a different ALS formulation.
        # Fix U and V, solve for W using least squares.
        # The full system is: for all (i,j,k), T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
        # This is a linear system in W (and similarly in U, V).
        
        # Let's build the full linear system for W.
        # Unknowns: W[k*n+i, r] for all i,k,r. Total: p*n*rank unknowns.
        # Equations: n*m*p equations.
        
        # For each (i,j,k), the equation is:
        #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # This is linear in W. Let's build the matrix.
        
        n_eq = n * m * p
        n_unk = p * n * rank
        
        A = np.zeros((n_eq, n_unk))
        b = np.ones(n_eq)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        # Unknown index for W[k*n+i, r]
                        unk_idx = (k * n + i) * rank + r
                        A[eq_idx, unk_idx] = U[i * m + j, r] * V[j * p + k, r]
                    eq_idx += 1
        
        # Solve least squares: A * w_flat = b
        w_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        W = w_flat.reshape((p * n, rank))
        
        # Similarly solve for V
        A = np.zeros((n_eq, n_unk))
        b = np.ones(n_eq)
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        # Unknown index for V[j*p+k, r]
                        unk_idx = (j * p + k) * rank + r
                        A[eq_idx, unk_idx] = U[i * m + j, r] * W[k * n + i, r]
                    eq_idx += 1
        v_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        V = v_flat.reshape((m * p, rank))
        
        # Similarly solve for U
        A = np.zeros((n_eq, n_unk))
        b = np.ones(n_eq)
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        # Unknown index for U[i*m+j, r]
                        unk_idx = (i * m + j) * rank + r
                        A[eq_idx, unk_idx] = V[j * p + k, r] * W[k * n + i, r]
                    eq_idx += 1
        u_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        U = u_flat.reshape((n * m, rank))
        
        # Check convergence
        T = _compute_decomposition_tensor(U, V, W, n, m, p)
        err = np.max(np.abs(T - T_target))
        if err < 1e-8:
            # Try rounding to half-integers
            U_r = _round_matrix_to_half_int(U)
            V_r = _round_matrix_to_half_int(V)
            W_r = _round_matrix_to_half_int(W)
            if _verify_decomposition(U_r, V_r, W_r, n, m, p, tol=1e-9):
                return (U_r, V_r, W_r)
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Try rank 18 first (theoretical lower bound)
    for rank in [18, 19, 20]:
        if time.time() > deadline:
            break
        for seed in range(10):
            if time.time() > deadline:
                break
            result = _try_rank(n, m, p, rank, max_iters=50, seed=seed)
            if result is not None:
                U, V, W = result
                # Convert to lists of lists
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
