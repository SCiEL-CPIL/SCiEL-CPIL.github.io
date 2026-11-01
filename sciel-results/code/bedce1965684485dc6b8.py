"""Matrix multiplication 2,3,7 — rank-minimal exact tensor decomposition."""
import time
import itertools
import numpy as np
from numpy.linalg import inv, solve, norm, LinAlgError

# EVOLVE-BLOCK-START

def _matmul_tensor(n, m, p):
    """Return the flat matrix-multiplication tensor of size (n*m*p,).
    Indexing convention:
        U row: i*m + j
        V row: j*p + k
        W row: k*n + i
    T[(i*m+j, j*p+k, k*n+i)] = 1
    """
    dim = n * m * p
    T = np.zeros(dim)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx * dim * dim + v_idx * dim + w_idx] = 1.0
    return T

def _decompose_to_matrices(T, U, V, W, n, m, p):
    """
    Given a target tensor T (flat) and candidate factor matrices U (nm × r),
    V (mp × r), W (pn × r), compute the residual tensor.
    Residual = T - sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    """
    r = U.shape[1]
    dim = n * m * p
    # Reconstruct the decomposed tensor
    decomposed = np.zeros(dim)
    for c in range(r):
        # Outer product of U[:,c], V[:,c], W[:,c]
        u = U[:, c]  # (nm,)
        v = V[:, c]  # (mp,)
        w = W[:, c]  # (pn,)
        # Tensor product: for each (a,b,c_idx), value = u[a]*v[b]*w[c_idx]
        # Flatten index: a * dim*dim + b * dim + c_idx
        for a in range(n * m):
            for b in range(m * p):
                for c_idx in range(p * n):
                    decomposed[a * dim * dim + b * dim + c_idx] += u[a] * v[b] * w[c_idx]
    return decomposed - T

def _rank1_decomposition(n, m, p):
    """
    For a single rank-1 tensor u ⊗ v ⊗ w, find u, v, w such that
    u[i*m+j] * v[j*p+k] * w[k*n+i] = T[i,j,k]
    This is only valid if T is actually rank-1, which it isn't for matmul.
    Instead, this is a placeholder for the schoolbook single term.
    """
    pass

def _try_rank(n, m, p, rank, T, timeout_s=100):
    """
    Attempt to find an exact decomposition of the matmul tensor with given rank.
    Returns (U, V, W) as lists of lists, or None if not found.
    Uses a combination of:
      1. Schoolbook baseline (rank = n*m*p)
      2. Random initialization + Newton's method on the tensor equation
      3. If Newton converges, verify exactness with rational rounding
    """
    start_time = time.time()
    dim = n * m * p
    num_terms = n * m * p  # number of non-zero entries in T
    
    # --- Method 1: Schoolbook (always works for rank = n*m*p) ---
    if rank == n * m * p:
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
        return U, V, W
    
    # For lower ranks, we need a smarter approach.
    # The matmul tensor T can be written as sum over (i,j,k) of e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    # A rank-r decomposition means: T = sum_{c=1}^{r} u_c ⊗ v_c ⊗ w_c
    # where u_c ∈ R^{nm}, v_c ∈ R^{mp}, w_c ∈ R^{pn}
    
    # Key insight: The matmul tensor has a special structure.
    # For 2×3×7, let's try to find a decomposition using the fact that
    # matrix multiplication can be expressed with fewer than nmp terms.
    
    # Strategy: Use the fact that if we fix the W factors, we can solve for U and V
    # from a linear system.
    #
    # T[a,b,c] = sum_{d=1}^{r} U[a,d] * V[b,d] * W[c,d]
    #
    # For fixed W, define for each c: M_c = W[c, :] (vector of length r)
    # Then T[:, :, c] = sum_d U[:,d] * V[:,d] * W[c,d]
    #                  = (U * diag(M_c) * V.T) as a matrix... no wait.
    #
    # Actually: T[a,b,c] = sum_d U[a,d] * (V[b,d] * W[c,d])
    # Let A_c[b,d] = V[b,d] * W[c,d]
    # Then T[:, :, c] = U * A_c (as matrix product? No, U is nm×r, A_c is mp×r)
    #
    # Let me think differently. Fix W. Then for each c, define:
    #   B_c = V * W[c, :]^T  ... this doesn't work dimensionally.
    #
    # Alternative: Fix W. Then the equation becomes:
    #   For each c, T[:, :, c] (an nm × mp matrix) = U * diag(W[c,:]) * V.T
    #   No, that's not right either.
    #
    # Let me be more careful:
    # T[a, b, c] = sum_{d=1}^{r} U[a, d] * V[b, d] * W[c, d]
    #
    # Fix W. Define for each c:
    #   S_c[a, b] = T[a, b, c]
    #   This must equal sum_d U[a, d] * (V[b, d] * W[c, d])
    #   Let M_c = diag(W[c, :])  (r × r diagonal)
    #   Then S_c = U * M_c * V.T  ... no:
    #   (U * M_c * V.T)[a, b] = sum_d sum_e U[a,d] M_c[d,e] V.T[e,b]
    #                           = sum_d U[a,d] W[c,d] V[b,d]
    # Yes! So S_c = U * diag(W[c,:]) * V.T
    #
    # So if we fix W, we need: for each c, S_c = U * diag(W[c,:]) * V.T
    # This is a system of p*n linear equations in the entries of U and V (with W fixed).
    # But U and V are unknowns, so it's bilinear.
    #
    # Better approach: Fix two of the three matrices, solve for the third.
    #
    # If we fix V and W, then for each (b, c):
    #   sum_d U[a, d] * V[b, d] * W[c, d] = T[a, b, c]
    #   Let q_{b,c}[d] = V[b, d] * W[c, d]
    #   Then U * q_{b,c}^T = T[:, b, c]  (vector of length nm)
    #   So U = [T[:, b, c]] * [q_{b,c}]^+  (pseudoinverse)
    #   This gives a linear system: for each (b,c), U * q_{b,c} = T[:, b, c]
    #   Stacking all (b,c) pairs: we get a system in U.
    #
    # Similarly, fix U and W, solve for V.
    # Fix U and V, solve for W.
    #
    # Alternating least squares (ALS) approach:
    #   Initialize W randomly.
    #   Solve for U (linear system, given V and W).
    #   Solve for V (linear system, given U and W).
    #   Solve for W (linear system, given U and V).
    #   Iterate until convergence.
    #   Then check if the result is exact.
    
    # ALS approach
    rng = np.random.default_rng(42)
    
    # Initialize
    U = rng.standard_normal((n * m, rank))
    V = rng.standard_normal((m * p, rank))
    W = rng.standard_normal((p * n, rank))
    
    # Precompute T as 3D array for easier access
    T3 = T.reshape(n * m, m * p, p * n)
    
    def solve_U(V, W):
        """Solve for U given V and W.
        T[a, b, c] = sum_d U[a,d] * V[b,d] * W[c,d]
        For fixed b, c: sum_d U[a,d] * (V[b,d]*W[c,d]) = T[a,b,c]
        Let q_{b,c} = V[b,:] * W[c,:]  (element-wise, length r)
        Then U * q_{b,c}^T = T[:, b, c]
        Stack over all (b, c):
        [q_{0,0}, q_{0,1}, ..., q_{mp-1,pn-1}]^T * U^T = T_flat^T
        """
        # Build the linear system: A * U_vec = T_vec
        # where A is (mp*pn) × (nm*rank)
        # A[(b,c), (a,d)] = V[b,d] * W[c,d]
        mp = m * p
        pn = p * n
        nm = n * m
        # A has shape (mp*pn, nm*rank)
        # For each (b, c), and for each (a, d):
        #   A[(b*pn + c), (a*rank + d)] = V[b, d] * W[c, d]
        # This is a Kronecker-like structure.
        # A = sum over d of kron(V[:, d] * W[:, d].T, I_nm) ... let's just build it.
        
        # More efficient: For each d, the contribution to A is:
        #   For all (b, c, a): A[(b,c), (a,d)] = V[b,d] * W[c,d]
        # This is: for each d, A_d = np.kron(W[:, d], V[:, d]) ... no.
        # Actually, A[(b,c), (a,d)] = V[b,d] * W[c,d] (independent of a for fixed d)
        # So for fixed d, the column block (all a) is the same: V[b,d]*W[c,d]
        
        # Let's build A as a sparse-like dense matrix
        # Shape: (mp * pn, nm * rank)
        A = np.zeros((mp * pn, nm * rank))
        for d in range(rank):
            # For this d, A[(b,c), (a,d)] = V[b,d] * W[c,d] for all a
            vec = np.outer(V[:, d], W[:, d])  # shape (mp, pn)
            vec_flat = vec.flatten()  # shape (mp*pn,)
            for a in range(nm):
                A[:, a * rank + d] = vec_flat
        
        T_vec = T3.reshape(-1)  # shape (nm*mp*pn,)
        # We need: for each (b,c), U[:, d] * (V[b,d]*W[c,d]) summed over d = T[:, b, c]
        # This is: U_vec @ A.T ... let me check dimensions.
        # A has shape (mp*pn, nm*rank)
        # We want: A @ U_vec = T_vec? 
        # (A @ U_vec)[(b,c)] = sum_{a,d} A[(b,c),(a,d)] * U_vec[a*rank+d]
        # = sum_{a,d} V[b,d]*W[c,d] * U[a,d]
        # But we want: sum_d U[a,d]*V[b,d]*W[c,d] for each a separately.
        # The above sums over a as well, which is wrong.
        
        # I need a different formulation. For each a, the equation is:
        # sum_d U[a,d] * (V[b,d]*W[c,d]) = T[a,b,c]
        # This is: U[a,:] @ (V[b,:]*W[c,:])^T = T[a,b,c]
        # For fixed a: U[a,:] is a row of U.
        # Let's form the system differently.
        # For each a, we have mp*pn equations:
        #   sum_d U[a,d] * V[b,d] * W[c,d] = T[a,b,c]
        # Let M_a be a matrix of shape (mp*pn, rank):
        #   M_a[(b,c), d] = V[b,d] * W[c,d]
        # Then M_a @ U[a,:] = T[a,:,:].flatten()
        # 
        # All a's are independent! So we solve rank separate least-squares problems,
        # one for each a.
        
        U_sol = np.zeros((nm, rank))
        for a in range(nm):
            # Build M_a: shape (mp*pn, rank)
            M_a = np.zeros((mp * pn, rank))
            for d in range(rank):
                M_a[:, d] = np.outer(V[:, d], W[:, d]).flatten()
            T_a = T3[a, :, :].flatten()
            try:
                U_sol[a, :] = solve(M_a, T_a)
            except LinAlgError:
                U_sol[a, :] = np.linalg.lstsq(M_a, T_a, rcond=None)[0]
        return U_sol
    
    def solve_V(U, W):
        """Solve for V given U and W.
        T[a, b, c] = sum_d U[a,d] * V[b,d] * W[c,d]
        For fixed a, c: sum_d V[b,d] * (U[a,d]*W[c,d]) = T[a,b,c]
        For each b, this is: V[b,:] @ (U[a,:]*W[c,:])^T = T[a,b,c]
        For fixed b: M_b[(a,c), d] = U[a,d]*W[c,d], M_b @ V[b,:] = T[:,b,:].flatten()
        """
        nm = n * m
        mp = m * p
        pn = p * n
        V_sol = np.zeros((mp, rank))
        for b in range(mp):
            M_b = np.zeros((nm * pn, rank))
            for d in range(rank):
                M_b[:, d] = np.outer(U[:, d], W[:, d]).flatten()
            T_b = T3[:, b, :].flatten()
            try:
                V_sol[b, :] = solve(M_b, T_b)
            except LinAlgError:
                V_sol[b, :] = np.linalg.lstsq(M_b, T_b, rcond=None)[0]
        return V_sol
    
    def solve_W(U, V):
        """Solve for W given U and V.
        T[a, b, c] = sum_d U[a,d] * V[b,d] * W[c,d]
        For fixed a, b: sum_d W[c,d] * (U[a,d]*V[b,d]) = T[a,b,c]
        For each c: M_c[(a,b), d] = U[a,d]*V[b,d], M_c @ W[c,:] = T[:,:,c].flatten()
        """
        nm = n * m
        mp = m * p
        pn = p * n
        W_sol = np.zeros((pn, rank))
        for c in range(pn):
            M_c = np.zeros((nm * mp, rank))
            for d in range(rank):
                M_c[:, d] = np.outer(U[:, d], V[:, d]).flatten()
            T_c = T3[:, :, c].flatten()
            try:
                W_sol[c, :] = solve(M_c, T_c)
            except LinAlgError:
                W_sol[c, :] = np.linalg.lstsq(M_c, T_c, rcond=None)[0]
        return W_sol
    
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    max_iters = 200
    for it in range(max_iters):
        if time.time() - start_time > timeout_s:
            break
        U = solve_U(V, W)
        V = solve_V(U, W)
        W = solve_W(U, V)
        
        # Compute error
        err = 0.0
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    val = sum(U[a, d] * V[b, d] * W[c, d] for d in range(rank))
                    err += (val - T3[a, b, c]) ** 2
        err = np.sqrt(err)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-10:
            break
    
    if best_U is not None and best_err < 1e-8:
        # Try to round to rationals (half-integers)
        U_round = np.round(best_U * 2) / 2
        V_round = np.round(best_V * 2) / 2
        W_round = np.round(best_W * 2) / 2
        
        # Verify
        err = 0.0
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    val = sum(U_round[a, d] * V_round[b, d] * W_round[c, d] for d in range(rank))
                    err += (val - T3[a, b, c]) ** 2
        if np.sqrt(err) < 1e-8:
            return (U_round.tolist(), V_round.tolist(), W_round.tolist())
    
    return None

def construct():
    n, m, p = 2, 3, 7
    start_time = time.time()
    deadline = start_time + 100  # Leave margin before 120s limit
    
    T = _matmul_tensor(n, m, p)
    
    # Try ranks from known lower bounds upward
    # Lower bound: max(nm, mp, pn, ceil(nmp / something)) 
    # For 2×3×7: nm=6, mp=21, pn=14, nmp=42
    # Strassen-type lower bounds suggest rank >= 36 or so for small matrices
    # Try a range of ranks
    
    best_result = None
    best_rank = n * m * p
    
    # Always compute schoolbook first as fallback
    schoolbook = _try_rank(n, m, p, n * m * p, T, timeout_s=10)
    if schoolbook is not None:
        best_result = schoolbook
        best_rank = n * m * p
    
    # Try lower ranks
    # Known: for 2×2×2, Strassen rank 7 < 8
    # For 2×3×7, try ranks from 30 to 41
    for rank in range(30, n * m * p):
        if time.time() > deadline:
            break
        result = _try_rank(n, m, p, rank, T, timeout_s=60)
        if result is not None:
            best_result = result
            best_rank = rank
            break  # Found a valid decomposition at this rank, stop
    
    if best_result is None:
        best_result = schoolbook
    
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
