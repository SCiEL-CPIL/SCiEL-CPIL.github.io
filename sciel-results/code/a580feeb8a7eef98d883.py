"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if (i,j,k) is a valid product index."""
    # T has shape (n*m, m*p, p*n) where T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _check_decomposition(U, V, W, T, tol=1e-9):
    """Check if U, V, W exactly decompose T."""
    recon = np.einsum('ac,bc,dc->abd', U, V, W)
    err = np.sum((recon - T) ** 2)
    return err < tol

def _snap_matrix(M, tol=1e-6):
    """Snap matrix entries to nearest integer or half-integer."""
    # Try integers first
    M_int = np.round(M)
    if np.allclose(M, M_int, atol=tol):
        return M_int
    
    # Try half-integers
    M_half = np.round(M * 2) / 2.0
    if np.allclose(M, M_half, atol=tol):
        return M_half
        
    return None

def _search_rank(n, m, p, rank, T, time_budget):
    """
    Search for a rank-rank decomposition.
    Strategy: Randomly initialize U and V with sparse integer/half-integer values.
    For each k, solve the linear system for W_k.
    Check if W can be snapped to integers/halves and if the full reconstruction works.
    """
    start_time = time.time()
    np.random.seed(42) # Deterministic for reproducibility in search, but we vary invariants
    
    # Values for U and V entries
    vals = [0, 1, -1, 0.5, -0.5]
    
    max_iters = 10000 # Will be limited by time
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        # 1. Randomly initialize U and V
        # Make them sparse to help with integer snapping
        density = 0.3 + 0.4 * np.random.rand()
        
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        
        for i in range(n * m):
            for r in range(rank):
                if np.random.rand() < density:
                    U[i, r] = np.random.choice(vals)
        
        for j in range(m * p):
            for r in range(rank):
                if np.random.rand() < density:
                    V[j, r] = np.random.choice(vals)
        
        # 2. For each k (output index), solve for W[:, k]
        # Equation: sum_r U[i, r] V[j, r] W[k, r] = T[i, j, k]
        # Let A be matrix where A[i, r] = U[i, r] * V[j, r] ? No, A depends on i and j.
        # We have a system for each k:
        # sum_r W[k, r] * (U[:, r] * V[:, r]) = T[:, :, k] (vectorized)
        
        W = np.zeros((p * n, rank))
        valid = True
        
        for k in range(p):
            # Build the linear system matrix for this k
            # Variables are W[k, 0..rank-1]
            # Equations are for each i, j
            A_k = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    idx = i * m + j
                    for r in range(rank):
                        A_k[idx, r] = U[i * m + j, r] * V[j * p + k, r]
            
            # Target vector
            b_k = np.zeros(n * m)
            for i in range(n):
                for j in range(m):
                    idx = i * m + j
                    b_k[idx] = T[i * m + j, j * p + k, k * n + i] # Wait, T index is (a, b, c)
                    # a = i*m+j
                    # b = j*p+k
                    # c = k*n+i
                    # So T[a, b, c] is the value.
            
            # Solve A_k @ W_k = b_k
            # Since rank > n*m (19 > 6), the system is underdetermined.
            # We use least squares, but we need an exact solution.
            # If the system is consistent, np.linalg.lstsq will find a solution with min norm.
            # However, if it's not consistent, it finds the closest.
            # We check consistency by residual.
            
            try:
                sol, residuals, rank_k, sv = np.linalg.lstsq(A_k, b_k, rcond=None)
                # Check if residual is small
                if residuals and residuals[0] > 1e-9:
                    valid = False
                    break
                W[k * n : (k+1) * n, :] = sol # Wait, W shape is (p*n, rank).
                # The index for W in the sum is c = k*n + i.
                # So W[c, r] is the coefficient.
                # My loop over k corresponds to the 'k' in the product index c = k*n + i.
                # So for a fixed k, we are solving for the row block of W?
                # No, W is indexed by c (which is k*n+i) and r.
                # The equation is sum_r U[i,r] V[j,r] W[c,r] = T[i,j,k] where c=k*n+i.
                # So for a fixed k, we have equations for all i, j.
                # The unknowns are W[c, r] for all i (so c varies with i) and r.
                # This couples the W rows for different i's of the same k.
                
                # Let's re-formulate.
                # For a fixed k, we have n*m equations.
                # Unknowns: W[k*n + i, r] for i in 0..n-1, r in 0..rank-1. Total n*rank unknowns.
                # Equations: for each i, j: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = T[i*m+j, j*p+k, k*n+i]
                
                # This is a block system.
                # Let's build the full A_k for the block of W rows corresponding to k.
                # A_k will be size (n*m) x (n*rank)
                
                A_k_block = np.zeros((n * m, n * rank))
                for i in range(n):
                    for j in range(m):
                        eq_idx = i * m + j
                        for ii in range(n):
                            for r in range(rank):
                                # Coefficient of W[k*n + ii, r]
                                # It appears in equation (i, j) only if ii == i?
                                # No, W is indexed by c = k*n + i.
                                # In the sum for T[i, j, k], the term is U[i, r] V[j, r] W[k*n + i, r].
                                # So W[k*n + i, r] only appears in the equation for (i, j, k) where the first index is i.
                                # So the system is block-diagonal with respect to i?
                                # Yes! For each i, we have m equations for the unknowns W[k*n + i, 0..rank-1].
                                # So we can solve for each i separately.
                                
                                pass # Logic corrected below
                
                valid = True
                for i in range(n):
                    # Solve for W[k*n + i, :]
                    A_i = np.zeros((m, rank))
                    b_i = np.zeros(m)
                    for j in range(m):
                        for r in range(rank):
                            A_i[j, r] = U[i * m + j, r] * V[j * p + k, r]
                        b_i[j] = T[i * m + j, j * p + k, k * n + i]
                    
                    sol_i, res_i, rank_i, sv_i = np.linalg.lstsq(A_i, b_i, rcond=None)
                    if res_i and res_i[0] > 1e-9:
                        valid = False
                        break
                    W[k * n + i, :] = sol_i
                
                if not valid:
                    break
                    
            except Exception:
                valid = False
                break
        
        if not valid:
            continue
            
        # 3. Snap W to integers/halves
        W_snap = _snap_matrix(W)
        if W_snap is None:
            continue
            
        # 4. Snap U and V to integers/halves (they were already chosen from vals, so they are exact)
        # But let's ensure they are clean
        U_clean = np.round(U * 2) / 2.0
        V_clean = np.round(V * 2) / 2.0
        
        # 5. Verify full decomposition
        if _check_decomposition(U_clean, V_clean, W_snap, T):
            return U_clean, V_clean, W_snap
            
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Deadline guard: must finish within 150s, use 140s budget
    total_start = time.time()
    time_budget = 140.0
    
    # Start with schoolbook rank 24 as fallback
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    # Try ranks from 19 up to 23
    # We know rank 19 is possible for 2x3x4
    for rank in [19, 20, 21, 22, 23]:
        if rank > rank_fb:
            continue
            
        elapsed = time.time() - total_start
        remaining = time_budget - elapsed
        if remaining < 2.0:
            break
        
        # Allocate time: give most to 19
        if rank == 19:
            rank_budget = remaining * 0.8
        else:
            rank_budget = remaining * 0.1
            
        result = _search_rank(n, m, p, rank, T, rank_budget)
        
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback to schoolbook
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
