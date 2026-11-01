import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    r = len(U[0]) if U else 0
    if r == 0:
        return False
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Compute decomposed tensor D[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    D = np.einsum('ar,br,cr->abc', Ua, Va, Wa, optimize=True)
    
    return np.allclose(D, T, atol=1e-9, rtol=1e-9)

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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

def _als_decompose(n, m, p, rank, U0, V0, W0, max_iter=500, tol=1e-12):
    """
    Alternating Least Squares for tensor decomposition.
    Given target tensor T, find U (nm x rank), V (mp x rank), W (pn x rank)
    such that sum_r U[:,r] outer V[:,r] outer W[:,r] approximates T.
    """
    U = np.array(U0, dtype=np.float64).copy()
    V = np.array(V0, dtype=np.float64).copy()
    W = np.array(W0, dtype=np.float64).copy()
    
    # Build target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    prev_error = np.inf
    for it in range(max_iter):
        # Update U: for each row a, solve (V*W^T) * U[:,r] = T[a,:,:]
        # Reshape: for fixed a, T[a,:,:] has shape (mp, pn)
        # We want: sum_r U[a,r] * V[:,r] * W[:,r]^T = T[a,:,:]
        # So: V @ diag(U[a,:]) @ W^T = T[a,:,:]
        # Equivalently: (W ⊗ I) ... let's use the standard ALS formulation
        
        # For U: fix V, W. For each a:
        #   sum_r U[a,r] * (V[:,r] ⊗ W[:,r]) = vec(T[a,:,:])
        # Let M_VW = [V[:,0]⊗W[:,0], ..., V[:,r-1]⊗W[:,r-1]] shape (mp*pn, rank)
        # Then M_VW @ U[a,:] = vec(T[a,:,:])
        
        # Precompute M_VW
        # M_VW[:, r] = V[:, r] ⊗ W[:, r]
        M_VW = np.zeros((m * p * p * n, rank), dtype=np.float64)
        for r in range(rank):
            M_VW[:, r] = np.kron(V[:, r], W[:, r])
        
        # Update U
        for a in range(n * m):
            target = T[a, :, :].flatten()
            # Solve least squares: M_VW @ U[a,:] = target
            try:
                U[a, :] = np.linalg.lstsq(M_VW, target, rcond=None)[0]
            except:
                U[a, :] = 0
        
        # For V: fix U, W. For each b:
        #   sum_r V[b,r] * (U[:,r] ⊗ W[:,r]) = vec(T[:,b,:])
        M_UW = np.zeros((n * m * p * n, rank), dtype=np.float64)
        for r in range(rank):
            M_UW[:, r] = np.kron(U[:, r], W[:, r])
        
        for b in range(m * p):
            target = T[:, b, :].flatten()
            try:
                V[b, :] = np.linalg.lstsq(M_UW, target, rcond=None)[0]
            except:
                V[b, :] = 0
        
        # For W: fix U, V. For each c:
        #   sum_r W[c,r] * (U[:,r] ⊗ V[:,r]) = vec(T[:,:,c])
        M_UV = np.zeros((n * m * m * p, rank), dtype=np.float64)
        for r in range(rank):
            M_UV[:, r] = np.kron(U[:, r], V[:, r])
        
        for c in range(p * n):
            target = T[:, :, c].flatten()
            try:
                W[c, :] = np.linalg.lstsq(M_UV, target, rcond=None)[0]
            except:
                W[c, :] = 0
        
        # Compute error
        D = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        error = np.sum((D - T) ** 2)
        
        if abs(prev_error - error) < tol * max(1.0, abs(error)):
            break
        prev_error = error
    
    return U, V, W, error

def _round_to_half_int(arr, tol=1e-6):
    """Round values to nearest half-integer if close enough."""
    rounded = np.round(arr * 2) / 2
    if np.allclose(arr, rounded, atol=tol, rtol=0):
        return rounded
    return arr

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4x5 matrix multiplication.
    
    Strategy:
    1. Start with the schoolbook decomposition (rank 40).
    2. Use ALS with multiple restarts to try to find lower-rank decompositions.
    3. Use time-bounded search.
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 40
    
    # Get the schoolbook solution as fallback
    schoolbook = _schoolbook()
    
    # Verify schoolbook solution
    if not _verify(schoolbook["U"], schoolbook["V"], schoolbook["W"]):
        raise ValueError("Schoolbook decomposition failed verification")
    
    best_solution = schoolbook
    best_rank = schoolbook_rank
    
    # Build target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Try ranks from schoolbook_rank-1 down to a reasonable minimum
    # For each rank, try multiple random restarts with ALS
    
    min_rank_to_try = max(20, schoolbook_rank - 8)  # Don't go too low
    
    for candidate_rank in range(schoolbook_rank - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline:
            break
        
        # Number of restarts depends on time available
        time_remaining = deadline - time.time()
        if time_remaining < 5:
            break
        
        # Try a few restarts
        num_restarts = min(20, max(3, int(time_remaining / 2)))
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + restart * 7919)
            
            # Initialize with random values
            U_init = np.random.randn(n * m, candidate_rank) * 0.5
            V_init = np.random.randn(m * p, candidate_rank) * 0.5
            W_init = np.random.randn(p * n, candidate_rank) * 0.5
            
            # Run ALS
            try:
                U_sol, V_sol, W_sol, error = _als_decompose(
                    n, m, p, candidate_rank, U_init, V_init, W_init,
                    max_iter=300, tol=1e-10
                )
            except Exception:
                continue
            
            if error < 1e-8:
                # Verify the solution
                D_sol = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol, optimize=True)
                if np.allclose(D_sol, T, atol=1e-8, rtol=1e-8):
                    # Try to round to half-integers for cleaner output
                    U_rounded = _round_to_half_int(U_sol)
                    V_rounded = _round_to_half_int(V_sol)
                    W_rounded = _round_to_half_int(W_sol)
                    
                    # Verify rounded version
                    D_rounded = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded, optimize=True)
                    if np.allclose(D_rounded, T, atol=1e-8, rtol=1e-8):
                        U_final, V_final, W_final = U_rounded, V_rounded, W_rounded
                    else:
                        U_final, V_final, W_final = U_sol, V_sol, W_sol
                    
                    U_list = [[float(x) for x in row] for row in U_final]
                    V_list = [[float(x) for x in row] for row in V_final]
                    W_list = [[float(x) for x in row] for row in W_final]
                    
                    best_solution = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = candidate_rank
                    break  # Found a valid solution for this rank, try lower
        
        if best_rank < schoolbook_rank:
            # Found a better solution, continue trying lower ranks
            continue
    
    # Final verification
    if not _verify(best_solution["U"], best_solution["V"], best_solution["W"]):
        # Fall back to schoolbook
        best_solution = schoolbook
        best_rank = schoolbook_rank
    
    return best_solution
# EVOLVE-BLOCK-END
