"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _build_rank28():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    S = [
        ({(0,1),(3,1)}, {(0,1),(3,1)}, {(0,1),(3,1)}),
        ({(2,1),(3,1)}, {(0,1)}, {(1,1),(3,-1)}),
        ({(0,1)}, {(1,1),(3,-1)}, {(2,1),(3,1)}),
        ({(3,1)}, {(2,1),(0,-1)}, {(0,1),(1,1)}),
        ({(0,1),(1,1)}, {(3,1)}, {(0,-1),(2,1)}),
        ({(2,1),(0,-1)}, {(0,1),(1,1)}, {(3,1)}),
        ({(1,1),(3,-1)}, {(2,1),(3,1)}, {(0,1)}),
    ]
    
    j_offsets = [0, 2]
    k_offsets = [0, 2]
    
    col = 0
    for j_off in j_offsets:
        for k_off in k_offsets:
            for u_terms, v_terms, w_terms in S:
                for (li, coeff) in u_terms:
                    i, j = li // 2, li % 2
                    gi, gj = i, j + j_off
                    U[gi * m + gj][col] += coeff
                for (li, coeff) in v_terms:
                    j, k = li // 2, li % 2
                    gj, gk = j + j_off, k + k_off
                    V[gj * p + gk][col] += coeff
                for (li, coeff) in w_terms:
                    k, i = li // 2, li % 2
                    gk, gi = k + k_off, i
                    W[gk * n + gi][col] += coeff
                col += 1
    
    actual_rank = col
    U = [row[:actual_rank] for row in U]
    V = [row[:actual_rank] for row in V]
    W = [row[:actual_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n=2, m=4, p=4, tol=1e-9):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_alternating_least_squares(rank, n, m, p, max_iter=200, tol=1e-10, time_limit=30.0):
    """Try to find an exact rank decomposition using ALS with rational reconstruction."""
    start_time = time.time()
    
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We want: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    # Initialize with random values
    np.random.seed(42)
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    # Normalize to avoid numerical issues
    for r in range(rank):
        norm_u = np.linalg.norm(U[:, r])
        norm_v = np.linalg.norm(V[:, r])
        norm_w = np.linalg.norm(W[:, r])
        if norm_u > 0 and norm_v > 0 and norm_w > 0:
            U[:, r] /= norm_u
            V[:, r] /= norm_v
            W[:, r] /= norm_w
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Update U: for each (i,j), solve for U[i,j,:] given V and W
        # The constraint is: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all k
        # This is a linear system in U[i,j,:]
        
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                # Build the system: A @ u = b where u = U[i,j,:]
                # A[k, r] = V[j,k,r] * W[k,i,r]
                # b[k] = 1
                A = np.zeros((p, rank))
                b = np.ones(p)
                for k in range(p):
                    for r in range(rank):
                        A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                
                # Solve least squares
                try:
                    u, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
                    U_new[i*m+j, :] = u
                except:
                    U_new[i*m+j, :] = U[i*m+j, :]
        
        # Update V
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                b = np.ones(n)
                for i in range(n):
                    for r in range(rank):
                        A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                try:
                    v, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
                    V_new[j*p+k, :] = v
                except:
                    V_new[j*p+k, :] = V[j*p+k, :]
        
        # Update W
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                b = np.ones(m)
                for j in range(m):
                    for r in range(rank):
                        A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                try:
                    w, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
                    W_new[k*n+i, :] = w
                except:
                    W_new[k*n+i, :] = W[k*n+i, :]
        
        U, V, W = U_new, V_new, W_new
        
        # Compute error
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    error += (s - 1.0) ** 2
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_error < tol:
            break
    
    if best_error < 1e-6:
        # Try to reconstruct exact rational values
        U_exact = [[0]*rank for _ in range(n*m)]
        V_exact = [[0]*rank for _ in range(m*p)]
        W_exact = [[0]*rank for _ in range(p*n)]
        
        for i in range(n*m):
            for r in range(rank):
                val = best_U[i, r]
                frac = Fraction(val).limit_denominator(1000)
                U_exact[i][r] = float(frac)
        
        for j in range(m*p):
            for r in range(rank):
                val = best_V[j, r]
                frac = Fraction(val).limit_denominator(1000)
                V_exact[j][r] = float(frac)
        
        for k in range(p*n):
            for r in range(rank):
                val = best_W[k, r]
                frac = Fraction(val).limit_denominator(1000)
                W_exact[k][r] = float(frac)
        
        if _verify_decomposition(U_exact, V_exact, W_exact, n, m, p, tol=1e-8):
            return {"U": U_exact, "V": V_exact, "W": W_exact}
    
    return None

def construct():
    start = time.time()
    deadline = start + 110.0  # Safety margin before 150s kill
    
    # Start with the known rank-28 Strassen-based decomposition
    result = _build_rank28()
    best_rank = len(result["U"][0])
    
    # Verify the base solution
    if not _verify_decomposition(result["U"], result["V"], result["W"]):
        pass
    
    # Try to find a lower-rank decomposition
    # Search from rank 27 down to 24
    for target_rank in range(best_rank - 1, 23, -1):
        if time.time() > deadline - 10:
            break
        
        print(f"Trying rank {target_rank}...")
        solution = _try_alternating_least_squares(
            target_rank, 2, 4, 4, 
            max_iter=150, tol=1e-10, 
            time_limit=min(30.0, deadline - time.time() - 5)
        )
        
        if solution is not None:
            result = solution
            best_rank = target_rank
            print(f"Found rank {best_rank} solution!")
            break
    
    return result
# EVOLVE-BLOCK-END
