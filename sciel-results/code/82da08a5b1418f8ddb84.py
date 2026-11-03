"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 21
    pn = p * n  # 21
    
    # Target: for each (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # For all other (u,v,w) combos, sum should be 0
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    U[u][col] = 1
                    V[v][col] = 1
                    W[w][col] = 1
                    col += 1
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = schoolbook()
    
    def verify(U, V, W, rank):
        """Check if decomposition is valid. Returns True if valid."""
        # Check target triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    s = 0.0
                    for c in range(rank):
                        s += U[u][c] * V[v][c] * W[w][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Check non-target triples
        target_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target_set.add((i * m + j, j * p + k, k * n + i))
        
        for u in range(nm):
            for v in range(mp):
                for w in range(pn):
                    if (u, v, w) not in target_set:
                        s = 0.0
                        for c in range(rank):
                            s += U[u][c] * V[v][c] * W[w][c]
                        if abs(s) > 1e-9:
                            return False
        return True
    
    def try_als(rank, n_restarts=3, max_iters=50):
        """Try to find a rank-r decomposition using ALS, then round."""
        if rank >= best_rank:
            return None
        
        for restart in range(n_restarts):
            if time.time() > deadline - 20:
                break
            
            # Random initialization
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            # ALS iterations
            for it in range(max_iters):
                if time.time() > deadline - 20:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        u_idx = i * m + j
                        # For each c, U[u_idx, c] should satisfy:
                        # sum_c U[u_idx,c] * V[v,c] * W[w,c] = 1 for target (i,j,k)
                        # This is complex; use gradient-like update
                        pass
                
                # Simplified: use coordinate descent on the error
                # Compute current error for each target triple
                # and adjust one factor at a time
                
                # Update U[u_idx, :] by solving least squares
                for i in range(n):
                    for j in range(m):
                        u_idx = i * m + j
                        # Build system: for each k, sum_c U[u_idx,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                        # This is nonlinear in U, so use linearization
                        # Fix V and W, solve for U[u_idx, :]
                        
                        # For each k, we need: sum_c U[u_idx,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
                        # Let A[c, k] = V[j*p+k,c] * W[k*n+i,c]
                        # Then U[u_idx, :] @ A = ones(p)
                        
                        A = np.zeros((rank, p))
                        for k in range(p):
                            for c in range(rank):
                                A[c, k] = V[j * p + k, c] * W[k * n + i, c]
                        
                        # Solve U[u_idx, :] @ A = ones(p)
                        # This is underdetermined if rank > p, overdetermined if rank < p
                        # Use least squares
                        try:
                            sol, residuals, rank_a, sv = np.linalg.lstsq(A.T, np.ones(p), rcond=None)
                            U[u_idx, :] = sol
                        except:
                            pass
                
                # Similarly update V
                for j in range(m):
                    for k in range(p):
                        v_idx = j * p + k
                        A = np.zeros((rank, n))
                        for i in range(n):
                            for c in range(rank):
                                A[c, i] = U[i * m + j, c] * W[k * n + i, c]
                        try:
                            sol, residuals, rank_a, sv = np.linalg.lstsq(A.T, np.ones(n), rcond=None)
                            V[v_idx, :] = sol
                        except:
                            pass
                
                # Similarly update W
                for k in range(p):
                    for i in range(n):
                        w_idx = k * n + i
                        A = np.zeros((rank, m))
                        for j in range(m):
                            for c in range(rank):
                                A[c, j] = U[i * m + j, c] * V[j * p + k, c]
                        try:
                            sol, residuals, rank_a, sv = np.linalg.lstsq(A.T, np.ones(m), rcond=None)
                            W[w_idx, :] = sol
                        except:
                            pass
            
            # After ALS, try to round to half-integers
            U_rounded = np.round(2 * U) / 2.0
            V_rounded = np.round(2 * V) / 2.0
            W_rounded = np.round(2 * W) / 2.0
            
            # Convert to list of lists
            U_list = U_rounded.tolist()
            V_list = V_rounded.tolist()
            W_list = W_rounded.tolist()
            
            # Verify
            if verify(U_list, V_list, W_list, rank):
                return U_list, V_list, W_list, rank
            
            # Try local perturbation search on rounded values
            # Try flipping some entries by ±0.5
            for _ in range(100):
                if time.time() > deadline - 20:
                    break
                
                U_try = [row[:] for row in U_list]
                V_try = [row[:] for row in V_list]
                W_try = [row[:] for row in W_list]
                
                # Randomly perturb a few entries
                for _ in range(random.randint(1, 5)):
                    choice = random.randint(0, 2)
                    if choice == 0:
                        idx = random.randint(0, nm - 1)
                        col = random.randint(0, rank - 1)
                        U_try[idx][col] += random.choice([-0.5, 0.5])
                    elif choice == 1:
                        idx = random.randint(0, mp - 1)
                        col = random.randint(0, rank - 1)
                        V_try[idx][col] += random.choice([-0.5, 0.5])
                    else:
                        idx = random.randint(0, pn - 1)
                        col = random.randint(0, rank - 1)
                        W_try[idx][col] += random.choice([-0.5, 0.5])
                
                if verify(U_try, V_try, W_try, rank):
                    return U_try, V_try, W_try, rank
        
        return None
    
    # Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = 21, but practical lower bound is higher
    # Try from 45 to 62
    for rank in range(45, min(63, best_rank)):
        if time.time() > deadline - 30:
            break
        
        result = try_als(rank)
        if result is not None:
            best_U, best_V, best_W, best_rank = result
            break
    
    # If no improvement found, return schoolbook
    if best_rank == n * m * p:
        best_U, best_V, best_W, best_rank = schoolbook()
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
