"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]
import time
import random
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Schoolbook decomposition (rank 84) as guaranteed fallback
    def schoolbook():
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

    # Verify a decomposition: check sum_r U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r] == 1 for all i,j,k
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Fallback result
    fallback = schoolbook()
    best_rank = n * m * p
    best_result = fallback

    # Try to find lower rank decompositions using structured approaches
    # Approach: Use the fact that for each j, the "slice" T_j is an n*p x n*p tensor
    # that must be decomposable with rank <= r.
    
    # Try bilinear methods: for each j, we have a bilinear map (n x p) x (n x p) -> (n x p)
    # Actually, the matrix multiplication tensor has a special structure.
    
    # Let's try a random search with local optimization for small ranks
    # We'll use the approach of trying to find sparse decompositions
    
    def try_random_search(max_rank, num_trials, seed=None):
        """Try to find a decomposition with rank <= max_rank using random initialization + local search."""
        if seed is not None:
            random.seed(seed)
        
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            
            r = max_rank
            # Random initialization with small integer values
            U = np.random.randint(-2, 3, size=(n*m, r)).astype(float)
            V = np.random.randint(-2, 3, size=(m*p, r)).astype(float)
            W = np.random.randint(-2, 3, size=(p*n, r)).astype(float)
            
            # Local search: try to fix errors by adjusting entries
            for it in range(50):
                if time.time() > deadline:
                    break
                
                # Compute current errors
                errors = {}
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for rr in range(r):
                                s += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                            errors[(i,j,k)] = s - 1
                
                # Check if we're close to a solution
                max_err = max(abs(v) for v in errors.values()) if errors else 0
                if max_err < 1e-6:
                    # Round to nearest integers/half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    if verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), r):
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                
                # Pick a random error to fix
                idx = random.choice(list(errors.keys()))
                i, j, k = idx
                err = errors[idx]
                if abs(err) < 1e-6:
                    continue
                
                # Try to adjust one of the U, V, or W entries to reduce error
                # The contribution from term rr is U[i*m+j,rr] * V[j*p+k,rr] * W[k*n+i,rr]
                # We want to change one factor to reduce err
                
                choice = random.choice(['U', 'V', 'W'])
                rr = random.randint(0, r-1)
                
                if choice == 'U':
                    # Current contribution: U[i*m+j,rr] * V[j*p+k,rr] * W[k*n+i,rr]
                    # New value should be: U[i*m+j,rr] + delta
                    # Delta should satisfy: delta * V[j*p+k,rr] * W[k*n+i,rr] ≈ -err
                    v_val = V[j*p+k, rr]
                    w_val = W[k*n+i, rr]
                    if abs(v_val * w_val) > 1e-6:
                        delta = -err / (v_val * w_val)
                        # Try rounding delta to nearest 0.5
                        for candidate in [round(delta*2)/2, round((delta+0.25)*2)/2, round((delta-0.25)*2)/2]:
                            old = U[i*m+j, rr]
                            U[i*m+j, rr] = old + candidate
                            # Recompute error
                            s = 0
                            for rr2 in range(r):
                                s += U[i*m+j, rr2] * V[j*p+k, rr2] * W[k*n+i, rr2]
                            new_err = s - 1
                            if abs(new_err) < abs(err) * 0.5:
                                break
                            else:
                                U[i*m+j, rr] = old
                
                elif choice == 'V':
                    u_val = U[i*m+j, rr]
                    w_val = W[k*n+i, rr]
                    if abs(u_val * w_val) > 1e-6:
                        delta = -err / (u_val * w_val)
                        for candidate in [round(delta*2)/2, round((delta+0.25)*2)/2, round((delta-0.25)*2)/2]:
                            old = V[j*p+k, rr]
                            V[j*p+k, rr] = old + candidate
                            s = 0
                            for rr2 in range(r):
                                s += U[i*m+j, rr2] * V[j*p+k, rr2] * W[k*n+i, rr2]
                            new_err = s - 1
                            if abs(new_err) < abs(err) * 0.5:
                                break
                            else:
                                V[j*p+k, rr] = old
                
                else:  # W
                    u_val = U[i*m+j, rr]
                    v_val = V[j*p+k, rr]
                    if abs(u_val * v_val) > 1e-6:
                        delta = -err / (u_val * v_val)
                        for candidate in [round(delta*2)/2, round((delta+0.25)*2)/2, round((delta-0.25)*2)/2]:
                            old = W[k*n+i, rr]
                            W[k*n+i, rr] = old + candidate
                            s = 0
                            for rr2 in range(r):
                                s += U[i*m+j, rr2] * V[j*p+k, rr2] * W[k*n+i, rr2]
                            new_err = s - 1
                            if abs(new_err) < abs(err) * 0.5:
                                break
                            else:
                                W[k*n+i, rr] = old
        
        return None

    # Try decreasing ranks from a reasonable upper bound
    # Known lower bound: max(n*p, m*p, n*m) = max(21, 28, 12) = 28
    # Upper bound: n*m*p = 84
    # Try a few ranks between the lower bound and schoolbook
    
    ranks_to_try = [40, 35, 30, 28]
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        result = try_random_search(r, num_trials=5, seed=42)
        if result is not None:
            best_rank = r
            best_result = result
            break
    
    if best_rank < n * m * p:
        return best_result
    
    return fallback
# EVOLVE-BLOCK-END
