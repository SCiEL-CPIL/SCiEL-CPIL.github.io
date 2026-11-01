"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120.0  # safety margin before 150s kill
    
    # Build schoolbook decomposition
    def build_schoolbook():
        terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        rank = len(terms)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for c, (i, j, k) in enumerate(terms):
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        return U, V, W

    # Verify decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    def verify(U, V, W, tol=1e-9):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    total = np.sum(u_val * v_val * w_val)
                    if abs(total - 1.0) > tol:
                        return False
        return True

    # Compute tensor error for a given decomposition
    def tensor_error(U, V, W):
        rank = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    total = np.sum(u_val * v_val * w_val)
                    err = abs(total - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    # Start with schoolbook
    U, V, W = build_schoolbook()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    if not verify(best_U, best_V, best_W):
        raise ValueError("Schoolbook decomposition failed verification")
    
    # Try to find rank-41 decomposition
    # Strategy: Try random rank-41 decompositions with small integer/half-integer entries
    # and use local search to minimize tensor error
    
    np.random.seed(42)
    
    # Candidate values for entries
    candidate_vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    # Try multiple random initializations for rank 41
    rank_to_try = best_rank - 1  # 41
    found_lower = False
    
    for attempt in range(200):
        if time.time() > deadline - 10:
            break
        
        # Random initialization with small values
        U_rand = np.random.choice(candidate_vals, size=(n*m, rank_to_try))
        V_rand = np.random.choice(candidate_vals, size=(m*p, rank_to_try))
        W_rand = np.random.choice(candidate_vals, size=(p*n, rank_to_try))
        
        # Local search: try to reduce error by adjusting individual entries
        best_err = tensor_error(U_rand, V_rand, W_rand)
        
        # If error is already small, we might have a solution
        if best_err < 1e-9:
            if verify(U_rand, V_rand, W_rand):
                best_rank = rank_to_try
                best_U, best_V, best_W = U_rand.copy(), V_rand.copy(), W_rand.copy()
                found_lower = True
                break
        
        # Try a few iterations of local search
        for it in range(50):
            if time.time() > deadline - 10:
                    break
            
            # Pick a random element to adjust
            mat = np.random.choice(['U', 'V', 'W'])
            if mat == 'U':
                row, col = np.random.randint(0, n*m), np.random.randint(0, rank_to_try)
                old_val = U_rand[row, col]
                new_val = np.random.choice(candidate_vals)
                if new_val == old_val:
                    continue
                U_rand[row, col] = new_val
                new_err = tensor_error(U_rand, V_rand, W_rand)
                if new_err < best_err - 1e-12:
                    best_err = new_err
                else:
                    U_rand[row, col] = old_val
            elif mat == 'V':
                row, col = np.random.randint(0, m*p), np.random.randint(0, rank_to_try)
                old_val = V_rand[row, col]
                new_val = np.random.choice(candidate_vals)
                if new_val == old_val:
                    continue
                V_rand[row, col] = new_val
                new_err = tensor_error(U_rand, V_rand, W_rand)
                if new_err < best_err - 1e-12:
                    best_err = new_err
                else:
                    V_rand[row, col] = old_val
            else:
                row, col = np.random.randint(0, p*n), np.random.randint(0, rank_to_try)
                old_val = W_rand[row, col]
                new_val = np.random.choice(candidate_vals)
                if new_val == old_val:
                    continue
                W_rand[row, col] = new_val
                new_err = tensor_error(U_rand, V_rand, W_rand)
                if new_err < best_err - 1e-12:
                    best_err = new_err
                else:
                    W_rand[row, col] = old_val
        
        if best_err < 1e-9:
            if verify(U_rand, V_rand, W_rand):
                best_rank = rank_to_try
                best_U, best_V, best_W = U_rand.copy(), V_rand.copy(), W_rand.copy()
                found_lower = True
                break
    
    # If we found a lower rank, try to go even lower
    if found_lower and best_rank > 30:
        for target_rank in range(best_rank - 1, max(20, best_rank - 5), -1):
            if time.time() > deadline - 10:
                break
            
            for attempt in range(50):
                if time.time() > deadline - 10:
                    break
                
                U_rand = np.random.choice(candidate_vals, size=(n*m, target_rank))
                V_rand = np.random.choice(candidate_vals, size=(m*p, target_rank))
                W_rand = np.random.choice(candidate_vals, size=(p*n, target_rank))
                
                best_err = tensor_error(U_rand, V_rand, W_rand)
                
                if best_err < 1e-9:
                    if verify(U_rand, V_rand, W_rand):
                        best_rank = target_rank
                        best_U, best_V, best_W = U_rand.copy(), V_rand.copy(), W_rand.copy()
                        break
                
                for it in range(30):
                    if time.time() > deadline - 10:
                        break
                    
                    mat = np.random.choice(['U', 'V', 'W'])
                    if mat == 'U':
                        row, col = np.random.randint(0, n*m), np.random.randint(0, target_rank)
                        old_val = U_rand[row, col]
                        new_val = np.random.choice(candidate_vals)
                        if new_val == old_val:
                            continue
                        U_rand[row, col] = new_val
                        new_err = tensor_error(U_rand, V_rand, W_rand)
                        if new_err < best_err - 1e-12:
                            best_err = new_err
                        else:
                            U_rand[row, col] = old_val
                    elif mat == 'V':
                        row, col = np.random.randint(0, m*p), np.random.randint(0, target_rank)
                        old_val = V_rand[row, col]
                        new_val = np.random.choice(candidate_vals)
                        if new_val == old_val:
                            continue
                        V_rand[row, col] = new_val
                        new_err = tensor_error(U_rand, V_rand, W_rand)
                        if new_err < best_err - 1e-12:
                            best_err = new_err
                        else:
                            V_rand[row, col] = old_val
                    else:
                        row, col = np.random.randint(0, p*n), np.random.randint(0, target_rank)
                        old_val = W_rand[row, col]
                        new_val = np.random.choice(candidate_vals)
                        if new_val == old_val:
                            continue
                        W_rand[row, col] = new_val
                        new_err = tensor_error(U_rand, V_rand, W_rand)
                        if new_err < best_err - 1e-12:
                            best_err = new_err
                        else:
                            W_rand[row, col] = old_val
                
                if best_err < 1e-9:
                    if verify(U_rand, V_rand, W_rand):
                        best_rank = target_rank
                        best_U, best_V, best_W = U_rand.copy(), V_rand.copy(), W_rand.copy()
                        break
    
    # Final verification
    if not verify(best_U, best_V, best_W):
        # Fall back to schoolbook
        best_U, best_V, best_W = build_schoolbook()
        best_rank = n * m * p
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
