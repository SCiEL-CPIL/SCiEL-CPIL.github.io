import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    def make_rank28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        r = 0
        # Part 1: A2 B2 (j=2, k=0..4)
        for i in range(n):
            for k in range(p):
                U[i * m + 2][r] = 1
                V[2 * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
        # Part 2: A1 B13 (j=0,1, k=4)
        for i in range(n):
            for j in range(2):
                U[i * m + j][r] = 1
                V[j * p + 4][r] = 1
                W[4 * n + i][r] = 1
                r += 1
        # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen
        U[0 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 0][r] = 1; V[1 * p + 1][r] = 1; W[0 * n + 0][r] = 1; W[1 * n + 1][r] = 1; r += 1
        U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 0][r] = 1; W[0 * n + 1][r] = 1; W[1 * n + 1][r] = -1; r += 1
        U[0 * m + 0][r] = 1; V[0 * p + 1][r] = 1; V[1 * p + 1][r] = -1; W[1 * n + 0][r] = 1; W[1 * n + 1][r] = 1; r += 1
        U[1 * m + 1][r] = 1; V[1 * p + 0][r] = 1; V[0 * p + 0][r] = -1; W[0 * n + 0][r] = 1; W[0 * n + 1][r] = 1; r += 1
        U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1; V[1 * p + 1][r] = 1; W[0 * n + 0][r] = -1; W[1 * n + 0][r] = 1; r += 1
        U[1 * m + 0][r] = 1; U[0 * m + 0][r] = -1; V[0 * p + 0][r] = 1; V[0 * p + 1][r] = 1; W[1 * n + 1][r] = 1; r += 1
        U[0 * m + 1][r] = 1; U[1 * m + 1][r] = -1; V[1 * p + 0][r] = 1; V[1 * p + 1][r] = 1; W[0 * n + 0][r] = 1; r += 1
        # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen
        U[0 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 2][r] = 1; V[1 * p + 3][r] = 1; W[2 * n + 0][r] = 1; W[3 * n + 1][r] = 1; r += 1
        U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 2][r] = 1; W[2 * n + 1][r] = 1; W[3 * n + 1][r] = -1; r += 1
        U[0 * m + 0][r] = 1; V[0 * p + 3][r] = 1; V[1 * p + 3][r] = -1; W[3 * n + 0][r] = 1; W[3 * n + 1][r] = 1; r += 1
        U[1 * m + 1][r] = 1; V[1 * p + 2][r] = 1; V[0 * p + 2][r] = -1; W[2 * n + 0][r] = 1; W[2 * n + 1][r] = 1; r += 1
        U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1; V[1 * p + 3][r] = 1; W[2 * n + 0][r] = -1; W[3 * n + 0][r] = 1; r += 1
        U[1 * m + 0][r] = 1; U[0 * m + 0][r] = -1; V[0 * p + 2][r] = 1; V[0 * p + 3][r] = 1; W[3 * n + 1][r] = 1; r += 1
        U[0 * m + 1][r] = 1; U[1 * m + 1][r] = -1; V[1 * p + 2][r] = 1; V[1 * p + 3][r] = 1; W[2 * n + 0][r] = 1; r += 1
        
        return {"U": U, "V": V, "W": W}
    
    def validate(U, V, W, rank):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for rr in range(rank):
                        s += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def try_rank_reduction(base_U, base_V, base_W, base_rank, target_rank, max_iters=5000):
        """Try to reduce rank by finding a linear combination that eliminates terms."""
        U = np.array(base_U, dtype=float)
        V = np.array(base_V, dtype=float)
        W = np.array(base_W, dtype=float)
        
        # Try removing each term and see if we can compensate with others
        for remove_idx in range(base_rank):
            if time.time() > deadline:
                return None
            # Remove term remove_idx
            U_rem = np.delete(U, remove_idx, axis=1)
            V_rem = np.delete(V, remove_idx, axis=1)
            W_rem = np.delete(W, remove_idx, axis=1)
            rem_rank = base_rank - 1
            
            if rem_rank >= target_rank:
                continue
            
            # Check if we can add a single new term to fix it
            # The error is: -U[*,remove_idx] * V[*,remove_idx] * W[*,remove_idx]
            # We need to find u,v,w such that u*v*w = error_tensor
            # This is generally not possible with a single rank-1 term unless error is rank-1
            
            # Instead, try local search: perturb remaining terms
            best_err = np.inf
            best_U, best_V, best_W = None, None, None
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                # Small perturbation
                U_try = U_rem.copy()
                V_try = V_rem.copy()
                W_try = W_rem.copy()
                
                # Perturb a random subset of entries
                for _ in range(3):
                    idx_u = np.random.randint(0, n*m)
                    idx_v = np.random.randint(0, m*p)
                    idx_w = np.random.randint(0, p*n)
                    col_u = np.random.randint(0, rem_rank)
                    col_v = np.random.randint(0, rem_rank)
                    col_w = np.random.randint(0, rem_rank)
                    
                    U_try[idx_u, col_u] += np.random.choice([-1, -0.5, 0.5, 1])
                    V_try[idx_v, col_v] += np.random.choice([-1, -0.5, 0.5, 1])
                    W_try[idx_w, col_w] += np.random.choice([-1, -0.5, 0.5, 1])
                
                # Validate
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for rr in range(rem_rank):
                                s += U_try[i*m+j, rr] * V_try[j*p+k, rr] * W_try[k*n+i, rr]
                            if abs(s - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    return {"U": U_try.tolist(), "V": V_try.tolist(), "W": W_try.tolist()}
        
        return None
    
    best = make_rank28()
    best_rank = 28
    
    # Try to reduce from rank 28 to 27
    for target_rank in range(27, best_rank):
        if time.time() > deadline:
            break
        result = try_rank_reduction(
            best["U"], best["V"], best["W"], best_rank, target_rank, max_iters=2000
        )
        if result is not None:
            best = result
            best_rank = target_rank
    
    # Fallback: quick random search for rank 27
    if best_rank > 27 and time.time() < deadline - 10:
        vals = [-1, -0.5, 0, 0.5, 1]
        for attempt in range(100):
            if time.time() > deadline:
                break
            U = np.random.choice(vals, size=(n*m, 27))
            V = np.random.choice(vals, size=(m*p, 27))
            W = np.random.choice(vals, size=(p*n, 27))
            if validate(U, V, W, 27):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = 27
                break
    
    return best
# EVOLVE-BLOCK-END
