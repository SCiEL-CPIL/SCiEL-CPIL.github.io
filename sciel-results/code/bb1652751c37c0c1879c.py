import time
import random
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1
    # Indices: i in [0, n), j in [0, m), k in [0, p)
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # Build the target tensor as a sparse dict of non-zero entries
    # T is defined by U, V, W such that T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # We want to find U, V, W such that this equals 1 for all i,j,k.
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Try to find a decomposition of rank r < rank_schoolbook
    # We search from lower ranks up, or start from schoolbook and try to reduce?
    # Searching for low rank is hard. Let's try ranks 55, 56, 57, 58, 59.
    
    # We will use a randomized greedy algorithm.
    # State: Residual tensor R (initially T)
    # At each step, pick a rank-1 tensor u v w, find best alpha, subtract alpha * u v w from R.
    # Constraint: alpha should be close to k/2 (integer or half-integer) to keep entries in that domain.
    # Actually, if we allow rational entries, we can just solve for alpha exactly.
    # But the problem requires integer/half-integer entries for U, V, W.
    # This implies alpha * u, alpha * v, alpha * w must have entries in {0, 0.5, 1, ...}.
    # If u, v, w have integer entries, alpha should be k/2.
    
    def get_target_tensor():
        # Returns a dictionary of non-zero entries (index tuple) -> value
        # Indices are (u_idx, v_idx, w_idx) corresponding to (i*m+j, j*p+k, k*n+i)
        # But it's easier to work with (i, j, k) and map to indices.
        # Let's store the residual as a dictionary mapping (i, j, k) to value.
        T = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[(i, j, k)] = 1.0
        return T

    def random_sparse_vector(length, max_nonzero=3):
        vec = [0.0] * length
        if length <= max_nonzero:
            for idx in range(length):
                vec[idx] = random.choice([-1.0, 1.0])
        else:
            indices = random.sample(range(length), max_nonzero)
            for idx in indices:
                vec[idx] = random.choice([-1.0, 1.0])
        return vec

    def try_find_decomposition(r, max_iterations=1000):
        # Returns U, V, W if found, else None
        # We store U, V, W as lists of lists (rank x dim)
        # Or we can store the vectors found so far.
        
        # To speed up, we represent the residual R as a dict of (i,j,k) -> val.
        # Initial R is T.
        
        # We will try multiple random restarts
        # Each restart:
        # 1. Initialize R = T
        # 2. For c in 1..r:
        #    - Generate random u, v, w
        #    - Calculate alpha = <R, u v w> / ||u v w||^2
        #    - Check if alpha is close to k/2. If so, round it? Or just use it?
        #      The problem requires U, V, W entries to be integer/half-integer.
        #      If we pick u, v, w with integer entries, and alpha is k/2, then the contributions are integer/half-integer.
        #      So we require alpha to be close to k/2.
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                return None
            
            R = get_target_tensor()
            U_cols = []
            V_cols = []
            W_cols = []
            
            success = True
            for c in range(r):
                # Generate random vectors
                # To improve chances, we can bias towards vectors that have non-zero entries where R is non-zero
                # But simple random might work for small r reduction.
                
                u = random_sparse_vector(n * m)
                v = random_sparse_vector(m * p)
                w = random_sparse_vector(p * n)
                
                # Calculate inner product <R, u v w>
                # R is defined on (i, j, k).
                # u[i*m+j], v[j*p+k], w[k*n+i]
                num = 0.0
                den = 0.0
                # We need to iterate over all (i,j,k) where R is non-zero?
                # Or just calculate the full inner product.
                # Since R is sparse (initially dense with 1s, but becomes sparse?), 
                # actually R starts dense. Subtracting rank-1 tensors makes it dense with small values.
                # We should iterate over all i,j,k.
                
                for i in range(n):
                    ui = u[i*m : (i+1)*m]
                    for j in range(m):
                        vj = v[j*p : (j+1)*p]
                        for k in range(p):
                            wk = w[k*n : (k+1)*n]
                            # Wait, w index is k*n + i.
                            # So w_val = w[k*n + i]
                            # u_val = u[i*m + j]
                            # v_val = v[j*p + k]
                            r_val = R.get((i, j, k), 0.0)
                            prod = u[i*m+j] * v[j*p+k] * w[k*n+i]
                            num += r_val * prod
                            den += prod * prod
                
                if den == 0:
                    continue
                
                alpha = num / den
                
                # Check if alpha is close to k/2
                # We want the resulting update to keep entries in integer/half-integer grid?
                # If u,v,w are integers, and alpha is k/2, then alpha*u is integer/half-integer.
                # So we check if alpha is close to a multiple of 0.5.
                alpha_rounded = round(alpha * 2) / 2.0
                if abs(alpha - alpha_rounded) > 1e-5:
                    # Not a valid half-integer alpha
                    # We could still use it if we allow rational U,V,W, but the constraint is on U,V,W entries.
                    # If we use alpha not in Z/2, then U,V,W entries might not be in Z/2.
                    # So we must reject this step or try to fix u,v,w.
                    # For now, reject.
                    continue
                
                # Update R
                # R = R - alpha_rounded * (u v w)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            prod = u[i*m+j] * v[j*p+k] * w[k*n+i]
                            if prod != 0:
                                current_val = R.get((i, j, k), 0.0)
                                new_val = current_val - alpha_rounded * prod
                                if abs(new_val) < 1e-9:
                                    if (i, j, k) in R:
                                        del R[(i, j, k)]
                                else:
                                    R[(i, j, k)] = new_val
                
                U_cols.append(u)
                V_cols.append(v)
                W_cols.append(w)
                
            # Check if R is empty (all zeros)
            if len(R) == 0:
                # Construct final matrices
                U = [[0.0] * r for _ in range(n * m)]
                V = [[0.0] * r for _ in range(m * p)]
                W = [[0.0] * r for _ in range(p * n)]
                
                for c in range(r):
                    for idx, val in enumerate(U_cols[c]):
                        if val != 0:
                            U[idx][c] = val
                    for idx, val in enumerate(V_cols[c]):
                        if val != 0:
                            V[idx][c] = val
                    for idx, val in enumerate(W_cols[c]):
                        if val != 0:
                            W[idx][c] = val
                
                # Final verification
                # Check a few random entries to be sure
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(r):
                                s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                            if abs(s - 1.0) > 1e-9:
                                success = False
                                break
                        if not success:
                            break
                    if not success:
                        break
                
                if success:
                    return U, V, W
        
        return None

    # Try ranks from 55 to 60
    # Start with lower ranks? Or start with 59?
    # 59 is most likely to be found quickly if it exists.
    # Let's try 59, 58, 57, 56, 55.
    
    best_result = None
    
    # We know rank 60 always works.
    # We'll generate the schoolbook solution as a fallback.
    def get_schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return U, V, W

    # Try to find a better rank
    for r in range(55, 60):
        if time.time() > deadline:
            break
        
        # Increase iterations for lower ranks?
        # For r=59, few iterations might suffice.
        # For r=55, more iterations needed.
        max_iter = 5000 if r < 58 else 2000
        
        res = try_find_decomposition(r, max_iter)
        if res:
            best_result = res
            break # Found a valid decomposition, return it
    
    if best_result:
        U, V, W = best_result
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        U, V, W = get_schoolbook()
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
