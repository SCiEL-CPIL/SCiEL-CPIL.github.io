import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Try to find a lower rank solution
    start_time = time.time()
    deadline = start_time + 100  # 100s limit to be safe
    
    # Precompute valid and invalid indices
    # Valid: (i, j, k) -> idx_u = i*m+j, idx_v = j*p+k, idx_w = k*n+i
    valid_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid_indices.append((i * m + j, j * p + k, k * n + i))
                
    # Invalid indices are all other combinations of (idx_u, idx_v, idx_w)
    # Total combinations: (n*m) * (m*p) * (p*n)
    # We only need to check a subset or use the full set if fast enough
    # 9 * 21 * 21 = 3969. This is small.
    all_indices = []
    for u in range(n * m):
        for v in range(m * p):
            for w in range(p * n):
                all_indices.append((u, v, w))
                
    valid_set = set(valid_indices)
    invalid_indices = [t for t in all_indices if t not in valid_set]

    def check_solution(U, V, W):
        # U, V, W are numpy arrays of shape (dim, R)
        # Calculate S(u, v, w) = sum_r U[u, r] V[v, r] W[w, r]
        # We need S = 8 for valid, 0 for invalid (scaled by 2)
        # Use matrix multiplication for speed?
        # S = U^T V * W? No.
        # S(u, v, w) = sum_r U[u, r] V[v, r] W[w, r]
        # This is a tensor contraction.
        # For small R, loop is fine.
        # But we have 3969 constraints.
        # R is ~30-60.
        # 3969 * 60 = 238k ops. Very fast.
        
        # Efficient check:
        # Compute M = U^T V (R x R? No, R x 1? No)
        # U is (9, R), V is (21, R), W is (21, R).
        # S(u, v, w) = sum_r U[u, r] V[v, r] W[w, r]
        
        # We can compute all S values at once?
        # Not easily with standard BLAS without expanding.
        # Loop over constraints is probably fast enough in Python if vectorized per constraint?
        # No, loop over constraints in Python is slow.
        # Vectorize:
        # For each r, add outer product? No.
        
        # Let's just loop over valid and invalid in Python, but use numpy dot products?
        # Or simply:
        # S = np.einsum('ur,vr,wr->uvw', U, V, W)
        # This creates a (9, 21, 21) array. Size 3969.
        # This is very fast.
        
        S = np.einsum('ur,vr,wr->uvw', U, V, W)
        
        # Check valid
        for u, v, w in valid_indices:
            if S[u, v, w] != 8:
                return False
        # Check invalid
        # We can check all invalid at once
        # Create a mask
        mask = np.ones((n*m, m*p, p*n), dtype=bool)
        for u, v, w in valid_indices:
            mask[u, v, w] = False
        if not np.all(S[mask] == 0):
            return False
            
        return True

    # Search for R
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from low to high
    for R in range(25, best_rank):
        if time.time() > deadline:
            break
            
        # Try a few random restarts
        found = False
        for restart in range(20):
            if time.time() > deadline:
                break
                
            # Initialize with random integers in {-2, 0, 2} (scaled by 2, so values are -1, 0, 1)
            # Actually, let's use {-1, 0, 1} for the scaled variables (so original are -0.5, 0, 0.5)
            # Or {-2, -1, 0, 1, 2}
            U = np.random.randint(-2, 3, (n * m, R))
            V = np.random.randint(-2, 3, (m * p, R))
            W = np.random.randint(-2, 3, (p * n, R))
            
            # Local search
            # Error function: sum (S_valid - 8)^2 + sum (S_invalid)^2
            def get_error(U, V, W):
                S = np.einsum('ur,vr,wr->uvw', U, V, W)
                err = 0
                for u, v, w in valid_indices:
                    err += (S[u, v, w] - 8) ** 2
                # Invalid
                # To speed up, we can just check a random subset of invalids?
                # No, we need 0.
                # Use the mask
                mask = np.ones((n*m, m*p, p*n), dtype=bool)
                for u, v, w in valid_indices:
                    mask[u, v, w] = False
                err += np.sum(S[mask] ** 2)
                return err

            current_error = get_error(U, V, W)
            
            # Simple hill climbing
            for step in range(500):
                if time.time() > deadline:
                    break
                if current_error == 0:
                    found = True
                    break
                    
                # Pick random element to change
                mat = np.random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    idx = np.random.randint(0, n * m)
                    r = np.random.randint(0, R)
                    old_val = U[idx, r]
                    U[idx, r] += np.random.choice([-1, 1])
                elif mat == 'V':
                    idx = np.random.randint(0, m * p)
                    r = np.random.randint(0, R)
                    old_val = V[idx, r]
                    V[idx, r] += np.random.choice([-1, 1])
                else:
                    idx = np.random.randint(0, p * n)
                    r = np.random.randint(0, R)
                    old_val = W[idx, r]
                    W[idx, r] += np.random.choice([-1, 1])
                    
                new_error = get_error(U, V, W)
                if new_error < current_error:
                    current_error = new_error
                else:
                    # Revert
                    if mat == 'U':
                        U[idx, r] = old_val
                    elif mat == 'V':
                        V[idx, r] = old_val
                    else:
                        W[idx, r] = old_val

            if found:
                best_rank = R
                best_U = U
                best_V = V
                best_W = W
                break

    if best_U is not None:
        # Convert to half-integers (divide by 2)
        U_out = (best_U / 2.0).tolist()
        V_out = (best_V / 2.0).tolist()
        W_out = (best_W / 2.0).tolist()
        return {"U": U_out, "V": V_out, "W": W_out}
    else:
        return construct() # Fallback to schoolbook
# EVOLVE-BLOCK-END
