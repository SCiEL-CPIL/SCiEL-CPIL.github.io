import numpy as np
import time
import random

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 9
    dim_w = p * n  # 6

    # Build target tensor T
    # T[a, b, c] = 1 if c = k*n + i and a = i*m + j and b = j*p + k
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    start_time = time.time()
    deadline = 140.0  # Leave 10s margin

    # Schoolbook rank 18 as fallback
    rank_fb = n * m * p
    U_fb = np.zeros((dim_u, rank_fb), dtype=np.int64)
    V_fb = np.zeros((dim_v, rank_fb), dtype=np.int64)
    W_fb = np.zeros((dim_w, rank_fb), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 2 # 1.0 * 2
                V_fb[j * p + k, col] = 2 # 1.0 * 2
                W_fb[k * n + i, col] = 2 # 1.0 * 2
                col += 1

    def compute_error(U, V, W):
        # U, V, W are int64 arrays (scaled by 2)
        # T_recon = sum_r U[a,r] V[b,r] W[c,r]
        # We want T_recon == 4 * T (since 2*2*2 = 8? No. 
        # If U,V,W are half-integers, let U=U_int/2.
        # Product is U_int V_int W_int / 8.
        # We want U_int V_int W_int / 8 == T.
        # So U_int V_int W_int == 8 T.
        # Wait, the seed program used:
        # T_recon = einsum(U_int, V_int, W_int)
        # return np.all(T_recon == 8 * T)
        # This implies U_int, V_int, W_int are the scaled values.
        # Let's stick to that.
        
        T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        # Error is the sum of squares of differences
        # We use integer arithmetic to avoid float issues
        diff = T_recon - 8 * T
        return np.sum(diff * diff)

    def try_discrete_search(r, max_restarts=50, max_iters_per_restart=20000):
        if time.time() > deadline - 5:
            return None

        # Target error is 0
        # We search in the space of integers for U, V, W (scaled by 2)
        # Entries are likely small. Let's restrict to [-4, 4] initially.
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break

            # Random initialization with small integers
            U = np.random.randint(-2, 3, size=(dim_u, r))
            V = np.random.randint(-2, 3, size=(dim_v, r))
            W = np.random.randint(-2, 3, size=(dim_w, r))
            
            # Ensure at least some non-zeros
            # If all zero, error is high.
            
            current_error = compute_error(U, V, W)
            best_error = current_error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

            # Simulated Annealing parameters
            T_temp = 100.0
            T_min = 0.1
            cooling = 0.999

            for it in range(max_iters_per_restart):
                if time.time() > deadline - 5:
                    return None # Abort this restart to check time
                
                # Pick a random matrix and entry to perturb
                mat_idx = random.choice([0, 1, 2])
                if mat_idx == 0:
                    i, j = np.random.randint(0, dim_u), np.random.randint(0, r)
                    old_val = U[i, j]
                    delta = random.choice([-2, -1, 1, 2])
                    new_val = old_val + delta
                    # Keep values bounded
                    if abs(new_val) > 10: continue
                    U[i, j] = new_val
                    new_error = compute_error(U, V, W)
                    if new_error > current_error:
                        # Revert if too bad? Or accept with prob?
                        if random.random() > np.exp(-(new_error - current_error) / T_temp):
                            U[i, j] = old_val
                    else:
                        current_error = new_error
                        if new_error < best_error:
                            best_error = new_error
                            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                            if best_error == 0:
                                return best_U, best_V, best_W
                elif mat_idx == 1:
                    i, j = np.random.randint(0, dim_v), np.random.randint(0, r)
                    old_val = V[i, j]
                    delta = random.choice([-2, -1, 1, 2])
                    new_val = old_val + delta
                    if abs(new_val) > 10: continue
                    V[i, j] = new_val
                    new_error = compute_error(U, V, W)
                    if new_error > current_error:
                        if random.random() > np.exp(-(new_error - current_error) / T_temp):
                            V[i, j] = old_val
                    else:
                        current_error = new_error
                        if new_error < best_error:
                            best_error = new_error
                            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                            if best_error == 0:
                                return best_U, best_V, best_W
                else:
                    i, j = np.random.randint(0, dim_w), np.random.randint(0, r)
                    old_val = W[i, j]
                    delta = random.choice([-2, -1, 1, 2])
                    new_val = old_val + delta
                    if abs(new_val) > 10: continue
                    W[i, j] = new_val
                    new_error = compute_error(U, V, W)
                    if new_error > current_error:
                        if random.random() > np.exp(-(new_error - current_error) / T_temp):
                            W[i, j] = old_val
                    else:
                        current_error = new_error
                        if new_error < best_error:
                            best_error = new_error
                            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                            if best_error == 0:
                                return best_U, best_V, best_W

                T_temp *= cooling
                if T_temp < T_min:
                    break

            if best_error == 0:
                return best_U, best_V, best_W

        return None

    # Try ranks 15, 16, 17
    for r in [15, 16, 17]:
        if time.time() > deadline - 10:
            break
        res = try_discrete_search(r)
        if res:
            U_int, V_int, W_int = res
            # Convert back to half-integers
            U = U_int / 2.0
            V = V_int / 2.0
            W = W_int / 2.0
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to rank 18
    U = U_fb / 2.0
    V = V_fb / 2.0
    W = W_fb / 2.0
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
