"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import numpy as np

DIMENSIONS = [2, 3, 6]
_START_TIME = time.time()

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # Safety margin before 120s evaluator kill
    n, m, p = DIMENSIONS

    # Default: schoolbook rank = n*m*p = 36
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

    best = schoolbook()
    best_rank = n * m * p

    def verify(U, V, W, rank):
        """Check if U, V, W correctly decompose the n x m x p matmul tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(rank):
                        s += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                    if abs(s - 1) > 1e-8:
                        return False
        return True

    def to_clean(mat):
        result = []
        for row in mat:
            clean_row = []
            for x in row:
                if abs(x - round(x)) < 1e-9:
                    clean_row.append(int(round(x)))
                elif abs(2 * x - round(2 * x)) < 1e-9:
                    clean_row.append(round(2 * x) / 2)
                else:
                    clean_row.append(x)
            result.append(clean_row)
        return result

    # Build the known rank-33 decomposition (3 blocks of rank-11 2x3x2)
    def build_rank33():
        rank = 33
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]

        strassen_terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [-1, 1, 0, 0]),
        ]

        U_block = [[0.0] * 11 for _ in range(6)]
        V_block = [[0.0] * 11 for _ in range(6)]
        W_block = [[0.0] * 11 for _ in range(4)]

        for t in range(7):
            ut, vt, wt = strassen_terms[t]
            for i in range(2):
                for j in range(2):
                    U_block[i * 3 + j][t] = ut[i * 2 + j]
            for j in range(2):
                for k in range(2):
                    V_block[j * 2 + k][t] = vt[j * 2 + k]
            for k in range(2):
                for i in range(2):
                    W_block[k * 2 + i][t] = wt[k * 2 + i]

        for di, (i_val, k_val) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            t = 7 + di
            U_block[i_val * 3 + 2][t] = 1
            V_block[2 * 2 + k_val][t] = 1
            W_block[k_val * 2 + i_val][t] = 1

        for block_idx in range(3):
            k_off = block_idx * 2
            for local_i in range(2):
                for local_j in range(3):
                    for t in range(11):
                        full_t = block_idx * 11 + t
                        U[local_i * 3 + local_j][full_t] = U_block[local_i * 3 + local_j][t]
            for local_j in range(3):
                for local_k in range(2):
                    for t in range(11):
                        full_t = block_idx * 11 + t
                        V[local_j * 2 + local_k][full_t] = V_block[local_j * 2 + local_k][t]
            for local_k in range(2):
                for local_i in range(2):
                    for t in range(11):
                        full_t = block_idx * 11 + t
                        W[(k_off + local_k) * 2 + local_i][full_t] = W_block[local_k * 2 + local_i][t]

        return U, V, W, rank

    # Try rank-33 first
    try:
        U33, V33, W33, r33 = build_rank33()
        if verify(U33, V33, W33, r33):
            best = {"U": to_clean(U33), "V": to_clean(V33), "W": to_clean(W33)}
            best_rank = r33
    except Exception:
        pass

    # Now try to find lower rank via structured search
    # Strategy: for each target rank from best_rank-1 down to 20,
    # try random initialization seeded from the rank-33 structure with some columns zeroed,
    # then do local search.
    
    rng = random.Random(42)
    np_rng = np.random.default_rng(42)

    # Convert rank-33 to numpy for faster operations
    U33_np, V33_np, W33_np, r33 = build_rank33()
    U33_np = np.array(U33_np, dtype=float)
    V33_np = np.array(V33_np, dtype=float)
    W33_np = np.array(W33_np, dtype=float)

    def tensor_residual(U_np, V_np, W_np, rank):
        """Compute max residual of the tensor equation."""
        max_res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(rank):
                        s += U_np[i * m + j, t] * V_np[j * p + k, t] * W_np[k * n + i, t]
                    res = abs(s - 1.0)
                    if res > max_res:
                        max_res = res
        return max_res

    def try_rank(target_rank, max_iters=200):
        """Try to find a valid decomposition of given rank."""
        if target_rank >= best_rank:
            return None
        
        # Initialize from rank-33 by taking first target_rank columns (or random subset)
        # Try several initialization strategies
        for init_strategy in range(5):
            if time.time() > deadline - 10:
                return None
            
            if init_strategy == 0:
                # Take first target_rank columns
                U_init = U33_np[:, :target_rank].copy()
                V_init = V33_np[:, :target_rank].copy()
                W_init = W33_np[:, :target_rank].copy()
            elif init_strategy == 1:
                # Take evenly spaced columns
                cols = np.linspace(0, r33 - 1, target_rank, dtype=int)
                U_init = U33_np[:, cols].copy()
                V_init = V33_np[:, cols].copy()
                W_init = W33_np[:, cols].copy()
            elif init_strategy == 2:
                # Random subset of columns
                cols = np_rng.choice(r33, size=target_rank, replace=False)
                U_init = U33_np[:, cols].copy()
                V_init = V33_np[:, cols].copy()
                W_init = W33_np[:, cols].copy()
            elif init_strategy == 3:
                # Random Gaussian initialization
                U_init = np_rng.standard_normal((n * m, target_rank)) * 0.5
                V_init = np_rng.standard_normal((m * p, target_rank)) * 0.5
                W_init = np_rng.standard_normal((p * n, target_rank)) * 0.5
            else:
                # Mix: some from rank-33, some random
                half = target_rank // 2
                cols = np_rng.choice(r33, size=half, replace=False)
                U_init = np.zeros((n * m, target_rank))
                V_init = np.zeros((m * p, target_rank))
                W_init = np.zeros((p * n, target_rank))
                U_init[:, :half] = U33_np[:, cols]
                V_init[:, :half] = V33_np[:, cols]
                W_init[:, :half] = W33_np[:, cols]
                U_init[:, half:] = np_rng.standard_normal((n * m, target_rank - half)) * 0.5
                V_init[:, half:] = np_rng.standard_normal((m * p, target_rank - half)) * 0.5
                W_init[:, half:] = np_rng.standard_normal((p * n, target_rank - half)) * 0.5

            # Local search: randomly perturb one entry at a time, accept if residual decreases
            current_res = tensor_residual(U_init, V_init, W_init, target_rank)
            
            if current_res < 1e-8:
                # Already valid!
                return (U_init.tolist(), V_init.tolist(), W_init.tolist(), target_rank)
            
            best_local_res = current_res
            best_local_U = U_init.copy()
            best_local_V = V_init.copy()
            best_local_W = W_init.copy()
            
            # Local search iterations
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                
                # Randomly pick one matrix and one entry to perturb
                mat_choice = rng.choice(['U', 'V', 'W'])
                if mat_choice == 'U':
                    row = rng.randint(0, n * m - 1)
                    col = rng.randint(0, target_rank - 1)
                    delta = rng.uniform(-0.5, 0.5)
                    old_val = U_init[row, col]
                    U_init[row, col] = old_val + delta
                    new_res = tensor_residual(U_init, V_init, W_init, target_rank)
                    if new_res < current_res:
                        current_res = new_res
                        if current_res < best_local_res:
                            best_local_res = current_res
                            best_local_U = U_init.copy()
                            best_local_V = V_init.copy()
                            best_local_W = W_init.copy()
                        if current_res < 1e-8:
                            return (U_init.tolist(), V_init.tolist(), W_init.tolist(), target_rank)
                    else:
                        U_init[row, col] = old_val
                
                elif mat_choice == 'V':
                    row = rng.randint(0, m * p - 1)
                    col = rng.randint(0, target_rank - 1)
                    delta = rng.uniform(-0.5, 0.5)
                    old_val = V_init[row, col]
                    V_init[row, col] = old_val + delta
                    new_res = tensor_residual(U_init, V_init, W_init, target_rank)
                    if new_res < current_res:
                        current_res = new_res
                        if current_res < best_local_res:
                            best_local_res = current_res
                            best_local_U = U_init.copy()
                            best_local_V = V_init.copy()
                            best_local_W = W_init.copy()
                        if current_res < 1e-8:
                            return (U_init.tolist(), V_init.tolist(), W_init.tolist(), target_rank)
                    else:
                        V_init[row, col] = old_val
                
                else:  # 'W'
                    row = rng.randint(0, p * n - 1)
                    col = rng.randint(0, target_rank - 1)
                    delta = rng.uniform(-0.5, 0.5)
                    old_val = W_init[row, col]
                    W_init[row, col] = old_val + delta
                    new_res = tensor_residual(U_init, V_init, W_init, target_rank)
                    if new_res < current_res:
                        current_res = new_res
                        if current_res < best_local_res:
                            best_local_res = current_res
                            best_local_U = U_init.copy()
                            best_local_V = V_init.copy()
                            best_local_W = W_init.copy()
                        if current_res < 1e-8:
                            return (U_init.tolist(), V_init.tolist(), W_init.tolist(), target_rank)
                    else:
                        W_init[row, col] = old_val
            
            # After local search, check if we found a valid solution
            if best_local_res < 1e-8:
                return (best_local_U.tolist(), best_local_V.tolist(), best_local_W.tolist(), target_rank)
        
        return None

    # Try ranks from best_rank-1 down to 20
    for target_rank in range(best_rank - 1, 19, -1):
        if time.time() > deadline - 15:
            break
        
        result = try_rank(target_rank, max_iters=100)
        if result is not None:
            U_list, V_list, W_list, rank = result
            # Final verification
            if verify(U_list, V_list, W_list, rank):
                best = {"U": to_clean(U_list), "V": to_clean(V_list), "W": to_clean(W_list)}
                best_rank = rank
                break  # Found the lowest rank we can get with this search

    return best
# EVOLVE-BLOCK-END
