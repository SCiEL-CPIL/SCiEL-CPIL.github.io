import numpy as np
import time
import random

def construct():
    N = 256
    deadline = time.time() + 145  # safety margin before 150s kill
    
    # Precompute all 4-APs
    ap_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(max_a):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    aps = np.array(ap_list, dtype=np.int32)
    M = len(aps)
    
    # Build inverse: for each position, which APs contain it
    # aps_inv[i] = list of (ap_index, position_in_ap)
    aps_inv = [[] for _ in range(N)]
    for m in range(M):
        for j in range(4):
            aps_inv[aps[m, j]].append((m, j))
    
    def count_mono(x):
        """x is ±1 array. Count APs where product == +1 (monochromatic)."""
        X = x[aps]
        prod_all = X.prod(axis=1)
        return int(np.sum(prod_all == 1))
    
    def compute_S(x):
        """S[i] = sum over APs containing i of (product of other 3 x-values)."""
        X = x[aps]
        prod_all = X.prod(axis=1)
        other_prods = prod_all[:, None] * X  # (M, 4)
        S = np.zeros(N, dtype=np.int32)
        for j in range(4):
            np.add.at(S, aps[:, j], other_prods[:, j])
        return S
    
    def bits_to_x(bits):
        return (1 - 2 * bits).astype(np.int8)
    
    def x_to_bits(x):
        return ((1 - x) // 2).astype(np.int8)
    
    def greedy_polish(bits, max_rounds=100):
        """Flip all positions that improve the objective simultaneously."""
        x = bits_to_x(bits.copy())
        for _ in range(max_rounds):
            S = compute_S(x)
            deltas = -2 * x * S
            flip_mask = deltas < 0
            if not np.any(flip_mask):
                break
            x[flip_mask] *= -1
        return x_to_bits(x)
    
    def single_flip_sa(bits, T_start, cooling, min_T, time_limit):
        """Single-flip simulated annealing."""
        x = bits_to_x(bits.copy())
        current_obj = count_mono(x)
        best_obj = current_obj
        best_x = x.copy()
        
        T = T_start
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            # Pick random position
            i = np.random.randint(N)
            S_i = S[i] if 'S' in dir() else 0
            # Recompute S only for position i
            S_i = 0
            for (m, j) in aps_inv[i]:
                prod = 1
                for jj in range(4):
                    if jj != j:
                        prod *= x[aps[m, jj]]
                S_i += prod
            
            delta = -2 * x[i] * S_i
            
            if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
                x[i] *= -1
                current_obj += delta // 2  # approx, recompute exactly periodically
                # Actually let's just track properly
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_x = x.copy()
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        # Final exact count
        final_obj = count_mono(x)
        if final_obj < best_obj:
            best_obj = final_obj
            best_x = x.copy()
        
        return x_to_bits(best_x), best_obj
    
    def batched_sa(bits, T_start, cooling, min_T, time_limit):
        """Batched SA: flip multiple positions per step."""
        x = bits_to_x(bits.copy())
        current_obj = count_mono(x)
        best_obj = current_obj
        best_x = x.copy()
        
        T = T_start
        end_time = time.time() + time_limit
        eval_count = 0
        
        while time.time() < end_time:
            S = compute_S(x)
            deltas = -2 * x * S
            
            # Decide flips
            flip_mask = np.zeros(N, dtype=bool)
            neg_mask = deltas < 0
            flip_mask[neg_mask] = True
            
            pos_mask = deltas > 0
            if np.any(pos_mask) and T > min_T:
                probs = np.exp(-deltas[pos_mask].astype(np.float64) / T)
                rand_vals = np.random.random(np.sum(pos_mask))
                pos_indices = np.where(pos_mask)[0]
                flip_mask[pos_indices[rand_vals < probs]] = True
            
            zero_mask = (deltas == 0)
            if np.any(zero_mask):
                rand_vals = np.random.random(np.sum(zero_mask))
                zero_indices = np.where(zero_mask)[0]
                flip_mask[zero_indices[rand_vals < 0.5]] = True
            
            if np.any(flip_mask):
                x[flip_mask] *= -1
                current_obj = count_mono(x)
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_x = x.copy()
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        return x_to_bits(best_x), best_obj
    
    def hill_climbing(bits, time_limit):
        """Iterated local search: greedy + perturb."""
        x = bits_to_x(bits.copy())
        best_x = x.copy()
        best_obj = count_mono(x)
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            # Greedy polish
            for _ in range(50):
                S = compute_S(x)
                deltas = -2 * x * S
                flip_mask = deltas < 0
                if not np.any(flip_mask):
                    break
                x[flip_mask] *= -1
            
            current_obj = count_mono(x)
            if current_obj < best_obj:
                best_obj = current_obj
                best_x = x.copy()
            
            # Perturb
            num_flips = np.random.randint(3, 12)
            indices = np.random.choice(N, size=num_flips, replace=False)
            x[indices] *= -1
        
        return x_to_bits(best_x), best_obj
    
    # Global best
    global_best_bits = None
    global_best_obj = float('inf')
    
    # Multiple diversification strategies
    def update_best(bits, obj):
        nonlocal global_best_bits, global_best_obj
        if obj < global_best_obj:
            global_best_obj = obj
            global_best_bits = bits.copy()
    
    # Strategy 1: Greedy from structured starts
    structured_starts = [
        np.zeros(N, dtype=np.int8),
        np.ones(N, dtype=np.int8),
        np.array([i % 2 for i in range(N)], dtype=np.int8),
        np.array([(i // 4) % 2 for i in range(N)], dtype=np.int8),
        np.array([(i // 8) % 2 for i in range(N)], dtype=np.int8),
        np.array([(i // 16) % 2 for i in range(N)], dtype=np.int8),
        np.array([(i * 7) % 2 for i in range(N)], dtype=np.int8),
        np.array([(i * 13) % 2 for i in range(N)], dtype=np.int8),
    ]
    
    for start in structured_starts:
        if time.time() > deadline - 10:
            break
        polished = greedy_polish(start, max_rounds=200)
        obj = count_mono(bits_to_x(polished))
        update_best(polished, obj)
    
    # Strategy 2: Random starts with greedy
    for _ in range(10):
        if time.time() > deadline - 30:
            break
        rand_bits = np.random.randint(0, 2, size=N, dtype=np.int8)
        polished = greedy_polish(rand_bits, max_rounds=200)
        obj = count_mono(bits_to_x(polished))
        update_best(polished, obj)
    
    # Strategy 3: Iterated local search from best
    if global_best_bits is not None:
        remaining = deadline - time.time() - 5
        if remaining > 5:
            il_bits, il_obj = hill_climbing(global_best_bits, time_limit=min(remaining * 0.5, 30))
            update_best(il_bits, il_obj)
    
    # Strategy 4: Batched SA from best
    if global_best_bits is not None:
        remaining = deadline - time.time() - 5
        if remaining > 5:
            sa_bits, sa_obj = batched_sa(global_best_bits, T_start=3.0, cooling=0.99998, min_T=0.005, time_limit=min(remaining * 0.4, 20))
            polished = greedy_polish(sa_bits, max_rounds=200)
            obj = count_mono(bits_to_x(polished))
            update_best(polished, obj)
    
    # Strategy 5: Multiple perturbations of best with greedy
    if global_best_bits is not None:
        remaining = deadline - time.time() - 5
        if remaining > 3:
            num_attempts = min(20, int(remaining / 2))
            for _ in range(num_attempts):
                if time.time() > deadline - 3:
                    break
                pert_bits = global_best_bits.copy()
                num_flips = np.random.randint(5, 25)
                indices = np.random.choice(N, size=num_flips, replace=False)
                pert_bits[indices] = 1 - pert_bits[indices]
                polished = greedy_polish(pert_bits, max_rounds=100)
                obj = count_mono(bits_to_x(polished))
                update_best(polished, obj)
    
    # Final polish
    if global_best_bits is not None:
        final = greedy_polish(global_best_bits, max_rounds=300)
        obj = count_mono(bits_to_x(final))
        update_best(final, obj)
    
    if global_best_bits is None:
        return [0] * N
    
    return [int(b) for b in global_best_bits]
