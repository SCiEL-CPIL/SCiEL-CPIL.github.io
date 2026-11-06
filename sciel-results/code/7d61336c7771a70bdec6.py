"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit

    # Precompute all arithmetic progressions of length >= 2 within [1, 256]
    ap_indices = []
    for start in range(1, n + 1):
        for step in range(1, n + 1):
            terms = list(range(start, n + 1, step))
            if len(terms) >= 2:
                ap_indices.append([t - 1 for t in terms])
    
    num_aps = len(ap_indices)
    
    # Build dense incidence matrix: M[ap_id, pos] = 1 if pos in ap
    M = np.zeros((num_aps, n), dtype=np.int8)
    for ap_id, idx in enumerate(ap_indices):
        for pos in idx:
            M[ap_id, pos] = 1
    
    # Precompute column vectors for fast delta calculation
    M_cols = [M[:, i].astype(np.int8) for i in range(n)]

    def compute_ap_sums(signs):
        """Vectorized: all AP sums via matrix-vector product."""
        return M @ signs

    def thue_morse_signs(n):
        """Generate Thue-Morse balanced signs."""
        signs = np.zeros(n, dtype=np.int8)
        for i in range(n):
            cnt = bin(i).count('1')
            signs[i] = 1 if cnt % 2 == 0 else -1
        return signs

    def local_search(signs, max_iters, deadline):
        signs = signs.copy().astype(np.int8)
        ap_sums = compute_ap_sums(signs)
        current_max = int(np.max(np.abs(ap_sums)))
        best_signs = signs.copy()
        best_max = current_max
        
        temperature = 8.0
        cooling = 0.9998
        rng = random.Random()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Decide move type: 1-flip (85%) or 2-flip (15%)
            is_2flip = rng.random() < 0.15
            
            if not is_2flip:
                # Targeted selection: 60% chance to pick from worst APs
                if rng.random() < 0.6 and current_max > 0:
                    worst_mask = np.abs(ap_sums) >= current_max - 1
                    worst_aps = np.where(worst_mask)[0]
                    if len(worst_aps) > 0:
                        target_ap = rng.choice(worst_aps.tolist())
                        positions_in_ap = np.where(M[target_ap] == 1)[0]
                        i = rng.choice(positions_in_ap.tolist())
                    else:
                        i = rng.randint(0, n - 1)
                else:
                    i = rng.randint(0, n - 1)
                
                # Single flip
                old_sign = signs[i]
                signs[i] = -old_sign
                delta_vec = -2 * old_sign * M_cols[i]
                ap_sums = ap_sums + delta_vec
                new_max = int(np.max(np.abs(ap_sums)))
                
                delta_obj = new_max - current_max
                accept = False
                if delta_obj <= 0:
                    accept = True
                elif temperature > 0.01:
                    if delta_obj < 50 * temperature:
                        accept = rng.random() < np.exp(-delta_obj / temperature)
                
                if accept:
                    current_max = new_max
                    if current_max < best_max:
                        best_max = current_max
                        best_signs = signs.copy()
                else:
                    signs[i] = old_sign
                    ap_sums = ap_sums - delta_vec
            else:
                # 2-flip move
                i1 = rng.randint(0, n - 1)
                i2 = rng.randint(0, n - 1)
                while i2 == i1:
                    i2 = rng.randint(0, n - 1)
                
                old_signs = [signs[i1], signs[i2]]
                signs[i1] = -old_signs[0]
                signs[i2] = -old_signs[1]
                delta_vec = -2 * old_signs[0] * M_cols[i1] - 2 * old_signs[1] * M_cols[i2]
                ap_sums = ap_sums + delta_vec
                new_max = int(np.max(np.abs(ap_sums)))
                
                delta_obj = new_max - current_max
                accept = False
                if delta_obj <= 0:
                    accept = True
                elif temperature > 0.01:
                    if delta_obj < 50 * temperature:
                        accept = rng.random() < np.exp(-delta_obj / temperature)
                
                if accept:
                    current_max = new_max
                    if current_max < best_max:
                        best_max = current_max
                        best_signs = signs.copy()
                else:
                    signs[i1] = old_signs[0]
                    signs[i2] = old_signs[1]
                    ap_sums = ap_sums - delta_vec
            
            temperature *= cooling
        
        return best_signs, best_max

    # Multiple restarts with different initializations
    best_overall = None
    best_overall_max = n + 1
    
    # Generate diverse balanced initializations
    initial_signs_list = []
    
    # Thue-Morse
    tm = thue_morse_signs(n)
    initial_signs_list.append(tm)
    
    # Negated Thue-Morse
    initial_signs_list.append(-tm)
    
    # Random balanced sequences with seed diversity
    seeds = [42, 123, 456, 789, 1000, 2024, 3141, 2718, 1618, 9999, 5555, 7777, 1111, 2222, 3333, 4444]
    for seed in seeds:
        rng = random.Random(seed)
        signs = np.array([rng.choice([-1, 1]) for _ in range(n)], dtype=np.int8)
        initial_signs_list.append(signs)
    
    # Structured initializations: alternating blocks
    for block_size in [4, 8, 16, 32]:
        signs = np.array([1 if (i // block_size) % 2 == 0 else -1 for i in range(n)], dtype=np.int8)
        initial_signs_list.append(signs)
    
    # More diverse structured patterns
    for period in [3, 5, 7]:
        pattern = []
        for i in range(n):
            if i % period < period // 2:
                pattern.append(1)
            else:
                pattern.append(-1)
        initial_signs_list.append(np.array(pattern, dtype=np.int8))
    
    # Run local search on each initialization
    for signs_init in initial_signs_list:
        if time.time() > deadline - 5.0:
            break
        
        signs, mx = local_search(signs_init, max_iters=200000, deadline=deadline)
        
        if mx < best_overall_max:
            best_overall_max = mx
            best_overall = signs.copy()
            if best_overall_max <= 2:
                break
    
    # Final greedy polish
    if best_overall is not None:
        signs = best_overall.copy().astype(np.int8)
        ap_sums = compute_ap_sums(signs)
        current_max = int(np.max(np.abs(ap_sums)))
        
        # Multiple greedy passes
        for pass_num in range(20):
            if time.time() > deadline - 1.0:
                break
            
            improved = True
            while improved and time.time() < deadline - 1.0:
                improved = False
                # Use different orderings for different passes
                order = list(range(n))
                if pass_num % 2 == 1:
                    order.reverse()
                else:
                    rng_order = random.Random(pass_num * 1000 + 7)
                    rng_order.shuffle(order)
                
                for i in order:
                    if time.time() > deadline - 1.0:
                        break
                    old_sign = signs[i]
                    signs[i] = -old_sign
                    delta_vec = -2 * old_sign * M_cols[i]
                    ap_sums = ap_sums + delta_vec
                    new_max = int(np.max(np.abs(ap_sums)))
                    if new_max < current_max:
                        current_max = new_max
                        improved = True
                    else:
                        signs[i] = old_sign
                        ap_sums = ap_sums - delta_vec
        
        if current_max < best_overall_max:
            best_overall_max = current_max
            best_overall = signs.copy()
    
    if best_overall is None:
        rng = random.Random(42)
        return [rng.choice([-1, 1]) for _ in range(n)]
    
    # Final verification and cleanup
    final_signs = [int(s) for s in best_overall]
    assert len(final_signs) == n
    assert all(s in [-1, 1] for s in final_signs)
    
    return final_signs
# EVOLVE-BLOCK-END
