import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    N = 256

    # Precompute all 4-term arithmetic progressions
    # A 4-AP is (a, a+d, a+2d, a+3d) with 1 <= a, a+3d <= N, d >= 1
    # Store 0-indexed positions
    ap_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            ap_list.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))

    num_aps = len(ap_list)
    aps = np.array(ap_list, dtype=np.int32)  # shape (num_aps, 4)

    # For each position, list of AP indices that contain it
    aps_at = [[] for _ in range(N)]
    for idx in range(num_aps):
        for pos in aps[idx]:
            aps_at[pos].append(idx)

    # Precompute "others" for each position: the other 3 positions in each AP containing i
    # Vectorized approach: for each position i, find all APs containing it, then mask out i
    others = [None] * N
    for i in range(N):
        if not aps_at[i]:
            others[i] = np.array([], dtype=np.int32).reshape(0, 3)
            continue
        ap_indices = np.array(aps_at[i], dtype=np.int32)
        sub_aps = aps[ap_indices]  # shape (M, 4)
        # Create mask: True where element != i
        mask = (sub_aps != i)  # shape (M, 4)
        # For each row, extract the 3 positions where mask is True
        # Since each row has exactly one False (the position i itself), we can use:
        others[i] = sub_aps[mask].reshape(len(ap_indices), 3)

    def count_mono(col):
        """Count monochromatic 4-APs given color array col (0/1)."""
        c = col[aps]  # shape (num_aps, 4)
        all_zero = np.all(c == 0, axis=1)
        all_one = np.all(c == 1, axis=1)
        return int(np.sum(all_zero) + np.sum(all_one))

    def fast_delta(col, i, new_val):
        """Compute change in mono count if we flip position i to new_val."""
        others_i = others[i]
        if others_i.size == 0:
            return 0
        current_val = col[i]
        if current_val == new_val:
            return 0
        cols_others = col[others_i]  # shape (M, 3)
        all_new = np.all(cols_others == new_val, axis=1).sum()
        all_old = np.all(cols_others == current_val, axis=1).sum()
        return int(all_new - all_old)

    def local_search(col, max_iters=500):
        """Greedy local search: flip any position if it improves the score.
        Uses random order to escape symmetric plateaus."""
        current_score = count_mono(col)
        improved = True
        iters = 0
        rng = random.Random(42)
        while improved and iters < max_iters and time.time() < deadline - 1.0:
            improved = False
            iters += 1
            order = list(range(N))
            rng.shuffle(order)
            for i in order:
                if time.time() > deadline - 1.0:
                    break
                new_val = 1 - col[i]
                delta = fast_delta(col, i, new_val)
                if delta < 0:
                    col[i] = new_val
                    current_score += delta
                    improved = True
        return col, current_score

    def sa_run(seed, max_steps, start_temp, end_temp):
        """Run simulated annealing. Returns best color array and its score."""
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)

        col = np_rng.randint(0, 2, size=N).astype(np.int32)
        current_score = count_mono(col)
        best_col = col.copy()
        best_score = current_score

        for step in range(max_steps):
            if time.time() > deadline - 2.0:
                break
            t = start_temp * (end_temp / start_temp) ** (step / max_steps)
            i = rng.randint(0, N - 1)
            new_val = 1 - col[i]
            delta = fast_delta(col, i, new_val)
            if delta <= 0:
                col[i] = new_val
                current_score += delta
                if current_score < best_score:
                    best_score = current_score
                    best_col = col.copy()
            else:
                if t > 0 and rng.random() < np.exp(-delta / t):
                    col[i] = new_val
                    current_score += delta

        return best_col, best_score

    # Generate diverse initial candidates
    def generate_candidates():
        candidates = []
        
        # Thue-Morse
        tm = np.zeros(N, dtype=np.int32)
        for i in range(N):
            tm[i] = bin(i).count('1') % 2
        candidates.append(tm.copy())
        
        # Alternating 2
        alt2 = np.array([(i // 2) % 2 for i in range(N)], dtype=np.int32)
        candidates.append(alt2.copy())
        
        # Alternating 4
        alt4 = np.array([(i // 4) % 2 for i in range(N)], dtype=np.int32)
        candidates.append(alt4.copy())
        
        # Random with various densities
        for density in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for seed in [1, 2, 3]:
                rng = np.random.RandomState(seed)
                c = (rng.random(N) < density).astype(np.int32)
                candidates.append(c.copy())
        
        # Structured: blocks of varying sizes
        for block_size in [3, 5, 7, 11]:
            c = np.zeros(N, dtype=np.int32)
            for i in range(N):
                c[i] = (i // block_size) % 2
            candidates.append(c.copy())
        
        # Walsh-like: based on bit patterns
        for bit in [0, 1, 2, 3]:
            c = np.zeros(N, dtype=np.int32)
            for i in range(N):
                c[i] = (i >> bit) % 2
            candidates.append(c.copy())
        
        # XOR of two bit positions
        for b1 in range(4):
            for b2 in range(b1+1, 4):
                c = np.zeros(N, dtype=np.int32)
                for i in range(N):
                    c[i] = ((i >> b1) ^ (i >> b2)) % 2
                candidates.append(c.copy())
        
        # Random perturbations of Thue-Morse
        for seed in [10, 20, 30]:
            rng = np.random.RandomState(seed)
            c = tm.copy()
            for _ in range(10):
                idx = rng.randint(0, N)
                c[idx] = 1 - c[idx]
            candidates.append(c.copy())
        
        # Additional: modular patterns with different moduli
        for mod in [3, 5, 7, 9]:
            c = np.zeros(N, dtype=np.int32)
            for i in range(N):
                c[i] = (i % mod) < (mod // 2)
            candidates.append(c.copy())
        
        # Additional: quadratic residue-like coloring
        for p in [3, 5, 7]:
            c = np.zeros(N, dtype=np.int32)
            for i in range(N):
                c[i] = 1 if (i * i) % p == 1 else 0
            candidates.append(c.copy())
        
        # Additional: sine-based smooth patterns
        for freq in [2, 3, 5]:
            c = np.zeros(N, dtype=np.int32)
            for i in range(N):
                c[i] = 1 if np.sin(2 * np.pi * freq * i / N) > 0 else 0
            candidates.append(c.copy())
        
        return candidates

    # Evaluate all candidates and pick the best
    best_overall = None
    best_overall_score = float('inf')
    
    candidates = generate_candidates()
    for c in candidates:
        s = count_mono(c)
        if s < best_overall_score:
            best_overall_score = s
            best_overall = c.copy()
    
    # Local search on the best candidate
    if best_overall is not None and time.time() < deadline - 1.0:
        col = best_overall.copy()
        col, final_score = local_search(col)
        if final_score < best_overall_score:
            best_overall_score = final_score
            best_overall = col
    
    # Multiple SA restarts with diverse starting points and adaptive parameters
    num_restarts = 120
    max_steps = 12000
    start_temp = 6.0
    end_temp = 0.005

    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break

        seed = 1000 + r * 7919
        # Vary temperature schedule slightly per restart for diversity
        temp_mult = 1.0 + 0.1 * (r % 5)
        best_col, best_score = sa_run(seed, max_steps, start_temp * temp_mult, end_temp)

        if best_score < best_overall_score:
            best_overall_score = best_score
            best_overall = best_col.copy()
    
    # Final local search polish on the best result found
    if best_overall is not None and time.time() < deadline - 1.0:
        col = best_overall.copy()
        col, final_score = local_search(col)
        if final_score < best_overall_score:
            best_overall_score = final_score
            best_overall = col

    return [int(c) for c in best_overall]
