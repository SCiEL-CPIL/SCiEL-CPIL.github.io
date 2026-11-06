import numpy as np
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 146.0  # Safety margin from 150s

    N = 256

    # Precompute all 4-term arithmetic progressions (0-indexed)
    ap_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            ap_list.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))

    aps = np.array(ap_list, dtype=np.int32)
    num_aps = len(aps)

    # For each position, precompute the other 3 positions in each AP containing it
    aps_at = [[] for _ in range(N)]
    for idx in range(num_aps):
        for pos in aps[idx]:
            aps_at[pos].append(idx)

    # Build others_for_pos: for position i, array of shape (num_aps_containing_i, 3)
    others_for_pos = [None] * N
    for i in range(N):
        if not aps_at[i]:
            others_for_pos[i] = np.array([], dtype=np.int32).reshape(0, 3)
        else:
            others_list = []
            for ap_idx in aps_at[i]:
                row = aps[ap_idx]
                others = [p for p in row if p != i]
                others_list.append(others)
            others_for_pos[i] = np.array(others_list, dtype=np.int32)

    def count_mono(col):
        c = col[aps]
        return int(np.sum((c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])))

    def fast_delta_vectorized(col):
        """Compute delta for all positions at once. Returns array of deltas."""
        deltas = np.zeros(N, dtype=np.int32)
        for i in range(N):
            others = others_for_pos[i]
            if others.shape[0] == 0:
                continue
            oc = col[others]
            all_same = (oc[:, 0] == oc[:, 1]) & (oc[:, 1] == oc[:, 2])
            if np.any(all_same):
                c_other = oc[:, 0]
                c_i = col[i]
                before = (c_other == c_i) & all_same
                after = (c_other == (1 - c_i)) & all_same
                deltas[i] = int(np.sum(after) - np.sum(before))
        return deltas

    def local_search_fast(col, score, max_passes=100):
        """Fast local search using vectorized delta computation."""
        improved = True
        passes = 0
        while improved and passes < max_passes and time.time() < deadline - 2.0:
            improved = False
            passes += 1
            order = list(range(N))
            random.shuffle(order)
            for i in order:
                if time.time() > deadline - 2.0:
                    break
                others = others_for_pos[i]
                if others.shape[0] == 0:
                    continue
                oc = col[others]
                all_same = (oc[:, 0] == oc[:, 1]) & (oc[:, 1] == oc[:, 2])
                if not np.any(all_same):
                    continue
                c_other = oc[:, 0]
                c_i = col[i]
                before = (c_other == c_i) & all_same
                after = (c_other == (1 - c_i)) & all_same
                d = int(np.sum(after) - np.sum(before))
                if d < 0:
                    col[i] = 1 - col[i]
                    score += d
                    improved = True
        return col, score

    def multi_flip_escape(col, score, num_flips, rng, max_rounds=50):
        """Try flipping multiple bits simultaneously to escape local minima."""
        for _ in range(max_rounds):
            if time.time() > deadline - 3.0:
                break
            flip_mask = np.zeros(N, dtype=bool)
            indices = rng.sample(range(N), num_flips)
            for i in indices:
                flip_mask[i] = True
            
            new_col = col.copy()
            new_col[flip_mask] = 1 - new_col[flip_mask]
            new_score = count_mono(new_col)
            
            if new_score < score:
                col = new_col
                score = new_score
            elif rng.random() < np.exp(-(new_score - score) / 3.0):
                col = new_col
                score = new_score
        
        return col, score

    def sa_run_fast(seed, max_steps, t0, t1):
        """Fast SA with vectorized delta."""
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)
        col = np_rng.randint(0, 2, size=N).astype(np.int32)
        cur = count_mono(col)
        best = col.copy()
        best_s = cur
        
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
            t = t0 * (t1 / t0) ** (step / max_steps)
            i = rng.randint(0, N - 1)
            others = others_for_pos[i]
            if others.shape[0] == 0:
                continue
            oc = col[others]
            all_same = (oc[:, 0] == oc[:, 1]) & (oc[:, 1] == oc[:, 2])
            if not np.any(all_same):
                d = 0
            else:
                c_other = oc[:, 0]
                c_i = col[i]
                before = (c_other == c_i) & all_same
                after = (c_other == (1 - c_i)) & all_same
                d = int(np.sum(after) - np.sum(before))
            
            if d <= 0 or (t > 1e-12 and rng.random() < np.exp(-d / t)):
                col[i] = 1 - col[i]
                cur += d
                if cur < best_s:
                    best_s = cur
                    best = col.copy()
        return best, best_s

    def thue_morse_pattern(n):
        """Generate Thue-Morse sequence of length n."""
        result = []
        for i in range(n):
            # Count number of 1s in binary representation
            c = bin(i).count('1')
            result.append(c % 2)
        return np.array(result, dtype=np.int32)

    def periodic_pattern(block, n):
        """Generate periodic pattern with given block size."""
        return np.array([(i // block) % 2 for i in range(n)], dtype=np.int32)

    best_col = None
    best_score = float('inf')

    # Phase 1: Multi-start with diverse initializations and fast local search
    initializations = [
        ('random', None),
        ('thue_morse', None),
        ('periodic_1', 1),
        ('periodic_2', 2),
        ('periodic_4', 4),
        ('periodic_8', 8),
        ('periodic_16', 16),
        ('periodic_32', 32),
    ]

    for trial in range(40):
        if time.time() > deadline - 30.0:
            break
        seed = 42 + trial * 97
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)

        init_type, init_param = initializations[trial % len(initializations)]
        
        if init_type == 'random':
            col = np_rng.randint(0, 2, size=N).astype(np.int32)
        elif init_type == 'thue_morse':
            col = thue_morse_pattern(N)
        else:  # periodic
            col = periodic_pattern(init_param, N)

        score = count_mono(col)
        col, score = local_search_fast(col, score, max_passes=100)

        # Multi-flip escape for some trials
        if trial % 3 == 0 and time.time() < deadline - 25.0:
            col, score = multi_flip_escape(col, score, 8, rng, max_rounds=30)
            col, score = local_search_fast(col, score, max_passes=50)

        if score < best_score:
            best_score = score
            best_col = col.copy()

    # Phase 2: SA Runs with fast delta
    for r in range(20):
        if time.time() > deadline - 20.0:
            break
        seed = 1000 + r * 211
        col, s = sa_run_fast(seed, 80000, 2.0, 0.01)
        col, s = local_search_fast(col, s, max_passes=80)
        
        # Multi-flip escape after SA
        if time.time() < deadline - 15.0:
            rng = random.Random(seed + 777)
            col, s = multi_flip_escape(col, s, 10, rng, max_rounds=40)
            col, s = local_search_fast(col, s, max_passes=50)
        
        if s < best_score:
            best_score = s
            best_col = col.copy()

    # Phase 3: Perturbation-refinement on best solution
    if best_col is not None and time.time() < deadline - 10.0:
        for perturb_trial in range(10):
            if time.time() > deadline - 5.0:
                break
            seed = 5000 + perturb_trial * 137
            rng = random.Random(seed)
            col = best_col.copy()
            
            # Perturb: flip random subset
            num_flips = rng.randint(5, 15)
            flip_indices = rng.sample(range(N), num_flips)
            for i in flip_indices:
                col[i] = 1 - col[i]
            
            s = count_mono(col)
            col, s = local_search_fast(col, s, max_passes=100)
            
            if s < best_score:
                best_score = s
                best_col = col.copy()

    # Phase 4: Final Polish
    if best_col is not None and time.time() < deadline - 2.0:
        col = best_col.copy()
        col, best_score = local_search_fast(col, best_score, max_passes=200)
        
        # Final multi-flip polish
        if time.time() < deadline - 1.0:
            rng = random.Random(12345)
            col, s = multi_flip_escape(col, best_score, 6, rng, max_rounds=30)
            col, s = local_search_fast(col, s, max_passes=100)
            if s < best_score:
                best_score = s
                best_col = col
        else:
            best_col = col

    return [int(c) for c in best_col]
