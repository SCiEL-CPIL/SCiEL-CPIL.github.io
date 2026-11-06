import numpy as np
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin from 150s

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

    def fast_delta(i, col):
        """Compute delta for flipping position i."""
        others = others_for_pos[i]
        if others.shape[0] == 0:
            return 0
        oc = col[others]
        all_same = (oc[:, 0] == oc[:, 1]) & (oc[:, 1] == oc[:, 2])
        if not np.any(all_same):
            return 0
        c_other = oc[:, 0]
        c_i = col[i]
        before = (c_other == c_i) & all_same
        after = (c_other == (1 - c_i)) & all_same
        return int(np.sum(after) - np.sum(before))

    def local_search_fast(col, score, max_passes=300):
        """Fast local search using precomputed deltas, multiple passes until convergence."""
        improved = True
        passes = 0
        while improved and passes < max_passes:
            if time.time() > deadline - 2.0:
                break
            improved = False
            passes += 1
            order = list(range(N))
            random.shuffle(order)
            for i in order:
                if time.time() > deadline - 2.0:
                    break
                d = fast_delta(i, col)
                if d < 0:
                    col[i] = 1 - col[i]
                    score += d
                    improved = True
        return col, score

    def multi_flip_escape(col, score, num_flips, max_rounds=80):
        """Try flipping multiple bits simultaneously to escape local minima."""
        rng = random.Random(int(time.time() * 1000) % 100000)
        for _ in range(max_rounds):
            if time.time() > deadline - 3.0:
                break
            indices = rng.sample(range(N), num_flips)
            new_col = col.copy()
            for i in indices:
                new_col[i] = 1 - new_col[i]
            new_score = count_mono(new_col)
            delta = new_score - score
            if delta < 0 or (delta > 0 and rng.random() < np.exp(-delta / 4.0)):
                col = new_col
                score = new_score
        return col, score

    def sa_run(seed, max_steps, t0, t1):
        rng = random.Random(seed)
        col = np.random.RandomState(seed).randint(0, 2, size=N).astype(np.int32)
        cur = count_mono(col)
        best = col.copy()
        best_s = cur
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
            t = t0 * (t1 / t0) ** (step / max_steps)
            i = rng.randint(0, N - 1)
            d = fast_delta(i, col)
            if d <= 0 or (t > 1e-12 and rng.random() < np.exp(-d / t)):
                col[i] = 1 - col[i]
                cur += d
                if cur < best_s:
                    best_s = cur
                    best = col.copy()
        return best, best_s

    def subgradient_step(col, score, temperature, rng):
        """
        Vectorized subgradient-style update: compute the 'pressure' on each position
        from APs where the other 3 are already monochromatic. Positions with high
        pressure toward their current color are more likely to be flipped.
        This allows correlated, large-scale moves.
        """
        # For each position, compute:
        #   n_same: number of APs where other 3 are same color as col[i]
        #   n_opp: number of APs where other 3 are same color as 1-col[i]
        # Flipping i changes score by n_opp - n_same.
        # We want to flip positions where n_opp > n_same (negative delta).
        # For high-temperature annealing, we can flip with probability based on delta.
        
        deltas = np.zeros(N, dtype=np.int32)
        for i in range(N):
            d = fast_delta(i, col)
            deltas[i] = d
        
        # Only consider positions with non-zero delta
        mask = deltas != 0
        if not np.any(mask):
            return col, score
        
        # For positions with negative delta, flipping is beneficial
        # For positions with positive delta, flipping hurts but may be accepted at high temp
        # Use a stochastic selection: flip with probability 1/(1+exp(delta/temperature))
        # This is a soft version: strongly prefer negative deltas, occasionally accept positive
        
        flip_prob = 1.0 / (1.0 + np.exp(deltas[mask].astype(np.float64) / max(temperature, 1e-6)))
        
        # Stochastically select which positions to flip
        rand_vals = rng.random(int(np.sum(mask)))
        to_flip = rand_vals < flip_prob
        
        if not np.any(to_flip):
            return col, score
        
        indices = np.where(mask)[0]
        flip_indices = indices[to_flip]
        
        # Apply flips
        new_col = col.copy()
        new_col[flip_indices] = 1 - new_col[flip_indices]
        new_score = count_mono(new_col)
        
        return new_col, new_score

    def subgradient_anneal(seed, max_steps, t0, t1):
        """Subgradient-based annealing that updates multiple positions per step."""
        rng = random.Random(seed)
        col = np.random.RandomState(seed).randint(0, 2, size=N).astype(np.int32)
        cur = count_mono(col)
        best = col.copy()
        best_s = cur
        
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
            t = t0 * (t1 / t0) ** (step / max_steps)
            col, cur = subgradient_step(col, cur, t, rng)
            if cur < best_s:
                best_s = cur
                best = col.copy()
        return best, best_s

    best_col = None
    best_score = float('inf')

    # Phase 1: Multi-start with diverse initializations + fast local search
    for trial in range(45):
        if time.time() > deadline - 50.0:
            break
        seed = 42 + trial * 97
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)

        # Diverse initializations
        r = rng.random()
        if r < 0.25:
            col = np_rng.randint(0, 2, size=N).astype(np.int32)
        elif r < 0.45:
            block = 1 + rng.randint(1, 12)
            col = np.array([(i // block) % 2 for i in range(N)], dtype=np.int32)
        elif r < 0.65:
            # Periodic with random period
            period = rng.randint(3, 20)
            col = np.array([(i % period) < period // 2 for i in range(N)], dtype=np.int32)
        elif r < 0.80:
            # Thue-Morse-like
            col = np.array([bin(i).count('1') % 2 for i in range(N)], dtype=np.int32)
        elif r < 0.90:
            # Random blocks of varying size
            col = np.zeros(N, dtype=np.int32)
            pos = 0
            while pos < N:
                length = rng.randint(1, 15)
                val = rng.randint(0, 1)
                col[pos:pos+length] = val
                pos += length
        else:
            # Walsh-like / Hadamard pattern
            col = np.array([((i >> 0) & 1) ^ ((i >> 1) & 1) ^ ((i >> 3) & 1) ^ ((i >> 5) & 1) for i in range(N)], dtype=np.int32)

        score = count_mono(col)
        col, score = local_search_fast(col, score, max_passes=250)

        # Multi-flip escape for some trials
        if trial % 3 == 0 and time.time() < deadline - 40.0:
            col, score = multi_flip_escape(col, score, 8, max_rounds=60)
            col, score = local_search_fast(col, score, max_passes=150)

        if score < best_score:
            best_score = score
            best_col = col.copy()

    # Phase 2: SA runs
    for r in range(20):
        if time.time() > deadline - 30.0:
            break
        seed = 1000 + r * 211
        col, s = sa_run(seed, 100000, 3.0, 0.005)
        col, s = local_search_fast(col, s, max_passes=250)

        if time.time() < deadline - 25.0:
            col, s = multi_flip_escape(col, s, 10, max_rounds=50)
            col, s = local_search_fast(col, s, max_passes=150)

        if s < best_score:
            best_score = s
            best_col = col.copy()

    # Phase 3: Subgradient annealing runs
    for r in range(15):
        if time.time() > deadline - 20.0:
            break
        seed = 5000 + r * 313
        col, s = subgradient_anneal(seed, 50000, 5.0, 0.01)
        col, s = local_search_fast(col, s, max_passes=250)

        if time.time() < deadline - 18.0:
            col, s = multi_flip_escape(col, s, 6, max_rounds=40)
            col, s = local_search_fast(col, s, max_passes=150)

        if s < best_score:
            best_score = s
            best_col = col.copy()

    # Phase 4: Focused SA on best solution with perturbations
    if best_col is not None and time.time() < deadline - 15.0:
        for r in range(15):
            if time.time() > deadline - 8.0:
                break
            col = best_col.copy()
            # Perturb random subset
            num_pert = random.randint(5, 25)
            indices = random.sample(range(N), num_pert)
            for i in indices:
                col[i] = 1 - col[i]
            s = count_mono(col)
            col, s = local_search_fast(col, s, max_passes=250)
            
            # Try subgradient escape from perturbed state
            if time.time() < deadline - 7.0:
                col2, s2 = subgradient_anneal(7000 + r * 137, 10000, 3.0, 0.05)
                # Blend: use subgradient result if better, else keep local search result
                if s2 < s:
                    col, s = col2, s2
                col, s = local_search_fast(col, s, max_passes=100)
            
            if s < best_score:
                best_score = s
                best_col = col.copy()

    # Phase 5: Final polish with intensive local search
    if best_col is not None and time.time() < deadline - 3.0:
        col = best_col.copy()
        col, best_score = local_search_fast(col, best_score, max_passes=500)
        best_col = col.copy()

        # Try one more multi-flip escape
        if time.time() < deadline - 2.0:
            col, s = multi_flip_escape(best_col, best_score, 4, max_rounds=30)
            col, s = local_search_fast(col, s, max_passes=200)
            if s < best_score:
                best_score = s
                best_col = col.copy()

    return [int(c) for c in best_col]
