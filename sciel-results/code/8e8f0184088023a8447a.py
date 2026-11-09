"""Combined: Vectorized SA with batching + multi-phase refinement."""
import time
import numpy as np
import random
import math


def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s kill
    n = 256

    # Precompute all 4-term APs as 0-indexed tuples
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            aps.append((i0, i1, i2, i3))

    aps_arr = np.array(aps, dtype=np.int32)
    M = len(aps_arr)

    # Build per-position list of AP indices (for SA)
    pos_aps = [[] for _ in range(n)]
    for idx in range(M):
        for pos in aps_arr[idx]:
            pos_aps[pos].append(idx)
    pos_aps_arr = [np.array(p, dtype=np.int32) if p else np.array([], dtype=np.int32) for p in pos_aps]
    
    # Precompute mask for vectorized greedy: pos_in_ap[p, k] is True if p is in AP k
    pos_in_ap = np.zeros((n, M), dtype=bool)
    for idx in range(M):
        for pos in aps_arr[idx]:
            pos_in_ap[pos, idx] = True

    # Thue-Morse initialization
    colors = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)

    def count_mono_vec(cols):
        c = cols[aps_arr]
        mono = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
        return int(np.sum(mono))

    def compute_mono_flags(cols):
        c = cols[aps_arr]
        return (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])

    current_mono = count_mono_vec(colors)
    best_colors = colors.copy()
    best_mono = current_mono
    mono_flag = compute_mono_flags(colors)

    # Phase 1: Vectorized Greedy polish
    def greedy_polish(cols, mono_flags, cur_count, time_budget):
        cols = cols.copy()
        mono_flags = mono_flags.copy()
        # C[k, j] is color of j-th element in AP k
        # We need to update C whenever cols changes, or recompute from cols each time.
        # Recomputing C is O(M*4) which is cheap.
        improved = True
        while improved and (time.time() - start_time) < time_budget:
            improved = False
            # Get current colors for all APs
            C = cols[aps_arr] # Shape (M, 4)
            # Compute old_mono (should match mono_flags, but let's trust mono_flags for speed if consistent)
            # Actually, let's recompute mono_flags from C to be safe if it's cheap?
            # mono_flags = (C[:,0]==C[:,1]) & (C[:,1]==C[:,2]) & (C[:,2]==C[:,3])
            
            # Compute delta for flipping each position p
            # We want to find p that minimizes sum of new_mono over APs containing p - sum of old_mono over APs containing p.
            # Let's compute new_mono for all p simultaneously.
            # This requires constructing C_flip for all p.
            # C_flip[p, k, j] = C[k, j] if aps_arr[k, j] != p else 1 - C[k, j]
            
            # Efficient way:
            # For each AP k, it has 4 positions.
            # If we flip p, APs not containing p are unchanged.
            # APs containing p change.
            
            # Let's use the mask pos_in_ap.
            # delta[p] = sum_k (pos_in_ap[p, k] * (new_mono_k(p) - mono_flags[k]))
            
            # To compute new_mono_k(p) for all p, k:
            # We can iterate over the 4 positions of an AP.
            # If p is at index j in AP k, then C_flip[k, j] is flipped.
            
            # This is tricky to fully vectorize without large memory.
            # Let's try a simpler heuristic: 
            # Just do a few passes of the original loop but optimized?
            # Or, since M is ~10k, and n=256, the original loop is 256 * (small ops).
            # The bottleneck was likely the number of passes or the time limit.
            # With 110s limit, the original might have been fine if the deadline was correct?
            # The error was "exceeded 120s". The code had deadline 145.
            # So it ran for 120s and was killed.
            # If we just lower the deadline to 110s, it will return a result.
            # The "improvement" is just not crashing.
            # Let's stick to the simple fix: Lower deadline.
            # The vectorized greedy is risky and complex.
            # Let's just ensure the SA loop is fast enough.
            pass # Placeholder, we will revert to simple loop but with better time management
            
            # Actually, let's just keep the original greedy_polish but ensure it respects the time budget strictly.
            # The original code checked time inside the loop.
            # The issue was the global deadline was too high.
            
            # Let's just use the original greedy_polish implementation but it's defined below?
            # No, I replaced it. Let's replace it back with the original but ensure it's fast.
            # The original was:
            # for pos in range(n): ...
            # This is O(N) per pass.
            # Let's just use the original logic.
            
            # Wait, I can just delete this complex vectorization attempt and use the original, 
            # but I must provide a replacement.
            # Let's provide a slightly optimized original: 
            # Pre-compute the "potential" delta? No, it depends on current state.
            
            # Let's just use the original code for greedy_polish but it was in the SEARCH block.
            # I will replace the vectorized attempt with the original code, 
            # but I need to make sure it's fast.
            # The original was fine, just the deadline was wrong.
            
            # Let's just return the original greedy_polish code.
            # But I can't "return" code, I must write it.
            
            # Okay, I will write the original greedy_polish code here.
            while improved and (time.time() - start_time) < time_budget:
                improved = False
                for pos in range(n):
                    if (time.time() - start_time) > time_budget:
                        break
                    ap_indices = pos_aps_arr[pos]
                    if len(ap_indices) == 0:
                        continue
                    old_mono = mono_flags[ap_indices]
                    cols[pos] = 1 - cols[pos]
                    new_colors = cols[aps_arr[ap_indices]]
                    new_mono = (new_colors[:, 0] == new_colors[:, 1]) & \
                               (new_colors[:, 1] == new_colors[:, 2]) & \
                               (new_colors[:, 2] == new_colors[:, 3])
                    delta = int(np.sum(new_mono) - np.sum(old_mono))
                    if delta < 0:
                        mono_flags[ap_indices] = new_mono
                        cur_count += delta
                        improved = True
                    else:
                        cols[pos] = 1 - cols[pos]
        return cols, mono_flags, cur_count

    colors, mono_flag, current_mono = greedy_polish(colors, mono_flag, current_mono, deadline - 100.0)
    if current_mono < best_mono:
        best_mono = current_mono
        best_colors = colors.copy()

    # Phase 2: Batched Simulated Annealing with restarts
    rng = np.random.RandomState(42)
    T = 2.0
    min_T = 0.05
    cooling = 0.9997
    batch_size = 32  # Increase batch size to reduce Python overhead

    while (time.time() - start_time) < deadline - 5.0:
        # Batch of random positions
        positions = rng.randint(0, n, size=batch_size)
        batch_delta = 0
        accepted_positions = []
        accepted_ap_indices = []
        accepted_new_mono = []
        accepted_old_mono = []

        for pos in positions:
            ap_indices = pos_aps_arr[pos]
            if len(ap_indices) == 0:
                continue
            old_mono = mono_flag[ap_indices]
            colors[pos] = 1 - colors[pos]
            new_colors = colors[aps_arr[ap_indices]]
            new_mono = (new_colors[:, 0] == new_colors[:, 1]) & \
                       (new_colors[:, 1] == new_colors[:, 2]) & \
                       (new_colors[:, 2] == new_colors[:, 3])
            d = int(np.sum(new_mono) - np.sum(old_mono))

            if d <= 0 or (T > 0.1 and rng.rand() < math.exp(-d / T)):
                batch_delta += d
                accepted_positions.append(pos)
                accepted_ap_indices.append(ap_indices)
                accepted_new_mono.append(new_mono)
                accepted_old_mono.append(old_mono)
            else:
                colors[pos] = 1 - colors[pos]

        # Apply all accepted flips at once
        if accepted_positions:
            for i, pos in enumerate(accepted_positions):
                mono_flag[accepted_ap_indices[i]] = accepted_new_mono[i]
            current_mono += batch_delta
            if current_mono < best_mono:
                best_mono = current_mono
                best_colors = colors.copy()

        T *= cooling
        if T < min_T:
            # Restart with perturbation
            T = 1.5
            for _ in range(10):
                p2 = rng.randint(n)
                ap_indices = pos_aps_arr[p2]
                if len(ap_indices) == 0:
                    continue
                old_mono = mono_flag[ap_indices]
                colors[p2] = 1 - colors[p2]
                new_colors = colors[aps_arr[ap_indices]]
                new_mono = (new_colors[:, 0] == new_colors[:, 1]) & \
                           (new_colors[:, 1] == new_colors[:, 2]) & \
                           (new_colors[:, 2] == new_colors[:, 3])
                d = int(np.sum(new_mono) - np.sum(old_mono))
                if d <= 0 or rng.rand() < 0.2:
                    mono_flag[ap_indices] = new_mono
                    current_mono += d
                    if current_mono < best_mono:
                        best_mono = current_mono
                        best_colors = colors.copy()
                else:
                    colors[p2] = 1 - colors[p2]

    # Phase 3: Final greedy polish
    if (time.time() - start_time) < deadline - 2.0:
        colors, mono_flag, current_mono = greedy_polish(best_colors, compute_mono_flags(best_colors), best_mono, deadline - 1.0)
        if current_mono < best_mono:
            best_mono = current_mono
            best_colors = colors.copy()
            
    # Ensure we return within time
    if time.time() > deadline - 0.5:
         return [int(c) for c in best_colors]

    # Verify
    actual = count_mono_vec(best_colors)
    if actual < best_mono:
        best_mono = actual

    return [int(c) for c in best_colors]
