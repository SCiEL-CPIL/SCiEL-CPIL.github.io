import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    N = 256

    # Precompute all 4-term arithmetic progressions (0-indexed)
    ap_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            ap_list.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))

    aps = np.array(ap_list, dtype=np.int32)
    num_aps = len(aps)

    # For each AP, store the 4 indices as a flat array
    # aps shape: (num_aps, 4)

    # Precompute for each index: list of (ap_index, position)
    index_aps = [[] for _ in range(N)]
    for i in range(num_aps):
        for j in range(4):
            idx = aps[i, j]
            index_aps[idx].append((i, j))

    # Convert to arrays for fast access
    # For each index, we store:
    #   ap_indices: array of AP indices that contain this index
    #   positions: array of positions (0-3) of this index within each AP
    idx_ap_arr = []
    idx_pos_arr = []
    for i in range(N):
        if index_aps[i]:
            arr = np.array(index_aps[i], dtype=np.int32)
            idx_ap_arr.append(arr[:, 0])
            idx_pos_arr.append(arr[:, 1])
        else:
            idx_ap_arr.append(np.array([], dtype=np.int32))
            idx_pos_arr.append(np.array([], dtype=np.int32))

    # Precompute: for each AP, which 4 positions it has
    # aps is already (num_aps, 4)

    # Fast score computation
    def get_score(colors):
        c0 = colors[aps[:, 0]]
        c1 = colors[aps[:, 1]]
        c2 = colors[aps[:, 2]]
        c3 = colors[aps[:, 3]]
        is_mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        return int(np.sum(is_mono))

    # Fast single-flip delta
    def delta_single(colors, idx):
        ap_idx = idx_ap_arr[idx]
        if len(ap_idx) == 0:
            return 0
        pos = idx_pos_arr[idx]

        # Get colors at all 4 positions for these APs
        c0 = colors[aps[ap_idx, 0]]
        c1 = colors[aps[ap_idx, 1]]
        c2 = colors[aps[ap_idx, 2]]
        c3 = colors[aps[ap_idx, 3]]

        old_mono = (c0 == c1) & (c1 == c2) & (c2 == c3)

        # Flip the relevant positions
        # pos is the position of idx in each AP
        n_c0 = c0.copy()
        n_c1 = c1.copy()
        n_c2 = c2.copy()
        n_c3 = c3.copy()

        m0 = (pos == 0)
        m1 = (pos == 1)
        m2 = (pos == 2)
        m3 = (pos == 3)

        n_c0[m0] ^= 1
        n_c1[m1] ^= 1
        n_c2[m2] ^= 1
        n_c3[m3] ^= 1

        new_mono = (n_c0 == n_c1) & (n_c1 == n_c2) & (n_c2 == n_c3)
        return int(np.sum(new_mono)) - int(np.sum(old_mono))

    # Precompute AP membership as a 2D boolean matrix for fast double-flip
    # ap_membership[i, ap_idx] = position of index i in AP ap_idx, or -1 if not present
    ap_membership = np.full((N, num_aps), -1, dtype=np.int8)
    for i in range(N):
        if len(idx_ap_arr[i]) > 0:
            ap_membership[i, idx_ap_arr[i]] = idx_pos_arr[i]

    # Double-flip delta using precomputed membership (vectorized)
    def delta_double(colors, i, j):
        # Get positions of i and j in all APs
        pos_i = ap_membership[i]  # shape (num_aps,), -1 if not present
        pos_j = ap_membership[j]

        # APs where either i or j appears
        mask_i = pos_i >= 0
        mask_j = pos_j >= 0
        affected = mask_i | mask_j
        if not np.any(affected):
            return 0

        affected_aps = aps[affected]  # shape (m, 4)
        c0 = colors[affected_aps[:, 0]]
        c1 = colors[affected_aps[:, 1]]
        c2 = colors[affected_aps[:, 2]]
        c3 = colors[affected_aps[:, 3]]

        old_mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        old_count = int(np.sum(old_mono))

        n_c0 = c0.copy()
        n_c1 = c1.copy()
        n_c2 = c2.copy()
        n_c3 = c3.copy()

        # Flip positions for i
        pi = pos_i[affected]
        n_c0[pi == 0] ^= 1
        n_c1[pi == 1] ^= 1
        n_c2[pi == 2] ^= 1
        n_c3[pi == 3] ^= 1

        # Flip positions for j
        pj = pos_j[affected]
        n_c0[pj == 0] ^= 1
        n_c1[pj == 1] ^= 1
        n_c2[pj == 2] ^= 1
        n_c3[pj == 3] ^= 1

        new_mono = (n_c0 == n_c1) & (n_c1 == n_c2) & (n_c2 == n_c3)
        new_count = int(np.sum(new_mono))

        return new_count - old_count

    # Greedy: repeatedly flip the best single or double flip
    def greedy_polish(colors, max_rounds=200):
        score = get_score(colors)
        for _ in range(max_rounds):
            if time.time() >= deadline - 3.0:
                break
            best_delta = 0
            best_move = None

            # Try all single flips
            for idx in range(N):
                d = delta_single(colors, idx)
                if d < best_delta:
                    best_delta = d
                    best_move = ('single', idx, None)

            # Try more systematic double flips: focus on indices in same APs
            # First, try pairs that share at least one AP
            for _ in range(500):
                if time.time() >= deadline - 3.0:
                    break
                i = np.random.randint(0, N)
                j = np.random.randint(0, N)
                if i == j:
                    continue
                d = delta_double(colors, i, j)
                if d < best_delta:
                    best_delta = d
                    best_move = ('double', i, j)

            # Also try structured doubles: flip two indices that are in the same AP
            # This can break monochromatic APs more effectively
            for _ in range(300):
                if time.time() >= deadline - 3.0:
                    break
                # Pick a random AP and two random positions in it
                ap_idx = np.random.randint(0, num_aps)
                pos_a, pos_b = np.random.choice(4, size=2, replace=False)
                i = int(aps[ap_idx, pos_a])
                j = int(aps[ap_idx, pos_b])
                d = delta_double(colors, i, j)
                if d < best_delta:
                    best_delta = d
                    best_move = ('double', i, j)

            if best_move is None:
                break

            if best_move[0] == 'single':
                colors[best_move[1]] ^= 1
            else:
                colors[best_move[1]] ^= 1
                colors[best_move[2]] ^= 1
            score += best_delta

        return score

    # Simulated Annealing with adaptive move selection
    def sa_search(colors, T_start, T_min, cooling, max_steps, p_double=0.35):
        score = get_score(colors)
        T = T_start
        step = 0
        consecutive_rejects = 0
        while T > T_min and step < max_steps:
            if time.time() >= deadline - 3.0:
                break
            step += 1

            # Adaptively increase double-flip probability when stuck
            eff_p_double = min(0.6, p_double + 0.1 * (consecutive_rejects // 10))

            if random.random() < eff_p_double:
                i = np.random.randint(0, N)
                j = np.random.randint(0, N)
                if i == j:
                    j = (j + 1) % N
                d = delta_double(colors, i, j)
                if d <= 0 or random.random() < np.exp(-d / T):
                    colors[i] ^= 1
                    colors[j] ^= 1
                    score += d
                    consecutive_rejects = 0
                else:
                    consecutive_rejects += 1
            else:
                idx = np.random.randint(0, N)
                d = delta_single(colors, idx)
                if d <= 0 or random.random() < np.exp(-d / T):
                    colors[idx] ^= 1
                    score += d
                    consecutive_rejects = 0
                else:
                    consecutive_rejects += 1

            T *= cooling

        return score

    # Build a structured initial coloring
    def structured_init():
        # Use a pattern based on residues mod some number
        # Try several patterns and pick the best
        candidates = []

        # Pattern 1: based on mod 4
        c = np.zeros(N, dtype=np.int32)
        for i in range(N):
            c[i] = (i % 4) // 2  # 0,0,1,1 repeating
        candidates.append(c.copy())

        # Pattern 2: based on mod 5
        c = np.zeros(N, dtype=np.int32)
        for i in range(N):
            r = i % 5
            c[i] = 0 if r in (0, 1) else 1
        candidates.append(c.copy())

        # Pattern 3: Thue-Morse-like
        c = np.zeros(N, dtype=np.int32)
        for i in range(N):
            # Count number of 1-bits in i
            c[i] = bin(i).count('1') % 2
        candidates.append(c.copy())

        # Pattern 4: based on mod 3
        c = np.zeros(N, dtype=np.int32)
        for i in range(N):
            c[i] = 0 if i % 3 != 0 else 1
        candidates.append(c.copy())

        # Pattern 5: alternating blocks of size 4
        c = np.zeros(N, dtype=np.int32)
        for i in range(N):
            block = i // 4
            c[i] = block % 2
        candidates.append(c.copy())

        # Pattern 6: based on mod 7
        c = np.zeros(N, dtype=np.int32)
        for i in range(N):
            r = i % 7
            c[i] = 0 if r in (0, 1, 2) else 1
        candidates.append(c.copy())

        # Random initializations
        for _ in range(10):
            c = np.random.randint(0, 2, N).astype(np.int32)
            candidates.append(c)

        best_c = None
        best_s = float('inf')
        for c in candidates:
            s = get_score(c)
            if s < best_s:
                best_s = s
                best_c = c.copy()
        return best_c, best_s

    # Main search
    best_colors = None
    best_score = float('inf')

    # Phase 1: Get a good initial solution
    init_colors, init_score = structured_init()
    if init_score < best_score:
        best_score = init_score
        best_colors = init_colors.copy()

    # Phase 2: Multiple SA runs with polish
    num_restarts = 12
    time_per_restart = (deadline - start_time - 5.0) / num_restarts

    for restart in range(num_restarts):
        if time.time() >= deadline - 5.0:
            break

        restart_deadline = min(start_time + time_per_restart * (restart + 1), deadline - 5.0)

        # Start from best or random
        if restart == 0:
            colors = best_colors.copy()
        elif restart < 6:
            # Perturb best with varied perturbation sizes
            colors = best_colors.copy()
            num_flips = random.randint(5, 30)
            for _ in range(num_flips):
                idx = np.random.randint(0, N)
                colors[idx] ^= 1
        else:
            # Mix of random and structured starts
            if restart % 2 == 0:
                colors = np.random.randint(0, 2, N).astype(np.int32)
            else:
                # Thue-Morse variant with random shift
                shift = np.random.randint(0, 4)
                colors = np.zeros(N, dtype=np.int32)
                for i in range(N):
                    colors[i] = bin((i + shift) % N).count('1') % 2

        current_score = get_score(colors)
        if current_score < best_score:
            best_score = current_score
            best_colors = colors.copy()

        # SA with slightly slower cooling for better exploration
        T_start = 15.0
        T_min = 0.03
        cooling = 0.9997
        max_steps = 40000
        p_double = 0.40

        current_score = sa_search(colors, T_start, T_min, cooling, max_steps, p_double)

        if current_score < best_score:
            best_score = current_score
            best_colors = colors.copy()

        # Greedy polish with more rounds
        if time.time() < restart_deadline - 2.0:
            current_score = greedy_polish(colors, max_rounds=150)
            if current_score < best_score:
                best_score = current_score
                best_colors = colors.copy()

    # Phase 3: Final intensive polish on best solution
    if best_colors is not None and time.time() < deadline - 3.0:
        colors = best_colors.copy()
        current_score = get_score(colors)

        # Multiple rounds of greedy polish
        for _ in range(8):
            if time.time() >= deadline - 3.0:
                break
            current_score = greedy_polish(colors, max_rounds=250)
            if current_score < best_score:
                best_score = current_score
                best_colors = colors.copy()

        # Short SA from best with higher double-flip probability
        if time.time() < deadline - 5.0:
            colors = best_colors.copy()
            # Perturb with varied amounts
            num_perturb = random.randint(8, 15)
            for _ in range(num_perturb):
                idx = np.random.randint(0, N)
                colors[idx] ^= 1
            current_score = get_score(colors)
            current_score = sa_search(colors, 10.0, 0.03, 0.9998, 25000, 0.45)
            current_score = greedy_polish(colors, max_rounds=200)
            if current_score < best_score:
                best_score = current_score
                best_colors = colors.copy()

    if best_colors is None:
        return [0] * N

    return [int(c) for c in best_colors]
