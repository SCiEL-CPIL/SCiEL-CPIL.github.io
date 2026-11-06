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

    # Double-flip delta: use a simpler approach
    # For two indices i and j, the affected APs are the union of their APs.
    # We compute old and new mono counts on just those APs.
    def delta_double(colors, i, j):
        ap_i = idx_ap_arr[i]
        ap_j = idx_ap_arr[j]
        pos_i = idx_pos_arr[i]
        pos_j = idx_pos_arr[j]

        if len(ap_i) == 0 and len(ap_j) == 0:
            return 0
        if len(ap_i) == 0:
            return delta_single(colors, j)
        if len(ap_j) == 0:
            return delta_single(colors, i)

        # Union of AP indices
        union_aps = np.unique(np.concatenate([ap_i, ap_j]))
        m = len(union_aps)

        c0 = colors[aps[union_aps, 0]]
        c1 = colors[aps[union_aps, 1]]
        c2 = colors[aps[union_aps, 2]]
        c3 = colors[aps[union_aps, 3]]

        old_mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        old_count = int(np.sum(old_mono))

        n_c0 = c0.copy()
        n_c1 = c1.copy()
        n_c2 = c2.copy()
        n_c3 = c3.copy()

        # Build position maps for i and j within union_aps
        # Use a lookup: for each AP in union_aps, what position is i at?
        # Since m is small (typically < 80), a simple approach works.
        # Map from AP index to its position in union_aps
        ap_to_pos = {}
        for k, ap in enumerate(union_aps):
            ap_to_pos[ap] = k

        # For index i
        for k in range(len(ap_i)):
            ap = ap_i[k]
            if ap in ap_to_pos:
                p = ap_to_pos[ap]
                pi = pos_i[k]
                if pi == 0:
                    n_c0[p] ^= 1
                elif pi == 1:
                    n_c1[p] ^= 1
                elif pi == 2:
                    n_c2[p] ^= 1
                elif pi == 3:
                    n_c3[p] ^= 1

        # For index j
        for k in range(len(ap_j)):
            ap = ap_j[k]
            if ap in ap_to_pos:
                p = ap_to_pos[ap]
                pj = pos_j[k]
                if pj == 0:
                    n_c0[p] ^= 1
                elif pj == 1:
                    n_c1[p] ^= 1
                elif pj == 2:
                    n_c2[p] ^= 1
                elif pj == 3:
                    n_c3[p] ^= 1

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

            # Try random double flips
            for _ in range(200):
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

            if best_move is None:
                break

            if best_move[0] == 'single':
                colors[best_move[1]] ^= 1
            else:
                colors[best_move[1]] ^= 1
                colors[best_move[2]] ^= 1
            score += best_delta

        return score

    # Simulated Annealing
    def sa_search(colors, T_start, T_min, cooling, max_steps, p_double=0.35):
        score = get_score(colors)
        T = T_start
        step = 0
        while T > T_min and step < max_steps:
            if time.time() >= deadline - 3.0:
                break
            step += 1

            if random.random() < p_double:
                i = np.random.randint(0, N)
                j = np.random.randint(0, N)
                if i == j:
                    j = (j + 1) % N
                d = delta_double(colors, i, j)
                if d <= 0 or random.random() < np.exp(-d / T):
                    colors[i] ^= 1
                    colors[j] ^= 1
                    score += d
            else:
                idx = np.random.randint(0, N)
                d = delta_single(colors, idx)
                if d <= 0 or random.random() < np.exp(-d / T):
                    colors[idx] ^= 1
                    score += d

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
    num_restarts = 15
    time_per_restart = (deadline - start_time - 5.0) / num_restarts

    for restart in range(num_restarts):
        if time.time() >= deadline - 5.0:
            break

        restart_deadline = min(start_time + time_per_restart * (restart + 1), deadline - 5.0)

        # Start from best or random
        if restart == 0:
            colors = best_colors.copy()
        elif restart < 5:
            # Perturb best
            colors = best_colors.copy()
            for _ in range(random.randint(5, 20)):
                idx = np.random.randint(0, N)
                colors[idx] ^= 1
        else:
            colors = np.random.randint(0, 2, N).astype(np.int32)

        current_score = get_score(colors)
        if current_score < best_score:
            best_score = current_score
            best_colors = colors.copy()

        # SA
        T_start = 12.0
        T_min = 0.05
        cooling = 0.9995
        max_steps = 50000
        p_double = 0.35

        current_score = sa_search(colors, T_start, T_min, cooling, max_steps, p_double)

        if current_score < best_score:
            best_score = current_score
            best_colors = colors.copy()

        # Greedy polish
        if time.time() < restart_deadline - 2.0:
            current_score = greedy_polish(colors, max_rounds=100)
            if current_score < best_score:
                best_score = current_score
                best_colors = colors.copy()

    # Phase 3: Final intensive polish on best solution
    if best_colors is not None and time.time() < deadline - 3.0:
        colors = best_colors.copy()
        current_score = get_score(colors)

        # Multiple rounds of greedy polish
        for _ in range(5):
            if time.time() >= deadline - 3.0:
                break
            current_score = greedy_polish(colors, max_rounds=200)
            if current_score < best_score:
                best_score = current_score
                best_colors = colors.copy()

        # Short SA from best
        if time.time() < deadline - 5.0:
            colors = best_colors.copy()
            # Perturb slightly
            for _ in range(10):
                idx = np.random.randint(0, N)
                colors[idx] ^= 1
            current_score = get_score(colors)
            current_score = sa_search(colors, 8.0, 0.05, 0.9998, 30000, 0.35)
            current_score = greedy_polish(colors, max_rounds=150)
            if current_score < best_score:
                best_score = current_score
                best_colors = colors.copy()

    if best_colors is None:
        return [0] * N

    return [int(c) for c in best_colors]
