import time
import numpy as np
import random
import math

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s/150s kill
    n = 256

    # Precompute all 4-term APs as index tuples (0-indexed)
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(max_a):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))

    M = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)

    # Build per-position list of AP indices and the indices of the OTHER elements in those APs
    pos_aps = [[] for _ in range(n)]
    pos_others = [[] for _ in range(n)]

    for idx, ap in enumerate(aps):
        for k in range(4):
            p = ap[k]
            pos_aps[p].append(idx)
            others = [ap[j] for j in range(4) if j != k]
            pos_others[p].append(others)

    # Precompute others arrays for vectorized delta
    pos_others_arr = [None] * n
    for p in range(n):
        if pos_others[p]:
            pos_others_arr[p] = np.array(pos_others[p], dtype=np.int32)

    def thue_morse(n):
        return bin(n).count('1') % 2

    def count_mono(col):
        a0 = col[aps_arr[:, 0]]
        a1 = col[aps_arr[:, 1]]
        a2 = col[aps_arr[:, 2]]
        a3 = col[aps_arr[:, 3]]
        mono = (a0 == a1) & (a1 == a2) & (a2 == a3)
        return int(np.sum(mono))

    def get_delta(pos, col):
        """Calculate delta in mono count if we flip col[pos]."""
        delta = 0
        others_list = pos_others[pos]
        for others in others_list:
            c1 = col[others[0]]
            c2 = col[others[1]]
            c3 = col[others[2]]
            if c1 == c2 and c2 == c3:
                if col[pos] == c1:
                    delta -= 1
                else:
                    delta += 1
        return delta

    def get_delta_vectorized(col):
        """Calculate delta for all positions at once."""
        deltas = np.zeros(n, dtype=np.int32)
        for p in range(n):
            if not pos_others[p]:
                continue
            others_arr = pos_others_arr[p]
            c1 = col[others_arr[:, 0]]
            c2 = col[others_arr[:, 1]]
            c3 = col[others_arr[:, 2]]
            all_same = (c1 == c2) & (c2 == c3)
            if col[p] == c1[0]:
                # If flipping makes it different from others when they're all same, delta -= count
                # If flipping makes it same as others when they're all same and different from col, delta += count
                # Actually: for each AP where others are all same:
                #   if col[p] == others: flipping breaks it (delta -= 1)
                #   if col[p] != others: flipping creates it (delta += 1)
                same_as_others = (c1 == col[p]) & all_same
                diff_from_others = (c1 != col[p]) & all_same
                deltas[p] = -np.sum(same_as_others) + np.sum(diff_from_others)
            else:
                same_as_others = (c1 == col[p]) & all_same
                diff_from_others = (c1 != col[p]) & all_same
                deltas[p] = -np.sum(same_as_others) + np.sum(diff_from_others)
        return deltas

    def greedy_polish(col, max_passes=200, random_order=False):
        """Run greedy local search until no improvement."""
        col = col.copy()
        cur_count = count_mono(col)
        improved = True
        passes = 0
        while improved and passes < max_passes and (time.time() - start_time) < deadline - 5.0:
            improved = False
            passes += 1
            if random_order:
                order = list(range(n))
                random.shuffle(order)
            else:
                order = range(n)
            for pos in order:
                if (time.time() - start_time) > deadline - 5.0:
                    break
                d = get_delta(pos, col)
                if d < 0:
                    col[pos] = 1 - col[pos]
                    cur_count += d
                    improved = True
        return col, cur_count

    def collective_walksat(col, cur_count, max_flips=None):
        """
        Collective WalkSAT: identify all positions in monochromatic APs,
        then flip a subset to shatter clusters. Use delta-aware selection.
        """
        a0 = col[aps_arr[:, 0]]
        a1 = col[aps_arr[:, 1]]
        a2 = col[aps_arr[:, 2]]
        a3 = col[aps_arr[:, 3]]
        mono_mask = (a0 == a1) & (a1 == a2) & (a2 == a3)

        if not np.any(mono_mask):
            return col, cur_count

        mono_aps = aps_arr[mono_mask]
        involved_positions = np.unique(mono_aps.ravel())

        if len(involved_positions) == 0:
            return col, cur_count

        # Compute deltas for involved positions and prefer those with most negative delta
        pos_list = list(involved_positions)
        pos_deltas = [(pos, get_delta(pos, col)) for pos in pos_list]
        
        # Sort by delta (most negative first)
        pos_deltas.sort(key=lambda x: x[1])
        
        # Flip positions with negative delta first, then some random from the rest
        num_flips = max_flips if max_flips is not None else max(3, min(10, len(pos_list) // 2))
        
        flipped = set()
        for pos, d in pos_deltas:
            if len(flipped) >= num_flips:
                break
            if d < 0:
                flipped.add(pos)
        
        # If we didn't fill up, add random ones from remaining
        if len(flipped) < num_flips:
            remaining = [pos for pos, d in pos_deltas if pos not in flipped]
            if remaining:
                extra = min(num_flips - len(flipped), len(remaining))
                random.shuffle(remaining)
                for pos in remaining[:extra]:
                    flipped.add(pos)
        
        for pos in flipped:
            col[pos] = 1 - col[pos]

        new_count = count_mono(col)
        return col, new_count

    def simulated_annealing(col, cur_count, T_init, cooling, min_T, max_steps, restart_strategy=True):
        """
        Simulated annealing with WalkSAT-style perturbations.
        """
        col = col.copy()
        cur_count = count_mono(col)
        best_col = col.copy()
        best_count = cur_count
        T = T_init
        no_improve = 0
        max_no_improve = 1500

        for step in range(max_steps):
            if (time.time() - start_time) > deadline - 5.0:
                break

            # Pick a random position
            pos = random.randint(0, n - 1)
            d = get_delta(pos, col)

            accept = False
            if d < 0:
                accept = True
                no_improve = 0
            elif d == 0:
                accept = True
                no_improve = 0
            else:
                if T > min_T:
                    prob = math.exp(-d / T)
                    if random.random() < prob:
                        accept = True
                        no_improve = 0

            if accept:
                col[pos] = 1 - col[pos]
                cur_count += d
                if cur_count < best_count:
                    best_col = col.copy()
                    best_count = cur_count
            else:
                no_improve += 1

            T *= cooling

            # If stuck, do a collective WalkSAT perturbation
            if no_improve > max_no_improve and restart_strategy:
                col, cur_count = collective_walksat(col, cur_count, max_flips=random.randint(5, 12))
                no_improve = 0
                T = max(T_init / 2, 0.5)

        return best_col, best_count

    # Phase 1: Greedy polish from multiple initializations
    rng = np.random.default_rng(42)
    random.seed(42)

    initializations = []

    # 1. Thue-Morse
    tm_colors = np.array([thue_morse(i) for i in range(n)], dtype=np.int8)
    initializations.append(tm_colors.copy())

    # 2. Random
    initializations.append(rng.integers(0, 2, size=n, dtype=np.int8))

    # 3. Thue-Morse with random flips
    for _ in range(8):
        flipped_tm = tm_colors.copy()
        mask = rng.random(n) < 0.08
        flipped_tm[mask] = 1 - flipped_tm[mask]
        initializations.append(flipped_tm)

    # 4. Block patterns
    for block_size in [2, 3, 4, 5, 8, 16]:
        block_colors = np.array([(i // block_size) % 2 for i in range(n)], dtype=np.int8)
        initializations.append(block_colors)

    # 5. More random initializations
    for _ in range(5):
        initializations.append(rng.integers(0, 2, size=n, dtype=np.int8))

    best_c = None
    best_count = float('inf')

    for init_colors in initializations:
        if (time.time() - start_time) > deadline - 10.0:
            break

        # Greedy polish
        c, count = greedy_polish(init_colors, max_passes=150, random_order=False)
        if count < best_count:
            best_c = c.copy()
            best_count = count

        # Random-order polish
        c_ro, count_ro = greedy_polish(c, max_passes=80, random_order=True)
        if count_ro < best_count:
            best_c = c_ro.copy()
            best_count = count_ro

    # Phase 2: Simulated Annealing with collective WalkSAT perturbations
    if (time.time() - start_time) < deadline - 10.0:
        # Run SA from the best solution found
        sa_col, sa_count = simulated_annealing(
            best_c, best_count,
            T_init=2.0, cooling=0.99994, min_T=0.02,
            max_steps=80000, restart_strategy=True
        )
        if sa_count < best_count:
            best_c = sa_col.copy()
            best_count = sa_count

        # Run SA from a few random perturbations of the best
        for _ in range(5):
            if (time.time() - start_time) > deadline - 10.0:
                break

            perturbed = best_c.copy()
            for _ in range(random.randint(8, 20)):
                p = random.randint(0, n - 1)
                perturbed[p] = 1 - perturbed[p]

            sa_col, sa_count = simulated_annealing(
                perturbed, count_mono(perturbed),
                T_init=1.5, cooling=0.99994, min_T=0.02,
                max_steps=40000, restart_strategy=True
            )
            if sa_count < best_count:
                best_c = sa_col.copy()
                best_count = sa_count

    # Phase 3: Final intensive polish with multiple passes
    if (time.time() - start_time) < deadline - 5.0:
        final_c, final_count = greedy_polish(best_c, max_passes=300, random_order=False)
        if final_count < best_count:
            best_c = final_c
            best_count = final_count

    if (time.time() - start_time) < deadline - 3.0:
        final_c, final_count = greedy_polish(best_c, max_passes=150, random_order=True)
        if final_count < best_count:
            best_c = final_c
            best_count = final_count

    # Phase 4: Final SA burst if time remains
    if (time.time() - start_time) < deadline - 5.0:
        sa_col, sa_count = simulated_annealing(
            best_c, best_count,
            T_init=1.0, cooling=0.99998, min_T=0.01,
            max_steps=50000, restart_strategy=True
        )
        if sa_count < best_count:
            best_c = sa_col.copy()
            best_count = sa_count

        final_c, final_count = greedy_polish(best_c, max_passes=200, random_order=False)
        if final_count < best_count:
            best_c = final_c
            best_count = final_count

    # Verify and return
    actual_count = count_mono(best_c)
    if actual_count < best_count:
        best_count = actual_count

    return [int(c) for c in best_c]
