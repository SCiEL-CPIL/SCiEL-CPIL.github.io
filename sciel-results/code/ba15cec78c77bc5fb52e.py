import math
import time
import random
import numpy as np

def construct():
    N = 256
    deadline = time.time() + 140

    # Precompute all 4-APs
    aps_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(max_a):
            aps_list.append((a, a+d, a+2*d, a+3*d))

    aps_np = np.array(aps_list, dtype=np.int32)  # shape (M, 4)
    M = len(aps_np)

    # For each position, precompute which APs it belongs to and the other 3 positions
    pos_aps = [[] for _ in range(N)]
    for idx, (a, b, c, d_pos) in enumerate(aps_list):
        pos_aps[a].append((idx, b, c, d_pos))
        pos_aps[b].append((idx, a, c, d_pos))
        pos_aps[c].append((idx, a, b, d_pos))
        pos_aps[d_pos].append((idx, a, b, c))

    # Convert to numpy arrays for fast lookup
    pos_aps_arr = []
    for i in range(N):
        if pos_aps[i]:
            arr = np.array(pos_aps[i], dtype=np.int32)
            pos_aps_arr.append(arr)
        else:
            pos_aps_arr.append(np.empty((0, 4), dtype=np.int32))

    def count_mono(cols_np):
        """Vectorized count of monochromatic 4-APs."""
        c0 = cols_np[aps_np[:, 0]]
        c1 = cols_np[aps_np[:, 1]]
        c2 = cols_np[aps_np[:, 2]]
        c3 = cols_np[aps_np[:, 3]]
        return int(np.sum((c0 == c1) & (c1 == c2) & (c2 == c3)))

    def compute_all_deltas(cols_np):
        """Compute flip deltas for all 256 positions in one vectorized pass."""
        c0 = cols_np[aps_np[:, 0]]
        c1 = cols_np[aps_np[:, 1]]
        c2 = cols_np[aps_np[:, 2]]
        c3 = cols_np[aps_np[:, 3]]

        all4_same = (c0 == c1) & (c1 == c2) & (c2 == c3)

        deltas = np.zeros(N, dtype=np.int32)

        if np.any(all4_same):
            idxs = np.where(all4_same)[0]
            for pos_col in range(4):
                positions = aps_np[idxs, pos_col]
                np.add.at(deltas, positions, -1)

        other3_same_diff0 = (c1 == c2) & (c2 == c3) & (c1 != c0)
        if np.any(other3_same_diff0):
            idxs = np.where(other3_same_diff0)[0]
            positions = aps_np[idxs, 0]
            np.add.at(deltas, positions, 1)

        other3_same_diff1 = (c0 == c2) & (c2 == c3) & (c0 != c1)
        if np.any(other3_same_diff1):
            idxs = np.where(other3_same_diff1)[0]
            positions = aps_np[idxs, 1]
            np.add.at(deltas, positions, 1)

        other3_same_diff2 = (c0 == c1) & (c1 == c3) & (c0 != c2)
        if np.any(other3_same_diff2):
            idxs = np.where(other3_same_diff2)[0]
            positions = aps_np[idxs, 2]
            np.add.at(deltas, positions, 1)

        other3_same_diff3 = (c0 == c1) & (c1 == c2) & (c0 != c3)
        if np.any(other3_same_diff3):
            idxs = np.where(other3_same_diff3)[0]
            positions = aps_np[idxs, 3]
            np.add.at(deltas, positions, 1)

        return deltas

    def greedy_polish(cols, current_cost):
        """Iteratively flip positions that reduce cost, using vectorized delta computation."""
        cols_np = np.array(cols, dtype=np.int8)
        improved = True
        max_iter = 50
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            improved = False
            deltas = compute_all_deltas(cols_np)
            neg_positions = np.where(deltas < 0)[0]
            if len(neg_positions) == 0:
                break
            for i in neg_positions:
                cols[i] = 1 - cols[i]
                cols_np[i] = 1 - cols_np[i]
                current_cost += deltas[i]
            improved = True
            if not improved:
                break
        # Second pass: single flips
        while improved:
            if time.time() > deadline - 5:
                break
            improved = False
            deltas = compute_all_deltas(cols_np)
            neg_positions = np.where(deltas < 0)[0]
            if len(neg_positions) == 0:
                break
            neg_positions = neg_positions[np.argsort(deltas[neg_positions])]
            for i in neg_positions:
                d = deltas[i]
                if d < 0:
                    cols[i] = 1 - cols[i]
                    cols_np[i] = 1 - cols_np[i]
                    current_cost += d
                    improved = True
        return cols, current_cost

    def thue_morse_coloring():
        colors = [0] * N
        for i in range(N):
            n = i + 1
            colors[i] = bin(n).count('1') % 2
        return colors

    def generate_structured_patterns():
        patterns = []
        patterns.append(thue_morse_coloring())

        for shift in range(1, 8):
            colors = [0] * N
            for i in range(N):
                n = (i + 1 + shift) % (N + 1)
                if n == 0:
                    n = N + 1
                colors[i] = bin(n).count('1') % 2
            patterns.append(colors)

        tm = thue_morse_coloring()
        patterns.append([1 - c for c in tm])

        for block in [2, 4, 8, 16]:
            p = [(i // block) % 2 for i in range(N)]
            patterns.append(p)

        patterns.append([i % 2 for i in range(N)])
        patterns.append([(i & 1) ^ ((i >> 1) & 1) for i in range(N)])
        patterns.append([(i & 1) ^ ((i >> 2) & 1) for i in range(N)])

        rng_init = random.Random(12345)
        for _ in range(4):
            patterns.append([rng_init.randint(0, 1) for _ in range(N)])

        scored = []
        for p in patterns:
            cost = count_mono(np.array(p, dtype=np.int8))
            scored.append((cost, p))
        scored.sort(key=lambda x: x[0])
        return [p for _, p in scored[:6]]

    initial_patterns = generate_structured_patterns()

    global_best_colors = initial_patterns[0][:]
    global_best_cost = count_mono(np.array(global_best_colors, dtype=np.int8))

    def parallel_tempering(initial_colors, num_replicas, time_limit, seed=None):
        """Parallel tempering (replica exchange) for spin glass optimization."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random

        # Initialize replicas: start from best and perturb
        replicas = []
        replica_costs = []
        for r in range(num_replicas):
            if r == 0:
                rep = initial_colors[:]
            elif r == 1:
                rep = initial_colors[:]
                # Small perturbation
                for _ in range(N // 10):
                    idx = rng.randint(0, N - 1)
                    rep[idx] = 1 - rep[idx]
            else:
                rep = initial_colors[:]
                # Larger perturbation for higher temperature replicas
                n_flips = int(N * (0.1 + 0.15 * r / max(1, num_replicas - 2)))
                for _ in range(n_flips):
                    idx = rng.randint(0, N - 1)
                    rep[idx] = 1 - rep[idx]
            replicas.append(rep)
            replica_costs.append(count_mono(np.array(rep, dtype=np.int8)))

        # Temperature schedule: geometric progression
        T_max = 3.0
        T_min = 0.005
        temperatures = np.geomspace(T_max, T_min, num_replicas)[::-1]  # index 0 = coldest

        best_cost = min(replica_costs)
        best_colors = replicas[replica_costs.index(best_cost)][:]

        end_time = time.time() + time_limit
        swap_interval = 50  # attempt swaps every this many iterations per replica

        iteration = 0
        while time.time() < end_time:
            # Update each replica
            for r in range(num_replicas):
                if time.time() > end_time:
                    break

                cols_np = np.array(replicas[r], dtype=np.int8)
                deltas = compute_all_deltas(cols_np)

                # Try to find a beneficial or neutral flip
                # Pick a random position, but bias toward negative deltas
                neg_mask = deltas < 0
                if np.any(neg_mask):
                    neg_positions = np.where(neg_mask)[0]
                    i = neg_positions[rng.randint(0, len(neg_positions) - 1)]
                    delta = int(deltas[i])
                    # Always accept if delta < 0
                    replicas[r][i] = 1 - replicas[r][i]
                    replica_costs[r] += delta
                    if replica_costs[r] < best_cost:
                        best_cost = replica_costs[r]
                        best_colors = replicas[r][:]
                else:
                    # No beneficial flips; try random flip with temperature-based acceptance
                    i = rng.randint(0, N - 1)
                    delta = int(deltas[i])
                    T = temperatures[r]
                    if delta <= 0:
                        replicas[r][i] = 1 - replicas[r][i]
                        replica_costs[r] += delta
                        if replica_costs[r] < best_cost:
                            best_cost = replica_costs[r]
                            best_colors = replicas[r][:]
                    else:
                        prob = math.exp(-delta / T)
                        if rng.random() < prob:
                            replicas[r][i] = 1 - replicas[r][i]
                            replica_costs[r] += delta
                            if replica_costs[r] < best_cost:
                                best_cost = replica_costs[r]
                                best_colors = replicas[r][:]

            iteration += 1

            # Attempt swaps between adjacent replicas
            if iteration % swap_interval == 0:
                for r in range(num_replicas - 1):
                    if time.time() > end_time:
                        break
                    # Swap between replica r (colder) and r+1 (hotter)
                    # Accept with probability min(1, exp((T_hot - T_cold) * (E_cold - E_hot)))
                    T_cold = temperatures[r]
                    T_hot = temperatures[r + 1]
                    E_cold = replica_costs[r]
                    E_hot = replica_costs[r + 1]

                    # Delta E for the swap: moving E_cold to hot chain, E_hot to cold chain
                    # Acceptance: exp((1/T_cold - 1/T_hot) * (E_hot - E_cold))
                    exponent = (1.0 / T_cold - 1.0 / T_hot) * (E_hot - E_cold)
                    if exponent > 0:
                        accept = True
                    else:
                        accept = rng.random() < math.exp(min(0, exponent))

                    if accept:
                        # Swap states
                        replicas[r], replicas[r + 1] = replicas[r + 1], replicas[r]
                        replica_costs[r], replica_costs[r + 1] = replica_costs[r + 1], replica_costs[r]

        return best_colors, best_cost

    # Run parallel tempering with multiple configurations
    num_restarts = 8
    time_per_restart = 130.0 / num_restarts

    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break

        if restart < len(initial_patterns):
            initial = initial_patterns[restart % len(initial_patterns)][:]
        elif restart < 6:
            initial = global_best_colors[:]
            n_flips = random.randint(N // 4, N // 2)
            for _ in range(n_flips):
                idx = random.randint(0, N - 1)
                initial[idx] = 1 - initial[idx]
        else:
            initial = global_best_colors[:]
            n_flips = random.randint(N // 8, N // 4)
            for _ in range(n_flips):
                idx = random.randint(0, N - 1)
                initial[idx] = 1 - initial[idx]

        num_replicas = 6 + (restart % 3)  # 6-8 replicas
        seed = 42 + restart * 137

        pt_colors, pt_cost = parallel_tempering(
            initial, num_replicas, time_per_restart * 0.75, seed=seed
        )

        polished_colors, polished_cost = greedy_polish(pt_colors, pt_cost)

        if polished_cost < global_best_cost:
            global_best_cost = polished_cost
            global_best_colors = polished_colors[:]

    # Final polish
    if time.time() < deadline - 5:
        final_colors, final_cost = greedy_polish(global_best_colors, global_best_cost)
        if final_cost < global_best_cost:
            global_best_cost = final_cost
            global_best_colors = final_colors

    return global_best_colors
