import math
import time
import random

def construct():
    N = 256
    deadline = time.time() + 110

    # Precompute all 4-APs
    aps_at = [[] for _ in range(N)]

    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(max_a):
            idxs = (a, a+d, a+2*d, a+3*d)
            for i in range(4):
                cur = idxs[i]
                others = [idxs[j] for j in range(4) if j != i]
                aps_at[cur].append(others)

    # Store all APs for counting
    aps = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(max_a):
            idxs = (a, a+d, a+2*d, a+3*d)
            aps.append(idxs)

    def count_mono(cols):
        cnt = 0
        for ap in aps:
            if cols[ap[0]] == cols[ap[1]] == cols[ap[2]] == cols[ap[3]]:
                cnt += 1
        return cnt

    def delta_for_flip(cols, i):
        """Calculate cost delta if we flip position i."""
        c_i = cols[i]
        delta = 0
        for others in aps_at[i]:
            j, k, l = others
            c_j = cols[j]
            c_k = cols[k]
            c_l = cols[l]
            if c_j == c_k == c_l:
                if c_i == c_j:
                    delta -= 1
                else:
                    delta += 1
        return delta

    def delta_for_multi_flip(cols, indices):
        """Calculate cost delta for flipping multiple positions simultaneously."""
        n = len(indices)
        if n == 1:
            return delta_for_flip(cols, indices[0])
        
        # For multi-flip, we need to consider interactions
        # An AP contributes to the delta if exactly 3 of its 4 positions are in the flip set
        # or if all 4 are in the flip set
        delta = 0
        flip_set = set(indices)
        
        for ap in aps:
            in_flip = sum(1 for idx in ap if idx in flip_set)
            if in_flip == 0 or in_flip == 4:
                continue
            
            # Get current colors of all 4 positions
            colors_ap = [cols[idx] for idx in ap]
            
            # Determine which positions are being flipped
            flip_mask = [1 if idx in flip_set else 0 for idx in ap]
            
            # Count how many are same-colored in current state
            c0 = colors_ap[0]
            current_mono = all(c == c0 for c in colors_ap)
            
            # After flipping, the new colors
            new_colors = [colors_ap[i] ^ flip_mask[i] for i in range(4)]
            nc0 = new_colors[0]
            new_mono = all(c == nc0 for c in new_colors)
            
            if new_mono and not current_mono:
                delta += 1
            elif current_mono and not new_mono:
                delta -= 1
        
        return delta

    def greedy_polish(colors, current_cost):
        """Try all single flips and apply any that reduce cost. Repeat until no improvement."""
        improved = True
        while improved:
            improved = False
            for i in range(N):
                d = delta_for_flip(colors, i)
                if d < 0:
                    colors[i] = 1 - colors[i]
                    current_cost += d
                    improved = True
        return colors, current_cost

    def greedy_multi_polish(colors, current_cost):
        """Try pairs of flips that reduce cost."""
        improved = True
        while improved:
            improved = False
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                for j in range(i+1, N):
                    d = delta_for_multi_flip(colors, [i, j])
                    if d < 0:
                        colors[i] = 1 - colors[i]
                        colors[j] = 1 - colors[j]
                        current_cost += d
                        improved = True
                        break
                if improved:
                    break
        return colors, current_cost

    def simulated_annealing(initial_colors, T_start, cooling, min_T, time_limit):
        colors = initial_colors[:]
        current_cost = count_mono(colors)
        best_cost = current_cost
        best_colors = colors[:]

        T = T_start
        end_time = time.time() + time_limit

        while time.time() < end_time:
            # Choose move type: 80% single flip, 15% double flip, 5% triple flip
            move_type = random.random()
            
            if move_type < 0.80:
                i = random.randint(0, N - 1)
                delta = delta_for_flip(colors, i)
                indices = [i]
            elif move_type < 0.95:
                i = random.randint(0, N - 1)
                j = random.randint(0, N - 1)
                if i == j:
                    j = (j + 1) % N
                delta = delta_for_multi_flip(colors, [i, j])
                indices = [i, j]
            else:
                i = random.randint(0, N - 1)
                j = random.randint(0, N - 1)
                k = random.randint(0, N - 1)
                # Ensure distinct
                if i == j:
                    j = (j + 1) % N
                if i == k or j == k:
                    k = (k + 1) % N
                if i == j:
                    j = (j + 1) % N
                if i == k or j == k:
                    k = (k + 1) % N
                delta = delta_for_multi_flip(colors, [i, j, k])
                indices = [i, j, k]

            if delta < 0:
                for idx in indices:
                    colors[idx] = 1 - colors[idx]
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
            elif delta == 0:
                for idx in indices:
                    colors[idx] = 1 - colors[idx]
            else:
                if T > 0:
                    prob = math.exp(-delta / T)
                else:
                    prob = 0.0

                if random.random() < prob:
                    for idx in indices:
                        colors[idx] = 1 - colors[idx]
                    current_cost += delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = colors[:]

            T *= cooling
            if T < min_T:
                T = min_T

        return best_colors, best_cost

    # Global best
    global_best_colors = None
    global_best_cost = float('inf')

    # Multiple restarts with different strategies
    num_restarts = 10
    time_per_restart = 105.0 / num_restarts

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Different initial colorings for diversity
        if restart == 0:
            # All zeros
            initial = [0] * N
        elif restart == 1:
            # All ones
            initial = [1] * N
        elif restart == 2:
            # Alternating
            initial = [i % 2 for i in range(N)]
        elif restart == 3:
            # Block pattern (size 8)
            initial = [(i // 8) % 2 for i in range(N)]
        elif restart == 4:
            # Block pattern (size 16)
            initial = [(i // 16) % 2 for i in range(N)]
        elif restart == 5:
            # Random
            initial = [random.randint(0, 1) for _ in range(N)]
        elif restart == 6:
            # Sinusoidal pattern
            initial = [1 if math.sin(i * 0.1) > 0 else 0 for i in range(N)]
        else:
            # Perturb previous best
            if global_best_colors is not None:
                initial = global_best_colors[:]
                num_flips = N // 6 + random.randint(0, N // 8)
                for _ in range(num_flips):
                    idx = random.randint(0, N - 1)
                    initial[idx] = 1 - initial[idx]
            else:
                initial = [random.randint(0, 1) for _ in range(N)]

        # Simulated annealing with varying parameters
        T_start = 0.8 + 0.4 * (restart % 3)
        cooling = 0.99985 - 0.00002 * (restart % 5)
        min_T = 0.003

        sa_colors, sa_cost = simulated_annealing(
            initial, T_start, cooling, min_T, time_per_restart * 0.75
        )

        # Greedy polish (single flips)
        polished_colors, polished_cost = greedy_polish(sa_colors, sa_cost)

        # Multi-flip polish
        if time.time() < deadline - 10:
            polished_colors, polished_cost = greedy_multi_polish(polished_colors, polished_cost)

        if polished_cost < global_best_cost:
            global_best_cost = polished_cost
            global_best_colors = polished_colors[:]

    # Final polish on the best result
    if global_best_colors is not None:
        final_colors, final_cost = greedy_polish(global_best_colors, global_best_cost)
        if final_cost < global_best_cost:
            global_best_cost = final_cost
            global_best_colors = final_colors
        
        # Final multi-flip polish
        if time.time() < deadline - 5:
            final_colors, final_cost = greedy_multi_polish(final_colors, final_cost)
            if final_cost < global_best_cost:
                global_best_cost = final_cost
                global_best_colors = final_colors

    return global_best_colors
