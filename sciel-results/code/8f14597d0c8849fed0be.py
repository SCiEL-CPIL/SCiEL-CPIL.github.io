import time
import random
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 110.0
    max_d = (n - 3) // 3

    # Precompute all 4-APs as (a, a+d, a+2d, a+3d)
    _ap_list = []
    for d in range(1, max_d + 1):
        for a in range(n - 3 * d):
            _ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    AP = np.array(_ap_list, dtype=np.int32)
    M = AP.shape[0]

    # Precompute for each position x: list of (ap_index, k) where x is the k-th element
    # This allows vectorized delta computation
    pos_to_aps = [[] for _ in range(n)]
    for idx in range(M):
        a, b, c, d = AP[idx]
        pos_to_aps[a].append((idx, 0))
        pos_to_aps[b].append((idx, 1))
        pos_to_aps[c].append((idx, 2))
        pos_to_aps[d].append((idx, 3))

    def count_mono(colors):
        c = colors[AP]
        eq = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
        return int(np.count_nonzero(eq))

    def delta_flip(colors, x):
        """Compute change in mono-AP count if we flip color at position x, using precomputed AP lists."""
        if not pos_to_aps[x]:
            return 0
        c = colors
        delta = 0
        for ap_idx, k in pos_to_aps[x]:
            a, b, cc, dd = AP[ap_idx]
            v0 = c[a]
            v1 = c[b]
            v2 = c[cc]
            v3 = c[dd]
            old_mono = (v0 == v1) and (v1 == v2) and (v2 == v3)
            # Compute new values after flipping position x
            if k == 0:
                nv0 = 1 - v0
                new_mono = (nv0 == v1) and (v1 == v2) and (v2 == v3)
            elif k == 1:
                nv1 = 1 - v1
                new_mono = (v0 == nv1) and (nv1 == v2) and (v2 == v3)
            elif k == 2:
                nv2 = 1 - v2
                new_mono = (v0 == v1) and (v1 == nv2) and (nv2 == v3)
            else:
                nv3 = 1 - v3
                new_mono = (v0 == v1) and (v1 == v2) and (v2 == nv3)
            delta += (1 if new_mono else 0) - (1 if old_mono else 0)
        return delta

    # Build structured initial colorings
    def thue_morse():
        return np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)

    def block2():
        return np.array([(i // 2) % 2 for i in range(n)], dtype=np.int8)

    def block4():
        return np.array([(i // 4) % 2 for i in range(n)], dtype=np.int8)

    def block8():
        return np.array([(i // 8) % 2 for i in range(n)], dtype=np.int8)

    def block16():
        return np.array([(i // 16) % 2 for i in range(n)], dtype=np.int8)

    def random_init():
        return np.random.randint(0, 2, size=n).astype(np.int8)

    # Generate multiple random seeds for diversity
    seeds = [thue_morse(), block2(), block4(), block8(), block16()]
    for _ in range(3):
        seeds.append(random_init())

    best_cost = float('inf')
    best_colors = None

    # Allocate time budget: ~75% for SA multi-start, ~25% for final polish
    total_budget = 110.0
    sa_budget = total_budget * 0.75
    sa_deadline = time.time() + sa_budget

    for seed_idx, colors in enumerate(seeds):
        if time.time() >= sa_deadline:
            break
        current_cost = count_mono(colors)
        if current_cost < best_cost:
            best_cost = current_cost
            best_colors = colors.copy()

        # Run SA from this seed with adaptive temperature
        T = 2.0
        cool_rate = 0.997
        min_T = 0.005
        local_best_cost = current_cost
        local_best_colors = colors.copy()
        no_improve_steps = 0
        max_no_improve = 5000

        while time.time() < sa_deadline:
            x = random.randint(0, n - 1)
            d_cost = delta_flip(colors, x)

            if d_cost < 0:
                colors[x] = 1 - colors[x]
                current_cost += d_cost
                no_improve_steps = 0
                if current_cost < local_best_cost:
                    local_best_cost = current_cost
                    local_best_colors = colors.copy()
            else:
                if T > min_T:
                    p = np.exp(-d_cost / T)
                    if random.random() < p:
                        colors[x] = 1 - colors[x]
                        current_cost += d_cost
                        if current_cost < local_best_cost:
                            local_best_cost = current_cost
                            local_best_colors = colors.copy()
                            no_improve_steps = 0
                        else:
                            no_improve_steps += 1
                    else:
                        no_improve_steps += 1
                else:
                    no_improve_steps += 1

            T *= cool_rate

            # Restart if stuck
            if no_improve_steps > max_no_improve:
                T = max(T * 2.0, 1.0)
                no_improve_steps = 0

        if local_best_cost < best_cost:
            best_cost = local_best_cost
            best_colors = local_best_colors.copy()

    # Final polish: multi-pass coordinate descent with random restarts of order
    polish_deadline = time.time() + (total_budget - sa_budget)
    if best_colors is not None and time.time() < polish_deadline:
        colors = best_colors.copy()
        current_cost = best_cost

        # Multiple polish passes with different sweep orders
        for pass_num in range(5):
            if time.time() >= polish_deadline:
                break
            improved = True
            while improved and time.time() < polish_deadline:
                improved = False
                # Alternate between forward and reverse sweep, and random order
                if pass_num % 3 == 0:
                    order = list(range(n))
                elif pass_num % 3 == 1:
                    order = list(range(n - 1, -1, -1))
                else:
                    order = list(range(n))
                    random.shuffle(order)

                for x in order:
                    if time.time() >= polish_deadline:
                        break
                    d_cost = delta_flip(colors, x)
                    if d_cost < 0:
                        colors[x] = 1 - colors[x]
                        current_cost += d_cost
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_colors = colors.copy()

        # If we still have time, do a short SA polish from the best solution
        if time.time() < polish_deadline - 5:
            colors = best_colors.copy()
            current_cost = best_cost
            T = 0.5
            cool_rate = 0.999
            min_T = 0.001
            local_best_cost = current_cost
            local_best_colors = colors.copy()
            no_improve = 0

            while time.time() < polish_deadline - 2:
                x = random.randint(0, n - 1)
                d_cost = delta_flip(colors, x)
                if d_cost < 0:
                    colors[x] = 1 - colors[x]
                    current_cost += d_cost
                    no_improve = 0
                    if current_cost < local_best_cost:
                        local_best_cost = current_cost
                        local_best_colors = colors.copy()
                else:
                    if T > min_T:
                        p = np.exp(-d_cost / T)
                        if random.random() < p:
                            colors[x] = 1 - colors[x]
                            current_cost += d_cost
                            if current_cost < local_best_cost:
                                local_best_cost = current_cost
                                local_best_colors = colors.copy()
                                no_improve = 0
                            else:
                                no_improve += 1
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1
                T *= cool_rate
                if no_improve > 3000:
                    T = max(T * 1.5, 0.3)
                    no_improve = 0

            if local_best_cost < best_cost:
                best_cost = local_best_cost
                best_colors = local_best_colors.copy()

    return best_colors.tolist()
