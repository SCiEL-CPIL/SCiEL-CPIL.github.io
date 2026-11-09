import time
import random
import numpy as np

def construct():
    n = 256
    # Keep a safety margin: evaluator kills at 150s, we target finishing by 110s.
    deadline = time.time() + 110.0
    max_d = (n - 3) // 3

    def count_mono(colors):
        count = 0
        for d in range(1, max_d + 1):
            for a in range(n - 3 * d):
                if colors[a] == colors[a + d] == colors[a + 2 * d] == colors[a + 3 * d]:
                    count += 1
        return count

    def delta_flip(colors, x):
        delta = 0
        for d in range(1, max_d + 1):
            for k in range(4):
                a = x - k * d
                if a < 0:
                    continue
                if a + 3 * d >= n:
                    continue
                v0 = colors[a]
                v1 = colors[a + d]
                v2 = colors[a + 2 * d]
                v3 = colors[a + 3 * d]
                old_mono = 1 if (v0 == v1 == v2 == v3) else 0
                if k == 0:
                    nv0 = 1 - v0
                    nv1 = v1
                    nv2 = v2
                    nv3 = v3
                elif k == 1:
                    nv0 = v0
                    nv1 = 1 - v1
                    nv2 = v2
                    nv3 = v3
                elif k == 2:
                    nv0 = v0
                    nv1 = v1
                    nv2 = 1 - v2
                    nv3 = v3
                else:
                    nv0 = v0
                    nv1 = v1
                    nv2 = v2
                    nv3 = 1 - v3
                new_mono = 1 if (nv0 == nv1 == nv2 == nv3) else 0
                delta += new_mono - old_mono
        return delta

    # Build a few structured initial colorings (multi-start within the same design family)
    def thue_morse():
        return np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)

    def block2():
        # Alternate colours in blocks of two (from inspiration)
        return np.array([(i // 2) % 2 for i in range(n)], dtype=np.int8)

    def block4():
        # Alternate colours in blocks of four
        return np.array([(i // 4) % 2 for i in range(n)], dtype=np.int8)

    def random_init():
        return np.random.randint(0, 2, size=n).astype(np.int8)

    seeds = [thue_morse(), block2(), block4(), random_init()]

    best_cost = float('inf')
    best_colors = None

    for colors in seeds:
        if time.time() >= deadline:
            break
        current_cost = count_mono(colors)
        if current_cost < best_cost:
            best_cost = current_cost
            best_colors = colors.copy()

        # Run a short SA from this seed
        T = 1.0
        cool_rate = 0.99
        min_T = 0.01
        local_best_cost = current_cost
        local_best_colors = colors.copy()

        while time.time() < deadline:
            x = random.randint(0, n - 1)
            d_cost = delta_flip(colors, x)

            if d_cost < 0:
                colors[x] = 1 - colors[x]
                current_cost += d_cost
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

            T *= cool_rate

        if local_best_cost < best_cost:
            best_cost = local_best_cost
            best_colors = local_best_colors.copy()

    return best_colors.tolist()
