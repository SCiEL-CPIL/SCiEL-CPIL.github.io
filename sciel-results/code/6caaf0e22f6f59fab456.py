import time
import random
import numpy as np

def construct():
    n = 128
    start_time = time.time()
    deadline = start_time + 85.0

    # Precompute all 4-APs
    aps = []
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            if a + 3 * d > n:
                break
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))

    # For each position, list of AP indices it participates in
    pos_to_aps = [[] for _ in range(n)]
    for i, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(i)

    def count_violations(colors):
        violations = 0
        for ap in aps:
            c = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
            if len(c) < 3:
                violations += 1
        return violations

    best_colors = None
    best_num_colors = float('inf')

    # Try multiple random restarts
    for attempt in range(30):
        if time.time() > deadline - 15.0:
            break

        num_colors = 8 + attempt
        colors = np.random.randint(0, num_colors, size=n)
        current_colors = colors.copy()
        current_violations = count_violations(current_colors)

        # Simulated annealing
        temp = 1.0
        cooling = 0.9998
        for iteration in range(10000):
            if time.time() > deadline - 15.0:
                break

            pos = np.random.randint(0, n)
            old_color = current_colors[pos]
            new_color = np.random.randint(0, num_colors)

            if new_color == old_color:
                continue

            # Compute delta violations
            current_colors[pos] = new_color
            new_violations = 0
            for ap_idx in pos_to_aps[pos]:
                ap = aps[ap_idx]
                c = {current_colors[ap[0]], current_colors[ap[1]], current_colors[ap[2]], current_colors[ap[3]]}
                if len(c) < 3:
                    new_violations += 1

            current_colors[pos] = old_color
            old_violations = 0
            for ap_idx in pos_to_aps[pos]:
                ap = aps[ap_idx]
                c = {current_colors[ap[0]], current_colors[ap[1]], current_colors[ap[2]], current_colors[ap[3]]}
                if len(c) < 3:
                    old_violations += 1

            delta = new_violations - old_violations
            current_colors[pos] = new_color

            if delta < 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
                current_violations += delta
                if current_violations < best_num_colors:
                    pass
            else:
                current_colors[pos] = old_color

            temp *= cooling

        # Check if valid
        if current_violations == 0:
            actual_colors = np.unique(current_colors)
            if len(actual_colors) < best_num_colors:
                best_num_colors = len(actual_colors)
                best_colors = current_colors.copy()

    if best_colors is None:
        # Fallback to modular coloring
        m = 43
        best_colors = np.array([x % m for x in range(1, n + 1)])

    return best_colors.tolist()
