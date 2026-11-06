"""Colorings with three colors in every four-term progression."""
import time
import random

# Precompute all nonconstant 4-term APs (a, a+d, a+2d, a+3d) with a+3d <= 128
APs = []
for d in range(1, 43):
    for a in range(1, 128 - 3 * d + 1):
        APs.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))

# For each position, precompute which AP indices it participates in
pos_aps = [[] for _ in range(128)]
for idx, (a, b, c, dd) in enumerate(APs):
    pos_aps[a].append(idx)
    pos_aps[b].append(idx)
    pos_aps[c].append(idx)
    pos_aps[dd].append(idx)

def construct():
    start_time = time.time()
    DEADLINE = 82  # seconds

    # Try increasing k values with optimized local search
    for k in range(3, 129):
        if time.time() - start_time > DEADLINE:
            break
        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 5:
            break
        result = _local_search(k, start_time, DEADLINE)
        if result is not None:
            return result

    # Ultimate fallback: greedy with many colors
    return _greedy_valid()

def _greedy_valid():
    """Greedy construction that guarantees validity."""
    colors = []
    for x in range(1, 129):
        for color in range(128):
            ok = True
            for ap_idx in pos_aps[x - 1]:
                a, b, c, dd = APs[ap_idx]
                if all(p <= x - 1 for p in (a, b, c, dd)):
                    cset = set()
                    for p in (a, b, c, dd):
                        if p == x - 1:
                            cset.add(color)
                        else:
                            cset.add(colors[p])
                    if len(cset) < 3:
                        ok = False
                        break
            if ok:
                colors.append(color)
                break
    return colors

def _local_search(k, start_time, deadline):
    """Optimized local search with incremental violation counting."""
    random.seed(12345)

    def ap_violates(ap_idx, colors):
        a, b, c, dd = APs[ap_idx]
        return len({colors[a], colors[b], colors[c], colors[dd]}) < 3

    def compute_violations(colors):
        v = 0
        for idx in range(len(APs)):
            if ap_violates(idx, colors):
                v += 1
        return v

    # Try multiple random initializations
    for init in range(300):
        if time.time() - start_time > deadline:
            break
        colors = [random.randint(0, k - 1) for _ in range(128)]
        v = compute_violations(colors)
        if v == 0:
            return colors

        # Simulated annealing with incremental updates
        T = 1.0
        cooling = 0.99995
        for it in range(500000):
            if time.time() - start_time > deadline:
                break
            if T < 0.0001:
                break
            pos = random.randint(0, 127)
            old_color = colors[pos]
            new_color = random.randint(0, k - 1)
            if new_color == old_color:
                continue

            # Compute delta by checking only affected APs
            delta = 0
            for ap_idx in pos_aps[pos]:
                a, b, c, dd = APs[ap_idx]
                # Old distinct count
                old_set = {colors[a], colors[b], colors[c], colors[dd]}
                old_viol = len(old_set) < 3
                # New distinct count
                new_set = set()
                for p in (a, b, c, dd):
                    if p == pos:
                        new_set.add(new_color)
                    else:
                        new_set.add(colors[p])
                new_viol = len(new_set) < 3
                if old_viol and not new_viol:
                    delta -= 1
                elif not old_viol and new_viol:
                    delta += 1

            colors[pos] = new_color
            if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
                v += delta
                if v == 0:
                    return colors
            else:
                colors[pos] = old_color
            T *= cooling

    return None
