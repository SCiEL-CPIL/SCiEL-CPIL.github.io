import time
import random
import numpy as np
from itertools import combinations

N = 16
E = N * (N - 1) // 2  # 120

def _edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

def _build_cycles_array():
    cycles = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        # Three 4-cycles on {a,b,c,d}
        cycles.append((_edge_index(a,b), _edge_index(b,c), _edge_index(c,d), _edge_index(d,a)))
        cycles.append((_edge_index(a,b), _edge_index(b,d), _edge_index(d,c), _edge_index(c,a)))
        cycles.append((_edge_index(a,c), _edge_index(c,b), _edge_index(b,d), _edge_index(d,a)))
    return np.array(cycles, dtype=np.int32)

_CYCLES = _build_cycles_array()

def _rainbow_count(colors_np):
    """Count number of rainbow 4-cycles (all 4 edges distinct colors)."""
    cycle_colors = colors_np[_CYCLES]
    sorted_c = np.sort(cycle_colors, axis=1)
    rainbow = (sorted_c[:,0] < sorted_c[:,1]) & (sorted_c[:,1] < sorted_c[:,2]) & (sorted_c[:,2] < sorted_c[:,3])
    return int(np.sum(rainbow))

def _distinct_count(colors_np):
    return len(set(colors_np.tolist()))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    random.seed(42)
    np.random.seed(42)

    best_colors = None
    best_distinct = 0

    # Precompute edge-to-cycle mapping for efficiency
    # For each edge, which cycles does it appear in?
    edge_cycles = [[] for _ in range(E)]
    for ci in range(len(_CYCLES)):
        for ei in _CYCLES[ci]:
            edge_cycles[ei].append(ci)

    for restart in range(500):
        if time.time() > deadline - 10.0:
            break

        # Initialize with structured coloring
        colors = np.zeros(E, dtype=np.int32)
        for i in range(N):
            for j in range(i + 1, N):
                idx = _edge_index(i, j)
                # Structured initial coloring
                colors[idx] = (i * 5 + j * 3) % 12

        # Local search
        for iteration in range(200000):
            if time.time() > deadline - 5.0:
                break

            rainbows = _rainbow_count(colors)

            if rainbows == 0:
                distinct = _distinct_count(colors)
                if distinct > best_distinct:
                    best_distinct = distinct
                    best_colors = colors.copy()

                # Try to increase diversity: pick random edge, assign new color
                e = random.randint(0, E - 1)
                old_c = colors[e]
                # Try a color not currently used, or a random one
                new_c = random.randint(0, 119)
                if new_c != old_c:
                    colors[e] = new_c
                    if _rainbow_count(colors) == 0:
                        distinct = _distinct_count(colors)
                        if distinct > best_distinct:
                            best_distinct = distinct
                            best_colors = colors.copy()
                    else:
                        colors[e] = old_c
            else:
                # Fix violations: pick a random rainbow cycle, recolor one edge
                cycle_colors = colors[_CYCLES]
                sorted_c = np.sort(cycle_colors, axis=1)
                rainbow_mask = (sorted_c[:, 0] < sorted_c[:, 1]) & \
                               (sorted_c[:, 1] < sorted_c[:, 2]) & \
                               (sorted_c[:, 2] < sorted_c[:, 3])
                rainbow_idx = np.where(rainbow_mask)[0]
                if len(rainbow_idx) == 0:
                    break
                ci = random.choice(rainbow_idx)
                e1, e2, e3, e4 = _CYCLES[ci]
                # Recolor one edge to match another in the cycle
                ei = random.choice([e1, e2, e3, e4])
                others = [e for e in [e1, e2, e3, e4] if e != ei]
                match_e = random.choice(others)
                colors[ei] = colors[match_e]

        if time.time() > deadline - 10.0:
            break

    if best_colors is not None:
        return best_colors.tolist()
    else:
        return [0] * E
