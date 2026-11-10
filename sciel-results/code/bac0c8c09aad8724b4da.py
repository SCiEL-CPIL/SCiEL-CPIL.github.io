import time
import random
import numpy as np
from itertools import combinations

N = 16
NEDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 120

def _get_edge_indices():
    idx_map = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            idx_map[(i, j)] = idx
            idx_map[(j, i)] = idx
            idx += 1

    cycles = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        cycles.append((idx_map[(a,b)], idx_map[(b,c)], idx_map[(c,d)], idx_map[(d,a)]))
        cycles.append((idx_map[(a,b)], idx_map[(b,d)], idx_map[(c,d)], idx_map[(a,c)]))
        cycles.append((idx_map[(a,c)], idx_map[(b,c)], idx_map[(b,d)], idx_map[(a,d)]))

    cycles_np = np.array(cycles, dtype=np.int32)

    edge_to_cycles = [[] for _ in range(NEDGES)]
    for ci, cycle in enumerate(cycles):
        for e in cycle:
            edge_to_cycles[e].append(ci)

    # For each edge, precompute the other edges in each cycle it belongs to
    edge_cycle_info = []
    for e in range(NEDGES):
        info = []
        for ci in edge_to_cycles[e]:
            cycle = cycles_np[ci]
            pos = -1
            for k in range(4):
                if cycle[k] == e:
                    pos = k
                    break
            other = [int(cycle[k]) for k in range(4) if k != pos]
            info.append((ci, other))
        edge_cycle_info.append(info)

    return idx_map, cycles_np, edge_to_cycles, edge_cycle_info

IDX_MAP, CYCLES, EDGE_TO_CYCLES, EDGE_CYCLE_INFO = _get_edge_indices()
NUM_CYCLES = len(CYCLES)

def _is_valid_full(colors_np):
    cycle_colors = colors_np[CYCLES]
    sorted_colors = np.sort(cycle_colors, axis=1)
    adjacent_equal = (sorted_colors[:, 0] == sorted_colors[:, 1]) | \
                     (sorted_colors[:, 1] == sorted_colors[:, 2]) | \
                     (sorted_colors[:, 2] == sorted_colors[:, 3])
    return bool(np.all(adjacent_equal))

def _is_valid_change(colors, e, new_color):
    """Check if setting colors[e]=new_color keeps validity, only checking affected cycles."""
    old_color = colors[e]
    if new_color == old_color:
        return True
    colors[e] = new_color
    for ci, other_edges in EDGE_CYCLE_INFO[e]:
        c0 = new_color
        c1 = colors[other_edges[0]]
        c2 = colors[other_edges[1]]
        c3 = colors[other_edges[2]]
        # Check if all four are distinct
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            colors[e] = old_color
            return False
    return True

def _count_distinct(colors_np):
    return len(np.unique(colors_np))

def _initial_valid_coloring():
    colors = np.zeros(NEDGES, dtype=np.int32)
    for i in range(N):
        for j in range(i + 1, N):
            idx = IDX_MAP[(i, j)]
            if i >= 8 and j >= 8:
                colors[idx] = 1
            else:
                colors[idx] = 0
    return colors

def _initial_algebraic_coloring(K):
    colors = np.zeros(NEDGES, dtype=np.int32)
    for i in range(N):
        for j in range(i + 1, N):
            idx = IDX_MAP[(i, j)]
            colors[idx] = (i + j) % K
    return colors

def _local_search(colors, deadline, rng, color_counts=None):
    best_colors = colors.copy()
    best_distinct = _count_distinct(colors)
    best_counts = np.bincount(colors, minlength=MAX_COLOR) if color_counts is None else color_counts.copy()

    # Track distinct count incrementally
    current_counts = np.bincount(colors, minlength=MAX_COLOR)
    current_distinct = int(np.sum(current_counts > 0))

    while time.time() < deadline:
        improved = False
        edge_order = list(range(NEDGES))
        rng.shuffle(edge_order)

        for e in edge_order:
            if time.time() >= deadline:
                break

            old_color = colors[e]

            # Compute allowed colors via constraint propagation
            allowed_colors = None
            for ci, other_edges in EDGE_CYCLE_INFO[e]:
                other_colors = [colors[oe] for oe in other_edges]
                other_set = set(other_colors)
                if len(other_set) == 3:
                    if allowed_colors is None:
                        allowed_colors = set(other_set)
                    else:
                        allowed_colors &= other_set

            if allowed_colors is not None:
                candidates = [c for c in allowed_colors if c != old_color]
                if not candidates:
                    continue
            else:
                # Prefer unused colors, then low-frequency
                unused = [c for c in range(MAX_COLOR) if current_counts[c] == 0]
                if unused:
                    rng.shuffle(unused)
                    candidates = unused[:15]
                else:
                    # Low frequency colors
                    used = [c for c in range(MAX_COLOR) if current_counts[c] > 0]
                    used.sort(key=lambda c: current_counts[c])
                    candidates = used[:15]
                candidates = [c for c in candidates if c != old_color]
                if not candidates:
                    continue

            new_color = rng.choice(candidates)

            if _is_valid_change(colors, e, new_color):
                # Update counts
                current_counts[old_color] -= 1
                current_counts[new_color] += 1
                if current_counts[old_color] == 0:
                    current_distinct -= 1
                if current_counts[new_color] == 1:
                    current_distinct += 1

                if current_distinct > best_distinct:
                    best_distinct = current_distinct
                    best_colors = colors.copy()
                    best_counts = current_counts.copy()
                    improved = True
            else:
                pass  # colors[e] was restored by _is_valid_change

        if not improved:
            # Perturbation
            num_perturb = rng.randint(1, 4)
            for _ in range(num_perturb):
                e = rng.randint(0, NEDGES - 1)
                old_color = colors[e]
                new_color = rng.randint(0, MAX_COLOR - 1)
                if _is_valid_change(colors, e, new_color):
                    current_counts[old_color] -= 1
                    current_counts[new_color] += 1
                    if current_counts[old_color] == 0:
                        current_distinct -= 1
                    if current_counts[new_color] == 1:
                        current_distinct += 1
                else:
                    pass

    return best_colors, best_distinct

def _polish(colors, deadline, rng):
    """Final polish: careful single-edge sweep to maximize distinct colors."""
    best_colors = colors.copy()
    best_distinct = _count_distinct(colors)
    current_counts = np.bincount(colors, minlength=MAX_COLOR)
    current_distinct = int(np.sum(current_counts > 0))

    sweeps = 0
    while time.time() < deadline:
        improved = False
        edge_order = list(range(NEDGES))
        rng.shuffle(edge_order)

        for e in edge_order:
            if time.time() >= deadline:
                break
            old_color = colors[e]
            # Try all unused colors first
            for c in range(MAX_COLOR):
                if c == old_color or current_counts[c] != 0:
                    continue
                if _is_valid_change(colors, e, c):
                    current_counts[old_color] -= 1
                    current_counts[c] += 1
                    current_distinct += 1
                    if current_distinct > best_distinct:
                        best_distinct = current_distinct
                        best_colors = colors.copy()
                        best_counts = current_counts.copy()
                    improved = True
                    break  # Only change once per edge per sweep

        if not improved:
            break
        sweeps += 1

    return best_colors, best_distinct

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin (90s limit)

    best_overall_colors = None
    best_overall_distinct = 0

    rng = random.Random(42)

    # Try algebraic colorings first
    for K in range(2, 25):
        colors = _initial_algebraic_coloring(K)
        if _is_valid_full(colors):
            distinct = _count_distinct(colors)
            if distinct > best_overall_distinct:
                best_overall_distinct = distinct
                best_overall_colors = colors.copy()

    # Also try some structural colorings
    # Coloring by (i//2 + j//2) % K
    for K in range(3, 15):
        colors = np.zeros(NEDGES, dtype=np.int32)
        for i in range(N):
            for j in range(i + 1, N):
                colors[IDX_MAP[(i, j)]] = (i // 2 + j // 2) % K
        if _is_valid_full(colors):
            distinct = _count_distinct(colors)
            if distinct > best_overall_distinct:
                best_overall_distinct = distinct
                best_overall_colors = colors.copy()

    # Multi-start local search with population diversity
    num_restarts = 0
    while time.time() < deadline - 8.0:
        # Generate diverse initial coloring
        r = rng.random()
        if r < 0.4:
            K = rng.randint(2, 12)
            colors = _initial_algebraic_coloring(K)
            if not _is_valid_full(colors):
                colors = _initial_valid_coloring()
        elif r < 0.7:
            colors = _initial_valid_coloring()
        else:
            # Perturbed algebraic
            K = rng.randint(2, 10)
            colors = _initial_algebraic_coloring(K)
            if not _is_valid_full(colors):
                colors = _initial_valid_coloring()

        # Add random valid perturbations
        for _ in range(rng.randint(5, 20)):
            e = rng.randint(0, NEDGES - 1)
            new_color = rng.randint(0, MAX_COLOR - 1)
            if _is_valid_change(colors, e, new_color):
                pass
            else:
                colors[e] = colors[e]  # unchanged

        # Run local search
        local_deadline = min(time.time() + 6.0, deadline - 6.0)
        colors, distinct = _local_search(colors, local_deadline, rng)

        if distinct > best_overall_distinct:
            best_overall_distinct = distinct
            best_overall_colors = colors.copy()

        num_restarts += 1

    # Final polish on the best coloring
    if best_overall_colors is not None:
        polish_deadline = deadline - 1.0
        best_overall_colors, best_overall_distinct = _polish(
            best_overall_colors, polish_deadline, rng
        )

    if best_overall_colors is None:
        best_overall_colors = _initial_valid_coloring()

    return best_overall_colors.tolist()
