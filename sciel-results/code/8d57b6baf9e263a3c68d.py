"""New design: structured backbone + greedy/local color packing with deadline guard."""
import time
import random

N = 16
M = N * (N - 1) // 2  # 120 edges

# Precompute edge index mapping
def _edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

# Precompute all 4-cycles: for each set of 4 vertices, the 3 cycle edge-tuples
def _precompute_cycles():
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Cycle 1: a-b-c-d-a  -> edges ab, bc, cd, da
                    e1 = (_edge_idx(a, b), _edge_idx(b, c), _edge_idx(c, d), _edge_idx(d, a))
                    # Cycle 2: a-b-d-c-a  -> edges ab, bd, dc, ca
                    e2 = (_edge_idx(a, b), _edge_idx(b, d), _edge_idx(c, d), _edge_idx(a, c))
                    # Cycle 3: a-c-b-d-a  -> edges ac, cb, bd, da
                    e3 = (_edge_idx(a, c), _edge_idx(b, c), _edge_idx(b, d), _edge_idx(d, a))
                    cycles.append((e1, e2, e3))
    return cycles

_CYCLES = _precompute_cycles()
NUM_CYCLES = len(_CYCLES)  # C(16,4) = 1820, each with 3 cycles

# For each edge, precompute which (cycle, position) it appears in
def _edge_cycle_index():
    edge_to_cycles = [[] for _ in range(M)]
    for ci, (c1, c2, c3) in enumerate(_CYCLES):
        for cyc in (c1, c2, c3):
            for pos, e in enumerate(cyc):
                edge_to_cycles[e].append((ci, pos))
    return edge_to_cycles

_EDGE_TO_CYCLES = _edge_cycle_index()


def _count_colors(colors):
    return len(set(colors))


def _is_valid(colors):
    """Check no 4-cycle has 4 distinct colors."""
    for c1, c2, c3 in _CYCLES:
        if len(set([colors[c1[0]], colors[c1[1]], colors[c1[2]], colors[c1[3]]])) == 4:
            return False
        if len(set([colors[c2[0]], colors[c2[1]], colors[c2[2]], colors[c2[3]]])) == 4:
            return False
        if len(set([colors[c3[0]], colors[c3[1]], colors[c3[2]], colors[c3[3]]])) == 4:
            return False
    return True


def _can_add_color(colors, edge, new_color):
    """Check if setting colors[edge]=new_color keeps all 4-cycles valid."""
    old_color = colors[edge]
    if old_color == new_color:
        return True
    colors[edge] = new_color
    for ci, pos in _EDGE_TO_CYCLES[edge]:
        # Check all 3 cycles of this 4-vertex set
        c1, c2, c3 = _CYCLES[ci]
        for cyc in (c1, c2, c3):
            c0, c1v, c2v, c3v = (colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]])
            if len(set([c0, c1v, c2v, c3v])) == 4:
                colors[edge] = old_color
                return False
    colors[edge] = old_color
    return True


def _try_change(colors, edge, new_color):
    """Try to change edge to new_color; return True if valid and applied."""
    if _can_add_color(colors, edge, new_color):
        colors[edge] = new_color
        return True
    return False


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill

    random.seed(42)

    # Phase 1: Structured backbone using vertex partition
    # Partition 16 vertices into 4 groups of 4
    # Color edges within same group with one base color, between groups with another
    # This creates many 4-cycles that share colors
    groups = [list(range(4 * g, 4 * g + 4)) for g in range(4)]

    colors = [0] * M
    used_colors = {0}

    # Assign base colors based on group structure
    # Edges within group g: color = 10 + g
    # Edges between group g and h (g<h): color = 20 + g * 4 + h
    for g in range(4):
        for h in range(g + 1, 4):
            col = 20 + g * 4 + h
            for u in groups[g]:
                for v in groups[h]:
                    idx = _edge_idx(u, v)
                    colors[idx] = col
                    used_colors.add(col)
        # Within group
        col = 10 + g
        for i in range(4):
            for j in range(i + 1, 4):
                idx = _edge_idx(groups[g][i], groups[g][j])
                colors[idx] = col
                used_colors.add(col)

    current_count = _count_colors(colors)

    # Verify backbone is valid
    if not _is_valid(colors):
        # Fallback: simpler structure
        colors = [0] * M
        for i in range(N):
            for j in range(i + 1, N):
                colors[_edge_idx(i, j)] = (i + j) % 5
        current_count = _count_colors(colors)

    # Phase 2: Local search - try to introduce new colors
    # Greedy: for each edge, try to assign a new unique color
    new_color = max(used_colors) + 1
    max_colors = 120

    # Edge ordering: randomize for diversity
    edge_order = list(range(M))
    random.shuffle(edge_order)

    # Multiple passes
    pass_count = 0
    improved = True
    while improved and time.time() < deadline:
        improved = False
        pass_count += 1
        random.shuffle(edge_order)
        for e in edge_order:
            if time.time() > deadline:
                break
            # Try to assign a new color to this edge
            if new_color >= max_colors:
                break
            # Try the new color
            if _try_change(colors, e, new_color):
                used_colors.add(new_color)
                new_color += 1
                current_count += 1
                improved = True

    # Phase 3: Local search - try to swap colors to free up space for new colors
    # For each edge, try changing its color to something else, then try new color on another edge
    # Simpler: hill-climbing on number of colors
    for iteration in range(200):
        if time.time() > deadline:
            break
        if new_color >= max_colors:
            break
        random.shuffle(edge_order)
        found = False
        for e1 in edge_order:
            if time.time() > deadline:
                break
            old_c1 = colors[e1]
            # Try changing e1 to a different existing color
            for new_c1 in list(used_colors):
                if new_c1 == old_c1:
                    continue
                if not _can_add_color(colors, e1, new_c1):
                    continue
                colors[e1] = new_c1
                # Now try to add a new color to some other edge
                for e2 in edge_order:
                    if e2 == e1:
                        continue
                    if new_color < max_colors and _try_change(colors, e2, new_color):
                        used_colors.add(new_color)
                        new_color += 1
                        found = True
                        break
                    if time.time() > deadline:
                        break
                if found or time.time() > deadline:
                    break
            if found or time.time() > deadline:
                break
        if not found:
            break

    # Final: ensure colors are in range 0..119
    result = []
    for c in colors:
        if c < 0 or c > 119:
            result.append(0)
        else:
            result.append(c)

    return result


# Quick self-test
if __name__ == "__main__":
    import time
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s")
    print(f"Colors used: {len(set(res))}")
    print(f"Valid: {_is_valid(res)}")
    print(f"First 20: {res[:20]}")
