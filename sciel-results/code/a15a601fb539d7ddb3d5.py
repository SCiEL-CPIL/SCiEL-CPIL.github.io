"""Search for max n with 4-colouring of K_n having no monochromatic C6."""
import time
import random

_DEADLINE = 115.0  # safety margin under 120s
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

# Precompute all 6-cycles (as tuples of vertex indices, closed) up to n=20.
# A 6-cycle is a sequence (v0,v1,v2,v3,v4,v5) of distinct vertices; the cycle
# edges are the 6 consecutive pairs (v_i, v_{i+1 mod 6}).
# We enumerate canonical cycles to avoid duplicates up to rotation+reversal.
def _all_c6(n):
    cycles = []
    # Fix v0=0 to break rotation symmetry.
    # Then fix v1 < v5 to break reversal symmetry.
    for v1 in range(1, n):
        for v2 in range(1, n):
            if v2 == v1: continue
            for v3 in range(1, n):
                if v3 in (v1, v2): continue
                for v4 in range(1, n):
                    if v4 in (v1, v2, v3): continue
                    for v5 in range(1, n):
                        if v5 in (v1, v2, v3, v4): continue
                        if v5 < v1: continue  # reversal break
                        cyc = (0, v1, v2, v3, v4, v5)
                        cycles.append(cyc)
    return cycles

_C6_CACHE = {}
def _cycles_for(n):
    if n not in _C6_CACHE:
        _C6_CACHE[n] = _all_c6(n)
    return _C6_CACHE[n]

def _edge_index(n, i, j):
    if i > j: i, j = j, i
    return i * n - i * (i + 1) // 2 + (j - i - 1)

def _is_valid(colors, n):
    """Check no colour class contains a C6. colors is list of length n*(n-1)//2."""
    cycles = _cycles_for(n)
    for cyc in cycles:
        v = cyc
        c0 = colors[_edge_index(n, v[0], v[1])]
        c1 = colors[_edge_index(n, v[1], v[2])]
        c2 = colors[_edge_index(n, v[2], v[3])]
        c3 = colors[_edge_index(n, v[3], v[4])]
        c4 = colors[_edge_index(n, v[4], v[5])]
        c5 = colors[_edge_index(n, v[5], v[0])]
        if c0 == c1 == c2 == c3 == c4 == c5:
            return False
    return True

def _backtrack(n, max_nodes=2000000):
    """Backtracking search for a valid 4-colouring of K_n.
    Returns colors list or None."""
    m = n * (n - 1) // 2
    colors = [0] * m
    cycles = _cycles_for(n)
    # Map each edge to the list of (cycle_pos, other_edges_needed) for quick conflict check.
    # For each edge e and each cycle containing e, record the other 5 edges of that cycle.
    edge_to_cycles = [[] for _ in range(m)]
    for cyc in cycles:
        v = cyc
        edges = []
        for k in range(6):
            a, b = v[k], v[(k + 1) % 6]
            edges.append(_edge_index(n, a, b))
        for k in range(6):
            others = [edges[j] for j in range(6) if j != k]
            edge_to_cycles[edges[k]].append(others)

    nodes = [0]
    def conflict_check(e, c):
        for others in edge_to_cycles[e]:
            if all(colors[oe] == c for oe in others):
                return True
        return False

    def bt(pos):
        if _remaining() < 5.0:
            return False
        nodes[0] += 1
        if nodes[0] % 20000 == 0 and nodes[0] > max_nodes:
            return False
        if pos == m:
            return True
        e = pos
        # Try colours in order 0,1,2,3
        for c in range(4):
            if conflict_check(e, c):
                continue
            colors[e] = c
            if bt(pos + 1):
                return True
        colors[e] = -1
        return False

    if bt(0):
        return colors
    return None

def _min_conflicts(colors, n, max_iter=200000):
    """Local search: minimise number of monochromatic C6s. Returns (colors, conflicts)."""
    m = n * (n - 1) // 2
    cycles = _cycles_for(n)
    # Build edge -> list of (cycle_idx, pos)
    edge_to_cycles = [[] for _ in range(m)]
    for ci, cyc in enumerate(cycles):
        v = cyc
        for k in range(6):
            a, b = v[k], v[(k + 1) % 6]
            e = _edge_index(n, a, b)
            edge_to_cycles[e].append((ci, k))

    def count_conflicts(cols):
        cnt = 0
        for ci, cyc in enumerate(cycles):
            v = cyc
            c0 = cols[_edge_index(n, v[0], v[1])]
            if c0 != cols[_edge_index(n, v[1], v[2])]: continue
            if c0 != cols[_edge_index(n, v[2], v[3])]: continue
            if c0 != cols[_edge_index(n, v[3], v[4])]: continue
            if c0 != cols[_edge_index(n, v[4], v[5])]: continue
            if c0 != cols[_edge_index(n, v[5], v[0])]: continue
            cnt += 1
        return cnt

    cur = colors[:]
    cur_conf = count_conflicts(cur)
    if cur_conf == 0:
        return cur, 0
    best = cur[:]
    best_conf = cur_conf
    for it in range(max_iter):
        if _remaining() < 5.0:
            break
        if it % 1000 == 0 and count_conflicts(best) == 0:
            return best, 0
        # Pick a random conflicting edge from a random conflicting cycle
        # Find a conflicting cycle
        found = False
        for _ in range(50):
            ci = random.randrange(len(cycles))
            cyc = cycles[ci]
            v = cyc
            c0 = cur[_edge_index(n, v[0], v[1])]
            if (c0 == cur[_edge_index(n, v[1], v[2])] == cur[_edge_index(n, v[2], v[3])] ==
                cur[_edge_index(n, v[3], v[4])] == cur[_edge_index(n, v[4], v[5])] ==
                cur[_edge_index(n, v[5], v[0])]):
                k = random.randrange(6)
                a, b = v[k], v[(k + 1) % 6]
                e = _edge_index(n, a, b)
                newc = (cur[e] + 1 + random.randrange(3)) % 4
                old = cur[e]
                cur[e] = newc
                new_conf = count_conflicts(cur)
                if new_conf < best_conf:
                    best = cur[:]
                    best_conf = new_conf
                elif new_conf > cur_conf:
                    cur[e] = old  # revert (simplified hill-climb)
                else:
                    pass
                found = True
                break
        if not found:
            # random perturbation
            e = random.randrange(m)
            cur[e] = random.randrange(4)
            new_conf = count_conflicts(cur)
            if new_conf < best_conf:
                best = cur[:]
                best_conf = new_conf
    return best, best_conf

def _greedy(n):
    """Greedy colouring: assign each edge a colour that doesn't immediately create a C6."""
    m = n * (n - 1) // 2
    colors = [-1] * m
    cycles = _cycles_for(n)
    edge_to_cycles = [[] for _ in range(m)]
    for cyc in cycles:
        v = cyc
        edges = []
        for k in range(6):
            a, b = v[k], v[(k + 1) % 6]
            edges.append(_edge_index(n, a, b))
        for k in range(6):
            others = [edges[j] for j in range(6) if j != k]
            edge_to_cycles[edges[k]].append(others)
    for e in range(m):
        placed = False
        for c in range(4):
            ok = True
            for others in edge_to_cycles[e]:
                if all(colors[oe] == c for oe in others):
                    ok = False
                    break
            if ok:
                colors[e] = c
                placed = True
                break
        if not placed:
            return None
    return colors

def construct():
    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    # Fallback known-good: the seed construction for n=9.
    random.seed(12345)
    # Try n from 20 down to 10; for each, try greedy then backtracking then min-conflicts.
    for n in range(20, 9, -1):
        if _remaining() < 10.0:
            break
        # Greedy first (fast)
        gc = _greedy(n)
        if gc is not None and _is_valid(gc, n):
            best_n = n
            best_colors = gc
            break
        # Backtracking
        if _remaining() > 15.0:
            bc = _backtrack(n)
            if bc is not None and _is_valid(bc, n):
                best_n = n
                best_colors = bc
                break
        # Min-conflicts repair from random start
        if _remaining() > 10.0:
            m = n * (n - 1) // 2
            for _ in range(3):
                if _remaining() < 8.0:
                    break
                rand_colors = [random.randrange(4) for _ in range(m)]
                rep, conf = _min_conflicts(rand_colors, n, max_iter=100000)
                if conf == 0 and _is_valid(rep, n):
                    best_n = n
                    best_colors = rep
                    break
            if best_n == n:
                break
    # Ensure we at least have the seed n=9
    if best_n < 9:
        best_n = 9
        best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    return {'n': best_n, 'colors': best_colors}
