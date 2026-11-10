"""Edge coloring of K_16: maximize distinct colors, no 4-cycle has 4 distinct colors."""
import time
import random
import sys

N = 16
M = N * (N - 1) // 2  # 120 edges

# Precompute edge index
def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

# Precompute all 4-cycles: for each 4-vertex subset, the 3 possible 4-cycles
# A 4-cycle on vertices {a,b,c,d} (a<b<c<d) can be:
#   (a,b,c,d): edges ab, bc, cd, da
#   (a,b,d,c): edges ab, bd, dc, ca
#   (a,c,b,d): edges ac, cb, bd, da
# Represent each 4-cycle as 4 edge indices.

def build_cycles():
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Cycle 1: a-b-c-d-a
                    e1 = [edge_idx(a, b), edge_idx(b, c), edge_idx(c, d), edge_idx(d, a)]
                    cycles.append(tuple(e1))
                    # Cycle 2: a-b-d-c-a
                    e2 = [edge_idx(a, b), edge_idx(b, d), edge_idx(d, c), edge_idx(c, a)]
                    cycles.append(tuple(e2))
                    # Cycle 3: a-c-b-d-a
                    e3 = [edge_idx(a, c), edge_idx(c, b), edge_idx(b, d), edge_idx(d, a)]
                    cycles.append(tuple(e3))
    return cycles

CYCLES = build_cycles()
NUM_CYCLES = len(CYCLES)

# For fast lookup: for each edge, which cycles contain it
EDGE_TO_CYCLES = [[] for _ in range(M)]
for ci, cyc in enumerate(CYCLES):
    for e in cyc:
        EDGE_TO_CYCLES[e].append(ci)

def count_colors(coloring):
    return len(set(coloring))

def is_valid(coloring):
    """Check no 4-cycle has 4 distinct colors."""
    for cyc in CYCLES:
        c0 = coloring[cyc[0]]
        c1 = coloring[cyc[1]]
        c2 = coloring[cyc[2]]
        c3 = coloring[cyc[3]]
        if len({c0, c1, c2, c3}) == 4:
            return False
    return True

def check_edge_change(coloring, edge, new_color):
    """Check if setting coloring[edge] = new_color keeps validity.
    Only cycles containing this edge need checking."""
    old_color = coloring[edge]
    coloring[edge] = new_color
    for ci in EDGE_TO_CYCLES[edge]:
        cyc = CYCLES[ci]
        c0 = coloring[cyc[0]]
        c1 = coloring[cyc[1]]
        c2 = coloring[cyc[2]]
        c3 = coloring[cyc[3]]
        if len({c0, c1, c2, c3}) == 4:
            coloring[edge] = old_color
            return False
    return True

def try_set_edge(coloring, edge, new_color):
    """Set edge to new_color if valid, return True if changed."""
    if coloring[edge] == new_color:
        return False
    old = coloring[edge]
    if check_edge_change(coloring, edge, new_color):
        return True
    coloring[edge] = old
    return False

def local_search(coloring, deadline, rng):
    """Improve coloring by trying to introduce new colors."""
    best_score = count_colors(coloring)
    current = list(coloring)
    
    while time.time() < deadline:
        made_progress = False
        # Try to introduce a new color on a random edge
        # Pick a random edge and try assigning it a new color
        for _ in range(2000):
            if time.time() >= deadline:
                break
            e = rng.randint(0, M - 1)
            # Try a new color not currently used
            used = set(current)
            # Try colors 0..M-1 that are not used
            # Or try to change an edge to a color that creates a new color
            # Strategy: pick an edge, try to give it a fresh color
            new_c = rng.randint(0, M - 1)
            if new_c not in used:
                old = current[e]
                if check_edge_change(current, e, new_c):
                    current[e] = new_c
                    made_progress = True
                    sc = count_colors(current)
                    if sc > best_score:
                        best_score = sc
                        coloring = list(current)
                    break
            else:
                # Try to change an edge to create diversity
                # Change edge to a random color
                new_c = rng.randint(0, M - 1)
                if new_c != current[e]:
                    if check_edge_change(current, e, new_c):
                        current[e] = new_c
                        sc = count_colors(current)
                        if sc > best_score:
                            best_score = sc
                            coloring = list(current)
                        made_progress = True
                        break
        
        if not made_progress:
            # Try more aggressive: try all edges with new colors
            progressed = False
            for e in range(M):
                if time.time() >= deadline:
                    break
                used = set(current)
                # Try a few new colors
                for nc in rng.sample(range(M), min(10, M)):
                    if nc not in used:
                        if check_edge_change(current, e, nc):
                            current[e] = nc
                            sc = count_colors(current)
                            if sc > best_score:
                                best_score = sc
                                coloring = list(current)
                            progressed = True
                            break
                if progressed:
                    break
            if not progressed:
                # Stuck, do a random perturbation
                for _ in range(20):
                    e = rng.randint(0, M - 1)
                    nc = rng.randint(0, M - 1)
                    if check_edge_change(current, e, nc):
                        current[e] = nc
                        sc = count_colors(current)
                        if sc > best_score:
                            best_score = sc
                            coloring = list(current)
                        break
    
    return coloring, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin for 90s limit
    
    best_coloring = [0] * M
    best_score = 1
    
    rng = random.Random(42)
    
    # Phase 1: Try to build up colors greedily
    # Start with all 0, try to assign new colors to edges
    current = [0] * M
    used_colors = {0}
    
    # Try to add colors one by one
    for target_color in range(1, M):
        if time.time() > deadline - 10:
            break
        # Try to find an edge that can take this new color
        # Prefer edges that are "safe" - in fewer cycles or cycles with repeated colors
        found = False
        # Shuffle edges for randomness
        edges = list(range(M))
        rng.shuffle(edges)
        for e in edges:
            if time.time() > deadline - 5:
                break
            if check_edge_change(current, e, target_color):
                current[e] = target_color
                found = True
                break
        if not found:
            # Try harder: maybe we need to rearrange
            pass
    
    sc = count_colors(current)
    if sc > best_score:
        best_score = sc
        best_coloring = list(current)
    
    # Phase 2: Local search to improve
    if time.time() < deadline - 5:
        current2, sc2 = local_search(current, deadline - 2, rng)
        if sc2 > best_score:
            best_score = sc2
            best_coloring = current2
    
    # Phase 3: More local search with different strategy
    if time.time() < deadline - 3:
        # Try to maximize by changing edges to unused colors
        cur = list(best_coloring)
        used = set(cur)
        # Try to find edges that can take new colors
        for _ in range(5000):
            if time.time() > deadline - 1:
                break
            e = rng.randint(0, M - 1)
            # Try to give this edge a new color
            candidates = [c for c in range(M) if c not in used]
            if candidates:
                nc = rng.choice(candidates)
                if check_edge_change(cur, e, nc):
                    cur[e] = nc
                    used.add(nc)
                    sc = count_colors(cur)
                    if sc > best_score:
                        best_score = sc
                        best_coloring = list(cur)
    
    # Verify final result
    if not is_valid(best_coloring):
        # Fallback to all zeros
        return [0] * M
    
    return best_coloring
