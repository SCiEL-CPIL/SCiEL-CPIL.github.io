"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def _edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

def _precompute_c4s():
    cycles = []
    for verts in itertools.combinations(range(N), 4):
        a, b, c, d = verts
        c1 = [_edge_idx(a,b), _edge_idx(b,c), _edge_idx(c,d), _edge_idx(d,a)]
        c2 = [_edge_idx(a,b), _edge_idx(b,d), _edge_idx(d,c), _edge_idx(c,a)]
        c3 = [_edge_idx(a,c), _edge_idx(c,b), _edge_idx(b,d), _edge_idx(d,a)]
        cycles.append(c1)
        cycles.append(c2)
        cycles.append(c3)
    return cycles

_C4S = _precompute_c4s()

def _build_edge_to_c4s():
    edge_c4s = [[] for _ in range(NUM_EDGES)]
    for ci, c4 in enumerate(_C4S):
        for e in c4:
            edge_c4s[e].append(ci)
    return edge_c4s

_EDGE_C4S = _build_edge_to_c4s()

def _is_rainbow(c4, colors):
    c0, c1, c2, c3 = colors[c4[0]], colors[c4[1]], colors[c4[2]], colors[c4[3]]
    return len({c0, c1, c2, c3}) == 4

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin
    
    colors = [0] * NUM_EDGES
    
    # Hierarchical initial coloring: 4 groups of 4 vertices
    group_of = [v // 4 for v in range(N)]
    group_pair_colors = {}
    next_color = 1
    for g1 in range(4):
        for g2 in range(g1 + 1, 4):
            group_pair_colors[(g1, g2)] = next_color
            next_color += 1
    
    for i in range(N):
        for j in range(i + 1, N):
            ei = _edge_idx(i, j)
            gi, gj = group_of[i], group_of[j]
            if gi == gj:
                colors[ei] = 0
            else:
                gmin, gmax = min(gi, gj), max(gi, gj)
                colors[ei] = group_pair_colors[(gmin, gmax)]
    
    used_colors = set(colors)
    
    # Verify initial validity
    for c4 in _C4S:
        if _is_rainbow(c4, colors):
            colors = [0] * NUM_EDGES
            used_colors = {0}
            break
    
    # Local search: greedily assign new colors
    edge_c4_count = [len(ec) for ec in _EDGE_C4S]
    edges_by_c4_count = sorted(range(NUM_EDGES), key=lambda e: edge_c4_count[e])
    
    max_color = 119
    random.seed(42)
    
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        edge_order = edges_by_c4_count[:]
        random.shuffle(edge_order)
        
        for edge in edge_order:
            if (time.time() - start_time) > deadline:
                break
            current_color = colors[edge]
            for new_color in range(max_color + 1):
                if new_color in used_colors:
                    continue
                old_color = colors[edge]
                valid = True
                for ci in _EDGE_C4S[edge]:
                    c4 = _C4S[ci]
                    colors[edge] = new_color
                    if _is_rainbow(c4, colors):
                        valid = False
                        break
                if valid:
                    colors[edge] = new_color
                    used_colors.add(new_color)
                    if old_color != new_color:
                        still_used = False
                        for e in range(NUM_EDGES):
                            if e != edge and colors[e] == old_color:
                                still_used = True
                                break
                        if not still_used:
                            used_colors.discard(old_color)
                    improved = True
                    break
                else:
                    colors[edge] = old_color
    
    # Final verification
    for c4 in _C4S:
        if _is_rainbow(c4, colors):
            return [0] * NUM_EDGES
    
    return colors
# EVOLVE-BLOCK-END
