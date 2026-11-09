"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_START_TIME = None
_DEADLINE = 135.0  # seconds, safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

# Precompute Q3 structure
_Q3_ADJ = {
    0: (1, 2, 4),
    1: (0, 3, 5),
    2: (0, 3, 6),
    3: (1, 2, 7),
    4: (0, 5, 6),
    5: (1, 4, 7),
    6: (2, 4, 7),
    7: (3, 5, 6)
}
_Q3_EDGES = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]

def _has_q3_fast(adj_matrix, vertices):
    """Check if subgraph induced by vertices contains Q3. adj_matrix is 16x16 bool."""
    if len(vertices) != 8:
        return False
    vlist = list(vertices)
    # Quick check: count induced edges
    ecount = 0
    for i in range(8):
        for j in range(i+1, 8):
            if adj_matrix[vlist[i]][vlist[j]]:
                ecount += 1
    if ecount < 12:
        return False
    
    # Backtracking with better ordering: try vertices with higher degree in induced subgraph first
    # For each vertex in vlist, count its degree within the induced subgraph
    deg = [0]*8
    for i in range(8):
        for j in range(8):
            if i != j and adj_matrix[vlist[i]][vlist[j]]:
                deg[i] += 1
    
    # Map Q3 vertices to candidate vertices. Use symmetry breaking: fix Q3 vertex 0 to vlist[0]
    # Actually, let's just do full backtracking but with pruning
    
    mapping = [-1]*8  # q3_v -> vertex in vlist
    used = [False]*8  # which vlist positions are used
    
    # Order of Q3 vertices to assign: start with vertex 0, then its neighbors, etc.
    # Use a fixed order that helps pruning: 0,1,2,4,3,5,6,7
    order = [0, 1, 2, 4, 3, 5, 6, 7]
    
    def backtrack(idx):
        if idx == 8:
            return True
        q3_v = order[idx]
        for pos in range(8):
            if used[pos]:
                continue
            cand = vlist[pos]
            # Check adjacency constraints with already-assigned Q3 vertices
            ok = True
            for q3_u in _Q3_ADJ[q3_v]:
                if mapping[q3_u] >= 0:
                    if not adj_matrix[cand][vlist[mapping[q3_u]]]:
                        ok = False
                        break
            if not ok:
                continue
            mapping[q3_v] = pos
            used[pos] = True
            if backtrack(idx + 1):
                return True
            used[pos] = False
            mapping[q3_v] = -1
        return False
    
    return backtrack(0)

def _add_edge_creates_q3(adj_matrix, u, v):
    """Check if adding edge (u,v) creates a Q3. adj_matrix is 16x16 mutable."""
    # Temporarily add the edge
    adj_matrix[u][v] = True
    adj_matrix[v][u] = True
    
    # A Q3 containing edge (u,v): in Q3, each edge connects two vertices.
    # Fix u -> Q3 vertex 0, v -> Q3 vertex 1 (by symmetry of the edge in Q3).
    # Then we need to map Q3 vertices 2,3,4,5,6,7 to 6 of the remaining 14 vertices.
    # Constraints:
    #   Q3 v0 (u) adjacent to v1(v), v2, v4
    #   Q3 v1 (v) adjacent to v0(u), v3, v5
    #   Q3 v2 adjacent to v0(u), v3, v6
    #   Q3 v3 adjacent to v1(v), v2, v7
    #   Q3 v4 adjacent to v0(u), v5, v6
    #   Q3 v5 adjacent to v1(v), v4, v7
    #   Q3 v6 adjacent to v2, v4, v7
    #   Q3 v7 adjacent to v3, v5, v6
    #
    # So: v2, v4 must be neighbors of u (besides v)
    #     v3, v5 must be neighbors of v (besides u)
    #     v6 must be neighbor of v2 AND v4
    #     v7 must be neighbor of v3 AND v5 AND v6
    #
    # Strategy: enumerate choices for v2, v4 from N(u)\{v}, and v3, v5 from N(v)\{u}
    # Then check if there exist v6, v7 satisfying constraints.
    
    # Get neighbors
    n_u = [x for x in range(16) if x != u and x != v and adj_matrix[u][x]]
    n_v = [x for x in range(16) if x != u and x != v and adj_matrix[v][x]]
    
    # For each pair (a,b) from n_u (mapping to Q3 v2, v4) and (c,d) from n_v (mapping to Q3 v3, v5)
    # We need:
    #   v6 adjacent to a, b, and one of {c,d} (actually v6 adjacent to v2=a and v4=b)
    #   v7 adjacent to c, d, and v6
    #   Also: a adjacent to c (v2-v3 edge), b adjacent to d (v4-v5 edge)? No wait:
    #   Q3 edges: 2-3, 2-6, 4-5, 4-6, 3-7, 5-7, 6-7
    #   So: a(v2)-c(v3), a(v2)-v6, b(v4)-d(v5), b(v4)-v6, c(v3)-v7, d(v5)-v7, v6-v7
    
    # Enumerate a, b from n_u (ordered, a != b) and c, d from n_v (ordered, c != d)
    for a in n_u:
        for b in n_u:
            if a == b:
                continue
            # a -> Q3 v2, b -> Q3 v4
            # Need: a adjacent to c (v3), b adjacent to d (v5)
            # So c must be in N(a) ∩ N(v), d must be in N(b) ∩ N(v)
            c_candidates = [x for x in n_v if adj_matrix[a][x]]
            d_candidates = [x for x in n_v if adj_matrix[b][x]]
            for c in c_candidates:
                for d in d_candidates:
                    if c == d:
                        continue
                    # Now need v6 adjacent to a, b, and v7 adjacent to c, d, v6
                    # v6 must be in N(a) ∩ N(b) \ {u,v,c,d}
                    v6_cands = [x for x in range(16) if x not in (u,v,a,b,c,d) and adj_matrix[a][x] and adj_matrix[b][x]]
                    for v6 in v6_cands:
                        # v7 must be in N(c) ∩ N(d) ∩ N(v6) \ {u,v,a,b,c,d,v6}
                        for v7 in range(16):
                            if v7 in (u,v,a,b,c,d,v6):
                                continue
                            if adj_matrix[c][v7] and adj_matrix[d][v7] and adj_matrix[v6][v7]:
                                adj_matrix[u][v] = False
                                adj_matrix[v][u] = False
                                return True
    adj_matrix[u][v] = False
    adj_matrix[v][u] = False
    return False

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Use adjacency matrix for speed
    adj = [[False]*16 for _ in range(16)]
    edge_count = 0
    
    # Start with K3,13: vertices 0,1,2 connected to 3..15
    for a in range(3):
        for b in range(3, 16):
            adj[a][b] = True
            adj[b][a] = True
            edge_count += 1
    
    best_adj = [row[:] for row in adj]
    best_count = edge_count
    
    # Candidate edges: all pairs not already present
    all_pairs = [(i, j) for i in range(16) for j in range(i+1, 16) if not adj[i][j]]
    
    random.seed(42)
    random.shuffle(all_pairs)
    
    for (u, v) in all_pairs:
        if _elapsed() > _DEADLINE:
            break
        if _add_edge_creates_q3(adj, u, v):
            continue
        adj[u][v] = True
        adj[v][u] = True
        edge_count += 1
        if edge_count > best_count:
            best_count = edge_count
            best_adj = [row[:] for row in adj]
    
    # Build edge list from best_adj
    result = []
    for i in range(16):
        for j in range(i+1, 16):
            if best_adj[i][j]:
                result.append([i, j])
    return result
# EVOLVE-BLOCK-END
