"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

def construct():
    n = 16
    
    # Q3 edges: vertices 0-7, edges between vertices differing in exactly one bit
    q3_edges = []
    for v in range(8):
        for i in range(3):
            nb = v ^ (1 << i)
            if nb > v:
                q3_edges.append((v, nb))
    
    all_8sets = list(itertools.combinations(range(n), 8))
    
    def is_q3_free(adj):
        """Check if the graph (given by adjacency matrix) is Q3-free."""
        for verts in all_8sets:
            vs = set(verts)
            # Quick degree check: all vertices must have degree >= 3 in the subgraph
            min_deg = min(sum(adj[v][u] for u in vs if u != v) for v in vs)
            if min_deg < 3:
                continue
            
            # Backtracking search for Q3 embedding
            assign = {}
            
            def bt(idx):
                if idx == 8:
                    # Verify all 12 edges of Q3 are present
                    for u, v in q3_edges:
                        gu, gv = assign[u], assign[v]
                        if not adj[gu][gv]:
                            return False
                    return True
                
                qv = idx
                for gv in verts:
                    if gv in assign.values():
                        continue
                    # Pruning: check edges to already-assigned Q3 vertices
                    compatible = True
                    for u in range(idx):
                        # Check if qv and u are adjacent in Q3
                        is_adj = any((qv == a and u == b) or (u == a and qv == b) for a, b in q3_edges)
                        if is_adj:
                            if not adj[gv][assign[u]]:
                                compatible = False
                                break
                    if not compatible:
                        continue
                    assign[qv] = gv
                    if bt(idx + 1):
                        return True
                    del assign[qv]
                return False
            
            if bt(0):
                return False
        return True
    
    # Try C5 blow-up construction
    # 5 parts in a cycle, sizes: 3, 3, 3, 3, 4
    adj = [[False] * n for _ in range(n)]
    part_sizes = [3, 3, 3, 3, 4]
    starts = [0]
    for s in part_sizes[:-1]:
        starts.append(starts[-1] + s)
    
    for i in range(5):
        j = (i + 1) % 5
        for u in range(starts[i], starts[i] + part_sizes[i]):
            for v in range(starts[j], starts[j] + part_sizes[j]):
                adj[u][v] = adj[v][u] = True
    
    if is_q3_free(adj):
        edges = [[u, v] for u in range(n) for v in range(u + 1, n) if adj[u][v]]
        return edges
    
    # Fallback: K_{3,13}
    return [[a, b] for a in range(3) for b in range(3, 16)]
# EVOLVE-BLOCK-END
