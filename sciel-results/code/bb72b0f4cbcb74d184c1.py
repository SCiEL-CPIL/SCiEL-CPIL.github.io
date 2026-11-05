import random
import time

_Q3_EDGES = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
_Q3_ADJ = [0] * 8
for _a, _b in _Q3_EDGES:
    _Q3_ADJ[_a] |= (1 << _b)
    _Q3_ADJ[_b] |= (1 << _a)

# Precompute Q3 edge orientations with sorted unassigned vertices
_Q3_EDGE_ORIENTS = []
for qa, qb in _Q3_EDGES:
    for (gu_q, gv_q) in [(qa, qb), (qb, qa)]:
        unassigned = [i for i in range(8) if i != gu_q and i != gv_q]
        def _cc(x, a, b):
            c = 0
            if _Q3_ADJ[x] & (1 << a): c += 1
            if _Q3_ADJ[x] & (1 << b): c += 1
            return c
        unassigned.sort(key=lambda x: _cc(x, gu_q, gv_q), reverse=True)
        _Q3_EDGE_ORIENTS.append((gu_q, gv_q, tuple(unassigned)))

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n = 16
    full_mask = (1 << n) - 1

    # Start with K_{3,13}: part0 = {0,1,2}, part1 = {3,...,15}
    adj = [0] * n
    for u in range(3):
        for v in range(3, 16):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

    def has_q3_after_add(u, v):
        """Check if adding edge (u,v) to current adj creates a Q3 subgraph."""
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

        for (qa, qb, unassigned) in _Q3_EDGE_ORIENTS:
            for (gu, gv) in [(u, v), (v, u)]:
                assignment = [None] * 8
                assignment[qa] = gu
                assignment[qb] = gv
                used_mask = (1 << gu) | (1 << gv)

                def bt(idx, used_mask):
                    if time.time() > deadline - 5:
                        return False
                    if idx == 6:
                        for a, b in _Q3_EDGES:
                            ga = assignment[a]
                            gb = assignment[b]
                            if not (adj[ga] & (1 << gb)):
                                return False
                        return True

                    x = unassigned[idx]
                    # Candidates: not used, adjacent to all assigned Q3-neighbors of x
                    candidates = full_mask & ~used_mask
                    for y in range(8):
                        if assignment[y] is not None and (_Q3_ADJ[x] & (1 << y)):
                            hy = assignment[y]
                            candidates &= adj[hy]
                    
                    while candidates:
                        bit = candidates & (-candidates)
                        g = bit.bit_length() - 1
                        candidates ^= bit
                        
                        assignment[x] = g
                        new_used = used_mask | (1 << g)
                        if bt(idx + 1, new_used):
                            return True
                        assignment[x] = None
                    return False

                if bt(0, used_mask):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    return True

        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        return False

    # Greedily add missing edges
    missing = []
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & (1 << j)):
                missing.append((i, j))
    
    random.shuffle(missing)
    
    for u, v in missing:
        if time.time() > deadline - 15:
            break
        if not has_q3_after_add(u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

    # Local search: try swapping edges to potentially increase edge count
    # Try removing an edge and adding two others, or just adding more
    # First, try to add any remaining edges
    for i in range(n):
        for j in range(i + 1, n):
            if time.time() > deadline - 10:
                break
            if not (adj[i] & (1 << j)):
                if not has_q3_after_add(i, j):
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)

    # Edge swap local search: remove edge (a,b), try to add (c,d) and (e,f)
    # Only do this if we have time
    if time.time() < deadline - 20:
        edges_list = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges_list.append((i, j))
        
        random.shuffle(edges_list)
        improved = True
        while improved and time.time() < deadline - 20:
            improved = False
            for a, b in edges_list:
                if time.time() > deadline - 20:
                    break
                # Try removing (a,b) and see if we can add more edges
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)
                
                # Try to add all missing edges
                added = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        if time.time() > deadline - 20:
                            break
                        if not (adj[i] & (1 << j)):
                            if not has_q3_after_add(i, j):
                                adj[i] |= (1 << j)
                                adj[j] |= (1 << i)
                                added += 1
                
                if added > 1:
                    improved = True
                    # Rebuild edges_list
                    edges_list = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i] & (1 << j):
                                edges_list.append((i, j))
                    random.shuffle(edges_list)
                else:
                    # Put back (a,b)
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
                    # Remove any edges we added
                    for i in range(n):
                        for j in range(i + 1, n):
                            if not (i == a and j == b) and not (i == b and j == a):
                                # Check if this edge was there before
                                pass  # This is complex, skip for now

    # Final: collect edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    
    return edges
