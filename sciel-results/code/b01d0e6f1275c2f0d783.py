import time
import random
import numpy as np

# Q3 edges (canonical, vertices 0..7)
_Q3_EDGES = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
_Q3_ADJ = [0] * 8
for _a, _b in _Q3_EDGES:
    _Q3_ADJ[_a] |= (1 << _b)
    _Q3_ADJ[_b] |= (1 << _a)

# Precompute Q3 edge orientations
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

def _bits(mask):
    while mask:
        lsb = mask & (-mask)
        yield lsb.bit_length() - 1
        mask ^= lsb

def construct():
    n = 16
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin

    full_mask = (1 << n) - 1

    def has_q3_after_add(adj, u, v, deadline):
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
                    if time.time() > deadline - 2:
                        return False
                    if idx == 6:
                        for a, b in _Q3_EDGES:
                            ga = assignment[a]
                            gb = assignment[b]
                            if not (adj[ga] & (1 << gb)):
                                return False
                        return True

                    x = unassigned[idx]
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

    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    def edge_list(adj):
        es = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    es.append([u, v])
        return es

    def adj_to_matrix(adj):
        A = np.zeros((n, n), dtype=np.int32)
        for u in range(n):
            mask = adj[u]
            while mask:
                lsb = mask & (-mask)
                v = lsb.bit_length() - 1
                A[u, v] = 1
                mask ^= lsb
        return A

    def trace_A4(A):
        """Compute trace(A^4) = number of 4-cycles (weighted) using NumPy."""
        A2 = A @ A
        A4 = A2 @ A2
        return int(np.trace(A4))

    def edge_removal_impact(A, u, v):
        """
        Compute how much removing edge (u,v) reduces trace(A^4).
        trace(A^4) counts closed walks of length 4.
        Removing edge (u,v) removes all closed walks that use edge (u,v).
        We compute the difference: trace(A^4) - trace((A - E_uv)^4)
        where E_uv has 1s at (u,v) and (v,u).
        
        Efficient computation: 
        The change in trace(A^4) when removing edge (u,v) can be computed
        by considering all 4-cycles that use edge (u,v).
        A 4-cycle using edge (u,v) has form u-v-x-y-u or u-v-y-x-u etc.
        More precisely, closed walks of length 4 that include edge (u,v):
        They must traverse u->v or v->u at some point.
        
        Let's use: delta = trace(A^4) - trace((A-E)^4)
        (A-E)^4 where E has E[u,v]=E[v,u]=1
        This is expensive to compute fully. Instead, let's compute the number
        of 4-cycles containing edge (u,v) directly.
        
        Number of 4-cycles through edge (u,v):
        A 4-cycle containing edge (u,v) is a path v-x-y-u of length 3 (no edge u-v in the path).
        So we need: sum over x in N(v)\\{u}, y in N(u)\\{v} such that x!=y and edge(x,y) exists.
        But we also need x-y to be an edge and x != y.
        
        Actually, let me just compute it via matrix:
        Let B = A - E where E[u,v]=E[v,u]=1
        trace(B^4) = trace((A-E)^4)
        trace(A^4) - trace(B^4) = impact
        
        For n=16, this is 16x16 matrix mult, very fast.
        """
        E = np.zeros((n, n), dtype=np.int32)
        E[u, v] = 1
        E[v, u] = 1
        B = A - E
        A2 = A @ A
        A4 = A2 @ A2
        B2 = B @ B
        B4 = B2 @ B2
        return int(np.trace(A4)) - int(np.trace(B4))

    best_edges = None
    best_count = -1

    # Strategy 1: Start with K3,13
    adj = [0] * n
    for a in range(3):
        for b in range(3, 16):
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
    
    cnt = count_edges(adj)
    best_count = cnt
    best_edges = edge_list(adj)

    # Greedy add
    missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
    
    # Multiple random restarts of the greedy phase
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
        
        # Reset to K3,13
        adj = [0] * n
        for a in range(3):
            for b in range(3, 16):
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
        
        cnt = count_edges(adj)
        
        if attempt == 0:
            pass
        else:
            random.shuffle(missing)
            
        for u, v in missing:
            if time.time() > deadline - 10:
                break
            if not (adj[u] & (1 << v)):
                if not has_q3_after_add(adj, u, v, deadline):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    cnt += 1
        
        if cnt > best_count:
            best_count = cnt
            best_edges = edge_list(adj)

    # Strategy 2: Local Search on the best found graph
    adj = [0] * n
    for u, v in best_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
    
    random.shuffle(missing)
    for u, v in missing:
        if time.time() > deadline - 5:
            break
        if not (adj[u] & (1 << v)):
            if not has_q3_after_add(adj, u, v, deadline):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                best_count += 1
                
    best_edges = edge_list(adj)

    # Strategy 3: Spectral heuristic - start from K16, remove edges to minimize 4-cycles
    if time.time() < deadline - 30:
        # Start with complete graph
        adj_k = [0] * n
        for u in range(n):
            for v in range(u+1, n):
                adj_k[u] |= (1 << v)
                adj_k[v] |= (1 << u)
        
        A = adj_to_matrix(adj_k)
        
        # Iteratively remove the edge whose removal most reduces trace(A^4)
        # But we need the result to be Q3-free, so we also check Q3-freeness
        for iteration in range(200):
            if time.time() > deadline - 25:
                break
            
            # Check if current graph is Q3-free
            # Quick check: try to find Q3
            is_q3_free = True
            for (qa, qb, unassigned) in _Q3_EDGE_ORIENTS[:6]:  # Check subset for speed
                for (gu, gv) in [(0, 1), (1, 0)]:
                    # Just check if edge exists
                    if not (adj_k[gu] & (1 << gv)):
                        continue
                    assignment = [None] * 8
                    assignment[qa] = gu
                    assignment[qb] = gv
                    used_mask = (1 << gu) | (1 << gv)
                    
                    found = False
                    def bt_check(idx, um):
                        nonlocal found
                        if found:
                            return
                        if idx == 6:
                            for a, b in _Q3_EDGES:
                                ga = assignment[a]
                                gb = assignment[b]
                                if not (adj_k[ga] & (1 << gb)):
                                    return
                            found = True
                            return
                        
                        x = unassigned[idx]
                        candidates = full_mask & ~um
                        for y in range(8):
                            if assignment[y] is not None and (_Q3_ADJ[x] & (1 << y)):
                                hy = assignment[y]
                                candidates &= adj_k[hy]
                        
                        while candidates:
                            bit = candidates & (-candidates)
                            g = bit.bit_length() - 1
                            candidates ^= bit
                            assignment[x] = g
                            bt_check(idx + 1, um | (1 << g))
                            assignment[x] = None
                            if found:
                                return
                    
                    bt_check(0, used_mask)
                    if found:
                        is_q3_free = False
                        break
                if not is_q3_free:
                    break
            
            if is_q3_free:
                # Current graph is Q3-free! Count edges
                cur_count = count_edges(adj_k)
                if cur_count > best_count:
                    best_count = cur_count
                    best_edges = edge_list(adj_k)
                # Try to add more edges
                break
            
            # Find edge to remove: pick edge with highest impact on trace(A^4)
            best_impact = -1
            best_edge = None
            all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
            random.shuffle(all_edges)
            
            # Evaluate a subset of edges for speed
            sample_size = min(40, len(all_edges))
            for u, v in all_edges[:sample_size]:
                if not (adj_k[u] & (1 << v)):
                    continue
                impact = edge_removal_impact(A, u, v)
                if impact > best_impact:
                    best_impact = impact
                    best_edge = (u, v)
            
            if best_edge is None:
                break
            
            # Remove the edge
            u, v = best_edge
            adj_k[u] &= ~(1 << v)
            adj_k[v] &= ~(1 << u)
            A = adj_to_matrix(adj_k)
            
            # Also try: after removing, check if Q3-free and try to add back
            # For now, just continue removing

    # Final: try to add edges to the spectral result
    if time.time() < deadline - 10:
        # Use the best graph found so far
        adj_final = [0] * n
        for u, v in best_edges:
            adj_final[u] |= (1 << v)
            adj_final[v] |= (1 << u)
        
        missing_final = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj_final[i] & (1 << j))]
        random.shuffle(missing_final)
        
        for u, v in missing_final:
            if time.time() > deadline - 5:
                break
            if not (adj_final[u] & (1 << v)):
                if not has_q3_after_add(adj_final, u, v, deadline):
                    adj_final[u] |= (1 << v)
                    adj_final[v] |= (1 << u)
                    best_count += 1
        
        best_edges = edge_list(adj_final)

    return best_edges
