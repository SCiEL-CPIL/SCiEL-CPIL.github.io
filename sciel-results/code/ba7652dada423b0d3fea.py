import time
import random
import numpy as np

# Q3 edges (canonical, vertices 0..7)
_Q3_EDGES = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
_Q3_ADJ = [0] * 8
for _a, _b in _Q3_EDGES:
    _Q3_ADJ[_a] |= (1 << _b)
    _Q3_ADJ[_b] |= (1 << _a)

# Precompute Q3 edge orientations for fast checking
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

def _has_q3_fast(adj, deadline):
    """Fast Q3 check using bitmask backtracking. Returns True if Q3 found."""
    full_mask = (1 << 16) - 1
    
    for (qa, qb, unassigned) in _Q3_EDGE_ORIENTS:
        for (gu, gv) in [(0, 1), (1, 0)]:
            if not (adj[gu] & (1 << gv)):
                continue
            
            assignment = [-1] * 8
            assignment[qa] = gu
            assignment[qb] = gv
            used_mask = (1 << gu) | (1 << gv)
            
            def bt(idx, um):
                if time.time() > deadline:
                    return False
                if idx == 6:
                    for a, b in _Q3_EDGES:
                        ga = assignment[a]
                        gb = assignment[b]
                        if not (adj[ga] & (1 << gb)):
                            return False
                    return True
                
                x = unassigned[idx]
                candidates = full_mask & ~um
                for y in range(8):
                    if assignment[y] != -1 and (_Q3_ADJ[x] & (1 << y)):
                        hy = assignment[y]
                        candidates &= adj[hy]
                
                while candidates:
                    bit = candidates & (-candidates)
                    g = bit.bit_length() - 1
                    candidates ^= bit
                    
                    assignment[x] = g
                    if bt(idx + 1, um | (1 << g)):
                        return True
                    assignment[x] = -1
                return False
            
            if bt(0, used_mask):
                return True
    
    return False

def _edge_removal_impact(adj_matrix, u, v):
    """Compute reduction in trace(A^4) if we remove edge (u,v).
    
    trace(A^4) = sum of (A^4)[i][i] = 8 * (number of 4-cycles)
    Removing edge (u,v) affects 4-cycles that use this edge.
    A 4-cycle using edge (u,v) has form u-v-x-y-u or u-v-y-x-u.
    For each common neighbor pair (x,y) where x~y and x!=v, y!=u,
    we count cycles through u-v-x-y-u.
    """
    # 4-cycles through edge (u,v): u-v-x-y-u where x is neighbor of v (x!=u),
    # y is neighbor of u (y!=v), and x~y
    # Count = sum over x in N(v)\{u} of |N(x) ∩ N(u)\{v}|
    nu = np.nonzero(adj_matrix[u])[0]
    nv = np.nonzero(adj_matrix[v])[0]
    
    # For each x in nv (excluding u), count neighbors of x that are also in nu (excluding v)
    impact = 0
    for x in nv:
        if x == u:
            continue
        # Count common neighbors of x and u, excluding v
        common = np.sum(adj_matrix[x, nu])
        # Subtract if v is a neighbor of x (it is, since x in nv)
        if adj_matrix[x, v]:
            common -= 1
        impact += common
    return impact

def construct():
    n = 16
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin
    
    best_edges = None
    best_count = -1
    
    def adj_to_matrix(adj):
        A = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(n):
                if adj[i] & (1 << j):
                    A[i, j] = 1
        return A
    
    def matrix_to_adj(A):
        adj = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if A[i, j]:
                    mask |= (1 << j)
            adj[i] = mask
        return adj
    
    def edge_list(adj):
        es = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    es.append([u, v])
        return es
    
    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2
    
    # Strategy: Start from K16, iteratively remove worst edges based on 4-cycle penalty
    for attempt in range(30):
        if time.time() > deadline - 20:
            break
        
        # Start with complete graph K16
        adj = [0] * n
        for i in range(n):
            for j in range(n):
                if i != j:
                    adj[i] |= (1 << j)
        
        A = adj_to_matrix(adj)
        
        # Iteratively remove edge with highest 4-cycle impact
        removed = 0
        while time.time() < deadline - 15:
            # Compute impact for each remaining edge
            best_impact = -1
            best_edge = None
            for u in range(n):
                for v in range(u+1, n):
                    if not (adj[u] & (1 << v)):
                        continue
                    imp = _edge_removal_impact(A, u, v)
                    if imp > best_impact:
                        best_impact = imp
                        best_edge = (u, v)
            
            if best_edge is None:
                break
            
            u, v = best_edge
            # Remove edge
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            A[u, v] = 0
            A[v, u] = 0
            removed += 1
            
            # Check if graph is Q3-free
            if _has_q3_fast(adj, deadline - 5):
                # Still has Q3, continue removing
                pass
            else:
                # Q3-free! Record and try to add back edges
                cnt = count_edges(adj)
                
                # Try to greedily add back edges
                missing = [(i, j) for i in range(n) for j in range(i+1, n) 
                          if not (adj[i] & (1 << j))]
                random.shuffle(missing)
                
                for eu, ev in missing:
                    if time.time() > deadline - 10:
                        break
                    # Try adding edge
                    adj[eu] |= (1 << ev)
                    adj[ev] |= (1 << eu)
                    if not _has_q3_fast(adj, deadline - 5):
                        cnt += 1
                    else:
                        adj[eu] &= ~(1 << ev)
                        adj[ev] &= ~(1 << eu)
                
                if cnt > best_count:
                    best_count = cnt
                    best_edges = edge_list(adj)
                
                # Local search: try swapping edges
                for swap_iter in range(5):
                    if time.time() > deadline - 10:
                        break
                    
                    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
                    random.shuffle(all_pairs)
                    
                    improved = False
                    for rem_u, rem_v in all_pairs:
                        if time.time() > deadline - 10:
                            break
                        if not (adj[rem_u] & (1 << rem_v)):
                            continue
                        
                        # Remove edge
                        adj[rem_u] &= ~(1 << rem_v)
                        adj[rem_v] &= ~(1 << rem_u)
                        
                        # Try to add a different edge
                        added = False
                        for add_u, add_v in all_pairs:
                            if (add_u, add_v) == (rem_u, rem_v):
                                continue
                            if adj[add_u] & (1 << add_v):
                                continue
                            
                            adj[add_u] |= (1 << add_v)
                            adj[add_v] |= (1 << add_u)
                            if not _has_q3_fast(adj, deadline - 5):
                                added = True
                                improved = True
                                break
                            else:
                                adj[add_u] &= ~(1 << add_v)
                                adj[add_v] &= ~(1 << add_u)
                        
                        if not added:
                            adj[rem_u] |= (1 << rem_v)
                            adj[rem_v] |= (1 << rem_u)
                        
                        if improved:
                            break
                    
                    if not improved:
                        break
                
                # Update best
                cnt = count_edges(adj)
                if cnt > best_count:
                    best_count = cnt
                    best_edges = edge_list(adj)
                
                break  # Found a Q3-free graph, move to next attempt
        
        # If we never found Q3-free, try a different randomization
        if best_count < 0 and attempt >= 5:
            # Fallback: use random sparse start
            adj = [0] * n
            all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_pairs)
            for u, v in all_pairs[:random.randint(30, 50)]:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            # Greedy fill
            missing = [(i, j) for i in range(n) for j in range(i+1, n) 
                      if not (adj[i] & (1 << j))]
            random.shuffle(missing)
            for eu, ev in missing:
                if time.time() > deadline - 10:
                    break
                adj[eu] |= (1 << ev)
                adj[ev] |= (1 << eu)
                if not _has_q3_fast(adj, deadline - 5):
                    pass
                else:
                    adj[eu] &= ~(1 << ev)
                    adj[ev] &= ~(1 << eu)
            
            cnt = count_edges(adj)
            if cnt > best_count:
                best_count = cnt
                best_edges = edge_list(adj)
    
    return best_edges
