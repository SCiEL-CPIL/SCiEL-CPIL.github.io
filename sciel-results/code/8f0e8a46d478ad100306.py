"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_N = 16
_E = _N * (_N - 1) // 2  # 120 edges
_DEADLINE_MARGIN = 20  # seconds to spare before the 150s kill

# Precompute Q3 structure
_Q3_EDGES = []
for i in range(8):
    for j in range(i + 1, 8):
        if (i ^ j) in (1, 2, 4):
            _Q3_EDGES.append((i, j))

_Q3_NEIGHBORS = [[] for _ in range(8)]
for i in range(8):
    for bit in (1, 2, 4):
        j = i ^ bit
        if j > i:
            _Q3_NEIGHBORS[i].append(j)
            _Q3_NEIGHBORS[j].append(i)

# Precompute: for each qv, bitmask of Q3 neighbors
_Q3_QV_NEIGHBOR_MASK = [0] * 8
for i in range(8):
    mask = 0
    for j in _Q3_NEIGHBORS[i]:
        mask |= (1 << j)
    _Q3_QV_NEIGHBOR_MASK[i] = mask

# Precompute: for each qv, list of Q3 neighbors that come before it in the backtracking order
# Backtracking order: 0, 1, 2, 3, 4, 5, 6, 7
_Q3_PRIOR_NEIGHBORS = [[] for _ in range(8)]
for i in range(8):
    for j in _Q3_NEIGHBORS[i]:
        if j < i:
            _Q3_PRIOR_NEIGHBORS[i].append(j)


def _has_q3_fast(adj_bits):
    """
    Fast Q3 subgraph detection using bitmask operations.
    adj_bits: list of 16 integers, each a 16-bit mask of neighbors.
    """
    assigned = [-1] * 8
    used_mask = 0
    full_mask = (1 << _N) - 1
    
    def backtrack(qv):
        nonlocal used_mask
        if qv == 8:
            return True
        
        candidates = (~used_mask) & full_mask
        
        # Filter: must be adjacent to all already-assigned Q3 neighbors of qv
        for pv in _Q3_PRIOR_NEIGHBORS[qv]:
            pgv = assigned[pv]
            candidates &= adj_bits[pgv]
        
        if candidates == 0:
            return False
        
        c = candidates
        while c:
            lsb = c & (-c)
            gv = lsb.bit_length() - 1
            c ^= lsb
            
            assigned[qv] = gv
            used_mask |= (1 << gv)
            if backtrack(qv + 1):
                return True
            used_mask &= ~(1 << gv)
            assigned[qv] = -1
        
        return False
    
    for gv0 in range(_N):
        assigned[0] = gv0
        used_mask = 1 << gv0
        if backtrack(1):
            return True
        used_mask = 0
        assigned[0] = -1
    
    return False


def _has_q3_with_edge(adj_bits, u, v):
    """
    Check if adding edge (u,v) to the current graph creates a Q3.
    Only checks Q3 embeddings that include edge (u,v).
    """
    # Q3 vertices: 0=000, 1=001, 2=010, 3=011, 4=100, 5=101, 6=110, 7=111
    # Fix edge (u,v) as Q3 edge (0,1)
    # Remaining: 2,3,4,5,6,7
    # 
    # Constraints:
    # 2: adj to u, 3, 6
    # 3: adj to v, 2, 7
    # 4: adj to u, 5, 6
    # 5: adj to v, 4, 7
    # 6: adj to 2, 4, 7
    # 7: adj to 3, 5, 6
    
    order = [2, 4, 3, 5, 6, 7]
    assigned = {}
    used_mask = (1 << u) | (1 << v)
    full_mask = (1 << _N) - 1
    
    # For each qv in order, which previously-assigned qvs it must connect to, and external constraint
    constraints = {
        2: ((), u),      # no prior q3 neighbors, must connect to u
        4: ((), u),      # no prior q3 neighbors, must connect to u
        3: ((2,), v),    # must connect to qv2 and v
        5: ((4,), v),    # must connect to qv4 and v
        6: ((2, 4), None),  # must connect to qv2 and qv4
        7: ((3, 5), None),  # must connect to qv3 and qv5
    }
    
    def backtrack(idx):
        nonlocal used_mask
        if idx == len(order):
            # Final check: edge 6-7
            return (adj_bits[assigned[6]] >> assigned[7]) & 1
        
        qv = order[idx]
        prior_qvs, ext = constraints[qv]
        
        candidates = (~used_mask) & full_mask
        
        if ext is not None:
            candidates &= adj_bits[ext]
        
        for nqv in prior_qvs:
            candidates &= adj_bits[assigned[nqv]]
        
        if candidates == 0:
            return False
        
        c = candidates
        while c:
            lsb = c & (-c)
            gv = lsb.bit_length() - 1
            c ^= lsb
            
            assigned[qv] = gv
            used_mask |= (1 << gv)
            if backtrack(idx + 1):
                return True
            used_mask &= ~(1 << gv)
            del assigned[qv]
        
        return False
    
    return backtrack(0)


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE_MARGIN
    
    all_edges = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
    
    best_edges = []
    best_count = 0
    best_adj = [0] * _N
    
    # Phase 1: Greedy edge addition with multiple orderings
    num_trials = 30
    
    for trial in range(num_trials):
        if time.time() > deadline - 8:
            break
        
        if trial == 0:
            edge_order = all_edges[:]
        elif trial == 1:
            edge_order = all_edges[::-1]
        else:
            random.seed(trial * 12345 + 67890)
            edge_order = all_edges[:]
            random.shuffle(edge_order)
        
        adj = [0] * _N
        edges = []
        
        for u, v in edge_order:
            if time.time() > deadline - 5:
                break
            
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if not _has_q3_with_edge(adj, u, v):
                edges.append([u, v])
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            best_adj = adj[:]
    
    # Phase 2: Local search - add missing edges
    if time.time() < deadline - 15:
        adj = best_adj[:]
        edges = best_edges[:]
        edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in edges)
        
        missing = [(u, v) for u, v in all_edges if (u, v) not in edge_set]
        random.shuffle(missing)
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for i in range(len(missing)):
                if time.time() > deadline - 7:
                    break
                u, v = missing[i]
                if (u, v) in edge_set:
                    continue
                
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                if not _has_q3_with_edge(adj, u, v):
                    edges.append([u, v])
                    edge_set.add((u, v))
                    missing.pop(i)
                    i -= 1
                    improved = True
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            best_adj = adj[:]
    
    # Phase 3: Edge swap local search - try removing one edge and adding two
    if time.time() < deadline - 10:
        adj = best_adj[:]
        edges = best_edges[:]
        edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in edges)
        
        for _swap_iter in range(20):
            if time.time() > deadline - 8:
                break
            
            made_swap = False
            # Try each existing edge for removal
            for e_idx in range(len(edges)):
                if time.time() > deadline - 7:
                    break
                
                u_rem, v_rem = edges[e_idx]
                
                # Temporarily remove edge
                adj[u_rem] &= ~(1 << v_rem)
                adj[v_rem] &= ~(1 << u_rem)
                
                # Try to add two missing edges
                added = []
                for u_add, v_add in all_edges:
                    if (u_add, v_add) in edge_set:
                        continue
                    if (u_add, v_add) == (u_rem, v_rem):
                        continue
                    
                    adj[u_add] |= (1 << v_add)
                    adj[v_add] |= (1 << u_add)
                    
                    if not _has_q3_with_edge(adj, u_add, v_add):
                        added.append((u_add, v_add))
                    else:
                        adj[u_add] &= ~(1 << v_add)
                        adj[v_add] &= ~(1 << u_add)
                    
                    if len(added) == 2:
                        break
                
                # If we added 2 edges, accept the swap
                if len(added) == 2:
                    # Re-add the removed edge is NOT done - we're replacing it
                    # But we need to check that the new edges don't conflict with each other
                    # Actually, we already checked each one individually with the current adj
                    # which includes the other new edge, so this should be fine
                    # Wait, no - when checking the second new edge, the first is already added
                    # so that's correct
                    
                    edges.pop(e_idx)
                    edge_set.discard((u_rem, v_rem))
                    for u_a, v_a in added:
                        edges.append([u_a, v_a])
                        edge_set.add((u_a, v_a))
                    made_swap = True
                    break
                else:
                    # Revert: re-add the removed edge
                    adj[u_rem] |= (1 << v_rem)
                    adj[v_rem] |= (1 << u_rem)
            
            if not made_swap:
                break
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            best_adj = adj[:]
    
    # Phase 4: Final polish - try to add any remaining edges
    if time.time() < deadline - 5:
        adj = best_adj[:]
        edges = best_edges[:]
        edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in edges)
        
        for u, v in all_edges:
            if time.time() > deadline - 3:
                break
            if (u, v) in edge_set:
                continue
            
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if not _has_q3_with_edge(adj, u, v):
                edges.append([u, v])
                edge_set.add((u, v))
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            best_adj = adj[:]
    
    return best_edges
# EVOLVE-BLOCK-END
