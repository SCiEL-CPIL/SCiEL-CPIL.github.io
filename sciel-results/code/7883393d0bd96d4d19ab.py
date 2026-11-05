"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_N = 16
_E = _N * (_N - 1) // 2  # 120 edges
_FULL_MASK = (1 << _N) - 1
_DEADLINE_MARGIN = 15  # seconds to spare before the 150s kill

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


def _compute_backtrack_order(a, b):
    """Compute optimal backtracking order for the 6 remaining Q3 vertices given fixed edge (a,b)."""
    remaining = [v for v in range(8) if v != a and v != b]
    
    order = []
    placed = {a, b}
    for _ in range(6):
        best_v = -1
        best_count = -1
        for v in remaining:
            if v in placed:
                continue
            count = sum(1 for n in _Q3_NEIGHBORS[v] if n in placed)
            if count > best_count:
                best_count = count
                best_v = v
        order.append(best_v)
        placed.add(best_v)
    
    return order

# Precompute for each of the 12 Q3 edges:
# - the 6 remaining vertices in backtrack order
# - for each position i in the order, which of {a,b} and order[0..i-1] are Q3-neighbors of order[i]
_Q3_EDGE_INFO = {}
for a, b in _Q3_EDGES:
    order = _compute_backtrack_order(a, b)
    # For each position i, precompute the list of Q3 neighbors of order[i] that are in {a,b} ∪ order[0..i-1]
    neighbor_info = []
    for i, v in enumerate(order):
        prior = set([a, b]) | set(order[:i])
        nbrs = [n for n in _Q3_NEIGHBORS[v] if n in prior]
        neighbor_info.append(nbrs)
    _Q3_EDGE_INFO[(a, b)] = (order, neighbor_info)


def _check_q3_with_fixed_edge(adj_bits, gu, gv, a, b, order, neighbor_info):
    """
    Check if there's a Q3 embedding where graph vertex gu maps to Q3 vertex a,
    gv maps to Q3 vertex b, and the remaining 6 Q3 vertices map to 6 distinct
    graph vertices from the remaining 14.
    """
    used_mask = (1 << gu) | (1 << gv)
    assigned = {}
    assigned[a] = gu
    assigned[b] = gv
    
    def backtrack(idx):
        nonlocal used_mask
        if idx == 6:
            return True
        
        qv = order[idx]
        candidates = (~used_mask) & _FULL_MASK
        
        # Must be adjacent to all prior Q3 neighbors
        for nqv in neighbor_info[idx]:
            candidates &= adj_bits[assigned[nqv]]
        
        if candidates == 0:
            return False
        
        c = candidates
        while c:
            lsb = c & (-c)
            gv2 = lsb.bit_length() - 1
            c ^= lsb
            
            assigned[qv] = gv2
            used_mask |= (1 << gv2)
            if backtrack(idx + 1):
                return True
            used_mask &= ~(1 << gv2)
            del assigned[qv]
        
        return False
    
    return backtrack(0)


def _has_q3_with_edge(adj_bits, u, v):
    """
    Check if adding edge (u,v) to the current graph creates a Q3.
    Only checks Q3 embeddings that include edge (u,v).
    Uses precomputed backtracking orders for speed.
    """
    # Try each undirected Q3 edge once, with u->a, v->b OR u->b, v->a
    for (a, b) in _Q3_EDGES:
        order, neighbor_info = _Q3_EDGE_INFO[(a, b)]
        
        # Try mapping: u->a, v->b
        if _check_q3_with_fixed_edge(adj_bits, u, v, a, b, order, neighbor_info):
            return True
        # Try mapping: u->b, v->a
        if _check_q3_with_fixed_edge(adj_bits, v, u, a, b, order, neighbor_info):
            return True
    
    return False


def _find_q3_witness(adj_bits, u, v):
    """
    Find a specific Q3 embedding that includes edge (u,v), if one exists.
    Returns a list of 8 (qv, gv) pairs or None.
    """
    for (a, b) in _Q3_EDGES:
        order, neighbor_info = _Q3_EDGE_INFO[(a, b)]
        
        for gu, gv, qa, qb in [(u, v, a, b), (v, u, a, b)]:
            used_mask = (1 << gu) | (1 << gv)
            assigned = {}
            assigned[qa] = gu
            assigned[qb] = gv
            
            def backtrack(idx):
                nonlocal used_mask
                if idx == 6:
                    return dict(assigned)
                
                qv = order[idx]
                candidates = (~used_mask) & _FULL_MASK
                
                for nqv in neighbor_info[idx]:
                    candidates &= adj_bits[assigned[nqv]]
                
                if candidates == 0:
                    return None
                
                c = candidates
                while c:
                    lsb = c & (-c)
                    gv2 = lsb.bit_length() - 1
                    c ^= lsb
                    
                    assigned[qv] = gv2
                    used_mask |= (1 << gv2)
                    result = backtrack(idx + 1)
                    if result is not None:
                        return result
                    used_mask &= ~(1 << gv2)
                    del assigned[qv]
                
                return None
            
            result = backtrack(0)
            if result is not None:
                # Convert to list of (qv, gv) pairs
                pairs = [(k, v) for k, v in result.items()]
                pairs.sort()
                return pairs
    
    return None


def _greedy_add_edges(adj, edges, edge_order, deadline):
    """
    Greedily add edges from edge_order to adj/edges, skipping those that create Q3.
    """
    for u, v in edge_order:
        if time.time() > deadline:
            break
        
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        if not _has_q3_with_edge(adj, u, v):
            edges.append([u, v])
        else:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
    
    return len(edges)


def _local_search(adj, edge_set, deadline, max_iter=200):
    """
    Local search: try to add missing edges, and if blocked, try edge swaps.
    Uses a set for edges to avoid list manipulation errors.
    Returns the number of edges in the improved graph.
    """
    all_edges = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
    missing = [(u, v) for u, v in all_edges if (u, v) not in edge_set]
    
    improved = True
    iter_count = 0
    while improved and time.time() < deadline and iter_count < max_iter:
        improved = False
        iter_count += 1
        random.shuffle(missing)
        
        for u, v in missing[:]:
            if time.time() > deadline:
                break
            if (u, v) in edge_set:
                continue
            
            # Try adding edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if not _has_q3_with_edge(adj, u, v):
                edge_set.add((u, v))
                if (u, v) in missing:
                    missing.remove((u, v))
                improved = True
            else:
                # Find the witness Q3
                witness = _find_q3_witness(adj, u, v)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
                if witness is None:
                    continue
                
                # Try to swap: remove one edge from the witness (not (u,v)) to unblock
                witness_edges = set()
                witness_qv_to_gv = {qv: gv for qv, gv in witness}
                for qv1, qv2 in _Q3_EDGES:
                    gv1 = witness_qv_to_gv.get(qv1)
                    gv2 = witness_qv_to_gv.get(qv2)
                    if gv1 is not None and gv2 is not None:
                        e = (min(gv1, gv2), max(gv1, gv2))
                        if e != (u, v):
                            witness_edges.add(e)
                
                # Try removing each witness edge and adding (u,v)
                for rem_u, rem_v in list(witness_edges):
                    if time.time() > deadline:
                        break
                    if (rem_u, rem_v) not in edge_set:
                        continue
                    
                    # Remove edge (rem_u, rem_v)
                    adj[rem_u] &= ~(1 << rem_v)
                    adj[rem_v] &= ~(1 << rem_u)
                    edge_set.remove((rem_u, rem_v))
                    
                    # Try adding (u, v)
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
                    if not _has_q3_with_edge(adj, u, v):
                        edge_set.add((u, v))
                        if (u, v) in missing:
                            missing.remove((u, v))
                        improved = True
                    else:
                        # Revert
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        adj[rem_u] |= (1 << rem_v)
                        adj[rem_v] |= (1 << rem_u)
                        edge_set.add((rem_u, rem_v))
    
    return len(edge_set)


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE_MARGIN
    
    all_edges = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
    
    best_edges = []
    best_count = 0
    best_adj = [0] * _N
    
    # Phase 1: Multiple greedy trials with different orderings
    num_trials = 30
    
    for trial in range(num_trials):
        if time.time() > deadline - 10:
            break
        
        # Create ordering
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
        
        _greedy_add_edges(adj, edges, edge_order, deadline - 5)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            best_adj = adj[:]
    
    # Phase 2: Local search with edge swaps
    if time.time() < deadline - 5 and best_count > 0:
        adj = best_adj[:]
        edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in best_edges)
        new_count = _local_search(adj, edge_set, deadline - 3, max_iter=100)
        
        if new_count > best_count:
            best_count = new_count
            best_edges = [list(e) for e in edge_set]
            best_adj = adj[:]
    
    # Phase 3: More local search iterations if time permits
    if time.time() < deadline - 2 and best_count > 0:
        adj = best_adj[:]
        edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in best_edges)
        new_count = _local_search(adj, edge_set, deadline - 1, max_iter=50)
        
        if new_count > best_count:
            best_count = new_count
            best_edges = [list(e) for e in edge_set]
            best_adj = adj[:]
    
    return best_edges
# EVOLVE-BLOCK-END
