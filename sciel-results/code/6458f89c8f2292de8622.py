"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

_N = 16
_E = _N * (_N - 1) // 2  # 120 edges
_DEADLINE_MARGIN = 15  # seconds to spare before the 150s kill

# Precompute Q3 structure
_Q3_EDGES = []
for i in range(8):
    for j in range(i + 1, 8):
        if (i ^ j) in (1, 2, 4):
            _Q3_EDGES.append((i, j))

# For backtracking: for each Q3 vertex, list of Q3 neighbors
_Q3_NEIGHBORS = [[] for _ in range(8)]
for i in range(8):
    for bit in (1, 2, 4):
        j = i ^ bit
        if j > i:
            _Q3_NEIGHBORS[i].append(j)
            _Q3_NEIGHBORS[j].append(i)

# Precompute all 8-vertex subsets of 16 vertices
_ALL_8SETS = list(itertools.combinations(range(_N), 8))

def _build_adj_bits(edge_list):
    """Build adjacency bitmask representation from edge list."""
    adj = [0] * _N
    for u, v in edge_list:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_q3_free_bitmask(adj_bits):
    """
    Check Q3-freeness using bitmask adjacency with optimized backtracking.
    Uses degree-based ordering and early pruning.
    """
    # Count degrees to order vertices
    degrees = [bin(adj_bits[i]).count('1') for i in range(_N)]
    
    # Order Q3 vertices by a heuristic: start with vertices that have
    # many constraints early. All Q3 vertices have degree 3, so use a
    # fixed ordering that helps pruning: 0, 1, 2, 3, 4, 5, 6, 7
    # But we can try multiple orderings.
    
    # For speed, use a single good ordering
    order = [0, 1, 2, 3, 4, 5, 6, 7]
    
    assigned = [-1] * 8  # q3_vertex -> graph_vertex
    used_mask = 0
    
    # Precompute for each Q3 vertex, which other Q3 vertices are its neighbors
    # and which positions in the order they come after
    qv_neighbors_after = [[] for _ in range(8)]
    for i in range(8):
        for j in _Q3_NEIGHBORS[i]:
            if order.index(j) > order.index(i):
                qv_neighbors_after[i].append(j)
    
    def backtrack(pos):
        nonlocal used_mask
        if pos == 8:
            return True
        qv = order[pos]
        for gv in range(_N):
            if (used_mask >> gv) & 1:
                continue
            # Check edges from qv to already-assigned Q3 neighbors
            ok = True
            for pv in _Q3_NEIGHBORS[qv]:
                ppos = order.index(pv)
                if ppos < pos:
                    if not (adj_bits[gv] >> assigned[pv]) & 1:
                        ok = False
                        break
            if not ok:
                continue
            # Additional pruning: check that for each unassigned neighbor,
            # there exists at least one available vertex that could connect
            # This is expensive, so skip for now
            assigned[qv] = gv
            used_mask |= (1 << gv)
            if backtrack(pos + 1):
                return True
            used_mask &= ~(1 << gv)
            assigned[qv] = -1
        return False
    
    return not backtrack(0)

def _add_edge_check(adj_bits, u, v):
    """
    Try adding edge (u,v) to the graph represented by adj_bits.
    Returns True if the resulting graph is Q3-free, False otherwise.
    Modifies adj_bits in place if the edge is added.
    """
    adj_bits[u] |= (1 << v)
    adj_bits[v] |= (1 << u)
    
    free = _is_q3_free_bitmask(adj_bits)
    
    if not free:
        adj_bits[u] &= ~(1 << v)
        adj_bits[v] &= ~(1 << u)
    
    return free

def _greedy_add_edges(adj_bits, candidate_edges, deadline):
    """
    Greedily add edges from candidate_edges in order, checking Q3-freeness.
    Returns list of added edges.
    """
    added = []
    for u, v in candidate_edges:
        if time.time() > deadline:
            break
        if _add_edge_check(adj_bits, u, v):
            added.append((u, v))
    return added

def _local_search(adj_bits, edges, deadline, max_iters=100):
    """
    Local search: try to improve by swapping edges.
    Remove one edge, try to add a different edge, if net gain and Q3-free, keep.
    """
    current_edges = set((min(u,v), max(u,v)) for u, v in edges)
    all_possible = set()
    for u in range(_N):
        for v in range(u+1, _N):
            all_possible.add((u, v))
    
    missing = list(all_possible - current_edges)
    random.shuffle(missing)
    
    improved = True
    iters = 0
    while improved and iters < max_iters and time.time() < deadline:
        improved = False
        iters += 1
        # Try removing a random edge and adding a missing one
        if len(current_edges) == 0:
            break
        e_remove = random.choice(list(current_edges))
        # Try adding each missing edge
        for e_add in missing:
            if time.time() > deadline:
                break
            # Remove e_remove, add e_add
            u1, v1 = e_remove
            u2, v2 = e_add
            adj_bits[u1] &= ~(1 << v1)
            adj_bits[v1] &= ~(1 << u1)
            adj_bits[u2] |= (1 << v2)
            adj_bits[v2] |= (1 << u2)
            
            if _is_q3_free_bitmask(adj_bits):
                # Accept the swap
                current_edges.remove(e_remove)
                current_edges.add(e_add)
                missing.remove(e_add)
                missing.append(e_remove)
                improved = True
                break
            else:
                # Revert
                adj_bits[u2] &= ~(1 << v2)
                adj_bits[v2] &= ~(1 << u2)
                adj_bits[u1] |= (1 << v1)
                adj_bits[v1] |= (1 << u1)
    
    return list(current_edges)

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE_MARGIN
    
    # Strategy: Start with K3,13 + K13 (117 edges) if Q3-free
    # Then try to add the 3 intra-A edges
    # Then do local search
    
    # K3,13 + K13: vertices 0,1,2 connect to 3..15, and 3..15 form a clique.
    edges = []
    for a in range(3):
        for b in range(3, 16):
            edges.append([a, b])
    for i in range(3, 16):
        for j in range(i + 1, 16):
            edges.append([i, j])
    
    adj_bits = _build_adj_bits(edges)
    
    if _is_q3_free_bitmask(adj_bits):
        # 117 edges, Q3-free. Try to add the 3 intra-A edges.
        intra_a = [(0,1), (0,2), (1,2)]
        for u, v in intra_a:
            if time.time() > deadline:
                break
            if _add_edge_check(adj_bits, u, v):
                edges.append([u, v])
        
        # Now do local search to try to improve
        if time.time() < deadline - 5:
            new_edges = _local_search(adj_bits, edges, deadline - 5, max_iters=50)
            if len(new_edges) > len(edges):
                edges = [[u, v] for u, v in new_edges]
                adj_bits = _build_adj_bits(edges)
        
        return edges
    
    # Fallback: Start with K3,13 and greedily add edges
    edges = [[a, b] for a in range(3) for b in range(3, 16)]
    adj_bits = _build_adj_bits(edges)
    
    # All candidate edges not yet in the graph
    all_edges = []
    for u in range(_N):
        for v in range(u + 1, _N):
            if not (adj_bits[u] >> v) & 1:
                all_edges.append((u, v))
    
    # Try multiple random orderings
    best_edges = list(edges)
    best_count = len(edges)
    
    for trial in range(5):
        if time.time() > deadline:
            break
        trial_edges = list(best_edges)
        trial_adj = _build_adj_bits(trial_edges)
        
        candidates = [(u, v) for u, v in all_edges 
                     if not (trial_adj[u] >> v) & 1]
        random.shuffle(candidates)
        
        added = _greedy_add_edges(trial_adj, candidates, deadline)
        trial_edges.extend(added)
        
        if len(trial_edges) > best_count:
            best_edges = trial_edges
            best_count = len(trial_edges)
    
    # Local search on best result
    if time.time() < deadline - 5:
        new_edges = _local_search(_build_adj_bits(best_edges), best_edges, deadline - 5, max_iters=30)
        if len(new_edges) > len(best_edges):
            best_edges = new_edges
    
    return [[u, v] for u, v in best_edges]
# EVOLVE-BLOCK-END
