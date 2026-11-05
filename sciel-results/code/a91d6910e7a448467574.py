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

# For fast Q3 detection: precompute for each Q3 vertex, its neighbors
# and use a specific vertex ordering for backtracking
# Order Q3 vertices to maximize early pruning: start with high-degree vertices
# All Q3 vertices have degree 3, so use natural order but fix vertex 0.

# Precompute: for each qv, list of (neighbor_qv, is_neighbor) as a bitmask of Q3 neighbors
_Q3_QV_NEIGHBOR_MASK = [0] * 8
for i in range(8):
    mask = 0
    for j in _Q3_NEIGHBORS[i]:
        mask |= (1 << j)
    _Q3_QV_NEIGHBOR_MASK[i] = mask


def _has_q3_fast(adj_bits):
    """
    Fast Q3 subgraph detection.
    Uses backtracking with bitmask operations and early pruning.
    adj_bits: list of 16 integers, each a 16-bit mask of neighbors.
    """
    # Fix Q3 vertex 0 to each graph vertex (16 choices)
    # Then backtrack on remaining 7 Q3 vertices
    # Use ordering: 0, 1, 2, 3, 4, 5, 6, 7
    # Pruning: when assigning qv, check all already-assigned neighbors
    
    assigned = [-1] * 8
    used_mask = 0
    
    def backtrack(qv):
        nonlocal used_mask
        if qv == 8:
            return True
        
        # Get candidate graph vertices: unused and adjacent to all assigned neighbors
        candidates = (~used_mask) & ((1 << _N) - 1)
        
        # Filter: must be adjacent to all already-assigned Q3 neighbors of qv
        for pv in _Q3_NEIGHBORS[qv]:
            if pv < qv:  # already assigned
                pgv = assigned[pv]
                # candidates must be in adj_bits[pgv]
                candidates &= adj_bits[pgv]
        
        if candidates == 0:
            return False
        
        # Try each candidate
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
    
    # Try fixing Q3 vertex 0 to each graph vertex
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
    This is much faster than full Q3 check.
    """
    # In Q3, fix the edge as connecting Q3 vertices 0 and 1 (000 and 001)
    # Q3 vertex 0 = 000, Q3 vertex 1 = 001
    # Remaining Q3 vertices: 2=010, 3=011, 4=100, 5=101, 6=110, 7=111
    # 
    # Required edges:
    # From Q3v0(=u): to Q3v2, Q3v4  => u must connect to assigned[2], assigned[4]
    # From Q3v1(=v): to Q3v3, Q3v5  => v must connect to assigned[3], assigned[5]
    # Among {2,3,4,5,6,7}:
    #   2-3, 2-6, 3-7, 4-5, 4-6, 5-7, 6-7
    #
    # We need to assign graph vertices to Q3 vertices 2,3,4,5,6,7
    # such that all required edges exist.
    #
    # Backtracking on 6 Q3 vertices with 14 candidate graph vertices.
    
    # Precompute neighbor requirements:
    # Q3v2 needs: adj to Q3v3, Q3v6, and adj to u (Q3v0)
    # Q3v3 needs: adj to Q3v2, Q3v7, and adj to v (Q3v1)
    # Q3v4 needs: adj to Q3v5, Q3v6, and adj to u (Q3v0)
    # Q3v5 needs: adj to Q3v4, Q3v7, and adj to v (Q3v1)
    # Q3v6 needs: adj to Q3v2, Q3v4, Q3v7
    # Q3v7 needs: adj to Q3v3, Q3v5, Q3v6
    
    # Ordering for backtracking: 2, 4, 3, 5, 6, 7
    # (start with those having external constraints)
    
    order = [2, 4, 3, 5, 6, 7]
    assigned = {}
    used_mask = (1 << u) | (1 << v)
    
    # Precompute for each Q3 vertex in order, which previously-assigned Q3 vertices it must connect to
    # and which external vertex (u or v) it must connect to
    constraints = {
        2: {'q3_neighbors': [], 'ext': u},      # must connect to u
        4: {'q3_neighbors': [], 'ext': u},      # must connect to u
        3: {'q3_neighbors': [2], 'ext': v},     # must connect to v and Q3v2
        5: {'q3_neighbors': [4], 'ext': v},     # must connect to v and Q3v4
        6: {'q3_neighbors': [2, 4], 'ext': None},  # must connect to Q3v2 and Q3v4
        7: {'q3_neighbors': [3, 5], 'ext': None},  # must connect to Q3v3 and Q3v5
    }
    # Also add forward constraints for completeness
    # Q3v2 also connects to Q3v6 (but 6 is assigned later)
    # Q3v4 also connects to Q3v6 (but 6 is assigned later)
    # Q3v3 also connects to Q3v7 (but 7 is assigned later)
    # Q3v5 also connects to Q3v7 (but 7 is assigned later)
    
    def backtrack(idx):
        nonlocal used_mask
        if idx == len(order):
            # Check remaining edges: 6-7
            if not (adj_bits[assigned[6]] >> assigned[7]) & 1:
                return False
            return True
        
        qv = order[idx]
        con = constraints[qv]
        
        # Candidates: unused graph vertices
        candidates = (~used_mask) & ((1 << _N) - 1)
        
        # Must connect to external vertex if specified
        if con['ext'] is not None:
            candidates &= adj_bits[con['ext']]
        
        # Must connect to already-assigned Q3 neighbors
        for nqv in con['q3_neighbors']:
            if nqv in assigned:
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
    
    # Phase 1: Greedy edge addition
    # Try edges in a specific order, add if doesn't create Q3
    
    all_edges = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
    
    # Use a heuristic order: prefer edges between "different" parts
    # For now, try natural order first, then random orders
    
    best_edges = []
    best_count = 0
    best_adj = [0] * _N
    
    # Try multiple orderings
    num_trials = 50  # will be limited by time
    
    for trial in range(num_trials):
        if time.time() > deadline - 5:
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
        
        for u, v in edge_order:
            if time.time() > deadline - 3:
                break
            
            # Try adding edge (u, v)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            # Check if this creates a Q3 (only check Q3s containing this edge)
            if not _has_q3_with_edge(adj, u, v):
                edges.append([u, v])
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
            best_adj = adj
    
    # Phase 2: Local search - try to improve by swapping edges
    if time.time() < deadline - 10:
        adj = best_adj[:]
        edges = best_edges[:]
        edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in edges)
        
        # Try adding missing edges in a good order
        missing = [(u, v) for u, v in all_edges if (u, v) not in edge_set]
        random.shuffle(missing)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for u, v in missing:
                if time.time() > deadline - 3:
                    break
                if (u, v) in edge_set:
                    continue
                
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                if not _has_q3_with_edge(adj, u, v):
                    edges.append([u, v])
                    edge_set.add((u, v))
                    missing.remove((u, v))
                    improved = True
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
            best_adj = adj
    
    return best_edges
# EVOLVE-BLOCK-END
