"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_N = 16
_E = _N * (_N - 1) // 2  # 120 edges
_DEADLINE = 140  # seconds, 10s safety margin before 150s kill

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


def _has_q3_with_edge(adj_bits, edge_u, edge_v):
    """
    Check if adding edge (edge_u, edge_v) to the graph creates a Q3.
    Only checks Q3 embeddings that contain this specific edge.
    Much faster than full Q3 detection.
    Returns True if a Q3 containing edge (edge_u, edge_v) exists.
    """
    # The new edge (edge_u, edge_v) must be part of the Q3.
    # In Q3, each edge connects two vertices that differ in exactly one bit.
    # We need to find a Q3 embedding where edge_u and edge_v play the roles
    # of two Q3 vertices connected by an edge.
    
    # For each pair of Q3 vertices (a, b) that are adjacent in Q3,
    # try assigning edge_u -> a, edge_v -> b and see if we can complete the embedding.
    
    # Precompute: for each Q3 vertex, which other Q3 vertices are its neighbors
    # and which are at distance 2 (for constraint checking)
    
    # We'll use backtracking. Fix edge_u and edge_v to be Q3 vertices a and b (adjacent).
    # Then assign the remaining 6 Q3 vertices to graph vertices.
    
    # To speed up, we can precompute for each Q3 vertex pair (a,b) whether they're adjacent.
    # Then for each such pair, try the assignment.
    
    # Order of Q3 vertices for backtracking: start with the two fixed, then
    # choose remaining in an order that maximizes constraints.
    
    # Let's define the Q3 vertices as 0..7 (3-bit numbers).
    # Adjacent if XOR is 1, 2, or 4.
    
    # For a given assignment of some Q3 vertices to graph vertices,
    # check if it's consistent: for each pair of assigned Q3 vertices that are
    # adjacent in Q3, the corresponding graph vertices must be adjacent;
    # for each pair that are NOT adjacent in Q3, the corresponding graph
    # vertices must NOT be adjacent (since we're looking for a subgraph isomorphic
    # to Q3, non-adjacent Q3 vertices must map to non-adjacent graph vertices).
    
    # Actually, for a subgraph (not induced), we only need: adjacent Q3 vertices
    # map to adjacent graph vertices. Non-adjacent Q3 vertices can map to anything.
    
    # Wait, the task says "no subgraph isomorphic to the 3-cube Q3". A subgraph
    # isomorphic to Q3 means there exists an injective mapping from Q3 vertices
    # to graph vertices such that for every edge in Q3, the corresponding graph
    # vertices are adjacent. It does NOT require non-edges in Q3 to map to
    # non-edges in the graph.
    
    # So we only need: for each edge (a,b) in Q3, adj_bits[graph_a] has graph_b.
    
    # This makes it easier. Let me redo.
    
    # For each adjacent pair (a, b) in Q3:
    #   Try assigning edge_u -> a, edge_v -> b
    #   Then backtrack on the remaining 6 Q3 vertices.
    
    # But we also need to try edge_u -> b, edge_v -> a (swapped).
    
    # Let's be more systematic. For each ordered pair (a, b) where a and b are
    # adjacent in Q3:
    #   assigned = {a: edge_u, b: edge_v}
    #   Check: adj_bits[edge_u] must have edge_v (it does, since we're adding it)
    #   Then backtrack on remaining Q3 vertices.
    
    # To make backtracking fast, order the remaining Q3 vertices by how many
    # constraints they have on already-assigned vertices.
    
    # For a Q3 vertex q, its constraints come from its neighbors in Q3 that
    # are already assigned. For each such neighbor p assigned to graph vertex g,
    # q must be assigned to a graph vertex that is adjacent to g (if q-p is an edge)
    # or not adjacent to g (if q-p is not an edge, but we don't care about that).
    
    # Since we only care about edges, for each unassigned Q3 vertex q, the
    # candidates are graph vertices that are adjacent to all assigned neighbors
    # of q in Q3.
    
    # Let me implement this.
    
    # Precompute for each Q3 vertex, its neighbors in Q3
    # Already done: _Q3_NEIGHBORS
    
    # For backtracking, we need to track which graph vertices are used.
    
    # Strategy: for each adjacent pair (a, b) in Q3:
    #   Try two assignments: (edge_u -> a, edge_v -> b) and (edge_u -> b, edge_v -> a)
    #   For each, do backtracking on remaining 6 Q3 vertices.
    
    # To avoid redundant work, note that (a, b) and (b, a) are the same edge in Q3,
    # but the assignments are different. So we need to try both.
    
    # Actually, let's just iterate over all ordered pairs (a, b) where a != b and
    # a and b are adjacent in Q3. There are 12 such unordered pairs, so 24 ordered pairs.
    
    # For each, fix the assignment and backtrack.
    
    # To speed up backtracking, order the remaining Q3 vertices by the number
    # of assigned neighbors (more constraints = better pruning).
    
    # Let's precompute the order of Q3 vertices for backtracking given the
    # initially assigned vertices.
    
    # Actually, since the Q3 is small (8 vertices), let's just try all 24 ordered
    # pairs and for each, do a simple backtracking.
    
    # For backtracking, we'll use a recursive function that assigns Q3 vertices
    # one by one.
    
    # To further speed up, we can precompute for each Q3 vertex and each set of
    # already-assigned neighbors, the candidate graph vertices.
    
    # Let me just implement it directly.
    
    for a in range(8):
        for b in range(8):
            if a == b:
                continue
            if b not in _Q3_NEIGHBORS[a]:
                continue
            
            # Try assignment: a -> edge_u, b -> edge_v
            # Check: edge_u and edge_v must be adjacent (they are, by assumption)
            
            # Now backtrack on remaining 6 Q3 vertices
            assigned = [-1] * 8
            assigned[a] = edge_u
            assigned[b] = edge_v
            used = (1 << edge_u) | (1 << edge_v)
            
            if _backtrack_q3(adj_bits, assigned, used):
                return True
    
    return False


def _backtrack_q3(adj_bits, assigned, used):
    """
    Backtrack to complete a Q3 embedding.
    assigned: list of 8, assigned[qv] = graph vertex or -1
    used: bitmask of used graph vertices
    Returns True if a valid embedding is found.
    """
    # Find the next unassigned Q3 vertex with the most constraints
    # (i.e., most assigned neighbors)
    best_qv = -1
    best_num_assigned_neighbors = -1
    
    for qv in range(8):
        if assigned[qv] != -1:
            continue
        # Count how many of qv's Q3 neighbors are already assigned
        num_assigned = 0
        for nb in _Q3_NEIGHBORS[qv]:
            if assigned[nb] != -1:
                num_assigned += 1
        if num_assigned > best_num_assigned_neighbors:
            best_num_assigned_neighbors = num_assigned
            best_qv = qv
    
    if best_qv == -1:
        # All assigned, check if it's a valid embedding
        # We only need to check that for each Q3 edge, the graph vertices are adjacent
        # But we've been checking this during backtracking, so it should be valid.
        return True
    
    qv = best_qv
    
    # For each candidate graph vertex, check if it's consistent with assigned neighbors
    for gv in range(_N):
        if (used >> gv) & 1:
            continue
        
        # Check against all assigned neighbors of qv in Q3
        ok = True
        for nb in _Q3_NEIGHBORS[qv]:
            if assigned[nb] != -1:
                # qv and nb are adjacent in Q3, so gv must be adjacent to assigned[nb]
                if not (adj_bits[gv] >> assigned[nb]) & 1:
                    ok = False
                    break
        
        if ok:
            assigned[qv] = gv
            if _backtrack_q3(adj_bits, assigned, used | (1 << gv)):
                return True
            assigned[qv] = -1
    
    return False


def _has_q3_fast(adj_bits):
    """
    Check if graph with adjacency bitmask contains Q3.
    Uses backtracking with pruning.
    """
    assigned = [-1] * 8
    used_mask = 0

    def backtrack(qv):
        nonlocal used_mask
        if qv == 8:
            return True
        for gv in range(_N):
            if (used_mask >> gv) & 1:
                continue
            ok = True
            for pv in range(qv):
                if pv in _Q3_NEIGHBORS[qv]:
                    if not (adj_bits[gv] >> assigned[pv]) & 1:
                        ok = False
                        break
            if ok:
                assigned[qv] = gv
                used_mask |= (1 << gv)
                if backtrack(qv + 1):
                    return True
                used_mask &= ~(1 << gv)
                assigned[qv] = -1
        return False

    return backtrack(0)


def _build_adj_bits(edge_list):
    """Build adjacency bitmask representation from edge list."""
    adj = [0] * _N
    for u, v in edge_list:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj


def construct():
    start_time = time.time()
    
    # Start with K16
    edges = [[u, v] for u in range(_N) for v in range(u + 1, _N)]
    adj_bits = _build_adj_bits(edges)
    
    # Phase 1: Greedy removal - remove edges to make graph Q3-free
    # While graph has Q3, find a Q3 and remove an edge from it
    while time.time() < start_time + _DEADLINE - 10:
        if not _has_q3_fast(adj_bits):
            break
        
        # Find a Q3 in the graph
        # We'll use the backtracking to find one, then remove an edge from it
        # To make this efficient, let's find a Q3 and remove a "critical" edge
        
        # For now, just remove a random edge from a found Q3
        # Actually, let's find a Q3 and remove the edge that appears in the most Q3s
        
        # Simpler: find any Q3, remove one of its edges
        # To find a Q3, use _has_q3_fast but modified to return the embedding
        
        # Let's modify the approach: find a Q3, then remove an edge from it
        # that maximizes the number of Q3s broken
        
        # For now, let's just do a simple greedy: find a Q3, remove a random edge from it
        # This is slow but should work within time
        
        # Actually, let's use a different strategy:
        # Start with K16, and iteratively remove edges that are part of many Q3s
        
        # To count how many Q3s an edge is part of, we'd need to enumerate all Q3s,
        # which is expensive. Instead, let's use a heuristic.
        
        # Alternative: use the _has_q3_with_edge function to guide removal
        # For each edge, check if removing it helps (i.e., if the graph without
        # that edge is closer to being Q3-free)
        
        # This is too expensive. Let's just do a simple greedy removal.
        
        # Find a Q3 in the current graph
        # We'll use a modified backtracking that returns the embedding
        q3_found = False
        assigned = [-1] * 8
        used_mask = 0
        
        def find_q3(qv):
            nonlocal used_mask, q3_found
            if q3_found:
                return True
            if qv == 8:
                q3_found = True
                return True
            for gv in range(_N):
                if (used_mask >> gv) & 1:
                    continue
                ok = True
                for pv in range(qv):
                    if pv in _Q3_NEIGHBORS[qv]:
                        if not (adj_bits[gv] >> assigned[pv]) & 1:
                            ok = False
                            break
                if ok:
                    assigned[qv] = gv
                    used_mask |= (1 << gv)
                    if find_q3(qv + 1):
                        return True
                    used_mask &= ~(1 << gv)
                    assigned[qv] = -1
            return False
        
        find_q3(0)
        
        if not q3_found:
            break
        
        # We found a Q3 in 'assigned'. Now remove an edge from it.
        # To choose which edge to remove, let's pick the one that appears
        # in the most 4-cycles (heuristic for breaking many Q3s)
        
        # For simplicity, let's just remove a random edge from the Q3
        # Actually, let's remove the edge that has the highest degree sum
        # (heuristic: high-degree vertices are more likely to be in many Q3s)
        
        # Get the edges of the Q3
        q3_edges = []
        for i in range(8):
            for j in range(i + 1, 8):
                if j in _Q3_NEIGHBORS[i]:
                    gu, gv = assigned[i], assigned[j]
                    q3_edges.append((gu, gv))
        
        # Pick the edge with highest degree sum
        best_edge = None
        best_score = -1
        for gu, gv in q3_edges:
            score = bin(adj_bits[gu]).count('1') + bin(adj_bits[gv]).count('1')
            if score > best_score:
                best_score = score
                best_edge = (gu, gv)
        
        # Remove the edge
        gu, gv = best_edge
        adj_bits[gu] &= ~(1 << gv)
        adj_bits[gv] &= ~(1 << gu)
        edges.remove([gu, gv]) if [gu, gv] in edges else edges.remove([gv, gu])
    
    # Phase 2: Try to add back edges greedily
    if time.time() < start_time + _DEADLINE - 5:
        # Try to add back all missing edges
        for u in range(_N):
            for v in range(u + 1, _N):
                if time.time() > start_time + _DEADLINE - 2:
                    break
                if (adj_bits[u] >> v) & 1:
                    continue
                # Try adding edge (u, v)
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                
                if not _has_q3_fast(adj_bits):
                    edges.append([u, v])
                else:
                    adj_bits[u] &= ~(1 << v)
                    adj_bits[v] &= ~(1 << u)
    
    return edges
# EVOLVE-BLOCK-END
