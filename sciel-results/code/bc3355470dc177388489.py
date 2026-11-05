"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

_N = 16
_E = _N * (_N - 1) // 2  # 120 edges
_DEADLINE_MARGIN = 15  # seconds to spare before the 150s kill

# Precompute Q3 structure
# Q3 vertices are 0..7, edges connect vertices differing in exactly 1 bit
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

# All permutations of Q3 vertices (8! = 40320) - precompute for speed
# Actually, we'll use a smarter approach: try all 8-vertex subsets and check
# if Q3 is a subgraph of the induced subgraph.

# Precompute all 8-vertex subsets of 16 vertices
_ALL_8SETS = list(itertools.combinations(range(_N), 8))

def _has_q3_fast(adj):
    """
    Check if graph with adjacency matrix adj (list of lists of bools) contains Q3.
    Uses the fact that Q3 is bipartite with parts of size 4.
    For each 8-vertex subset, check if Q3 is a subgraph.
    Optimization: Q3 is bipartite (4,4). For a set S of 8 vertices,
    we need to find a bipartition (A,B) of S with |A|=|B|=4 such that
    all 12 edges of Q3 are present in the induced subgraph on S.
    
    Actually, Q3 as a subgraph doesn't require the induced subgraph to be exactly Q3.
    We just need 12 specific edges to be present.
    
    More efficient: enumerate all ways to assign Q3 vertices to 8 distinct graph vertices.
    Use backtracking with pruning.
    """
    # Use backtracking: assign Q3 vertex 0 to a graph vertex, then Q3 vertex 1, etc.
    # Order Q3 vertices by degree (all degree 3) - no help.
    # Instead, use the bipartite structure.
    
    # Q3 bipartition: even parity = {0,3,5,6}, odd parity = {1,2,4,7}
    # Edges only go between even and odd.
    q3_even = [0, 3, 5, 6]  # even parity
    q3_odd = [1, 2, 4, 7]   # odd parity
    
    # For each choice of 4 vertices for the "even" part and 4 for the "odd" part:
    # Check if there's a bijection from q3_even to the 4 chosen vertices and
    # from q3_odd to the other 4 such that all Q3 edges are present.
    
    # This is: for each 8-subset S, for each 4-subset A of S,
    # check if the bipartite graph between A and S\A contains Q3.
    
    # Even more efficient: precompute for each pair (u,v) whether edge exists.
    # For a given 4-set A and 4-set B, check if the 4x4 bipartite graph
    # contains Q3 as a subgraph.
    
    # Q3 bipartite edges: each even vertex connects to 3 odd vertices.
    # q3_even[0]=0 connects to q3_odd: 0^1=1, 0^2=2, 0^4=4 -> indices 0,1,2 in q3_odd
    # q3_even[1]=3 connects to: 3^1=2, 3^2=1, 3^4=7 -> indices 1,0,3 in q3_odd
    # q3_even[2]=5 connects to: 5^1=4, 5^2=7, 5^4=1 -> indices 2,3,1 in q3_odd
    # q3_even[3]=6 connects to: 6^1=7, 6^2=4, 6^4=2 -> indices 3,2,0 in q3_odd
    
    # So the bipartite adjacency (even_idx -> odd_indices):
    # 0: [0,1,2]
    # 1: [0,1,3]
    # 2: [1,2,3]
    # 3: [0,2,3]
    
    # For a given 4-set A (mapped to even positions) and 4-set B (mapped to odd positions),
    # we need to check if there's a permutation of B such that the edge pattern matches.
    # There are 4! = 24 permutations of B.
    
    # Precompute the required pattern:
    # req[e][o] = 1 if q3_even[e] is connected to q3_odd[o] in Q3
    req = [[0]*4 for _ in range(4)]
    req[0] = [1,1,1,0]
    req[1] = [1,1,0,1]
    req[2] = [0,1,1,1]
    req[3] = [1,0,1,1]
    
    # For each 8-set, for each 4-subset as "even" part:
    for S in _ALL_8SETS:
        S_set = set(S)
        # Try all 4-subsets of S as the "even" part
        for A in itertools.combinations(S, 4):
            B = tuple(v for v in S if v not in A)
            # Check if there's a bijection from A to even positions and B to odd positions
            # such that all required edges are present.
            # Try all 24 permutations of B (assigning to odd positions 0,1,2,3)
            # And all 24 permutations of A (assigning to even positions 0,1,2,3)
            # Total: 24*24 = 576 checks per (S,A) pair.
            # Number of (S,A) pairs: C(16,8)*C(8,4) = 12870*70 = 900900
            # Total checks: 900900 * 576 = 518,918,400 - too slow!
            
            # Better: fix the assignment of A to even positions (24 ways) and
            # for each, check if any permutation of B works (24 ways).
            # But we can do better: for a fixed assignment of A to even positions,
            # the required edges from each even position to odd positions are fixed.
            # So for odd position o, we need a specific set of even positions to connect to.
            # This is a bipartite matching problem with specific constraints.
            
            # Actually, let's just do the 24x24 check but with early termination.
            # 576 * 900900 is too much. Let me think of a better approach.
            
            pass  # This approach is too slow. Let me use a different method.
    
    # Alternative: backtracking on Q3 vertices.
    # Assign Q3 vertex 0 to some graph vertex, then Q3 vertex 1, etc.
    # With pruning based on already-assigned neighbors.
    
    assigned = [-1] * 8  # q3_vertex -> graph_vertex
    used = [False] * _N
    
    def backtrack(qv):
        if qv == 8:
            return True
        for gv in range(_N):
            if used[gv]:
                continue
            # Check edges from qv to already-assigned Q3 vertices
            ok = True
            for pv in range(qv):
                if pv in _Q3_NEIGHBORS[qv]:
                    if not adj[gv][assigned[pv]]:
                        ok = False
                        break
            if ok:
                assigned[qv] = gv
                used[gv] = True
                if backtrack(qv + 1):
                    return True
                used[gv] = False
                assigned[qv] = -1
        return False
    
    return backtrack(0)

def _has_q3_bitmask(adj_bits):
    """
    Check Q3 using bitmask adjacency. adj_bits[u] is an int bitmask of neighbors of u.
    Faster than list-of-lists for the inner loop.
    """
    assigned = [-1] * 8
    used_mask = 0
    
    def backtrack(qv):
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
                used_mask_bit = 1 << gv
                if backtrack(qv + 1):
                    return True
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

def _is_q3_free_bitmask(adj_bits):
    """Check Q3-freeness using bitmask adjacency."""
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
    
    return not backtrack(0)

def _add_edge_check(adj_bits, u, v):
    """
    Try adding edge (u,v) to the graph represented by adj_bits.
    Returns True if the resulting graph is Q3-free, False otherwise.
    Modifies adj_bits in place if the edge is added (caller must restore if False).
    """
    # Temporarily add the edge
    adj_bits[u] |= (1 << v)
    adj_bits[v] |= (1 << u)
    
    # Check Q3-freeness
    free = _is_q3_free_bitmask(adj_bits)
    
    if not free:
        # Remove the edge
        adj_bits[u] &= ~(1 << v)
        adj_bits[v] &= ~(1 << u)
    
    return free

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE_MARGIN  # 135s total, 15s margin
    
    # Strategy: Start with a dense random graph, then remove edges to eliminate Q3.
    # Or: Start with K16 and remove edges until Q3-free.
    # Or: Start with a good construction and greedily add edges.
    
    # Best known construction: K3,13 + K13 = 117 edges (if Q3-free)
    # Let's verify this first.
    
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
        # Great! 117 edges. Try to add the 3 intra-A edges.
        for a in range(3):
            for b in range(a + 1, 3):
                if time.time() > deadline:
                    break
                if _add_edge_check(adj_bits, a, b):
                    edges.append([a, b])
                else:
                    # Edge was not added (already restored in _add_edge_check)
                    pass
        # Now we have a Q3-free graph. Try to add any remaining edges.
        # The only remaining edges are intra-A (3 edges) which we just tried.
        # Actually, all edges are now either present or rejected.
        # Let's do a local search: try swapping edges.
        
        # For now, return what we have.
        return edges
    
    # If K3,13 + K13 is not Q3-free, use a different approach.
    # Start with K3,13 (39 edges, definitely Q3-free since no 4 vertices in A)
    # and greedily add edges.
    
    edges = [[a, b] for a in range(3) for b in range(3, 16)]
    adj_bits = _build_adj_bits(edges)
    
    # All candidate edges not yet in the graph
    all_edges = []
    for u in range(_N):
        for v in range(u + 1, _N):
            if not (adj_bits[u] >> v) & 1:
                all_edges.append((u, v))
    
    # Shuffle for randomization
    random.shuffle(all_edges)
    
    # Greedily add edges
    for u, v in all_edges:
        if time.time() > deadline:
            break
        if _add_edge_check(adj_bits, u, v):
            edges.append([u, v])
    
    # Local search: try to swap edges to get more edges
    # Try: remove edge e1, add edge e2, if Q3-free and net gain, keep.
    # This is complex. For now, return the greedy result.
    
    return edges
# EVOLVE-BLOCK-END
