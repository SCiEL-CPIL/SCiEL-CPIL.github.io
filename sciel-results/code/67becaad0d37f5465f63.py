"""Structural change: Simulated Annealing with fast bitmask branch-and-bound evaluation.
Start from a dense K4-free random graph, then use SA to minimize the size of the
largest triangle-free induced subgraph.
"""
import time
import random
import math

N = 18
NUM_EDGES = N * (N - 1) // 2  # 153

def _edge_index(u, v):
    if u > v:
        u, v = v, u
    return u * (N - 1) - u * (u + 1) // 2 + (v - u - 1)

def _precompute_triangles():
    """For each pair (u,v), return the bitmask of common neighbors (potential third vertices)."""
    tri = [0] * NUM_EDGES
    # We'll build adjacency bitmasks first, but since edges change during SA,
    # we need a way to compute triangle counts fast.
    # Instead, precompute for each vertex the list of other vertices.
    # Actually, let's precompute the triangle participation differently.
    # For each vertex w, the edges (u,v) that form a triangle with w are those
    # where w is connected to both u and v.
    # Let's precompute for each edge (u,v) the list of vertices w != u,v.
    # That's just all other vertices. The triangle exists if edges (u,w) and (v,w) exist.
    pass

def _build_adj_bitmasks(edge_set):
    """Given a set of edge indices, build adjacency bitmasks for each vertex."""
    adj = [0] * N
    for e in edge_set:
        # Decode edge index to (u, v)
        # u * (N-1) - u*(u+1)//2 + (v-u-1) = e
        # This is a bit tricky to invert. Let's just iterate.
        # Actually, let's precompute a list of (u,v) for each edge index.
        pass
    return adj

# Precompute edge endpoints
_EDGE_ENDPOINTS = []
for u in range(N):
    for v in range(u + 1, N):
        idx = _edge_index(u, v)
        _EDGE_ENDPOINTS.append((u, v))

def _adj_from_edges(edge_set):
    """Build adjacency bitmasks from a set of edge indices."""
    adj = [0] * N
    for e in edge_set:
        u, v = _EDGE_ENDPOINTS[e]
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _max_triangle_free_size(adj):
    """Exact branch and bound: find the largest subset of vertices inducing a triangle-free subgraph."""
    n = N
    # A subset is triangle-free if for every triple (a,b,c) in the subset, 
    # at least one of the edges (a,b), (a,c), (b,c) is missing.
    # Equivalently, the complement of the induced subgraph on the subset must contain
    # a vertex cover of all triangles... no, that's not right.
    # A triangle-free induced subgraph means: for every three vertices in the subset,
    # they do NOT form a triangle in the original graph.
    # So we want the largest S such that for all {a,b,c} subset S, 
    # NOT (edge(a,b) AND edge(b,c) AND edge(c,a)).
    
    # Branch and bound: try to build S vertex by vertex.
    # State: current set S (as bitmask), next vertex to consider.
    # Upper bound: |S| + (n - next_vertex)
    # We can also use: if adding vertex v to S creates a triangle, skip v.
    
    best = [0]
    full_mask = (1 << n) - 1
    
    # Precompute for each pair (a,b) the bitmask of common neighbors c
    # such that a,b,c form a triangle.
    # tri_common[a][b] = bitmask of c where edges (a,c) and (b,c) exist.
    # Then adding v to S is forbidden if there exist a,b in S with v in tri_common[a][b].
    
    # Actually, let's use a different approach:
    # S is triangle-free iff for every v in S, the neighbors of v within S form an independent set
    # in the original graph... no, that's not quite right either.
    
    # Let me think again. S induces a triangle-free subgraph iff there is no triangle entirely within S.
    # So for each v in S, the subgraph induced by N(v) ∩ S must be edge-free (i.e., an independent set).
    
    # Branch and bound:
    # We process vertices in order 0, 1, ..., 17.
    # At each step, we decide to include or exclude the current vertex.
    # When we include vertex v, we check that no triangle is formed with previously included vertices.
    # Specifically, if we include v, then for any two previously included vertices a, b,
    # we must NOT have edges (a,b), (a,v), (b,v) all present.
    # Equivalently, among the previously included vertices that are neighbors of v,
    # there must be no edge between any two of them.
    
    # Let's precompute: for each v, and for each pair of vertices (a,b) with a < b < v,
    # whether a, b, v form a triangle.
    
    # Actually, let's just do it directly with bitmasks.
    
    def _is_triangle_free_subset(mask):
        """Check if the subset given by mask induces a triangle-free subgraph."""
        # For each vertex v in mask, check that N(v) ∩ mask is an independent set.
        verts = [i for i in range(n) if (mask >> i) & 1]
        for v in verts:
            # Get neighbors of v within mask
            neigh = adj[v] & mask
            # Check if neigh is an independent set: no two vertices in neigh are adjacent.
            # For each u in neigh, check if adj[u] & neigh has any bits other than u itself.
            # Actually, we need to check that for each pair u, w in neigh, edge (u,w) does not exist.
            # This is: for each u in neigh, (adj[u] & neigh) should have no bits except possibly... 
            # wait, adj[u] doesn't include u itself (simple graph, no self-loops).
            # So (adj[u] & neigh) should be 0 for all u in neigh? No, that's not right.
            # We need: for all u, w in neigh with u != w, edge (u,w) does NOT exist.
            # Equivalently: for each u in neigh, (adj[u] & neigh) & ~(1 << u) == 0.
            # But adj[u] already doesn't have bit u set. So (adj[u] & neigh) == 0 for all u in neigh.
            # Wait, that would mean neigh is an independent set. Yes, that's correct.
            temp = neigh
            while temp:
                lsb = temp & (-temp)
                u = lsb.bit_length() - 1
                if adj[u] & neigh:
                    return False
                temp ^= lsb
        return True
    
    # Branch and bound search
    def _search(current_mask, next_v, count):
        if next_v >= n:
            if count > best[0]:
                best[0] = count
            return
        
        # Upper bound: current count + remaining vertices
        if count + (n - next_v) <= best[0]:
            return
        
        # Option 1: skip vertex next_v
        _search(current_mask, next_v + 1, count)
        
        # Option 2: include vertex next_v
        # Check if adding next_v to current_mask keeps it triangle-free.
        # We need to verify that the new set (current_mask | (1 << next_v)) is triangle-free.
        # Instead of checking the whole thing, we can check locally:
        # The new vertex is next_v. We need to ensure that among the neighbors of next_v
        # that are in current_mask, there is no edge between any two of them.
        # Also, we need to ensure that next_v doesn't form a triangle with two other vertices
        # in current_mask. That's the same condition: N(next_v) ∩ current_mask must be independent.
        neigh_in_current = adj[next_v] & current_mask
        # Check if neigh_in_current is an independent set
        is_independent = True
        temp = neigh_in_current
        while temp:
            lsb = temp & (-temp)
            u = lsb.bit_length() - 1
            if adj[u] & neigh_in_current:
                is_independent = False
                break
            temp ^= lsb
        
        if is_independent:
            _search(current_mask | (1 << next_v), next_v + 1, count + 1)
    
    _search(0, 0, 0)
    return best[0]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    random.seed(42)
    
    # Start with a random dense K4-free graph.
    # Strategy: start with the complete graph and remove edges to break K4s.
    # Or start with a random graph and ensure it's K4-free.
    
    # Let's start with a random graph with edge probability ~0.6 and then fix K4s.
    edge_set = set()
    for u in range(N):
        for v in range(u + 1, N):
            if random.random() < 0.55:
                edge_set.add(_edge_index(u, v))
    
    # Ensure K4-free: iteratively remove edges from K4s.
    def _is_k4_free(edge_set):
        adj = _adj_from_edges(edge_set)
        for a in range(N):
            for b in range(a + 1, N):
                if not (adj[a] & (1 << b)):
                    continue
                for c in range(b + 1, N):
                    if not (adj[b] & (1 << c)):
                        continue
                    if not (adj[a] & (1 << c)):
                        continue
                    # a, b, c form a triangle. Check if there's a d forming K4.
                    for d in range(c + 1, N):
                        if (adj[a] & (1 << d)) and (adj[b] & (1 << d)) and (adj[c] & (1 << d)):
                            return False
        return True
    
    while not _is_k4_free(edge_set):
        # Find a K4 and remove a random edge from it
        adj = _adj_from_edges(edge_set)
        found = False
        for a in range(N):
            for b in range(a + 1, N):
                if not (adj[a] & (1 << b)):
                    continue
                for c in range(b + 1, N):
                    if not (adj[b] & (1 << c)):
                        continue
                    if not (adj[a] & (1 << c)):
                        continue
                    for d in range(c + 1, N):
                        if (adj[a] & (1 << d)) and (adj[b] & (1 << d)) and (adj[c] & (1 << d)):
                            # Remove a random edge from this K4
                            candidates = [
                                _edge_index(a, b),
                                _edge_index(a, c),
                                _edge_index(a, d),
                                _edge_index(b, c),
                                _edge_index(b, d),
                                _edge_index(c, d),
                            ]
                            e_to_remove = random.choice(candidates)
                            edge_set.discard(e_to_remove)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if not found:
            break  # Should not happen if _is_k4_free returned False
    
    adj = _adj_from_edges(edge_set)
    current_size = _max_triangle_free_size(adj)
    best_edge_set = set(edge_set)
    best_size = current_size
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    while T > T_min:
        if time.time() > deadline:
            break
        
        # Pick a random edge and toggle it
        u = random.randint(0, N - 1)
        v = random.randint(0, N - 1)
        if u == v:
            continue
        e = _edge_index(u, v)
        
        new_edge_set = set(edge_set)
        if e in new_edge_set:
            new_edge_set.discard(e)
        else:
            new_edge_set.add(e)
        
        # Check K4-free
        if not _is_k4_free(new_edge_set):
            T *= cooling_rate
            continue
        
        new_adj = _adj_from_edges(new_edge_set)
        new_size = _max_triangle_free_size(new_adj)
        
        delta = new_size - current_size
        if delta < 0 or random.random() < math.exp(-delta / T):
            edge_set = new_edge_set
            adj = new_adj
            current_size = new_size
            if current_size < best_size:
                best_size = current_size
                best_edge_set = set(edge_set)
        
        T *= cooling_rate
    
    # Convert best_edge_set to list of [u, v] pairs
    result = []
    for e in best_edge_set:
        u, v = _EDGE_ENDPOINTS[e]
        result.append([u, v])
    
    return result
