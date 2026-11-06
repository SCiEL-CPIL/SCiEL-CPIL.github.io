"""C5-free graph with no independent 9-set. Maximize n."""
import time
import itertools
from collections import defaultdict

_START = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _check_c5(adj, n):
    """Check if graph has a 5-cycle. adj is list of sets."""
    for a in range(n):
        for b in itertools.combinations(adj[a], 2):
            b1, b2 = b
            if b1 < b2:
                for c in adj[b2]:
                    if c > b2 and c != a:
                        for d in adj[c]:
                            if d > c and d != a and d != b1 and d != b2:
                                if b1 in adj[d]:
                                    return True
    return False

def _check_c5_fast(adj, n):
    """Faster C5 check using edge-based approach."""
    # A 5-cycle: v0-v1-v2-v3-v4-v0
    # For each pair of non-adjacent vertices, check if they have 2 common neighbors
    # that are also non-adjacent to each other (forming a path of length 2)
    # Actually: v0-v1-v2-v3-v4-v0 means v1,v4 are both neighbors of v0
    # and v1-v2-v3 is a path, v2-v3 are edges, v3-v4 edge, v4-v0 edge
    # Better: for each edge (u,v), check if there's a path of length 3 from u to v
    # through non-adjacent intermediates... this is getting complex.
    # Use: for each 4-tuple of distinct vertices a,b,c,d with edges a-b, b-c, c-d,
    # check if d-a is an edge and a-c is NOT an edge (to avoid triangles)
    # and b-d is NOT an edge (to avoid 4-cycles being miscounted)
    # Actually for a simple C5 check:
    # v0-v1-v2-v3-v4-v0: need edges (v0,v1),(v1,v2),(v2,v3),(v3,v4),(v4,v0)
    # and non-edges (v0,v2),(v0,v3),(v1,v3),(v1,v4),(v2,v4) for induced,
    # but we just need it as a subgraph, so just need the 5 edges.
    
    # For each pair (a, b) that are adjacent, look for a path of length 4 from a to b
    # That's expensive. Let's do:
    # For each triple (a, b, c) with a-b and b-c edges and a != c:
    #   For each d in adj[c] where d != a, d != b:
    #     For each e in adj[d] where e != b, e != c:
    #       If e == a: found C5 (a-b-c-d-a... wait that's only 4)
    # 
    # Let me be more careful. C5: v0-v1-v2-v3-v4-v0
    # Pick v0 and v1 as an edge. Then v2 is neighbor of v1 (not v0),
    # v3 is neighbor of v2 (not v0, v1), v4 is neighbor of v3 (not v0,v1,v2)
    # and v4 must be neighbor of v0.
    
    for v0 in range(n):
        for v1 in adj[v0]:
            for v2 in adj[v1]:
                if v2 == v0:
                    continue
                for v3 in adj[v2]:
                    if v3 == v0 or v3 == v1:
                        continue
                    for v4 in adj[v3]:
                        if v4 == v0 or v4 == v1 or v4 == v2:
                            continue
                        if v0 in adj[v4]:
                            return True
    return False

def _alpha(adj, n):
    """Compute independence number exactly for small n."""
    # For n up to ~35, use a greedy + branch approach
    # Convert to complement: independent set in G = clique in complement
    comp_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if j not in adj[i]:
                comp_adj[i].add(j)
                comp_adj[j].add(i)
    
    # Find max clique in complement
    best = 0
    # Use Bron-Kerbosch with pivot
    def bronk(R, P, X):
        nonlocal best
        if not P and not X:
            best = max(best, len(R))
            return
        # Pivot
        if P:
            pivot = next(iter(P))
            candidates = P - comp_adj[pivot]
        else:
            candidates = set()
        
        for v in list(candidates):
            P2 = P & comp_adj[v]
            bronk(R | {v}, P2, X & comp_adj[v])
            P.remove(v)
            X.add(v)
            if time.time() > _DEADLINE - 20:
                return
        return
    
    bronk(set(), set(range(n)), set())
    return best

def _is_valid(adj, n):
    """Check C5-free and alpha <= 8."""
    if _check_c5_fast(adj, n):
        return False
    if _alpha(adj, n) > 8:
        return False
    return True

def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 140.0
    
    # Start with 8 disjoint K4s: n=32, C5-free, alpha=8
    n = 32
    adj = [set() for _ in range(n)]
    for b in range(8):
        for i in range(4):
            for j in range(i+1, 4):
                u = 4*b + i
                v = 4*b + j
                adj[u].add(v)
                adj[v].add(u)
    
    # Verify base
    if not _is_valid(adj, n):
        # Fallback to 7 K4s
        n = 28
        adj = [set() for _ in range(n)]
        for b in range(7):
            for i in range(4):
                for j in range(i+1, 4):
                    u = 4*b + i
                    v = 4*b + j
                    adj[u].add(v)
                    adj[v].add(u)
        return {'n': n, 'edges': [[min(u,v), max(u,v)] for u in range(n) for v in sorted(adj[u])]}
    
    best_n = n
    best_adj = [s.copy() for s in adj]
    
    # Try to add vertices one by one
    # Strategy: add vertex with edges to existing vertices
    # A new vertex v connected to a set S of existing vertices.
    # C5 constraint: no 5-cycle can involve v.
    # A 5-cycle through v: v-a-b-c-d-v where a,d in S, b,c not in S (or in S)
    # Actually: v-a-b-c-d-v needs edges va, ab, bc, cd, dv
    # So if we add v connected to S, we need to ensure no such cycle exists.
    
    # Simpler approach: try all possible edge sets for new vertex, check validity
    # But that's 2^32 possibilities. Instead, use a heuristic:
    # Connect new vertex to a small number of vertices (to keep alpha low)
    # but enough to avoid creating C5s.
    
    # Key insight: if new vertex connects to a clique of size >= 3 in G,
    # then any C5 through v would need a path of length 3 between two neighbors
    # of v in G. If all neighbors of v form a clique, then any two neighbors
    # are adjacent, so a path v-a-b-c-d-v with a,d neighbors of v would need
    # a-b and c-d edges... but if a and d are adjacent (clique), this might still work.
    
    # Let's try: add vertex connected to all 4 vertices of one K4 block.
    # Then v + that K4 forms K5. No C5 in K5. 
    # But wait, does this create a C5 with vertices from other blocks?
    # v-a-b-c-d-v: a is in the K4, d must be neighbor of v, so d also in K4.
    # b is neighbor of a, c is neighbor of b and d.
    # If a,d in same K4, and b,c from other blocks... 
    # a-b edge means b adjacent to a. In our graph, a is only adjacent to its K4 mates.
    # So b must be in the same K4. Similarly c. So b,c in K4.
    # Then a-b-c-d is a path in K4, and v connects to a and d. 
    # In K4, a-b-c-d is a path of length 3. With v-a and v-d, we get v-a-b-c-d-v = 5-cycle!
    # So connecting to a full K4 creates a C5. Bad.
    
    # Connect to 3 vertices of a K4 (a triangle): v + triangle = K4.
    # C5 through v: v-a-b-c-d-v. a,d neighbors of v (in the triangle).
    # a-b: b neighbor of a. a is in K4, so b in K4 or... a only connects to K4 mates.
    # If b in K4: b-c: c neighbor of b. If c in K4: c-d: d in K4 (neighbor of v).
    # a-b-c-d in K4: path of length 3 in K4 between two vertices that are both neighbors of v.
    # In a K4, any two vertices have a path of length 2 between them (via the other two).
    # Path of length 3: a-b-c-d where all in K4. In K4, a-b-c-d means a,d not adjacent? No, in K4 all pairs adjacent.
    # a-b-c-d: edges ab, bc, cd. In K4 these exist. And d-a edge exists. So a-b-c-d-a is a 4-cycle in K4.
    # With v-a and v-d: v-a-b-c-d-v. Is this a 5-cycle? Vertices v,a,b,c,d all distinct. Edges: va,ab,bc,cd,dv. Yes!
    # So even connecting to a triangle in K4 creates a C5.
    
    # Connect to 2 vertices of a K4 (an edge): v + edge = triangle (K3).
    # C5 through v: v-a-b-c-d-v. a,d neighbors of v (the two K4 vertices).
    # a-b: b neighbor of a. In K4, a connects to 3 others. If b is the third vertex of K4 (not d):
    # b-c: c neighbor of b. If c is d: then we have v-a-b-d-v which is a 4-cycle, not 5.
    # For 5-cycle: v-a-b-c-d-v with 5 distinct vertices. a,d in K4.
    # b neighbor of a: b in K4 (the other two) or... a only connects to K4 mates.
    # So b in K4. c neighbor of b: c in K4 or other.
    # If c in K4: then c is the remaining vertex. d is the other neighbor of v in K4.
    # v-a-b-c-d-v: a,b,c,d all in K4. That's a 4-cycle in K4 plus v. But we need 5 distinct vertices.
    # v,a,b,c,d are 5 distinct. Edges: va, ab, bc, cd, dv. In K4, ab,bc,cd all exist. va,dv exist.
    # So yes, this is a C5!
    
    # Connect to 1 vertex of a K4: v + single = just an edge.
    # C5 through v: v-a-b-c-d-v. a is the K4 vertex. d must be neighbor of v, so d=a.
    # But d=a means v-a-...-a-v, which repeats a. Not a simple cycle.
    # So no C5 through v if v connects to only 1 vertex per K4 block? 
    # Wait, d must be a neighbor of v. If v only connects to a, then d=a, and the cycle would be v-a-b-c-a-v which has a repeated. Not valid.
    # So connecting v to exactly 1 vertex total: no C5 through v (any C5 through v needs at least 2 neighbors of v).
    # But then alpha increases by 1 (v is independent of all but 1 vertex).
    
    # So the strategy: connect new vertex to a small set S such that:
    # 1. No C5 is created
    # 2. Alpha doesn't increase beyond 8
    
    # For condition 1: if |S| <= 1, no C5 through v. If |S| >= 2, need to check.
    # For condition 2: v is in an independent set iff it's not adjacent to that set.
    # If v connects to S, then v can be in an independent set only if the set avoids S.
    
    # Let's try a different approach: use a known construction.
    # The 32-vertex graph: 8 K4s works.
    # Can we do 33? Try adding a vertex connected to 1 vertex from each of several K4s.
    
    # Actually, let me try a completely different approach:
    # Use the complement. We need: G is C5-free, alpha(G) <= 8.
    # Equivalently: complement of G is K9-free, and G has no C5.
    
    # Let me just try to add vertices with careful edge selection.
    
    for attempt in range(200):
        if time.time() > _DEADLINE - 30:
            break
        
        # Try adding a vertex with a random small set of edges
        new_n = n + 1
        new_adj = [s.copy() for s in adj]
        new_adj.append(set())
        
        # Choose a small set of neighbors for the new vertex
        # Try different sizes
        import random
        random.seed(attempt * 12345 + 67890)
        
        # Try connecting to 1, 2, or 3 vertices
        for edge_count in [1, 2, 3]:
            if time.time() > _DEADLINE - 30:
                break
            
            neighbors = set(random.sample(range(n), min(edge_count, n)))
            for nb in neighbors:
                new_adj[n].add(nb)
                new_adj[nb].add(n)
            
            if _check_c5_fast(new_adj, new_n):
                continue
            
            if _alpha(new_adj, new_n) <= 8:
                # Valid! Update
                n = new_n
                adj = new_adj
                best_n = n
                best_adj = [s.copy() for s in adj]
                break  # try to add another vertex
            else:
                # Try different neighbors
                continue
    
    # Build edge list
    edges = []
    for u in range(best_n):
        for v in sorted(best_adj[u]):
            if u < v:
                edges.append([u, v])
    
    return {'n': best_n, 'edges': edges}

# EVOLVE-BLOCK-END
