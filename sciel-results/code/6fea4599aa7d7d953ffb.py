import time
import itertools
import random

def construct():
    start = time.time()
    DEADLINE = 140.0

    def time_left():
        return DEADLINE - time.time()

    def make_adj(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def has_c5(n, adj):
        """Check if graph has a 5-cycle as a subgraph using bitmasks."""
        # A 5-cycle: v0-v1-v2-v3-v4-v0
        # Fix v0 = 0, try all combinations
        # For efficiency: for each pair (v0, v1) with edge, for each (v1, v2) with edge,
        # v2 != v0, for each (v2, v3) with edge, v3 not in {v0,v1},
        # for each (v3, v4) with edge, v4 not in {v0,v1,v2},
        # check (v4, v0) edge exists
        # This is O(n^5) worst case but with early pruning much faster.
        # Better: for each 3-path v0-v1-v2-v3, check if there's v4 adjacent to both v0 and v3
        # and v4 not in {v0,v1,v2}.
        
        # Even better: count 5-cycles via:
        # For each edge (u,v), count common neighbors, etc.
        # Simplest exact: enumerate 5-cycles by fixing the smallest vertex.
        
        for v0 in range(n):
            nbrs0 = [u for u in range(v0+1, n) if (adj[v0] >> u) & 1]
            for v1 in nbrs0:
                nbrs1 = [u for u in range(v1+1, n) if (adj[v1] >> u) & 1 and u != v0]
                for v2 in nbrs1:
                    # v2 must not be adjacent to v0 for a proper 5-cycle? No, it can be.
                    # Actually for a 5-cycle subgraph, we just need the 5 edges to exist.
                    # v0-v1-v2-v3-v4-v0. v2 can be adjacent to v0 (chord), that's fine.
                    nbrs2 = [u for u in range(v2+1, n) if (adj[v2] >> u) & 1 and u != v0 and u != v1]
                    for v3 in nbrs2:
                        # Need v4 adjacent to both v3 and v0, v4 > v3 (to avoid duplicates)
                        # v4 not in {v0, v1, v2, v3}
                        common = adj[v3] & adj[v0]
                        # v4 must be > v3, not in {v0,v1,v2}
                        for v4 in range(v3+1, n):
                            if (common >> v4) & 1 and v4 != v0 and v4 != v1 and v4 != v2:
                                return True
        return False

    def alpha_ge(n, adj, target):
        """Check if independence number >= target using branch and bound with bitmasks."""
        # Maximum independent set via branch and bound
        # State: set of candidate vertices (as bitmask), current size
        best = [0]
        
        def dfs(candidates, count):
            if count >= target:
                return True
            if candidates == 0:
                return False
            if count + bin(candidates).count('1') < target:
                return False
            if time.time() > DEADLINE:
                return False
            
            # Pick a vertex from candidates
            # Heuristic: pick vertex with most neighbors in candidates (to reduce branching)
            v = candidates & (-candidates)  # lowest set bit
            v_idx = v.bit_length() - 1
            
            # Branch 1: include v
            new_cands = candidates & ~adj[v_idx] & ~(v)
            if dfs(new_cands, count + 1):
                return True
            
            # Branch 2: exclude v
            if dfs(candidates & ~v, count):
                return True
            
            return False
        
        all_verts = (1 << n) - 1
        return dfs(all_verts, 0)

    def validate(n, adj):
        """Check C5-free and alpha < 9."""
        if time.time() > DEADLINE:
            return False
        if has_c5(n, adj):
            return False
        if alpha_ge(n, adj, 9):
            return False
        return True

    # Base: 8 disjoint K4s
    n_base = 32
    edges_base = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i+1, 4):
                edges_base.append([base+i, base+j])
    
    adj_base = make_adj(n_base, edges_base)
    
    if not validate(n_base, adj_base):
        # Shouldn't happen, but fallback to 7 K4s
        n_fb = 28
        edges_fb = []
        for b in range(7):
            base = 4 * b
            for i in range(4):
                for j in range(i+1, 4):
                    edges_fb.append([base+i, base+j])
        return {'n': n_fb, 'edges': edges_fb}
    
    best_n = n_base
    best_edges = edges_base[:]
    
    # Try to add vertices 32, 33, 34
    # For each new vertex, try different connection patterns
    # Key constraint: adding vertex v connected to set S of existing vertices
    # must not create a C5 and must keep alpha < 9
    
    for new_n in range(33, 36):
        if time.time() > DEADLINE - 5:
            break
        new_v = new_n - 1
        
        # Try random connection patterns
        # The new vertex connects to some subset of existing vertices
        # For C5-freedom: if new_v connects to u and w, then there must be no
        # path of length 3 from u to w in the existing graph (otherwise u-new_v-w-...-u forms C5)
        # Wait, more precisely: a C5 through new_v: new_v-a-b-c-d-new_v
        # So we need: a,d adjacent to new_v, and a-b-c-d is a path of length 3 in G
        
        # Strategy: try sparse connections first (fewer edges from new_v)
        # Actually, for alpha: new_v in an independent set means new_v is not connected to others in the set.
        # So if new_v is connected to many vertices, it's harder to include in independent set.
        
        # Let's try: connect new_v to one vertex in each K4 (8 edges total)
        # This way, any independent set containing new_v can't contain those 8 vertices,
        # but can contain the other 3 from each K4... wait no.
        # Independent set: if new_v is in it, it can't be adjacent to any member.
        # If new_v is adjacent to exactly one vertex in each K4, then an independent set
        # containing new_v can pick at most one from each K4 minus the connected one.
        # From each K4, the connected vertex is excluded, leaving 3 vertices (a K3).
        # From a K3, we can pick at most 1. So independent set with new_v: 1 (new_v) + 8*1 = 9. That's too many!
        
        # So connecting to exactly one per K4 gives alpha >= 9. Bad.
        
        # Connect to 2 vertices in each K4? Then from each K4, 2 vertices excluded, 2 remain (an edge).
        # Independent set with new_v: 1 + 8*1 = 9. Still too many!
        
        # Connect to 3 vertices in each K4? 1 vertex remains per K4.
        # Independent set with new_v: 1 + 8*1 = 9. Still 9!
        
        # Connect to 4 vertices in each K4 (all of them)? 
        # Independent set with new_v: 1 + 0 = 1. Good for alpha.
        # But C5: new_v connected to all in a K4. Take new_v, a, b, c, d all in same K4.
        # new_v-a-b-c-d-new_v: all edges exist (K4 + new_v universal to that K4). That's a C5! Bad.
        
        # So we need a more nuanced approach. Let me think...
        # 
        # What if new_v is NOT connected to all vertices in any single K4?
        # For C5-freedom through new_v: no two neighbors of new_v can have a path of length 3 between them.
        # In 8 disjoint K4s, a path of length 3 exists between two vertices iff they're in the same K4
        # (since K4 has paths of all lengths between any two non-adjacent... wait, in K4 all pairs are adjacent).
        # In a K4, any two vertices are adjacent, so a path of length 3 between u and v in K4:
        # u-w-x-v where w,x are the other two vertices. Yes, that exists.
        # So if new_v is connected to two vertices in the same K4, there's a C5 through new_v.
        # 
        # Therefore: new_v can be connected to at most ONE vertex in each K4!
        # 
        # But we showed that connecting to one per K4 gives alpha >= 9.
        # 
        # So it's impossible to add a vertex to 8 disjoint K4s while maintaining both properties?
        # 
        # Wait, let me recheck the alpha calculation.
        # If new_v is connected to exactly one vertex in each K4:
        # Independent set containing new_v: new_v + (from each K4, pick vertices not adjacent to new_v)
        # In each K4, 1 vertex is adjacent to new_v, 3 are not. The 3 non-adjacent form a K3.
        # From K3, max independent set is 1.
        # So total: 1 + 8*1 = 9. Yes, alpha >= 9.
        #
        # What if new_v is connected to 0 vertices in some K4 and 1 in others?
        # Say connected to 1 vertex in k K4s and 0 in (8-k) K4s.
        # Independent set with new_v: 1 + k*1 + (8-k)*1 = 1 + 8 = 9. Still 9!
        # Because even in a K4 with no connection to new_v, we can still pick 1 vertex.
        #
        # So ANY single vertex added to 8 disjoint K4s will have alpha >= 9?
        # Unless... the new vertex's connections affect things differently.
        #
        # Actually wait: if new_v is connected to 0 vertices (isolated), then independent set
        # can include new_v + one from each K4 = 1 + 8 = 9. Still 9.
        #
        # It seems like 8 K4s + any single vertex always has alpha >= 9.
        # 
        # So we can't simply add a vertex. We need to modify the structure.
        #
        # Alternative: use a different base graph. What about 7 K4s + some other structure?
        # Or merge some K4s?
        #
        # Actually, let me reconsider. The upper bound is 35, best known is 32.
        # Maybe 32 is optimal and we should just return 8 K4s cleanly.
        #
        # But let me try one more idea: what if we don't use disjoint K4s?
        # What about a different C5-free graph with alpha < 9 and n > 32?
        #
        # For now, let's just return the 8 K4s construction which gives n=32.
        # This is a structural improvement over the seed's n=28.
    
    return {'n': best_n, 'edges': best_edges}
