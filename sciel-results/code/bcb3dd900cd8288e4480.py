import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def elapsed():
        return time.time() - start_time

    def time_left():
        return deadline - time.time()

    def has_c5(adj, n):
        """Check if graph has a 5-cycle. adj is list of sets."""
        # Find all 5-cycles: v0-v1-v2-v3-v4-v0 with all distinct
        # Use adjacency lists for speed
        for v0 in range(n):
            for v1 in adj[v0]:
                if v1 <= v0:
                    continue
                for v2 in adj[v1]:
                    if v2 in (v0, v1):
                        continue
                    for v3 in adj[v2]:
                        if v3 in (v0, v1, v2):
                            continue
                        for v4 in adj[v3]:
                            if v4 in (v0, v1, v2, v3):
                                continue
                            if v0 in adj[v4]:
                                return True
        return False

    def has_independent_set_9(adj, n):
        """Check if there's an independent set of size 9."""
        # Use Bron-Kerbosch style or simple backtracking
        # For n<=35, we can use a smarter approach
        # Find maximum independent set
        # Use recursive search with pruning
        
        def max_independent_set(vertices, adj, n):
            if not vertices:
                return 0
            if len(vertices) <= 1:
                return len(vertices)
            
            # Greedy upper bound: at most len(vertices)
            # Pick a vertex, branch on include/exclude
            # Use a heuristic: pick vertex with max degree in induced subgraph
            
            # For speed, use a simple approach
            best = 0
            verts = list(vertices)
            
            # Upper bound
            if len(verts) + best <= 8:  # We only care if >= 9
                return 0
            
            # Pick a pivot
            v = verts[0]
            # Include v: can't include its neighbors
            non_neighbors = set(verts) - set(adj[v]) - {v}
            best = max(best, 1 + max_independent_set(non_neighbors, adj, n))
            
            # Exclude v
            best = max(best, max_independent_set(set(verts) - {v}, adj, n))
            
            return best
        
        # This is too slow. Let me use a different approach.
        # For n=33-35, check all 9-subsets is C(33,9) ≈ 10 million - too slow.
        # Use a smarter algorithm.
        
        # Use the fact that we can use a greedy coloring of the complement
        # or use a branch and bound
        
        # Actually, let me use a simpler approach: try to find an independent set of size 9
        # using backtracking with pruning
        
        def find_ind_set_9(adj, n):
            # Use recursive search
            # State: current set, candidates
            candidates = set(range(n))
            
            def search(current, candidates):
                if len(current) >= 9:
                    return True
                if not candidates:
                    return False
                # Upper bound: len(current) + len(candidates) < 9
                if len(current) + len(candidates) < 9:
                    return False
                
                # Pick a vertex from candidates
                # Heuristic: pick one with fewest neighbors in candidates (easier to extend)
                v = min(candidates, key=lambda x: len(adj[x] & candidates))
                
                # Branch 1: include v
                new_candidates = candidates - adj[v] - {v}
                if search(current | {v}, new_candidates):
                    return True
                
                # Branch 2: exclude v
                if search(current, candidates - {v}):
                    return True
                
                return False
            
            return search(set(), set(range(n)))
        
        return find_ind_set_9(adj, n)

    def is_valid(n, adj):
        if has_c5(adj, n):
            return False
        if has_independent_set_9(adj, n):
            return False
        return True

    def adj_to_edges(adj, n):
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append([i, j])
        return edges

    # Known valid: 8 copies of K4
    def make_8k4():
        n = 32
        adj = [set() for _ in range(n)]
        for b in range(8):
            for i in range(4):
                for j in range(i+1, 4):
                    u = 4*b + i
                    v = 4*b + j
                    adj[u].add(v)
                    adj[v].add(u)
        return n, adj

    n_best, adj_best = make_8k4()
    
    # Verify the known solution
    if not is_valid(n_best, adj_best):
        # Shouldn't happen, but fallback
        pass

    # Try to extend to n=33
    # Strategy 1: Add a vertex connected to some vertices
    # To keep alpha < 9, the new vertex should connect to enough vertices
    # To keep C5-free, need to be careful
    
    # Try random approaches for n=33, 34, 35
    random.seed(42)
    
    for target_n in [33, 34, 35]:
        if time_left() < 10:
            break
        
        # Try multiple random graphs
        for attempt in range(200):
            if time_left() < 5:
                break
            
            # Generate random graph with edge probability p
            # Higher p -> smaller alpha, but more likely to have C5
            p = random.uniform(0.6, 0.9)
            
            adj = [set() for _ in range(target_n)]
            for i in range(target_n):
                for j in range(i+1, target_n):
                    if random.random() < p:
                        adj[i].add(j)
                        adj[j].add(i)
            
            # Quick check: does it have C5?
            if has_c5(adj, target_n):
                continue
            
            # Check alpha < 9
            if has_independent_set_9(adj, target_n):
                continue
            
            # Valid!
            n_best = target_n
            adj_best = adj
            break
        
        if n_best == target_n:
            break  # Found valid graph for this n, try larger
    
    # Also try structured approach for n=33
    if n_best < 33 and time_left() > 15:
        # Start from 8xK4, try adding vertex 32
        base_n, base_adj = make_8k4()
        
        for attempt in range(500):
            if time_left() < 5:
                break
            
            adj = [set(s) for s in base_adj]
            adj.append(set())
            n = 33
            
            # Connect vertex 32 to random subset of existing vertices
            # To keep alpha < 9, vertex 32 should be connected to at least some vertices
            # in each "independent set candidate"
            
            # Random connection pattern
            for i in range(32):
                if random.random() < 0.5:
                    adj[32].add(i)
                    adj[i].add(32)
            
            if has_c5(adj, n):
                continue
            
            if has_independent_set_9(adj, n):
                continue
            
            n_best = 33
            adj_best = adj
            break
    
    # Try modifying the 8xK4 structure more cleverly
    if n_best < 33 and time_left() > 15:
        # Try: 7xK4 + one K4 with an extra vertex connected
        # Or: merge two K4s into a larger C5-free component
        
        # A C5-free graph on 5 vertices: K5 minus an edge? No, K5 has C5.
        # K4 + isolated vertex: C5-free, alpha=2
        # K4 + vertex connected to 1 vertex of K4: C5-free?
        #   The new vertex v connected to u in K4. Can we form C5?
        #   v-u-a-b-c-v? Need u-a, a-b, b-c, c-v. But v only connects to u. So no.
        #   Actually v only has one neighbor, so it can't be in a cycle of length >= 3.
        #   So C5-free. Alpha: the 3 non-neighbors of u in K4 plus... 
        #   In K4, removing u's neighbor, the other 3 form K3, alpha=1. Plus v is not adjacent to them.
        #   So alpha of K4+{v,u}: {v, a, b, c} where a,b,c are the other 3 in K4. That's 4.
        #   Wait, v is only adjacent to u. So v, a, b, c are mutually non-adjacent? 
        #   a,b,c are in K4 so they are adjacent to each other. So {v,a,b,c} is not independent.
        #   Independent sets: {v, a} where a is not u. Size 2. Or {a,b,c} is not independent (they're a clique).
        #   Actually in K4, any 2 vertices are adjacent. So alpha(K4) = 1.
        #   Adding v adjacent only to u: alpha = max(alpha of K4 with u removed + v, ...) 
        #   K4 - u is K3, alpha=1. v is not adjacent to K3. So {v} + independent set of K3 = size 2.
        #   Or independent set not containing v: alpha(K4)=1.
        #   So alpha = 2.
        
        # Let me try: 6xK4 + two components of 5 vertices each that are C5-free with small alpha
        # 6*4 + 5 + 5 = 34
        
        # What C5-free graphs on 5 vertices have alpha <= 2?
        # K5: has C5. No.
        # K5 - e: has C5? K5 minus edge (u,v). Cycle u-a-b-c-v-a? No, v-a and u-v missing... 
        #   Let me check: vertices 0,1,2,3,4 with edge (0,1) missing.
        #   0-2-3-4-1-0? 1-0 missing. 
        #   0-2-1-3-4-0? 2-1 yes, 1-3 yes, 3-4 yes, 4-0 yes, 0-2 yes. Yes! C5 exists.
        # K5 - 2 edges: depends.
        # Wheel W5: center + C4. C4 is not C5. Center connected to all. 
        #   C5? center-a-b-c-d-center? That's 5 edges, 5 vertices. a-b-c-d is path in C4. 
        #   center-a, a-b, b-c, c-d, d-center. Yes, C5! So W5 has C5.
        # 
        # Let me think differently. C5-free graphs on 5 vertices:
        # - K4 + isolated: C5-free, alpha=2
        # - K4 + vertex connected to k vertices of K4 (k=0,1,2,3,4)
        #   k=0: K4 + isolated, alpha=2
        #   k=1: as above, alpha=2
        #   k=2: v connected to u,w in K4. C5? v-u-x-w-v? Need u-x, x-w. In K4, u-x and x-w exist for x being the other vertex. So v-u-x-w-v is a 4-cycle, not 5. 
        #     For C5: need 5 distinct vertices. v,u,x,w,? Only 4 vertices in K4 plus v = 5. 
        #     v-u-a-b-w-v? Need u-a, a-b, b-w, w-v, v-u. In K4: u-a yes, a-b yes, b-w yes. w-v yes, v-u yes. 
        #     But a,b,w are 3 vertices in K4, and we need them distinct from u. K4 has 4 vertices: u,w,a,b. 
        #     So v-u-a-b-w-v: 5 vertices v,u,a,b,w. Edges: v-u, u-a, a-b, b-w, w-v. All exist! C5!
        #   So k=2 gives C5.
        #   k=3: v connected to 3 vertices of K4. Definitely has C5 (similar argument).
        #   k=4: v connected to all 4. K5, has C5.
        # 
        # So only k=0 or k=1 work for C5-free on 5 vertices with a K4 component.
        # 
        # What about other C5-free graphs on 5 vertices?
        # - C4 + vertex: C4 is 0-1-2-3-0. Add vertex 4.
        #   If 4 isolated: C5-free, alpha=2 (from C4) + 1 = 3? C4 has alpha=2, plus isolated = 3.
        #   If 4 connected to one vertex: alpha could be 3 or less.
        # - P5: path of length 4. C5-free, alpha=3.
        # - Star K1,4: C5-free, alpha=4.
        # 
        # For alpha <= 2 on 5 vertices, C5-free:
        #   K4 + isolated: alpha=2. C5-free.
        #   K4 + vertex connected to 1: alpha=2. C5-free.
        #   Are there others? 
        #   Complement has no independent set of size 3 in original, so complement has no triangle... 
        #   This is getting complex.
        
        # Let me just try the K4+1 approach: 6xK4 + 2x(K4+vertex connected to 1)
        # 6*4 + 2*5 = 34. Alpha = 6*1 + 2*2 = 10. Too high!
        # 
        # I need alpha < 9 total. With 8 components of K4, alpha=8.
        # If I merge two K4s into a 5-vertex C5-free component with alpha=2, I gain 1 vertex but keep alpha the same.
        # So 7xK4 + 1x(K4+1) = 7*4+5 = 33 vertices, alpha = 7*1 + 2 = 9. NOT valid (need < 9).
        # 
        # If I use K4+vertex connected to 1, alpha=2. So 7xK4 + 1x(K4+1) has alpha=9. Fails.
        # 
        # What if I connect the extra vertex to 2 vertices? But that creates C5 as shown.
        # 
        # Alternative: make a 6-vertex C5-free component with alpha=2?
        #   K5 has C5. K5-e has C5. 
        #   K4 + 2 vertices: 
        #     Both connected to same vertex of K4: 
        #       v1,v2 connected to u in K4. 
        #       C5? v1-u-a-b-v2-? v2 only connects to u. So v2 can't be in a cycle with 5 vertices unless...
        #       Actually v1-u-a-b-c-v1? c is the 4th vertex of K4. v1-u, u-a, a-b, b-c, c-v1? c-v1 doesn't exist (v1 only connects to u).
        #       So no C5 involving v1 in a 5-cycle. Similarly for v2.
        #       What about v1-u-v2-?-?-v1? v2 only connects to u, so v1-u-v2 is a path, can't extend.
        #       So C5-free!
        #       Alpha: v1,v2 are not adjacent. They're both non-adjacent to the other 3 in K4 (except u).
        #       {v1, v2, a} where a is not u: v1-a no, v2-a no, a is in K4. So {v1,v2,a} is independent? 
        #       v1 and v2 not adjacent, v1 and a not adjacent, v2 and a not adjacent. Yes! Size 3.
        #       So alpha >= 3. Too high.
        
        # This approach isn't working easily. Let me just do random search.
        
        pass

    return {'n': n_best, 'edges': adj_to_edges(adj_best, n_best)}
