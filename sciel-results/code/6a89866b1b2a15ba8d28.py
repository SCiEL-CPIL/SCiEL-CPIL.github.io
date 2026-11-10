"""Dense Q3-free graph on 16 vertices via Maximum Weight Independent Set on Q3-completion graph."""
import time
import random

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    n = 16
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    m = len(all_edges)  # 120 edges
    
    # Edge index mapping
    edge_idx = {}
    for i, (u, v) in enumerate(all_edges):
        edge_idx[(u, v)] = i
        edge_idx[(v, u)] = i
    
    # Generate all Q3 embeddings in K16
    # A Q3 has 8 vertices and 12 edges
    # We need to find all sets of 12 edges that form a Q3
    # Strategy: enumerate all 8-subsets of vertices, then all Q3 structures on them
    
    # First, generate all distinct Q3 edge patterns on 8 labeled vertices
    # Q3 vertices are {0,1}^3, edges connect vertices differing in 1 bit
    q3_vertices = [(a, b, c) for a in range(2) for b in range(2) for c in range(2)]
    q3_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            vi, vj = q3_vertices[i], q3_vertices[j]
            diff = sum(1 for x, y in zip(vi, vj) if x != y)
            if diff == 1:
                q3_edges.append((i, j))
    
    # q3_edges has 12 edges on vertices 0..7
    # For each permutation of 8 vertices from 0..15, map the Q3 structure
    # But we need to avoid duplicates due to automorphisms
    
    # Better approach: for each 8-subset of vertices, find all Q3 subgraphs
    # Use a canonical form to avoid duplicates
    
    def get_q3_edge_sets(vertices):
        """Given 8 vertices, find all Q3 edge sets (as frozensets of edge indices)."""
        # Map vertices to 0..7
        vert_map = {v: i for i, v in enumerate(vertices)}
        
        # Try all bijections from Q3 vertices to these 8 vertices
        # But use symmetry: fix vertex 0 of Q3 to the first vertex
        # Then try permutations of remaining 7 vertices (7! = 5040)
        
        # Actually, let's just try all 8! = 40320 permutations
        # and use a set to deduplicate
        
        edge_sets = set()
        
        # For efficiency, fix the first Q3 vertex to the first graph vertex
        # Then permute the remaining 7
        import itertools
        
        base_vertex = vertices[0]
        remaining_vertices = vertices[1:]
        
        # The Q3 vertex 0 maps to base_vertex
        # Q3 vertices 1..7 map to some permutation of remaining_vertices
        
        # But we also need to consider that Q3 vertices other than 0 could map to base_vertex
        # due to automorphisms. To be safe, let's try all 8! but with early dedup.
        
        # Alternative: generate all 8! permutations but use a canonical representation
        
        # For now, let's be simpler: try all 8! permutations
        # 8! = 40320, times C(16,8) = 12870 is too much (518M)
        
        # Better: for each 8-subset, there are at most 8! / 48 = 840 distinct Q3s
        # But generating them is still expensive
        
        # Let's use a different approach: generate all Q3s in K16 directly
        # by choosing 8 vertices and checking all possible Q3 structures
        
        # Actually, the most efficient approach for n=16:
        # Precompute all Q3 edge sets once at module level or use a smarter enumeration
        
        return None
    
    # Let's use a completely different approach:
    # Generate all Q3s by choosing a "root" vertex and building up
    
    # Actually, for the MWIS formulation, we need the conflict graph:
    # Two edges e1, e2 conflict if they are part of a common Q3
    # We can compute this by: for each edge, find all Q3s containing it
    
    # Let's generate all Q3s first using a more efficient method
    
    def generate_all_q3s():
        """Generate all Q3 subgraphs in K16 as sets of edge indices."""
        all_q3s = []
        seen = set()
        
        # Q3 structure: 8 vertices, 12 edges
        # Vertices of Q3: {0,1}^3
        # Edges: pairs differing in exactly 1 coordinate
        
        # For each 8-subset of {0..15}, check all possible Q3 embeddings
        # To make this efficient, we use the fact that in Q3, 
        # the vertices can be partitioned into two sets of 4 (by parity)
        # and edges only go between the two sets
        
        # Actually, let's just enumerate all 8-subsets and for each,
        # try to find Q3 structures using backtracking
        
        # For n=16, C(16,8) = 12870, and for each we need to check
        # if it contains a Q3. This is feasible if we're smart.
        
        # A Q3 on 8 vertices has the property that:
        # - It's 3-regular
        # - It has 8 vertices, 12 edges
        # - It's bipartite with parts of size 4
        # - Each vertex has degree 3
        
        # For a given 8-vertex set, a Q3 subgraph is a spanning 3-regular subgraph
        # that is isomorphic to Q3.
        
        # Let's use a different strategy: directly enumerate Q3s by
        # choosing 8 vertices and all ways to assign them to Q3 positions
        
        # Since we're in K16, any 8 vertices can host a Q3 if we can find
        # a valid labeling.
        
        # For each 8-subset S of {0..15}:
        #   For each bijection f: {0..7} -> S:
        #     Check if the 12 edges of Q3 (under f) are all present in K16
        #     (they always are, since K16 is complete)
        #     So every 8-subset hosts 8!/48 = 840 distinct Q3s
        
        # Total Q3s: 12870 * 840 = 10,810,800
        # This is too many to store explicitly, but we can work with them implicitly
        
        # For the conflict graph, we need: for each pair of edges, do they
        # belong to a common Q3?
        
        # Two edges e1=(u1,v1) and e2=(u2,v2) belong to a common Q3 iff
        # there exists a Q3 containing both edges.
        
        # This is equivalent to: can we complete {e1, e2} to a Q3?
        
        # For efficiency, let's precompute for each edge which other edges
        # it can be paired with in a Q3.
        
        # An edge (u,v) in Q3: u and v differ in 1 bit.
        # The other 10 edges of the Q3 are determined by the choice of
        # the remaining 6 vertices and how they connect.
        
        # Given edge (u,v), in Q3, u and v are adjacent. 
        # u has 2 other neighbors, v has 2 other neighbors.
        # The 4 neighbors (2 of u, 2 of v) form two disjoint pairs.
        # Then 2 more vertices at distance 2, and 1 at distance 3.
        
        # This is getting complex. Let's use a brute-force but bounded approach:
        # For each pair of edges, check if they can be in a common Q3
        # by trying to extend them.
        
        # Actually, for the MWIS, we can use a different characterization:
        # Two edges conflict if removing both doesn't prevent all Q3s,
        # i.e., they are in some common Q3.
        
        # Let's precompute the conflict relation using the following:
        # For each edge e, find all Q3s containing e, and mark all other
        # edges in those Q3s as conflicting with e.
        
        # To find all Q3s containing a given edge (u,v):
        # In Q3, (u,v) is an edge. Let's say u=000, v=001 (wlog).
        # Then u's other neighbors are 010, 100.
        # v's other neighbors are 011, 101.
        # The vertex at distance 2 from u: 110 (adjacent to 010 and 100)
        # The vertex at distance 2 from v: 111 (adjacent to 011 and 101)
        # The vertex at distance 3: 110? No, 111 is at distance 2 from u.
        # Let me redo this.
        
        # Q3 vertices: 000, 001, 010, 011, 100, 101, 110, 111
        # Edges: (000,001), (000,010), (000,100),
        #         (001,011), (001,101),
        #         (010,011), (010,110),
        #         (100,101), (100,110),
        #         (011,111), (101,111), (110,111)
        
        # If edge is (000, 001), then:
        # 000's other neighbors: 010, 100
        # 001's other neighbors: 011, 101
        # 010's neighbors: 000, 011, 110
        # 100's neighbors: 000, 101, 110
        # 011's neighbors: 001, 010, 111
        # 101's neighbors: 001, 100, 111
        # 110's neighbors: 010, 100, 111
        # 111's neighbors: 011, 101, 110
        
        # So the 12 edges are determined. The 6 other vertices are:
        # 010, 100, 011, 101, 110, 111
        
        # Given an edge (u,v) in our graph, to find all Q3s containing it:
        # We need to choose 6 other vertices from the remaining 14, and
        # check if they can be arranged to form a Q3 with u and v.
        
        # This is still complex. Let's use a simpler approach for the conflict graph:
        
        # Two edges e1=(a,b) and e2=(c,d) conflict if there exists a Q3
        # containing both. We can check this by:
        # 1. If they share a vertex, say a=c, then b and d must be at
        #    distance 2 in Q3 (i.e., there's a common neighbor in Q3).
        # 2. If they're disjoint, we need to check if they can be part of a Q3.
        
        # For the MWIS approach, let's use a different formulation:
        # Instead of finding the maximum set of edges with no Q3,
        # let's directly search for dense Q3-free graphs using
        # a more effective local search.
        
        # Given the complexity, let's abandon the MWIS formulation and use
        # a very effective local search with multiple restarts.
        
        return None
    
    # Let's use a highly optimized local search approach instead
    # The key insight: we need a fast Q3 detection and a good search strategy
    
    def creates_q3_fast(u, v, adj):
        """
        Check if adding edge (u,v) creates a Q3.
        adj[i] is a bitmask of neighbors of i.
        """
        neigh_u = adj[u] & ~(1 << v)
        neigh_v = adj[v] & ~(1 << u)
        
        # In Q3, u and v are adjacent. 
        # u has 2 other neighbors in the Q3, v has 2 other neighbors.
        # Let's denote u's other Q3-neighbors as a, b
        # and v's other Q3-neighbors as c, d.
        # Then a, b are not adjacent to each other (in Q3),
        # c, d are not adjacent to each other (in Q3),
        # a is adjacent to c or d (exactly one),
        # b is adjacent to the other of c, d,
        # and there are 2 more vertices e, f at distance 2,
        # and 1 vertex g at distance 3.
        
        # Actually, let's use the structure more directly:
        # Given u and v adjacent in Q3:
        # - u has 3 neighbors in Q3: v, a, b
        # - v has 3 neighbors in Q3: u, c, d
        # - a and b are at distance 2 from each other (via u)
        # - c and d are at distance 2 from each other (via v)
        # - a is adjacent to exactly one of {c, d} (say c)
        # - b is adjacent to exactly one of {c, d} (say d)
        # - There's a vertex e adjacent to a, b, and one of {c, d}
        #   Actually, e is at distance 2 from u and v.
        #   e is adjacent to a, b, and... let me think.
        
        # Let me use the explicit Q3 structure:
        # u=000, v=001
        # a=010, b=100 (u's other neighbors)
        # c=011, d=101 (v's other neighbors)
        # e=110 (adjacent to a=010 and b=100)
        # f=111 (adjacent to c=011 and d=101)
        # g=110? No, 110 is e. 
        # Wait, I have 8 vertices: 000,001,010,011,100,101,110,111
        # u=000, v=001, a=010, c=011, b=100, d=101, e=110, f=111
        
        # Edges:
        # u-v: 000-001
        # u-a: 000-010, u-b: 000-100
        # v-c: 001-011, v-d: 001-101
        # a-c: 010-011, a-e: 010-110
        # b-d: 100-101, b-e: 100-110
        # c-f: 011-111, d-f: 101-111
        # e-f: 110-111
        
        # So the structure is:
        # u is adjacent to v, a, b
        # v is adjacent to u, c, d
        # a is adjacent to u, c, e
        # b is adjacent to u, d, e
        # c is adjacent to v, a, f
        # d is adjacent to v, b, f
        # e is adjacent to a, b, f
        # f is adjacent to c, d, e
        
        # Key properties for checking:
        # Given u, v (the edge we're adding):
        # 1. Choose a, b from N(u) \ {v} (u's other neighbors in Q3)
        # 2. Choose c, d from N(v) \ {u} (v's other neighbors in Q3)
        # 3. a must be adjacent to exactly one of {c, d} (say c)
        # 4. b must be adjacent to exactly one of {c, d} (say d)
        # 5. There must exist e adjacent to a, b, and f
        # 6. There must exist f adjacent to c, d, and e
        # 7. e and f are distinct and not in {u,v,a,b,c,d}
        
        # Simplified check:
        # For each pair (a,b) from N(u)\{v}:
        #   For each pair (c,d) from N(v)\{u}:
        #     Check if a is adjacent to exactly one of c,d
        #     Check if b is adjacent to exactly one of c,d
        #     Then find e = common neighbor of a and b (excluding u)
        #     Find f = common neighbor of c and d (excluding v)
        #     Check if e and f are adjacent and distinct from all others
        
        # This is O(deg(u)^2 * deg(v)^2) which for dense graphs is O(n^4) = 65536
        # Too slow for many calls.
        
        # Let's optimize:
        # Precompute for each vertex, the list of neighbors
        # Use bitmasks for efficiency
        
        # For each a in N(u)\{v}:
        #   For each b in N(u)\{v}\{a}:
        #     common_ab = N(a) & N(b) & ~(1<<u)  # candidates for e
        #     For each c in N(v)\{u}:
        #       For each d in N(v)\{u}\{c}:
        #         # Check a-c and b-d or a-d and b-c
        #         ac = (adj[a] >> c) & 1
        #         ad = (adj[a] >> d) & 1
        #         bc = (adj[b] >> c) & 1
        #         bd = (adj[b] >> d) & 1
        #         
        #         # Case 1: a-c and b-d
        #         if ac and bd and not ad and not bc:
        #           # Find f = common neighbor of c and d excluding v
        #           common_cd = (adj[c] & adj[d]) & ~(1 << v)
        #           # f must be in common_cd and adjacent to e
        #           # e is in common_ab
        #           # Check if there exist e in common_ab, f in common_cd
        #           # such that e-f is an edge and e != f
        #           for e_bit in common_ab:
        #             e_idx = e_bit.bit_length() - 1
        #             f_cand = common_cd & ~(1 << e_idx)
        #             if f_cand & adj[e_idx]:
        #               return True
        #         
        #         # Case 2: a-d and b-c
        #         if ad and bc and not ac and not bd:
        #           common_cd = (adj[c] & adj[d]) & ~(1 << v)
        #           for e_bit in common_ab:
        #             e_idx = e_bit.bit_length() - 1
        #             f_cand = common_cd & ~(1 << e_idx)
        #             if f_cand & adj[e_idx]:
        #               return True
        
        # This is still complex. Let's use the original approach but optimized.
        
        # Original approach from the current program:
        a2_bits = neigh_u
        while a2_bits:
            a2 = a2_bits & (-a2_bits)
            a2_bits -= a2
            a2_idx = a2.bit_length() - 1
            
            a4_bits = neigh_u & ~(1 << a2_idx)
            while a4_bits:
                a4 = a4_bits & (-a4_bits)
                a4_bits -= a4
                a4_idx = a4.bit_length() - 1
                
                # a2 and a4 are u's other two Q3-neighbors
                # a3 must be a neighbor of a2 (and v)
                # a5 must be a neighbor of a4 (and v)
                a3_candidates = neigh_v & adj[a2_idx]
                a5_candidates = neigh_v & adj[a4_idx]
                
                a3_candidates &= ~(1 << a2_idx) & ~(1 << a4_idx)
                
                if not a3_candidates or not a5_candidates:
                    continue
                
                a3_bits = a3_candidates
                while a3_bits:
                    a3 = a3_bits & (-a3_bits)
                    a3_bits -= a3
                    a3_idx = a3.bit_length() - 1
                    
                    a5_cand = a5_candidates & ~(1 << a2_idx) & ~(1 << a4_idx) & ~(1 << a3_idx)
                    if not a5_cand:
                        continue
                    
                    a5_bits = a5_cand
                    while a5_bits:
                        a5 = a5_bits & (-a5_bits)
                        a5_bits -= a5
                        a5_idx = a5.bit_length() - 1
                        
                        used_mask = (1 << u) | (1 << v) | (1 << a2_idx) | (1 << a3_idx) | (1 << a4_idx) | (1 << a5_idx)
                        a6_candidates = (adj[a2_idx] & adj[a4_idx]) & ~used_mask
                        if not a6_candidates:
                            continue
                        
                        a7_base = (adj[a3_idx] & adj[a5_idx]) & ~used_mask
                        
                        a6_bits = a6_candidates
                        while a6_bits:
                            a6 = a6_bits & (-a6_bits)
                            a6_bits -= a6
                            a6_idx = a6.bit_length() - 1
                            
                            a7_cand = a7_base & ~(1 << a6_idx)
                            if not a7_cand:
                                continue
                            
                            if adj[a6_idx] & a7_cand:
                                return True
        return False

    def count_edges(adj):
        count = 0
        for u in range(n):
            count += bin(adj[u]).count('1')
        return count // 2

    def edges_from_adj(adj):
        result = []
        for u in range(n):
            mask = adj[u]
            while mask:
                bit = mask & (-mask)
                v = bit.bit_length() - 1
                if v > u:
                    result.append([u, v])
                mask -= bit
        return result

    def greedy_add(adj, edge_order):
        added = 0
        for u, v in edge_order:
            if time.time() > deadline:
                break
            if adj[u] & (1 << v):
                continue
            if not creates_q3_fast(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                added += 1
        return added

    def local_search(adj, max_time=10):
        """
        Local search with edge swaps and perturbations.
        """
        current_count = count_edges(adj)
        best_adj = adj[:]
        best_count = current_count
        start_time = time.time()
        
        # Get missing edges
        missing = [(u, v) for u in range(n) for v in range(u+1, n) if not (adj[u] & (1 << v))]
        
        # Phase 1: Multiple greedy passes
        for pass_i in range(5):
            if time.time() - start_time > max_time or time.time() > deadline - 5:
                break
            if not missing:
                break
            shuffled = missing[:]
            random.shuffle(shuffled)
            greedy_add(adj, shuffled)
            current_count = count_edges(adj)
            if current_count > best_count:
                best_count = current_count
                best_adj = adj[:]
            missing = [(u, v) for u in range(n) for v in range(u+1, n) if not (adj[u] & (1 << v))]
        
        # Phase 2: Edge swaps
        for _ in range(50):
            if time.time() - start_time > max_time or time.time() > deadline - 5:
                break
            
            missing_now = [(u, v) for u in range(n) for v in range(u+1, n) if not (adj[u] & (1 << v))]
            if not missing_now:
                break
            
            found_improvement = False
            for _attempt in range(20):
                if time.time() - start_time > max_time or time.time() > deadline - 5:
                    break
                if not missing_now:
                    break
                
                u_miss, v_miss = random.choice(missing_now)
                
                # Find existing edges to potentially remove
                existing_edges = []
                for neighbor in range(n):
                    if neighbor != u_miss and neighbor != v_miss:
                        if adj[u_miss] & (1 << neighbor):
                            existing_edges.append((u_miss, neighbor))
                        if adj[v_miss] & (1 << neighbor):
                            existing_edges.append((v_miss, neighbor))
                
                if not existing_edges:
                    continue
                
                u_rem, v_rem = random.choice(existing_edges)
                
                trial = best_adj[:]
                trial[u_rem] &= ~(1 << v_rem)
                trial[v_rem] &= ~(1 << u_rem)
                
                if not creates_q3_fast(u_miss, v_miss, trial):
                    trial[u_miss] |= (1 << v_miss)
                    trial[v_miss] |= (1 << u_miss)
                
                # Try to add more edges
                missing_trial = [(u, v) for u in range(n) for v in range(u+1, n) if not (trial[u] & (1 << v))]
                random.shuffle(missing_trial)
                greedy_add(trial, missing_trial)
                
                new_count = count_edges(trial)
                if new_count > best_count:
                    best_count = new_count
                    best_adj = trial[:]
                    adj = trial[:]
                    current_count = new_count
                    found_improvement = True
                    missing_now = [(u, v) for u in range(n) for v in range(u+1, n) if not (adj[u] & (1 << v))]
                    break
            
            if not found_improvement:
                # Random perturbation
                existing = [(u, v) for u in range(n) for v in range(u+1, n) if (best_adj[u] & (1 << v))]
                if not existing:
                    break
                
                # Remove 1-2 random edges
                for _ in range(random.randint(1, 2)):
                    u_rem, v_rem = random.choice(existing)
                    trial = best_adj[:]
                    trial[u_rem] &= ~(1 << v_rem)
                    trial[v_rem] &= ~(1 << u_rem)
                    best_adj = trial[:]
                
                missing_trial = [(u, v) for u in range(n) for v in range(u+1, n) if not (best_adj[u] & (1 << v))]
                random.shuffle(missing_trial)
                greedy_add(best_adj, missing_trial)
                
                new_count = count_edges(best_adj)
                if new_count > best_count:
                    best_count = new_count
                    adj = best_adj[:]
                    current_count = new_count
        
        return best_adj, best_count

    random.seed(42)
    best_adj = [0] * n
    best_count = -1
    
    # Run multiple restarts
    num_restarts = 100
    for i in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        adj = [0] * n
        order = all_edges[:]
        random.shuffle(order)
        greedy_add(adj, order)
        
        adj, cnt = local_search(adj, max_time=5)
        
        if cnt > best_count:
            best_count = cnt
            best_adj = adj
    
    # Final polish
    if time.time() < deadline - 2:
        remaining = [(u, v) for u in range(n) for v in range(u+1, n) 
                     if not (best_adj[u] & (1 << v))]
        
        for _ in range(5):
            if time.time() > deadline - 1:
                break
            random.shuffle(remaining)
            greedy_add(best_adj, remaining)
            remaining = [(u, v) for u in range(n) for v in range(u+1, n) 
                         if not (best_adj[u] & (1 << v))]
            if not remaining:
                break
        
        best_count = count_edges(best_adj)

    return edges_from_adj(best_adj)
