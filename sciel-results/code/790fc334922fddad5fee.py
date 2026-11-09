import time
import itertools
import random

def _has_c5_fast(n, adj_mat):
    """Check if graph has a C5. adj_mat is a list of bitmasks (ints).
    Returns True if C5 exists."""
    # Efficient C5 detection using bitmask operations.
    # For each vertex a, for each pair of neighbors (b, c):
    #   Check if there's a path b-w-x-c of length 3 in G-a with all vertices distinct.
    # This finds cycles a-b-w-x-c-a (5 vertices, 5 edges).
    
    for a in range(n):
        neighbors_a = [v for v in range(n) if (adj_mat[a] >> v) & 1]
        for i in range(len(neighbors_a)):
            b = neighbors_a[i]
            for j in range(i + 1, len(neighbors_a)):
                c = neighbors_a[j]
                # w must be neighbor of b and c, and not a
                candidates_w = (adj_mat[b] & adj_mat[c]) & ~(1 << a)
                while candidates_w:
                    w_bit = candidates_w & -candidates_w
                    w = w_bit.bit_length() - 1
                    candidates_w ^= w_bit
                    # x must be neighbor of w and c, and not a, b, w, c
                    candidates_x = (adj_mat[w] & adj_mat[c]) & ~((1 << a) | (1 << b) | (1 << w) | (1 << c))
                    if candidates_x:
                        return True
    return False

def _has_independent_set_k(n, adj_mat, k=9):
    """Check if graph has independent set of size k. adj_mat is list of bitmasks.
    Returns True if such a set exists."""
    # Use backtracking with bitmask representation and aggressive pruning
    
    found = [False]
    
    # Precompute non-neighbor masks for each vertex (for independent set branching)
    # When we pick v, new candidates = candidates & ~(adj_mat[v] | (1 << v))
    
    # For faster degree counting, precompute popcount for small masks
    # Use int.bit_count() if available (Python 3.10+), else bin().count('1')
    def popcount(x):
        return x.bit_count() if hasattr(int, 'bit_count') else bin(x).count('1')
    
    def search(chosen_count, candidates):
        if found[0]:
            return
        if chosen_count >= k:
            found[0] = True
            return
        # Pruning: if even adding all candidates can't reach k
        if chosen_count + popcount(candidates) < k:
            return
        if candidates == 0:
            return
        
        # Pick a vertex with minimum degree in the induced subgraph
        # This maximizes pruning in the "include" branch
        best_v = -1
        best_deg = float('inf')
        temp = candidates
        while temp:
            v = (temp & (-temp)).bit_length() - 1
            temp &= temp - 1
            deg = popcount(adj_mat[v] & candidates)
            if deg < best_deg:
                best_deg = deg
                best_v = v
        
        v = best_v
        bit = 1 << v
        
        # Branch 1: include v
        new_candidates = candidates & ~(adj_mat[v] | bit)
        search(chosen_count + 1, new_candidates)
        if found[0]:
            return
        
        # Branch 2: exclude v
        search(chosen_count, candidates & ~bit)
    
    search(0, (1 << n) - 1)
    return found[0]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Base: 8 disjoint K4s, n=32
    def make_8k4_edges():
        edges = []
        for b in range(8):
            base = 4 * b
            for i in range(4):
                for j in range(i+1, 4):
                    edges.append([base+i, base+j])
        return edges
    
    # Projective plane of order 5: 31 vertices (16 points + 16 lines - 1 for normalization)
    # Actually: projective plane of order q has q^2+q+1 points and same number of lines
    # For q=5: 31 points, 31 lines, total 62 vertices in incidence graph
    # But we want a graph on 31 vertices. Let's use a different approach.
    # 
    # Alternative: Use the complement of a known C5-free graph with small alpha.
    # Or: construct a bipartite graph with 31 vertices total.
    #
    # Let's use: 16 vertices on one side, 15 on the other (or 15+16).
    # Actually, the inspiration says "incidence graph of projective plane of order 5, 
    # which has 31 vertices". The projective plane of order 5 has 31 points and 31 lines,
    # so the incidence graph has 62 vertices. That doesn't match.
    #
    # Perhaps they mean a graph ON 31 vertices derived from the projective plane structure.
    # Let's try: take 31 vertices, arrange them as points of PG(2,5), and connect two
    # vertices if the corresponding points are collinear with some fixed line? No, that's complex.
    #
    # Simpler interpretation: use a bipartite graph with parts of size ~15 and ~16,
    # where edges represent incidence in a partial projective plane structure.
    # Key property: bipartite => no odd cycles => C5-free.
    # Alpha: max independent set in bipartite graph. For a regular bipartite graph
    # with parts A and B, alpha >= max(|A|, |B|). So if we have parts 15 and 16,
    # alpha >= 16 > 8. That's too large.
    #
    # Wait, the inspiration says alpha = 6. For a bipartite graph, alpha = n - (min vertex cover).
    # By Konig's theorem, min vertex cover = max matching. So alpha = n - max_matching.
    # For alpha = 6 with n = 31: max_matching = 25.
    #
    # Let me just try the 8-K4 approach more aggressively, and also try
    # a structured approach with the projective plane idea adapted.
    #
    # Actually, let me re-read: "incidence graph of the projective plane of order 5,
    # which has 31 vertices". Maybe they're using a quotient or a specific subgraph.
    # Let me just focus on making the 8-K4 approach work for n=33,34,35.
    
    best_n = 32
    best_edges = make_8k4_edges()
    
    def verify(n, edges):
        """Verify C5-free and alpha <= 8. Returns (c5_free, alpha_ok)."""
        adj_mat = [0] * n
        for u, v in edges:
            adj_mat[u] |= (1 << v)
            adj_mat[v] |= (1 << u)
        
        # Check C5-free
        if _has_c5_fast(n, adj_mat):
            return False, True  # Has C5, invalid
        
        # Check alpha <= 8 (i.e., no independent set of size 9)
        if _has_independent_set_k(n, adj_mat, 9):
            return True, False  # C5-free but has alpha >= 9
        
        return True, True  # Valid
    
    # Strategy: 8 K4s + extra vertices
    # Each extra vertex must be connected to enough vertices to prevent C5
    # and to keep alpha <= 8.
    #
    # Key insight: In 8 K4s, the independent set can pick at most 1 from each K4,
    # giving alpha = 8. Adding a vertex v: if v is independent of some set S of 8
    # vertices (one per K4), then we have an independent set of size 9.
    # So v must be adjacent to at least one vertex in every possible choice of
    # one-per-K4. Equivalently, the set of K4s from which v is NOT adjacent to
    # any vertex must be empty... no, that's not right.
    #
    # Actually: an independent set of size 9 = v + 8 others, one from each K4,
    # none adjacent to v. So for each K4_i, the vertices of K4_i NOT adjacent to v
    # must be non-empty for v to be in an independent set with one from each K4.
    # To prevent this: for at least one K4_i, v must be adjacent to ALL vertices of K4_i.
    #
    # For C5-freeness: v connected to all 4 of K4_i creates K5-like structure but
    # K4 is already a clique. The issue is paths of length 4 between neighbors of v.
    # If v is connected to all of K4_0 and one vertex in each other K4, we need to
    # check that no C5 is formed.
    
    # For n=33: vertex 32 connected to all 4 in one K4 and one in each other K4
    def try_n33_structured():
        edges = make_8k4_edges()
        # Vertex 32: connect to all 4 in K4_0
        for v in range(4):
            edges.append([32, v])
        # Connect to one in each K4_1..K4_7
        for b in range(1, 8):
            edges.append([32, 4*b])
        return edges
    
    # For n=34: two extra vertices
    def try_n34_structured():
        edges = make_8k4_edges()
        # Vertex 32: all of K4_0, one in each other
        for v in range(4):
            edges.append([32, v])
        for b in range(1, 8):
            edges.append([32, 4*b])
        # Vertex 33: all of K4_1, one in each other
        for v in range(4, 8):
            edges.append([33, v])
        edges.append([33, 0])  # one from K4_0
        for b in range(2, 8):
            edges.append([33, 4*b])
        # Connect 32 and 33? If both are in K4_0 fully, they share neighbors.
        # Don't connect them to avoid potential C5 issues, or do connect them.
        # Let's not connect them first.
        return edges
    
    # For n=35: three extra vertices
    def try_n35_structured():
        edges = make_8k4_edges()
        # Vertex 32: all of K4_0, one in each other
        for v in range(4):
            edges.append([32, v])
        for b in range(1, 8):
            edges.append([32, 4*b])
        # Vertex 33: all of K4_1, one in each other
        for v in range(4, 8):
            edges.append([33, v])
        edges.append([33, 0])
        for b in range(2, 8):
            edges.append([33, 4*b])
        # Vertex 34: all of K4_2, one in each other
        for v in range(8, 12):
            edges.append([34, v])
        edges.append([34, 0])
        edges.append([34, 4])
        for b in range(3, 8):
            edges.append([34, 4*b])
        return edges
    
    # Try structured constructions from highest n down
    for target_n, builder in [(35, try_n35_structured), (34, try_n34_structured), (33, try_n33_structured)]:
        if time.time() >= deadline - 5:
            break
        if target_n <= best_n:
            continue
        
        edges = builder()
        c5_free, alpha_ok = verify(target_n, edges)
        if c5_free and alpha_ok:
            best_n = target_n
            best_edges = edges
            break
    
    # If structured didn't work, try randomized search with more trials
    if best_n < 35 and time.time() < deadline - 10:
        for target_n in [35, 34, 33]:
            if target_n <= best_n:
                continue
            if time.time() >= deadline - 5:
                break
            
            n_extra = target_n - 32
            # More trials with improved heuristics
            for trial in range(5000):
                if time.time() >= deadline - 5:
                    break
                
                random.seed(trial * 7919 + target_n * 104729)
                edges = make_8k4_edges()
                
                for k in range(n_extra):
                    v = 32 + k
                    # Connect to all 4 in one random K4 (to ensure alpha constraint)
                    chosen_k4 = random.randint(0, 7)
                    base = 4 * chosen_k4
                    for i in range(4):
                        edges.append([v, base + i])
                    # Connect to one random vertex in each other K4
                    for b in range(8):
                        if b != chosen_k4:
                            edges.append([v, 4*b + random.randint(0, 3)])
                    # Connect to other extra vertices with some probability
                    for w in range(32, 32+k):
                        if random.random() < 0.4:
                            edges.append([v, w])
                
                c5_free, alpha_ok = verify(target_n, edges)
                if c5_free and alpha_ok:
                    best_n = target_n
                    best_edges = edges
                    break
            if best_n >= target_n:
                break
    
    return {'n': best_n, 'edges': best_edges}
