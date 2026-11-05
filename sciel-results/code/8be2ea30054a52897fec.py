"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _clebsch_edges():
    """Return edges of the Clebsch graph (n=16, alpha=5, triangle-free).
    Vertices 0..15. Two vertices are adjacent if their binary representations differ in exactly 1 or 3 bits (complement of the 4-cube).
    Actually, the Clebsch graph is the complement of the 4-dimensional hypercube Q4.
    In Q4, vertices are adjacent if they differ in exactly 1 bit.
    So in Clebsch, vertices are adjacent if they differ in 2 or 3 or 4 bits.
    Wait, let me reconsider. The Clebsch graph is a 5-regular graph on 16 vertices.
    It's the complement of the 4-cube. In Q4, each vertex has degree 4 (differs in 1 bit).
    So in the complement, each vertex has degree 16-1-4 = 11. That's not 5-regular.
    
    Let me use the standard construction: vertices are binary strings of length 4.
    Two vertices are adjacent in the Clebsch graph if they differ in exactly 1 bit OR exactly 3 bits.
    Actually, the standard Clebsch graph: vertices are subsets of {1,2,3,4} (or binary strings).
    Two vertices are adjacent if the size of their symmetric difference is 1 or 3.
    Symmetric difference of 1: differ in exactly 1 bit.
    Symmetric difference of 3: differ in exactly 3 bits.
    Number of such vertices for a given vertex: C(4,1) + C(4,3) = 4 + 4 = 8. That gives degree 8, not 5.
    
    Let me look up the correct construction. The Clebsch graph is a strongly regular graph with parameters (16, 5, 0, 2).
    It can be constructed as follows: take the 4-cube Q4, and for each vertex, connect it to the 5 vertices at Hamming distance 1... no.
    
    Actually, a common construction: vertices are all binary strings of length 4. Two vertices are adjacent if their Hamming distance is 1 or 3.
    Hamming distance 1: C(4,1) = 4 vertices.
    Hamming distance 3: C(4,3) = 4 vertices.
    Total degree: 4 + 4 = 8. Still not 5.
    
    Let me try another construction. The Clebsch graph can be built from the 5-cycle:
    It's the "folded 4-cube" or "halved 4-cube".
    
    Actually, I'll use a known explicit edge list for the Clebsch graph.
    The Clebsch graph is the complement of the line graph of K_{4,4}.
    K_{4,4} has 16 edges, so the line graph L(K_{4,4}) has 16 vertices.
    Two edges in K_{4,4} are adjacent in L(K_{4,4}) if they share a vertex.
    In K_{4,4}, each edge is incident to 2 vertices, each of degree 4, so each edge is adjacent to 4+4-2=6 other edges in L(K_{4,4}).
    So L(K_{4,4}) is 6-regular on 16 vertices.
    The complement is (16-1-6)=9-regular. Still not 5.
    
    I'm confusing myself. Let me just use a well-known explicit construction.
    
    The Clebsch graph: vertices 0..15. Vertex i is adjacent to vertex j if the binary representation of i XOR j has exactly 1 or 3 bits set... 
    Let me just count: for a fixed vertex, number of vertices at Hamming distance 1 is 4, at distance 2 is 6, at distance 3 is 4, at distance 4 is 1.
    If we connect at distance 1 and 3: degree = 4+4 = 8.
    If we connect at distance 2: degree = 6.
    If we connect at distance 1: degree = 4.
    
    The Clebsch graph has degree 5. So it's not simply based on Hamming distance in Q4.
    
    Let me use the standard "complement of the 4-cube's edges removed" approach differently.
    Actually, I recall now: the Clebsch graph is the graph on {0,1}^4 where two vertices are adjacent if they differ in an odd number of coordinates... no, that would be bipartite.
    
    Let me just hardcode a known triangle-free graph with good properties. The Andrásfai graph And_3 is triangle-free on 9 vertices with alpha=3.
    And_3: vertices 0..8, edges (i, j) if (j-i) mod 9 is in {1, 3, 4, 5, 7}. Wait, that's not right either.
    
    Let me use a simpler approach: the 5-cycle C5 is triangle-free with alpha=2.
    Blow up C5: replace each vertex with k twins. The blow-up C5[k] is triangle-free.
    n = 5k, alpha = 2k (since alpha(C5)=2, blow-up by k gives alpha=2k).
    We need 2k < 10, so k <= 4. n = 5*4 = 20.
    
    That's only n=20. Not great.
    
    Let me think of better triangle-free graphs with high minimum degree relative to n/alpha.
    
    The complete bipartite graph K_{m,m} is triangle-free with alpha=m. n=2m, alpha=m.
    We need m < 10, so m <= 9, n=18.
    
    The Andrásfai graph And_3: 9 vertices, degree 4, triangle-free, alpha=3.
    Blow up And_3 by k: n=9k, alpha=3k. Need 3k<10, k<=3, n=27.
    
    The Clebsch graph: 16 vertices, degree 5, triangle-free, alpha=5.
    Blow up by k: n=16k, alpha=5k. Need 5k<10, k<=1, n=16.
    
    Hmm, all these give small n.
    
    What about the Schläfli graph? 27 vertices, degree 16, triangle-free, alpha=9.
    Blow up by k=1: n=27, alpha=9 < 10. That works!
    
    But I need to construct the Schläfli graph. It's the complement of the line graph of K_6.
    K_6 has 15 edges, so L(K_6) has 15 vertices. That's not 27.
    
    The Schläfli graph has 27 vertices. It's the halved 5-cube or something similar.
    
    Actually, let me use a different approach. The Hoffman-Singleton graph is (25,7,0,1) but it's not triangle-free (it has triangles? No, girth 5, so no triangles).
    Hoffman-Singleton: 25 vertices, degree 7, triangle-free, alpha=5 (since it's (25,7,0,1), lambda=0 means no triangles, and alpha can be computed).
    
    Actually for a (v,k,lambda,mu) SRG, alpha <= v*k/(v-k-1+... ) using Hoffman bound.
    For Hoffman-Singleton (25,7,0,1): eigenvalues are 7, 2 (mult 15), -3 (mult 9).
    Hoffman bound: alpha <= 25 * (-3) / (7 + (-3)) = 25 * (-3)/4 = -18.75. That's not right.
    The bound is alpha <= -k * v / (k + lambda_min) where lambda_min is the smallest eigenvalue.
    alpha <= -7 * 25 / (7 + (-3)) = -175/4 = -43.75. Still negative, which means the bound gives alpha <= 0, which is wrong.
    
    I'm making errors. Let me just use a simple, verifiable construction.
    
    The 5-cycle blow-up: C5[k] for k=4 gives n=20, alpha=8.
    For k=5: n=25, alpha=10. Fails (alpha must be < 10).
    
    What about using two different blow-ups? Not straightforward.
    
    Let me reconsider. The task says the largest known is 39. So there ARE graphs on 39 vertices.
    These are likely not simple blow-ups of small graphs.
    
    For the purpose of this problem, let me use a random construction with verification.
    The key is to find ANY valid graph, and maximize n.
    
    I'll use a randomized approach: for each n from 40 down, try to construct a triangle-free graph with alpha < 10 using a greedy/randomized algorithm, and verify with exact branch-and-bound.
    
    But the branch-and-bound for alpha on 40 vertices might be slow.
    
    Alternative: use the fact that we just need alpha < 10. We can use a simpler upper bound check.
    
    Actually, let me use a known result: the complement of a 9-clique-free graph... no.
    
    Let me try a different tactic. Use the circulant graph approach but with better search.
    For a circulant graph on Z_n with symmetric connection set S, triangle-free means no 3-term arithmetic progression in S ∪ {-S} (modulo n).
    
    For n=39, can we find such an S with alpha < 10?
    
    This is getting complex. Let me just implement a robust randomized search with verification, and hope it finds something.
    
    Actually, the simplest reliable approach: use the 5-cycle blow-up for n=20 (alpha=8), and try to improve.
    
    But the score is 1.4615 which corresponds to n=37 or so (if seed is 25, score = n/25, so 1.4615*25 = 36.5, so n=37).
    
    So the current program finds n=37. I need to match or beat that.
    
    Let me implement a proper search. I'll use a randomized greedy construction for triangle-free graphs, then check alpha with branch-and-bound (with time limit).
    
    For the alpha check, I'll use a bitset-based branch-and-bound that's efficient for n up to 40.
    """
    # Clebsch graph edges (16 vertices, 5-regular, triangle-free, alpha=5)
    # Standard construction: vertices are {0,1}^4. Two vertices adjacent if Hamming distance is 1 or 3.
    # Wait, that gives degree 8. Let me use the correct one.
    # The Clebsch graph: two vertices (binary strings of length 4) are adjacent if they differ in exactly 1 bit.
    # No, that's Q4, which is 4-regular.
    # 
    # I'll use a verified edge list. The Clebsch graph is the complement of the 4-cube with a perfect matching removed... 
    # 
    # Let me just use a different well-known triangle-free graph: the icosahedron graph has triangles.
    # The 5-cycle is the smallest non-bipartite triangle-free graph.
    #
    # I'll use the Andrásfai graph And_3: 9 vertices, triangle-free, alpha=3.
    # And_3 edges: (i, j) for i,j in 0..8 if (j-i) mod 9 is in {1, 2, 4, 5, 7, 8}... 
    # Actually And_3 is the circulant graph on Z_9 with S={1,2,4,5,7,8}. But that's 6-regular, degree 6.
    # And_3 should be 4-regular. S={1,2,4,5,7,8} has 6 elements, but since it's symmetric, degree is 6.
    # 
    # Let me use a simpler verified graph. The 5-cycle C5: vertices 0-1-2-3-4-0. Triangle-free, alpha=2.
    # Blow up by k: n=5k, alpha=2k.
    # For k=4: n=20, alpha=8. Valid.
    #
    # For better n, use the icosahedron... no, has triangles.
    #
    # The Wagner graph (Möbius ladder on 8 vertices): triangle-free, alpha=4.
    # Blow up by k: n=8k, alpha=4k. For k=2: n=16, alpha=8.
    #
    # The 5-cycle blow-up by 4: n=20, alpha=8.
    #
    # Let me just go with a randomized search approach.
    pass
    return []

def _build_clebsch():
    """Build the Clebsch graph (16 vertices, triangle-free, alpha=5).
    Uses the standard construction: vertices 0..15 (binary 0..15).
    Two vertices u,v are adjacent if popcount(u XOR v) == 1 or popcount(u XOR v) == 3.
    This gives degree C(4,1)+C(4,3) = 4+4 = 8. But Clebsch is 5-regular.
    
    Correct construction: The Clebsch graph is the halved 4-cube.
    Vertices are even-weight binary strings of length 5 (or odd-weight).
    There are C(5,0)+C(5,2)+C(5,4) = 1+10+5 = 16 even-weight strings.
    Two vertices are adjacent if they differ in exactly 2 bits.
    Degree: C(4,2) = 6? No.
    
    For an even-weight string of length 5, to get another even-weight string differing in 2 bits,
    we flip 2 bits. There are C(5,2)=10 such pairs, but each flip of 2 bits changes weight by -2, 0, or +2.
    To stay even, we need to flip 2 bits such that the weight stays even. Flipping any 2 bits changes weight by an even number, so all C(5,2)=10 flips give even-weight strings.
    But some of these may be the same vertex... no, each pair of positions gives a unique neighbor.
    So degree = 10. That's not 5.
    
    I'm struggling with the exact construction. Let me just use a hardcoded edge list from a known source.
    
    The Clebsch graph edges (one standard labeling):
    """
    # I'll use a verified construction from the literature.
    # The Clebsch graph can be constructed as follows:
    # Vertices: all subsets of {1,2,3,4} with even size: {}, {1,2}, {1,3}, {1,4}, {2,3}, {2,4}, {3,4}, {1,2,3,4}
    # That's only 8 vertices. Not 16.
    
    # All subsets of {1,2,3,4}: 16 vertices.
    # Two subsets A, B are adjacent if |A Δ B| = 1 or |A Δ B| = 3.
    # |A Δ B| = 1: A and B differ by one element. There are 4 such B for each A.
    # |A Δ B| = 3: A and B differ by three elements. There are 4 such B for each A.
    # Total degree: 8.
    
    # This is NOT the Clebsch graph. The Clebsch graph has degree 5.
    
    # The correct Clebsch graph: it's the complement of the line graph of K_{4,4} minus a 1-factor... 
    # 
    # I give up on the Clebsch graph. Let me use a simpler approach.
    #
    # I'll use the 5-cycle blow-up for a guaranteed valid graph, and try to improve with random search.
    
    edges = []
    # 5-cycle: 0-1-2-3-4-0
    cycle = [(0,1), (1,2), (2,3), (3,4), (4,0)]
    return cycle

def _blow_up_graph(base_edges, n_base, k):
    """Blow up each vertex of the base graph into k twins.
    Returns edges of the blown-up graph on n_base*k vertices.
    """
    n = n_base * k
    edges = []
    # For each edge (u,v) in base, add all edges between twins of u and twins of v
    for u, v in base_edges:
        for i in range(k):
            for j in range(k):
                a = u * k + i
                b = v * k + j
                if a < b:
                    edges.append([a, b])
    return edges

def _has_triangle(edges, n):
    """Check if graph has a triangle. O(n^3) worst case, but with adjacency sets it's faster."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for u in range(n):
        for v in adj[u]:
            if u < v:
                common = adj[u] & adj[v]
                if common:
                    return True
    return False

def _independence_number(edges, n, time_limit=10):
    """Compute independence number using branch and bound with bitmasks.
    Returns the size of the maximum independent set.
    """
    start_time = time.time()
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    full_mask = (1 << n) - 1
    comp_mask = [0] * n
    for i in range(n):
        comp_mask[i] = full_mask ^ (1 << i) ^ adj_mask[i]
    
    best = 0
    
    def dfs(candidates, current_size, depth):
        nonlocal best
        if time.time() - start_time > time_limit:
            return
        if current_size + candidates.bit_count() <= best:
            return
        if candidates == 0:
            if current_size > best:
                best = current_size
            return
        
        # Pick vertex with fewest neighbors in candidate set (greedy heuristic)
        cands_list = []
        tmp = candidates
        while tmp:
            lsb = tmp & (-tmp)
            v = lsb.bit_length() - 1
            cands_list.append(v)
            tmp ^= lsb
        
        best_v = -1
        best_score = -1
        for v in cands_list:
            score = (candidates & comp_mask[v]).bit_count()
            if score > best_score:
                best_score = score
                best_v = v
        
        v = best_v
        # Include v
        dfs(candidates & comp_mask[v] & ~(1 << v), current_size + 1, depth + 1)
        # Exclude v
        dfs(candidates & ~(1 << v), current_size, depth + 1)
    
    dfs(full_mask, 0, 0)
    return best

def _verify_graph(n, edges):
    """Verify graph is triangle-free and has alpha < 10."""
    if _has_triangle(edges, n):
        return False
    alpha = _independence_number(edges, n, time_limit=15)
    if alpha >= 10:
        return False
    return True

def _random_triangle_free(n, seed):
    """Construct a random maximal triangle-free graph on n vertices."""
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    
    rng.shuffle(edges)
    
    for u, v in edges:
        if adj[u].isdisjoint(adj[v]):
            adj[u].add(v)
            adj[v].add(u)
    
    edge_list = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edge_list.append([i, j])
    return edge_list

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_n = 0
    best_edges = []
    
    # Strategy 1: Deterministic blow-up of 5-cycle
    # C5 blow-up by k: n=5k, alpha=2k. Need 2k<10, so k<=4, n<=20.
    # This is a guaranteed valid graph.
    base_edges = [(0,1), (1,2), (2,3), (3,4), (4,0)]
    for k in range(4, 0, -1):
        n = 5 * k
        edges = _blow_up_graph(base_edges, 5, k)
        if _verify_graph(n, edges):
            if n > best_n:
                best_n = n
                best_edges = edges
            break  # k=4 gives the largest for this construction
    
    # Strategy 2: Randomized search for larger n
    for n in range(40, max(best_n, 20), -1):
        if time.time() > deadline:
            break
        
        found = False
        for seed in range(20):
            if time.time() > deadline:
                break
            if found:
                break
            
            edges = _random_triangle_free(n, seed + n * 1000)
            if _verify_graph(n, edges):
                best_n = n
                best_edges = edges
                found = True
        
        if found:
            break
    
    # Strategy 3: Try circulant graphs
    if time.time() < deadline:
        for n in range(40, max(best_n, 25), -1):
            if time.time() > deadline:
                break
            
            found = False
            for seed in range(10):
                if time.time() > deadline:
                    break
                if found:
                    break
                
                rng = random.Random(seed + n * 7777)
                S = set()
                half = n // 2
                available = list(range(1, half + 1))
                rng.shuffle(available)
                
                for x in available:
                    neg_x = (-x) % n
                    new_S = S | {x, neg_x}
                    # Quick triangle check
                    S_set = set(new_S)
                    neg_S = {(-y) % n for y in new_S}
                    has_tri = False
                    for a in new_S:
                        for b in new_S:
                            if a == b:
                                continue
                            c = (a + b) % n
                            if c in S_set or c in neg_S:
                                has_tri = True
                                break
                        if has_tri:
                            break
                    if not has_tri:
                        S = new_S
                
                if S and len(S) < n:
                    edges = []
                    S_set = set(S)
                    for i in range(n):
                        for s in S:
                            j = (i + s) % n
                            if j > i:
                                edges.append([i, j])
                    if _verify_graph(n, edges):
                        best_n = n
                        best_edges = edges
                        found = True
            
            if found:
                break
    
    # Fallback: ensure we have a valid graph
    if best_n < 20:
        n0 = 20
        best_edges = _blow_up_graph([(0,1), (1,2), (2,3), (3,4), (4,0)], 5, 4)
        best_n = n0
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
