"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import math
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n = 16
    target_edges = 65
    
    def build_adj(edges):
        """Build adjacency bitmasks and degrees from edge list."""
        adj = [0] * n
        deg = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        return adj, deg
    
    def compute_score_from_adj(adj, deg):
        """Compute max triangle degree sum using bitmasks."""
        max_sum = 0
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] >> j) & 1:
                    continue
                # Common neighbors of i and j, with index > j
                common = adj[i] & adj[j] & ~((1 << (j + 1)) - 1)
                while common:
                    k = common & (-common)
                    k_idx = k.bit_length() - 1
                    s = deg[i] + deg[j] + deg[k_idx]
                    if s > max_sum:
                        max_sum = s
                    common &= common - 1
        return max_sum
    
    def compute_score(edges):
        """Compute the maximum triangle degree sum."""
        adj, deg = build_adj(edges)
        return compute_score_from_adj(adj, deg)
    
    def is_valid(edges):
        """Check if edges form a simple graph with exactly target_edges edges."""
        if len(edges) != target_edges:
            return False
        seen = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in seen:
                return False
            seen.add(key)
        return True
    
    # Strategy: K_{6,10} + 5 edges in the 10-vertex part
    # Vertices 0-5 are the "small" part, 6-15 are the "large" part
    # Base degrees: small=10, large=6
    # Adding 5 internal edges: each endpoint gets +1 degree
    # Triangle score = 10 + deg(i) + deg(j) for internal edge (i,j)
    # To get score <= 23, need deg(i) + deg(j) <= 13 for all internal edges
    # i.e., (6+a_i) + (6+a_j) <= 13 => a_i + a_j <= 1
    # where a_v = number of internal edges incident to v
    
    # MILP-inspired approach:
    # Let x_e be binary variable for each possible internal edge e
    # Constraints: sum(x_e) = 5, and for each internal edge e=(i,j):
    #   deg(i) + deg(j) <= 13 (for score <= 23)
    #   deg(v) = 6 + sum_{e incident to v} x_e
    # So constraint becomes: 12 + sum_{e incident to i} x_e + sum_{e incident to j} x_e <= 13
    # i.e., sum of x_e over edges incident to i or j (but not e itself counted twice) <= 1
    
    # This means: for any two adjacent vertices i,j in the internal graph,
    # the total degree in the internal graph of i and j combined (counting e once) <= 1
    # Wait, let me reconsider. deg(i) = 6 + d_i where d_i is internal degree
    # Score for triangle (s, i, j) where s is small, (i,j) internal edge:
    #   10 + (6 + d_i) + (6 + d_j) = 22 + d_i + d_j
    # For score <= 23: d_i + d_j <= 1
    # But d_i >= 1 and d_j >= 1 for any edge (i,j), so d_i + d_j >= 2
    # This means score <= 23 is IMPOSSIBLE with K_{6,10} + 5 internal edges!
    
    # So we need a different approach. Let's try K_{8,8} + 1 edge.
    # Vertices 0-7: part A (8 vertices), vertices 8-15: part B (8 vertices)
    # K_{8,8} has 64 edges. Add 1 edge within one part, say (0,1) in part A.
    # Degrees: vertices 0,1 in A: 8+1=9, other A vertices: 8, B vertices: 8
    # Triangles: (0,1) with any B vertex: score = 9+9+8 = 26
    # Or (0,1) with... wait, (0,1) is an edge in A, and both connect to all B.
    # Triangle (0,1,b) for b in B: deg(0)=9, deg(1)=9, deg(b)=8, sum=26
    
    # What about K_{6,10} but with different internal structure?
    # If we add edges NOT in the large part but... no, we must have 65 edges.
    
    # Let's try: modify the bipartition. What if we use a non-complete bipartite base?
    # Actually, let's try a different design: start from K_{8,8} and add edges
    # in BOTH parts to balance degrees.
    
    # K_{8,8}: 64 edges. Need 1 more edge.
    # If we add edge (a1, a2) in part A: deg(a1)=9, deg(a2)=9, others in A: 8, B: 8
    # Max triangle sum: 9+9+8 = 26 (for any b in B connected to both a1,a2)
    
    # Alternative: remove some cross edges and add more internal edges.
    # Remove k cross edges, add k+1 internal edges.
    # This changes the structure significantly.
    
    # Let's try a systematic search with the MILP constraint insight:
    # For K_{6,10} + 5 internal edges, the minimum possible max score is 24
    # (achieved by a perfect matching in the 10-vertex part).
    # To beat 24, we need a fundamentally different structure.
    
    # Try: K_{7,9} + 2 edges? 7*9 = 63, need 2 more edges.
    # Part A: 7 vertices, Part B: 9 vertices.
    # Base degrees: A=9, B=7.
    # Add 2 internal edges. If both in part B (9 vertices):
    #   Two B vertices get degree 8, others in B: 7, A: 9
    #   Triangle (a, b1, b2) where (b1,b2) is internal edge: 9+8+8=25
    #   If the 2 edges share a vertex: one B vertex has degree 9, two have 8
    #     Triangle with the degree-9 vertex and a degree-8 neighbor: 9+9+8=26
    #   If matching: two pairs, each endpoint degree 8: max = 9+8+8=25
    
    # K_{5,11}: 55 edges, need 10 more. Too many internal edges.
    
    # Let's try K_{8,8} with 1 edge removed from cross and 2 added internally.
    # Remove edge (a1, b1), add (a1, a2) and (b1, b2).
    # Degrees: a1: 8-1+1=8, a2: 8+1=9, b1: 8-1+1=8, b2: 8+1=9, others: 8
    # Triangles:
    #   (a1, a2, b): for b in B\{b1}, deg(a1)=8, deg(a2)=9, deg(b)=8, sum=25
    #   (b1, b2, a): for a in A\{a1}, deg(b1)=8, deg(b2)=9, deg(a)=8, sum=25
    #   (a1, a2, b1)? No, a1-b1 edge was removed.
    #   (a1, b2, a2)? a1-b2 exists, b2-a2? No, a2 is in A, b2 in B, they're connected.
    #     Triangle (a1, a2, b2): a1-a2 exists, a1-b2 exists, a2-b2 exists.
    #     deg(a1)=8, deg(a2)=9, deg(b2)=9, sum=26. Bad.
    
    # This is getting complex. Let's do a comprehensive local search with
    # multiple starting points and a better scoring function.
    
    # Key insight from MILP: for K_{6,10}+5 internal, min score is 24.
    # We need to explore other graph families.
    
    # Let's try a general local search over all 65-edge graphs, but guided
    # by the degree sum objective.
    
    # Start with K_{6,10} + matching (score 24) and try to improve.
    
    base_edges_k610 = [[u, v] for u in range(6) for v in range(6, 16)]
    initial_internal = [[6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
    best_edges = base_edges_k610 + initial_internal
    best_score = compute_score(best_edges)
    
    # Also try K_{8,8} + 1 edge
    k88_edges = [[u, v] for u in range(8) for v in range(8, 16)]
    k88_plus1 = k88_edges + [[0, 1]]
    k88_score = compute_score(k88_plus1)
    if k88_score < best_score:
        best_score = k88_score
        best_edges = k88_plus1[:]
    
    # Local search with edge swaps
    def local_search(edges, max_iters=20000):
        """Try to improve score via edge swaps."""
        current = edges[:]
        adj, deg = build_adj(current)
        current_score = compute_score_from_adj(adj, deg)
        
        # Precompute all possible edges
        all_edges = set()
        for i in range(n):
            for j in range(i+1, n):
                all_edges.add((i, j))
        
        edge_set = set()
        for u, v in current:
            edge_set.add((min(u,v), max(u,v)))
        
        improved = True
        iters = 0
        no_improve_count = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline - 5.0:
                break
            
            # Try random edge swaps
            for attempt in range(20):
                if time.time() > deadline - 5.0:
                    break
                
                # Pick a random edge to remove
                u1, v1 = random.choice(list(edge_set))
                
                # Pick a random non-edge to add
                u2, v2 = random.choice(list(all_edges - edge_set))
                
                # Perform swap
                new_adj = adj[:]
                new_deg = deg[:]
                
                # Remove edge (u1, v1)
                new_adj[u1] &= ~(1 << v1)
                new_adj[v1] &= ~(1 << u1)
                new_deg[u1] -= 1
                new_deg[v1] -= 1
                
                # Add edge (u2, v2)
                new_adj[u2] |= (1 << v2)
                new_adj[v2] |= (1 << u2)
                new_deg[u2] += 1
                new_deg[v2] += 1
                
                new_score = compute_score_from_adj(new_adj, new_deg)
                
                if new_score < current_score:
                    # Accept
                    adj = new_adj
                    deg = new_deg
                    current_score = new_score
                    edge_set.remove((u1, v1))
                    edge_set.add((u2, v2))
                    
                    # Rebuild edge list
                    current = [[i, j] for i, j in edge_set]
                    
                    improved = True
                    no_improve_count = 0
                    break
                else:
                    no_improve_count += 1
            
            if no_improve_count > 500:
                break
        
        return current, current_score
    
    # Run local search from best starting point
    ls_edges, ls_score = local_search(best_edges, max_iters=15000)
    if ls_score < best_score:
        best_score = ls_score
        best_edges = ls_edges[:]
    
    # Try multiple random restarts with diverse starting points
    all_possible_edges = [(a, b) for a in range(n) for b in range(a+1, n)]
    
    for restart in range(50):
        if time.time() > deadline - 10.0:
            break
        
        # Random 65-edge graph
        random.shuffle(all_possible_edges)
        random_edges = [[a, b] for a, b in all_possible_edges[:65]]
        
        rand_score = compute_score(random_edges)
        if rand_score < best_score:
            best_score = rand_score
            best_edges = random_edges[:]
        
        # Local search from random start
        if time.time() > deadline - 10.0:
            continue
        ls_edges2, ls_score2 = local_search(random_edges, max_iters=5000)
        if ls_score2 < best_score:
            best_score = ls_score2
            best_edges = ls_edges2[:]
    
    # Final verification
    if not is_valid(best_edges):
        best_edges = base_edges_k610 + [[6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
    
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
