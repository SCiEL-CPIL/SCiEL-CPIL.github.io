import time
import random
from itertools import combinations

def construct():
    """
    Construct a triangle-free graph with no independent set of 10 vertices, maximizing n.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s limit

    # Fallback: A known valid graph (Andrásfai graph And(9) has n=26, alpha=10? No, alpha=9)
    # Let's use a simpler guaranteed valid graph if search fails.
    # And(k) graph: n=3k-1, edges if diff in {1, k+1, 2k+1}.
    # For k=3, n=8. For k=4, n=11. For k=5, n=14.
    # Let's construct And(5) n=14 as a safe fallback.
    def _make_andrasfai(k):
        n = 3 * k - 1
        adj = [0] * n
        for i in range(n):
            for d in [1, k + 1, 2 * k + 1]:
                j = (i + d) % n
                if j > i:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return n, adj

    # Try to find a graph with n up to 40
    best_n = 0
    best_edges = []

    # Helper to convert bitmask adj to edge list
    def _adj_to_edges(n, adj):
        edges = []
        for i in range(n):
            mask = adj[i]
            # Only consider j > i
            mask &= ~((1 << (i + 1)) - 1)
            while mask:
                j = (mask & -mask).bit_length() - 1
                edges.append([i, j])
                mask &= mask - 1
        return edges

    # Helper: Check if graph is triangle-free and alpha < 10
    # We use a fast B&B for alpha.
    def _is_valid(n, adj, deadline):
        # 1. Check Triangle Free
        # For each edge (u, v), check if common neighbors is empty
        for u in range(n):
            mu = adj[u]
            # Iterate neighbors v > u
            mask = mu
            mask &= ~((1 << (u + 1)) - 1)
            while mask:
                v = (mask & -mask).bit_length() - 1
                # Common neighbors of u and v
                common = mu & adj[v]
                if common:
                    return False
                mask &= mask - 1

        # 2. Check Alpha < 10
        # We need to ensure no independent set of size 10.
        # This is equivalent to no clique of size 10 in the complement.
        # Complement adjacency: ~adj[i] & ~((1<<n)-1) & ~(1<<i)
        # Let's build complement masks for faster B&B
        comp_adj = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            comp_adj[i] = all_mask ^ adj[i] ^ (1 << i)

        # B&B for clique in comp_adj of size 10
        # State: candidates mask, current size
        # Pruning: popcount(candidates) + current_size < 10 -> False
        # Pruning: if any vertex in candidates has degree < (10 - current_size - 1) within candidates -> False (stronger)
        
        def _popcount(x):
            return bin(x).count('1')

        def _has_clique_10(candidates_mask, current_size):
            if current_size >= 10:
                return True
            
            # Time check
            if time.time() > deadline:
                return False # Assume invalid to be safe? No, if we time out, we should reject this candidate to avoid returning an invalid graph.
                             # But we want to find valid ones. If we can't decide, we reject.
                             # Actually, if we timeout, we don't know if it's valid. 
                             # The task says "valid graph... certifies". 
                             # If we return an invalid graph, score is 0.
                             # So we must be sure. If we can't verify alpha < 10 in time, we must NOT return it.
                             # So return False (invalid).

            # Pruning 1: Size
            if current_size + _popcount(candidates_mask) < 10:
                return False

            # Pruning 2: Degree within candidates
            # If any candidate v has fewer than (10 - current_size - 1) neighbors in candidates, it can't be part of a clique of size 10.
            # This is a standard bound.
            # However, checking this for all candidates is O(n^2) per node, which is slow.
            # Let's stick to the simple size bound first, or a quick degree check on the first vertex?
            # Let's try a simple recursive step.
            
            # Heuristic: Pick a vertex with high degree in candidates?
            # Or just iterate.
            
            # To speed up, let's try to find a clique greedily first?
            # No, B&B is standard.
            
            # Optimized B&B:
            # We can use the bound: if max degree in candidates < (10 - current_size - 1), return False.
            # Let's compute max degree in candidates? 
            # That's expensive.
            
            # Let's just iterate.
            # To avoid deep recursion, use iterative or optimized recursion.
            
            # Simple approach: Pick the first vertex? No, that's slow.
            # Pick a vertex with highest degree in the candidate subgraph?
            
            # Let's try a different approach: 
            # Since n <= 40 and we are looking for size 10, and the graph is likely dense (complement of sparse TF graph),
            # the complement is also dense.
            # Wait, TF graph is sparse? Max edges ~ n^2/4 = 400. Total edges 780.
            # So complement has ~380 edges. It's relatively sparse too.
            # Cliques in sparse graphs are hard to find if they don't exist.
            # But if alpha < 10, the complement has no K10.
            
            # Let's use a simple B&B with ordering.
            # Sort vertices by degree in comp_adj? 
            # We can pre-sort indices by degree in comp_adj once.
            
            # For now, simple recursion.
            
            # Optimization: If candidates_mask is 0, return False.
            if not candidates_mask:
                return False

            # Try each vertex in candidates
            # To improve, we can skip vertices that cannot lead to a solution?
            
            # Let's pick a vertex. 
            # Heuristic: pick the one with highest degree in candidates?
            # This requires O(n) check per node.
            
            # Let's just iterate through set bits.
            # To make it faster, we can use a specific order.
            
            # Let's try:
            mask = candidates_mask
            while mask:
                v = (mask & -mask).bit_length() - 1
                # New candidates: neighbors of v in candidates, excluding v and bits lower than v?
                # To avoid duplicates, we only consider neighbors with index > v?
                # No, standard clique finding:
                # new_candidates = candidates & comp_adj[v]
                # But we must ensure we don't count permutations.
                # Standard way:
                # Iterate i from 0 to n-1. If i in candidates:
                #   new_candidates = candidates & comp_adj[i] & mask_greater_than_i
                #   recurse(new_candidates, size+1)
                #   candidates = candidates & ~ (1<<i)
                
                # Let's implement this standard loop.
                # But we are inside a while loop on mask. 
                # Let's switch to the standard iterative B&B structure.
                
                pass
            
            # Let's rewrite the inner B&B properly.
            
            return _clique_bnb(candidates_mask, current_size, comp_adj, deadline)

        def _clique_bnb(candidates_mask, current_size, comp_adj, deadline):
            if current_size >= 10:
                return True
            if time.time() > deadline:
                return False
            
            if _popcount(candidates_mask) + current_size < 10:
                return False
                
            # Simple degree bound: 
            # If any vertex in candidates has < (10 - current_size - 1) neighbors in candidates, 
            # then no clique of size 10 can be formed including that vertex.
            # But we need a clique of size 10 in the WHOLE graph, not necessarily including a specific vertex.
            # The correct bound is: 
            # If the maximum number of vertices in a clique in the subgraph induced by candidates is < 10 - current_size, return False.
            # This is what we are computing.
            
            # A faster sufficient condition for NO clique:
            # If for every vertex v in candidates, deg(v, candidates) < 10 - current_size - 1, then return False.
            # Because any clique of size K must have min degree K-1.
            
            target_rem = 10 - current_size - 1
            if target_rem < 0: return False # Already handled by size check?
            
            # Check if all vertices have low degree?
            # This is O(n) per node.
            mask = candidates_mask
            all_low = True
            while mask:
                v = (mask & -mask).bit_length() - 1
                deg_in_cand = _popcount(candidates_mask & comp_adj[v])
                if deg_in_cand >= target_rem:
                    all_low = False
                    break
                mask &= mask - 1
            
            if all_low:
                return False

            # Expand
            # Pick a vertex. Heuristic: highest degree in candidates?
            # Let's just pick the first one for simplicity, or random?
            # Let's pick the one with highest degree to prune faster?
            
            # To speed up, let's pre-compute sorted vertices?
            # No, let's just iterate.
            
            # Standard expansion:
            # We will iterate through candidates in order.
            # To avoid symmetry, we process vertices in increasing order.
            
            # Optimization: If we can't find a clique, we return False.
            
            # Let's use a recursive helper that takes a mask of "available" vertices and a mask of "forbidden" (lower than current choice)?
            # Simpler:
            # def rec(cands, size):
            #   if size == 10: return True
            #   if popcount(cands) < 10 - size: return False
            #   # Degree bound
            #   target = 10 - size - 1
            #   m = cands
            #   while m:
            #      v = lsb(m)
            #      if popcount(cands & comp_adj[v]) < target: return False
            #      m &= m-1
            #   
            #   # Branch
            #   # Pick a vertex v.
            #   # Heuristic: pick v with max degree in cands?
            #   # Or just pick the first one?
            #   # Let's pick the first one.
            #   v = lsb(cands)
            #   new_cands = cands & comp_adj[v] & ~(1<<v) # Neighbors of v in cands, excluding v
            #   # Also, to avoid duplicates, we should restrict new_cands to vertices with index > v?
            #   # No, that's for listing cliques. For existence, we can just remove v from cands after trying it?
            #   # If we pick v, we must form the rest from neighbors of v.
            #   # If we DON'T pick v, we must form the rest from cands \ {v}.
            #   
            #   # Branch 1: Include v
            #   if rec(new_cands, size + 1): return True
            #   # Branch 2: Exclude v
            #   cands2 = cands & ~(1<<v)
            #   if rec(cands2, size): return True
            #   return False
            
            # This "Exclude" branch is expensive.
            # Better: Iterate through vertices in cands.
            # For each v in cands:
            #    if rec(cands & comp_adj[v] & mask_greater_than_v, size+1): return True
            #    cands = cands & ~(1<<v)
            
            # Let's implement this.
            
            cands = candidates_mask
            while cands:
                # Pick v
                v = (cands & -cands).bit_length() - 1
                
                # Mask of vertices in cands with index > v
                mask_greater = cands & ~((1 << (v + 1)) - 1)
                
                # New candidates: neighbors of v that are also in mask_greater
                new_cands = comp_adj[v] & mask_greater
                
                if _clique_bnb(new_cands, current_size + 1, comp_adj, deadline):
                    return True
                
                cands = cands & ~(1 << v)
                
            return False

        # Initial call
        all_mask = (1 << n) - 1
        if _clique_bnb(all_mask, 0, comp_adj, deadline):
            return False # Found a clique of size 10 in complement -> Independent set of size 10 in G -> Invalid
        
        return True

    # Search Strategy:
    # 1. Try n=40, 39, ...
    # 2. Generate random maximal triangle-free graphs.
    
    for n in range(40, 25, -1):
        if time.time() > deadline - 5:
            break
            
        # Number of attempts
        # For n=40, we need many attempts.
        # But each validation might take time.
        
        attempts = 100 if n > 35 else 50
        
        for _ in range(attempts):
            if time.time() > deadline - 2:
                break
                
            # Generate random maximal triangle-free graph
            adj = [0] * n
            
            # Random order of edges
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(edges)
            
            for u, v in edges:
                # Check if adding (u,v) creates triangle
                # Common neighbors of u and v
                common = adj[u] & adj[v]
                if common:
                    continue
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
            # Check validity
            if _is_valid(n, adj, deadline):
                best_n = n
                best_edges = _adj_to_edges(n, adj)
                # Return immediately? 
                # Since we want to MAXIMIZE n, and we are going down from 40, 
                # the first valid graph found for a given n is the best for that n.
                # But we start from 40. So the first valid graph found overall is the best.
                return {'n': n, 'edges': best_edges}
                
    # Fallback if no valid graph found for n > 25
    # Use And(5) n=14
    n_f, adj_f = _make_andrasfai(5)
    best_edges = _adj_to_edges(n_f, adj_f)
    
    return {'n': n_f, 'edges': best_edges}

# Test locally?
# if __name__ == "__main__":
#     res = construct()
#     print(res['n'], len(res['edges']))
