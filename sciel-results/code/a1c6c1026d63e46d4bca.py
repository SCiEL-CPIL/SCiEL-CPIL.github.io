import numpy as np
from itertools import combinations

def construct():
    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

    def bron_kerbosch(graph, n):
        adj = [set() for _ in range(n)]
        for u, v in graph:
            adj[u].add(v)
            adj[v].add(u)
        
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if R:
                    cliques.append(frozenset(R))
                return
            
            # Pivot selection
            pivot = None
            max_degree = -1
            for v in P.union(X):
                deg = len(P.intersection(adj[v]))
                if deg > max_degree:
                    max_degree = deg
                    pivot = v
            
            for v in list(P - adj[pivot]):
                R_new = R | {v}
                P_new = P & adj[v]
                X_new = X & adj[v]
                
                bk(R_new, P_new, X_new)
                
                P.remove(v)
                X.add(v)
                
        bk(set(), set(range(n)), set())
        return [list(c) for c in cliques]

    def solve_transversal(n, edges, cliques):
        relevant_cliques = [c for c in cliques if len(c) >= 2]
        
        if not relevant_cliques:
            return []
        
        # Map vertices to bitmasks of cliques they belong to
        # And cliques to bitmasks of vertices?
        # Let's use a list of sets for cliques, and precompute vertex -> set of clique indices
        
        num_cliques = len(relevant_cliques)
        # For fast "remove all cliques containing v", we precompute a bitmask for each vertex
        # representing which cliques it hits.
        
        # However, if num_cliques is large (e.g. > 1000), Python int bitmasks are still fine.
        
        vertex_clique_mask = [0] * n
        for i, clique in enumerate(relevant_cliques):
            bit = 1 << i
            for v in clique:
                vertex_clique_mask[v] |= bit
        
        # Precompute clique masks for lower bound?
        # Lower bound: Max number of disjoint cliques.
        # We can compute a greedy packing bound.
        
        best_solution = None
        best_size = n + 1 # Upper bound
        
        # Initial greedy solution to set a good upper bound
        covered_mask = (1 << num_cliques) - 1
        remaining = covered_mask
        greedy_sel = []
        # Simple greedy: pick vertex with max coverage
        # We need to track counts. 
        # For simplicity and speed in B&B, we might just start with B&B.
        # But a good upper bound helps pruning.
        
        # Let's do a quick greedy to get an initial upper bound
        rem = covered_mask
        curr_sel = []
        while rem:
            best_v = -1
            best_cov = 0
            for v in range(n):
                cov = bin(rem & vertex_clique_mask[v]).count('1')
                if cov > best_cov:
                    best_cov = cov
                    best_v = v
            if best_cov == 0: break
            curr_sel.append(best_v)
            rem &= ~vertex_clique_mask[best_v]
        
        if rem == 0:
            best_size = len(curr_sel)
            best_solution = curr_sel[:]
        else:
            # If greedy failed to cover (shouldn't happen if graph is connected enough or we pick all), 
            # but if it did, we need a valid solution. 
            # Actually, if rem != 0, it means some cliques are not covered by the greedy? 
            # Greedy always picks a vertex that covers something if rem != 0.
            # So rem should become 0.
            pass

        def lower_bound(rem_mask):
            # Greedy packing of disjoint cliques
            # Count how many disjoint cliques we can pick from rem_mask
            count = 0
            temp = rem_mask
            # To make it faster, we can just iterate through cliques?
            # Or just return 1 if rem_mask != 0? No, that's weak.
            # A better bound: 
            # Sort cliques by size? 
            # For now, use a simple heuristic: 
            # Pick the clique with the fewest available vertices?
            # Since we don't have vertex availability in rem_mask directly (it's clique availability),
            # we can just pick any clique, count it, and remove it and its neighbors?
            # Disjointness in terms of cliques: Two cliques are "disjoint" for the bound if they don't share any vertices?
            # No, the lower bound for Hitting Set is the size of a maximum set of pairwise disjoint sets.
            # Two cliques C1, C2 are disjoint if C1 ∩ C2 = empty.
            # If they share a vertex, one vertex can hit both.
            # So we want max number of pairwise vertex-disjoint cliques.
            
            # This is a Max Independent Set on the intersection graph of cliques.
            # For small N, we can approximate.
            # Let's use a simple greedy packing:
            # Pick the smallest clique, add to count, remove all cliques that intersect it.
            
            # Precompute intersection masks? 
            # clique_mask[i] is the bitmask of vertices in clique i? No, we have vertex_clique_mask.
            # We need clique_vertex_mask to check intersection.
            
            # This is getting complex for the inner loop.
            # Let's use a simpler bound: 
            # If rem_mask is not 0, LB = 1.
            # Better: LB = ceil(popcount(rem_mask) / max_clique_size)? No.
            
            # Let's stick to LB = 1 if rem != 0 for now, and rely on branching order.
            # Or, precompute a static lower bound? No, it depends on rem.
            
            # Alternative: 
            # Count the number of cliques. If we have K cliques, and max clique size is S, 
            # one vertex can hit at most S cliques? No, a vertex can hit many cliques.
            
            # Let's use the "Maximum Disjoint Cliques" bound computed greedily.
            # To do this fast, we need to know which cliques intersect which.
            # We can precompute a matrix or masks.
            
            # Given time constraints, I will use a simple recursive B&B with LB=1 for non-empty rem.
            # This is effectively a depth-first search with pruning based on current size + 1.
            # It might be slow if the solution is large.
            
            # Let's improve LB: 
            # Find the clique with the minimum number of vertices in the *remaining* graph?
            # No, the cliques are fixed.
            
            # Let's try LB = 1 if rem != 0.
            return 1 if rem_mask else 0

        def branch(rem_mask, curr_size, curr_sel):
            nonlocal best_size, best_solution
            
            if curr_size + lower_bound(rem_mask) >= best_size:
                return
            
            if rem_mask == 0:
                if curr_size < best_size:
                    best_size = curr_size
                    best_solution = curr_sel[:]
                return
            
            # Choose a clique to branch on
            # Pick the first set bit in rem_mask
            # Or pick the clique with the smallest size?
            # Let's pick the clique with the smallest number of vertices that are still "available"?
            # All vertices are available.
            # Let's pick the clique with the smallest size (precomputed).
            
            # Find index of the smallest clique in rem_mask
            # We can iterate through set bits.
            min_idx = -1
            min_len = float('inf')
            
            # Optimization: Precompute clique sizes
            # And iterate
            temp = rem_mask
            while temp:
                # Get lowest set bit index
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                if len(relevant_cliques[idx]) < min_len:
                    min_len = len(relevant_cliques[idx])
                    min_idx = idx
                temp ^= lsb
                
            if min_idx == -1:
                return
                
            # Branch on each vertex in this clique
            clique = relevant_cliques[min_idx]
            # Order vertices by frequency in rem_mask (most frequent first) to prune faster?
            # Or least frequent?
            # Usually, most frequent is better for covering.
            
            verts = sorted(clique, key=lambda v: bin(rem_mask & vertex_clique_mask[v]).count('1'), reverse=True)
            
            for v in verts:
                # Include v
                new_rem = rem_mask & ~vertex_clique_mask[v]
                new_sel = curr_sel + [v]
                branch(new_rem, curr_size + 1, new_sel)
                
                # Note: We do NOT have an "exclude v" branch in standard Set Cover B&B
                # because we MUST pick at least one vertex from the chosen clique.
                # By iterating all v in the clique, we cover all possibilities of which vertex
                # from this clique is the one that hits it.
                
        branch(covered_mask, 0, [])
        
        if best_solution is None:
            # Fallback to greedy if B&B fails or doesn't find anything (shouldn't happen)
            return [v for v in range(n)]
            
        return sorted(best_solution)

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges = instance(n, p, seed)
        cliques = bron_kerbosch(edges, n)
        transversal = solve_transversal(n, edges, cliques)
        out.append(sorted(transversal))
        
    return out
