import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    # Target n=20 is the known bound. We aim to construct a valid graph on 20 vertices.
    # Strategy: Greedy Sequential Addition + Local Search Repair
    
    def count_violations(adj, n):
        """
        Count the number of subsets of size 6 that induce a regular subgraph.
        Returns 0 if valid.
        """
        count = 0
        # Precompute degrees for speed? No, induced subgraph degrees change.
        # Vectorized check for all subsets is too slow if done naively in Python loops for n=20.
        # C(20,6) = 38760.
        # For each subset, we need to check if the induced subgraph is regular.
        
        # Optimization: Use bitmasks or just efficient numpy slicing?
        # Let's try a semi-vectorized approach or a fast Python loop with precomputed masks.
        
        # For n=20, 38760 checks.
        # Each check involves extracting a 6x6 submatrix and checking degree uniformity.
        
        for subset in combinations(range(n), 6):
            # Extract rows
            sub_adj = adj[np.ix_(subset, subset)]
            # Calculate degrees
            degrees = sub_adj.sum(axis=1)
            # Check if all degrees are equal
            if len(set(degrees.tolist())) == 1:
                count += 1
                if count > 0: # We only care if > 0 for validity, but for greedy we want min count
                    pass # Continue counting to get a score for greedy? 
                    # Actually, for greedy, we want to minimize the count.
                    # Breaking early is faster for "is it valid?" but slower for "how bad is it?".
                    # Let's just count all to get a better gradient for greedy.
                    # But 38760 * (cost of check) might be high.
                    # Let's estimate: 38760 * ~1us = 0.04s. This is acceptable for a few iterations.
        
        return count

    def is_valid(adj, n):
        # Faster check: stop at first violation
        for subset in combinations(range(n), 6):
            sub_adj = adj[np.ix_(subset, subset)]
            degrees = sub_adj.sum(axis=1)
            if len(set(degrees.tolist())) == 1:
                return False
        return True

    def greedy_construct(target_n):
        """
        Build graph vertex by vertex.
        """
        if target_n <= 0:
            return None
            
        adj = np.zeros((target_n, target_n), dtype=bool)
        
        # Start with 5 vertices (any graph on 5 vertices is valid wrt 6-subsets)
        # Let's start with a random graph on 5 vertices to break symmetry?
        # Or just empty. Empty is fine.
        
        for k in range(6, target_n):
            # We are adding vertex k.
            # We need to decide its edges to vertices 0..k-1.
            # There are 2^(k-1) possibilities.
            # For k=20, 2^19 = 524,288 possibilities.
            # Checking each possibility requires checking if it creates new regular induced subgraphs of size 6.
            # A new regular induced subgraph of size 6 must include the new vertex k.
            # So we only need to check subsets of size 5 from {0..k-1} combined with k.
            
            best_edges = None
            min_violations = float('inf')
            
            # To speed up, we can sample if k is large, or be exhaustive if k is small.
            # For k up to 10, 2^9=512 is fine.
            # For k=20, 2^19 is too many to check exhaustively with full subset checks.
            
            # Optimization: Only check subsets involving the new vertex.
            # Number of such subsets is C(k-1, 5).
            # For k=20, C(19,5) = 11628.
            # Total checks per vertex addition = 2^(k-1) * 11628.
            # For k=10: 512 * 126 = 64512.
            # For k=20: 524288 * 11628 = ~6e9. Too slow.
            
            # So we need a heuristic for large k.
            # Random sampling of edge configurations for the new vertex.
            
            num_trials = 50
            if k < 12:
                # Exhaustive for small k
                candidates = []
                for mask in range(1 << (k-1)):
                    candidates.append(mask)
                # Shuffle to avoid bias
                random.shuffle(candidates)
            else:
                # Random sampling
                candidates = [random.getrandbits(k-1) for _ in range(num_trials)]
                
            for mask in candidates:
                # Apply mask to adj[k, :k] and adj[:k, k]
                # Note: adj is symmetric
                
                # Create a temporary copy of the relevant part? 
                # Or just simulate the check.
                
                # To check violations, we need to know the degrees of the induced subgraphs.
                # For a subset S of {0..k-1} of size 5, let T = S U {k}.
                # The induced subgraph on T is regular if:
                # 1. The induced subgraph on S has degrees d_i (for i in S).
                # 2. The degree of k in T is d_k.
                # 3. For each i in S, its degree in T is d_i + (1 if edge(i,k) else 0).
                # 4. All these degrees must be equal.
                
                # Let's precompute the "potential" degrees for all 5-subsets of {0..k-1}.
                # This is expensive to recompute every time.
                
                # Simpler approach for greedy step:
                # Just try the mask, update adj, and check only the new subsets.
                
                # Save state
                old_row = adj[k, :k].copy()
                
                # Set new row
                for i in range(k):
                    if mask & (1 << i):
                        adj[k, i] = True
                        adj[i, k] = True
                    else:
                        adj[k, i] = False
                        adj[i, k] = False
                        
                # Count violations in new subsets (those containing k)
                # Subsets are combinations of 5 from 0..k-1
                violations = 0
                for s in combinations(range(k), 5):
                    # s is a tuple of 5 indices
                    # Subset is s + (k,)
                    # Check if induced subgraph on s U {k} is regular
                    
                    # Get degrees of s in the current graph (excluding k's influence for a moment? No, including edges within s)
                    # Degrees of vertices in s within the subgraph induced by s U {k}
                    
                    # deg_i = (number of edges from i to other vertices in s) + (1 if edge(i,k) else 0)
                    # deg_k = (number of edges from k to vertices in s)
                    
                    # Let's compute deg_k first
                    deg_k = 0
                    for i in s:
                        if adj[k, i]:
                            deg_k += 1
                            
                    # Check if deg_k matches any possible degree
                    # If deg_k is 0, all others must be 0.
                    # If deg_k is 5, all others must be 5.
                    # etc.
                    
                    # Efficient check:
                    # Calculate degrees for vertices in s induced by s
                    # This is C(k,5) * C(5,2) operations.
                    # For k=20, 11628 * 10 = 116k ops.
                    # For 50 trials, 5.8M ops.
                    # In Python, this is ~0.5s - 1s per vertex addition.
                    # 14 vertices to add (6 to 20). Total ~10-15s. Acceptable.
                    
                    # Let's implement the check
                    # degs = [0]*5
                    # for i_idx, i in enumerate(s):
                    #   for j_idx, j in enumerate(s):
                    #     if i != j and adj[i, j]:
                    #       degs[i_idx] += 1
                    #   degs[i_idx] += 1 if adj[i, k] else 0
                    
                    # if all(d == degs[0] for d in degs) and degs[0] == deg_k:
                    #   violations += 1
                    
                    # Optimization: Precompute degrees within s?
                    # Since s is small, direct calculation is fine.
                    
                    degs = []
                    for i in s:
                        d = 0
                        for j in s:
                            if i != j and adj[i, j]:
                                d += 1
                        if adj[i, k]:
                            d += 1
                        degs.append(d)
                        
                    if all(d == degs[0] for d in degs) and degs[0] == deg_k:
                        violations += 1
                            
                if violations < min_violations:
                    min_violations = violations
                    best_edges = mask
                    
                # Restore state
                adj[k, :k] = old_row
                adj[:k, k] = old_row # adj is symmetric, so this restores both if we just set row k? 
                # Wait, adj[:k, k] was modified. 
                # adj[i, k] = adj[k, i]. So setting adj[k, :k] restores adj[k, i].
                # But adj[i, k] is the same memory? No, adj is a 2D array.
                # adj[k, i] and adj[i, k] are different elements.
                # We modified both.
                # So we need to restore both.
                adj[:k, k] = old_row
                # Wait, old_row is 1D. adj[:k, k] is 1D.
                # This works.
                
                if min_violations == 0:
                    break # Found a perfect addition
            
            # Apply best_edges
            if best_edges is not None:
                for i in range(k):
                    if best_edges & (1 << i):
                        adj[k, i] = True
                        adj[i, k] = True
                    else:
                        adj[k, i] = False
                        adj[i, k] = False
            else:
                # Fallback: random
                for i in range(k):
                    if random.random() < 0.5:
                        adj[k, i] = True
                        adj[i, k] = True
                        
        return adj

    def local_search(adj, n, max_iters=200):
        """
        Try to remove all violations by flipping edges.
        """
        current_violations = count_violations(adj, n)
        if current_violations == 0:
            return adj
            
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Pick a random violation
            # We need to find one.
            viol_subset = None
            for subset in combinations(range(n), 6):
                sub_adj = adj[np.ix_(subset, subset)]
                degrees = sub_adj.sum(axis=1)
                if len(set(degrees.tolist())) == 1:
                    viol_subset = subset
                    break
                    
            if viol_subset is None:
                return adj
                
            # Pick a random edge within this subset to flip
            # Or pick a random edge in the whole graph?
            # Flipping an edge in the violating subset is more likely to help.
            u, v = random.sample(list(viol_subset), 2)
            
            # Flip edge u-v
            adj[u, v] = not adj[u, v]
            adj[v, u] = not adj[u, v]
            
            new_violations = count_violations(adj, n)
            
            if new_violations < current_violations:
                current_violations = new_violations
                if current_violations == 0:
                    return adj
            else:
                # Revert
                adj[u, v] = not adj[u, v]
                adj[v, u] = not adj[u, v]
                
                # Simulated annealing step?
                # For now, just accept if equal to escape local minima?
                if new_violations == current_violations and random.random() < 0.1:
                    pass # Keep change
                else:
                    pass # Revert already done? No, we reverted above.
                    # Let's add a small probability to accept equal
                    # Actually, the code above reverts if not strictly better.
                    # Let's modify to accept equal with prob 0.1
                    # But I already reverted. Let's just stick to hill climbing for simplicity first.
                    
        return adj

    # Main loop
    # Try to construct n=20 first.
    # If fails, try 19, 18...
    
    best_adj = None
    best_n = 0
    
    for target_n in [20, 19, 18, 17, 16]:
        if time.time() > deadline - 10:
            break
            
        found = False
        for restart in range(5):
            if time.time() > deadline - 5:
                break
                
            # Randomize initial 5 vertices? 
            # Greedy starts with empty 5-vertex graph.
            # Let's add some randomness to the start or the greedy choices.
            # The greedy step already has randomness in candidate selection for k>=12.
            
            adj = greedy_construct(target_n)
            
            if adj is None:
                continue
                
            # Check if valid
            if is_valid(adj, target_n):
                best_adj = adj
                best_n = target_n
                found = True
                break
                
            # If not valid, try local search
            adj_ls = local_search(adj.copy(), target_n, max_iters=50)
            if is_valid(adj_ls, target_n):
                best_adj = adj_ls
                best_n = target_n
                found = True
                break
                
            # If still not valid, maybe try a different restart
            # We can also try to "repair" by adding more random restarts
            
        if found:
            break # We want max n, so if we found 20, stop. If 19, stop? 
            # Yes, we iterate 20, 19, 18... so if we find 20, we break.
            # If we fail 20, we try 19.
            
    if best_adj is not None and best_n > 0:
        # Extract edges
        edges = []
        for i in range(best_n):
            for j in range(i+1, best_n):
                if best_adj[i, j]:
                    edges.append([i, j])
        return {'n': best_n, 'edges': edges}
        
    # Fallback: Return a small valid graph (e.g. n=5)
    # Any graph on 5 vertices is valid.
    return {'n': 5, 'edges': []}
