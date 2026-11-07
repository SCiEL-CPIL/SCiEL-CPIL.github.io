import math
import time

def construct():
    N = 256
    # 1. Identify primes and forced nodes
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    # Forced nodes: 1 and primes > 128
    forced = [1]
    remaining_candidates = []
    for i in range(2, N + 1):
        if is_prime(i) and i > 128:
            forced.append(i)
        else:
            remaining_candidates.append(i)
    
    # The sum of forced nodes is always included
    base_sum = sum(forced)
    
    # 2. Build conflict graph for remaining_candidates
    # Map values to indices for faster access
    val_to_idx = {v: i for i, v in enumerate(remaining_candidates)}
    n_rem = len(remaining_candidates)
    
    # Adjacency list: adj[i] is a set of indices j such that gcd(v_i, v_j) > 1
    adj = [set() for _ in range(n_rem)]
    for i in range(n_rem):
        vi = remaining_candidates[i]
        for j in range(i + 1, n_rem):
            vj = remaining_candidates[j]
            if math.gcd(vi, vj) > 1:
                adj[i].add(j)
                adj[j].add(i)
                
    # Weights
    weights = remaining_candidates
    
    # 3. Branch and Bound
    # We want to find the max weight independent set in the graph defined by adj.
    
    # Initial best
    best_sum = 0
    best_set = []
    
    # Deadline
    start_time = time.time()
    time_limit = 50.0
    
    # Global variables for B&B
    # To avoid passing many args, we use closure or class.
    # Let's use a recursive function with nonlocal.
    
    # Greedy coloring for upper bound
    # Colors are assigned to nodes. 
    # We process nodes in the order they appear in the candidate list.
    # For a subset of candidates, we need to compute a coloring bound.
    
    def color_bound(candidates_idx, adj_subset):
        # candidates_idx: list of indices in the original graph that are in the current candidate set
        # We need to compute the greedy coloring of the subgraph induced by candidates_idx
        # and return the sum of max weights per color.
        
        # To make it fast, we can just do a simple greedy coloring on the fly.
        # However, doing this at every node of the B&B tree might be expensive.
        # Standard Tomita algorithm reorders vertices and colors them.
        
        # Simplified approach:
        # The upper bound is sum of weights of all candidates.
        # This is weak.
        
        # Better: 
        # Use the fact that the graph is sparse.
        # Let's try a simple heuristic bound first: sum of weights.
        # If that's too slow, we implement coloring.
        
        pass

    # Let's implement a standard B&B with a simple bound first and see if it's fast enough.
    # Given N=250, even $2^{250}$ is impossible, but pruning is strong.
    # Actually, the "sum of all remaining weights" bound is often sufficient for such sparse graphs if we branch on high-weight nodes.
    
    # Let's implement a more sophisticated bound: 
    # The maximum weight independent set is bounded by the sum of weights of nodes in each "clique cover" or similar?
    # No, coloring bound: Partition V into independent sets $C_1, C_2, \dots$. 
    # The MWIS can pick at most 1 node from each clique? No, 1 node from each independent set? No.
    # If we color the graph with $k$ colors, the nodes of the same color form an independent set.
    # This doesn't help bound the MWIS directly as sum of max weights.
    # Wait, if the graph is colored with $k$ colors, any clique has at most 1 node of each color? No.
    # Any independent set can contain multiple nodes of the same color? No, nodes of the same color are independent.
    # So an independent set can contain ALL nodes of a given color? No, that's the definition of the color class being independent.
    # So if we color the graph, the nodes of color 1 form an independent set.
    # The MWIS is a subset of V.
    # The coloring bound usually works for MAX CLIQUE (dual problem).
    # For MWIS, we can use the bound: Sum of max weights of nodes in each color class of the COMPLEMENT graph?
    # Or simply: The graph is sparse.
    
    # Let's stick to a simpler, effective strategy:
    # 1. Sort nodes by weight descending.
    # 2. Use a recursive search that decides Include/Exclude.
    # 3. Prune if current_sum + sum(remaining_weights) <= best_sum.
    
    # To speed up, we can pre-compute suffix sums of sorted weights?
    # But the set of remaining nodes changes non-trivially (we remove neighbors).
    
    # Alternative: Use a library? No, standard library only.
    
    # Let's try the B&B with the simple bound. If it times out, we need a better bound.
    # Given the structure (conflict based on small primes), the graph is very sparse.
    # The "sum of remaining weights" bound is likely strong enough because excluding a node removes its neighbors, which have significant weight.
    
    def solve_bnb():
        nonlocal best_sum, best_set
        
        # Order nodes by weight descending
        # We work with a list of available node indices
        # To allow fast removal, we maintain a set or list.
        
        # Let's use a stack-based approach or recursion.
        # Recursion depth <= 250.
        
        # We need to track which nodes are available.
        # Using a boolean array `available` of size n_rem.
        
        available = [True] * n_rem
        current_sum = 0
        
        # Precompute weight list
        w = weights
        
        def dfs(available, current_sum):
            nonlocal best_sum, best_set
            nonlocal start_time, time_limit
            
            if time.time() - start_time > time_limit:
                return False # Signal timeout
            
            # Check if any available
            avail_indices = [i for i in range(n_rem) if available[i]]
            if not avail_indices:
                if current_sum > best_sum:
                    best_sum = current_sum
                    # Reconstruct set? We need to track the set.
                    # This is inefficient.
                    pass
                return True
            
            # Upper bound: current_sum + sum(w[i] for i in avail_indices)
            ub = current_sum + sum(w[i] for i in avail_indices)
            if ub <= best_sum:
                return True
            
            # Pick a node to branch on.
            # Heuristic: Pick the available node with the highest weight.
            # Or pick one that conflicts with many high-weight nodes?
            # Highest weight is standard.
            
            # Find max weight available
            max_idx = -1
            max_w = -1
            for i in avail_indices:
                if w[i] > max_w:
                    max_w = w[i]
                    max_idx = i
            
            # Branch 1: Include max_idx
            # New available: remove max_idx and its neighbors
            new_available = available[:]
            new_available[max_idx] = False
            for neighbor in adj[max_idx]:
                if new_available[neighbor]:
                    new_available[neighbor] = False
            
            if dfs(new_available, current_sum + max_w):
                return False # Propagate timeout
            
            # Branch 2: Exclude max_idx
            new_available = available[:]
            new_available[max_idx] = False
            
            if dfs(new_available, current_sum):
                return False
                
            return True

        # We need to track the actual set of nodes for the best solution.
        # Modifying the above to track the set is messy with copies.
        # Let's use a global `current_set` list that we append/pop.
        
        current_set = []
        
        def dfs_track(available, current_sum):
            nonlocal best_sum, best_set
            nonlocal start_time, time_limit
            
            if time.time() - start_time > time_limit:
                return False
            
            avail_indices = [i for i in range(n_rem) if available[i]]
            if not avail_indices:
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_set = current_set[:]
                return True
            
            ub = current_sum + sum(w[i] for i in avail_indices)
            if ub <= best_sum:
                return True
            
            max_idx = -1
            max_w = -1
            for i in avail_indices:
                if w[i] > max_w:
                    max_w = w[i]
                    max_idx = i
            
            # Include
            new_available = available[:]
            new_available[max_idx] = False
            for neighbor in adj[max_idx]:
                if new_available[neighbor]:
                    new_available[neighbor] = False
            
            current_set.append(remaining_candidates[max_idx])
            if dfs_track(new_available, current_sum + max_w):
                current_set.pop()
                return False
            current_set.pop()
            
            # Exclude
            new_available = available[:]
            new_available[max_idx] = False
            
            if dfs_track(new_available, current_sum):
                return False
                
            return True

        initial_available = [True] * n_rem
        dfs_track(initial_available, 0)
        
        # Check timeout
        if time.time() - start_time > time_limit:
             # Return best found so far
             pass

        return best_sum, best_set

    rem_sum, rem_set = solve_bnb()
    
    final_set = forced + rem_set
    # Ensure unique and int
    final_set = list(set(final_set))
    
    return final_set
