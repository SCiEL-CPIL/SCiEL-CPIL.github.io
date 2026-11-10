import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave 10s margin for 120s limit
    
    # We aim for n=20 first, as it is the known best.
    # We will try to find a valid graph for n=20.
    # If it fails, we try n=19, etc.
    
    best_n = 5
    best_edges = []
    
    # Try n=20 down to 10
    for n in range(20, 9, -1):
        if time.time() - start_time > time_limit:
            break
            
        found = _find_valid_graph(n, time_limit - (time.time() - start_time))
        if found is not None:
            return found
            
        # If we didn't find one for this n, try smaller n
        # But usually n=20 is found quickly.
        
    # Fallback to a trivial valid graph
    return {'n': 5, 'edges': []}

def _find_valid_graph(n, time_limit):
    start_time = time.time()
    
    # Initialize random graph
    # adj[i] is an integer bitmask where bit j is set if (i,j) is an edge
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    current_violations = _get_violations(adj, n)
    
    if not current_violations:
        return _to_edges(adj, n)
        
    # Local Search
    # We want to reduce the number of regular induced subgraphs of size >= 6
    # A violation is a tuple (mask, k) where mask is a subset of size k>=6 and all degrees in induced subgraph are equal.
    
    # To speed up, we can focus on violations of size 6, as they are the most numerous.
    # If we eliminate all size 6 violations, larger ones might disappear or be fewer.
    
    # Heuristic: Pick a random violation, pick a random edge to flip that might break it.
    # Flipping an edge (u,v) changes degrees of u and v in any subset containing both.
    
    # Let's use a simpler approach: Randomly flip edges, accept if it reduces total violations.
    # To be more targeted, we can look at a specific violation and try to break it.
    
    # Precompute masks for all subsets of size 6? No, too many.
    # Instead, when we have a list of violations, pick one.
    
    # Optimization: Only check violations of size 6 for the "cost" function?
    # The problem states "No set of 6 or more".
    # If a set of size 7 is regular, it contains regular subsets of size 6?
    # Not necessarily. A 7-vertex regular graph (e.g., 7-cycle) has induced subgraphs of size 6 which are paths (not regular).
    # So we must check all sizes >= 6.
    
    # However, checking all subsets of size 6..n is expensive.
    # For n=20, C(20,6) = 38760. C(20,7) = 77520. Sum is manageable if done efficiently.
    
    # Let's implement a fast violation checker.
    
    def get_violations_fast(adj, n):
        violations = []
        # Check sizes 6 to n
        # For efficiency, we can iterate through combinations.
        # But generating combinations is slow in Python.
        # Alternative: Iterate through all masks? 2^20 is 1M. 
        # For each mask, check if size >= 6 and if regular.
        # Checking regularity: 
        #   degs = []
        #   for v in mask:
        #       degs.append(popcount(adj[v] & mask))
        #   if all(d == degs[0] for d in degs):
        #       violations.append(mask)
        
        # 1M masks * 20 vertices * popcount is ~20M ops. Might be too slow in Python loop.
        # We need to optimize.
        
        # Let's stick to checking specific subsets or using a smarter heuristic.
        # Actually, for n=20, we can just check all subsets of size 6.
        # If there are no regular subsets of size 6, is it possible to have one of size 7?
        # Yes. Example: 7-cycle. No 6-vertex induced subgraph is regular (degrees in 7-cycle induced subgraph of size 6 are 1,1,1,1,2,2 or similar, not all equal).
        # So we MUST check sizes > 6.
        
        # Strategy: 
        # 1. Check all subsets of size 6.
        # 2. Check all subsets of size 7.
        # 3. ... up to n.
        
        # To make this fast, we can use bit operations.
        
        all_violations = []
        
        # Generate combinations
        # It is faster to iterate k from 6 to n and generate combinations.
        
        from itertools import combinations
        
        for k in range(6, n + 1):
            for comb in combinations(range(n), k):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                
                # Check if regular
                # Get degrees
                is_reg = True
                first_deg = -1
                for v in comb:
                    deg = bin(adj[v] & mask).count('1')
                    if first_deg == -1:
                        first_deg = deg
                    elif deg != first_deg:
                        is_reg = False
                        break
                if is_reg:
                    all_violations.append(mask)
                    
        return all_violations

    # The above get_violations_fast is still slow due to combinations generation and inner loop.
    # Let's try a different approach for the search:
    # Start with a known good structure or random, and use a "penalty" method.
    
    # Given the time constraints, let's try to find a graph by random restarts and local search on size-6 violations only first, then verify full validity.
    # If we eliminate all size-6 violations, we then check for larger ones.
    
    # Revised Plan:
    # 1. Random init.
    # 2. Local search to minimize count of regular induced subgraphs of size 6.
    # 3. If count is 0, check for regular induced subgraphs of size 7..n.
    # 4. If any found, add them to the list and continue local search minimizing total violations.
    
    # To speed up step 2, we need a fast way to count size-6 violations.
    # Precompute all masks of size 6? There are 38760.
    # For each mask, we can compute the degrees quickly?
    # Actually, flipping an edge (u,v) affects only violations that contain both u and v.
    
    # Let's implement a simpler local search:
    # Pick a random violation (size 6).
    # Pick a random edge (u,v) within that violation.
    # Flip it.
    # Check if it reduces the number of violations (recompute violations? No, too slow).
    # Approximate: Assume it helps.
    
    # Better: 
    # Use a simulated annealing or greedy approach with a fast cost function.
    # Cost = number of regular induced subgraphs of size 6.
    
    # Let's try to just run a greedy search that fixes one violation at a time.
    
    # 1. Get violations (size 6).
    # 2. Pick one.
    # 3. Try flipping each edge in the induced subgraph.
    # 4. Check if the specific violation is broken.
    # 5. If yes, accept.
    # 6. Repeat until no violations.
    # 7. Then check for larger violations.
    
    # This is O(V * E * Cost_Check).
    # Cost_Check for one violation is O(6).
    # This is very fast.
    
    # Implementation:
    
    def is_regular(mask, adj, n):
        # mask is an integer
        # get vertices
        vs = []
        for i in range(n):
            if mask & (1 << i):
                vs.append(i)
        if len(vs) < 6:
            return False
        degs = []
        for v in vs:
            degs.append(bin(adj[v] & mask).count('1'))
        return all(d == degs[0] for d in degs)
        
    def find_violations_size(adj, n, k):
        # Find all masks of size k that are regular
        from itertools import combinations
        viols = []
        for comb in combinations(range(n), k):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            if is_regular(mask, adj, n):
                viols.append(mask)
        return viols
        
    # Main Loop
    # We will iteratively remove violations.
    # Start with k=6.
    
    # To avoid getting stuck, we allow some randomization.
    
    max_iters = 5000
    iters = 0
    
    while time.time() - start_time < time_limit and iters < max_iters:
        iters += 1
        
        # Find any violation of size 6
        viols6 = find_violations_size(adj, n, 6)
        if not viols6:
            # Check larger sizes
            has_larger = False
            for k in range(7, n + 1):
                viols_k = find_violations_size(adj, n, k)
                if viols_k:
                    has_larger = True
                    # Pick one larger violation to fix
                    mask = viols_k[0]
                    # Try to fix it by flipping an edge within it
                    vs = [i for i in range(n) if mask & (1 << i)]
                    # Try flipping edges within vs
                    improved = False
                    for i in range(len(vs)):
                        for j in range(i+1, len(vs)):
                            u = vs[i]
                            v = vs[j]
                            # Flip edge u-v
                            bit_u = 1 << v
                            bit_v = 1 << u
                            adj[u] ^= bit_u
                            adj[v] ^= bit_v
                            
                            # Check if this specific violation is gone
                            # And ideally doesn't create new size-6 violations
                            # Simple check: is this mask still regular?
                            if not is_regular(mask, adj, n):
                                improved = True
                                break
                            else:
                                # Revert
                                adj[u] ^= bit_u
                                adj[v] ^= bit_v
                        if improved:
                            break
                    if improved:
                        continue
                    else:
                        # If we can't fix it by internal flip, try external?
                        # Or just break and try another violation?
                        # For now, if stuck, break and try random restart or accept.
                        break
                else:
                    pass
            
            if not has_larger:
                # No violations of size 6..n
                return _to_edges(adj, n)
            else:
                # If we broke out of the loop because we couldn't fix a larger violation
                # We might be stuck. Let's try a random move.
                # Pick random u, v and flip
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u != v:
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                continue
                
        # If we have size 6 violations
        # Pick a random one
        mask = random.choice(viols6)
        vs = [i for i in range(n) if mask & (1 << i)]
        
        # Try flipping each edge within vs
        improved = False
        for i in range(len(vs)):
            for j in range(i+1, len(vs)):
                u = vs[i]
                v = vs[j]
                bit_u = 1 << v
                bit_v = 1 << u
                adj[u] ^= bit_u
                adj[v] ^= bit_v
                
                # Check if this specific violation is gone
                if not is_regular(mask, adj, n):
                    # Check if we introduced new violations? 
                    # We just accept for now. The next iteration will check.
                    improved = True
                    break
                else:
                    # Revert
                    adj[u] ^= bit_u
                    adj[v] ^= bit_v
            if improved:
                break
        
        if not improved:
            # Stuck on this violation. Try a random move elsewhere to escape local minimum.
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v:
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
            else:
                # If stuck, break and try next n? Or restart?
                break

    # If we didn't find a valid graph, return None
    return None

def _to_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return {'n': n, 'edges': edges}

# Quick sanity check
if __name__ == "__main__":
    res = construct()
    print(res)
