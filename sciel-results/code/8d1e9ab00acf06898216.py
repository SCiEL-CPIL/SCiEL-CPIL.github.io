import math
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Moduli range [20, 200]
    moduli = list(range(20, 201))
    N = len(moduli)
    
    # Precompute gcds
    # gcds[i][j] is gcd(moduli[i], moduli[j])
    gcds = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            gcds[i][j] = math.gcd(moduli[i], moduli[j])
            gcds[j][i] = gcds[i][j]
            
    # Precompute reciprocal sums for pruning
    # rem_sum[i] = sum(1/m for m in moduli[i:])
    rem_sum = [0.0] * (N + 1)
    for i in range(N-1, -1, -1):
        rem_sum[i] = rem_sum[i+1] + 1.0 / moduli[i]
        
    best_score = 0.0
    best_solution = []
    
    # Greedy initialization to get a good lower bound
    def greedy_init():
        selected = []
        # domains[i] is a set of valid residues for moduli[i]
        # Initially all residues are valid
        domains = [set(range(m)) for m in moduli]
        
        for i in range(N):
            m = moduli[i]
            if not domains[i]:
                continue # Skip if no valid residues
            
            # Choose residue with minimum impact on future domains
            min_cost = float('inf')
            best_r = None
            for r in domains[i]:
                cost = 0
                for j in range(i+1, N):
                    g = gcds[i][j]
                    if g > 1:
                        # Count how many residues in domains[j] are congruent to r mod g
                        # Optimization: Precompute counts? Or just iterate.
                        # N is small, domains are small.
                        count = sum(1 for rj in domains[j] if rj % g == r % g)
                        cost += count
                if cost < min_cost:
                    min_cost = cost
                    best_r = r
            
            if best_r is not None:
                selected.append((m, best_r))
                # Update domains for future moduli
                for j in range(i+1, N):
                    g = gcds[i][j]
                    if g > 1:
                        target = best_r % g
                        # Filter domains[j]
                        domains[j] = {rj for rj in domains[j] if rj % g != target}
        
        score = sum(1.0/m for m, r in selected)
        return selected, score

    init_sol, init_score = greedy_init()
    best_score = init_score
    best_solution = init_sol
    
    # Backtracking Search
    # To avoid recursion depth issues and overhead, we might use an explicit stack or just recursion with sys.setrecursionlimit
    sys.setrecursionlimit(10000)
    
    # We need to track the domains efficiently.
    # Copying lists of sets is expensive.
    # We will pass `domains` as a list of sets.
    
    def dfs(idx, selected, current_score, domains):
        nonlocal best_score, best_solution
        
        # Time check
        if time.time() > deadline:
            return
            
        # Pruning: Score bound
        if current_score + rem_sum[idx] <= best_score:
            return
            
        # Base case
        if idx == N:
            if current_score > best_score:
                best_score = current_score
                best_solution = selected
            return
        
        # Check if any future domain is empty
        # We can check this lazily or eagerly.
        # Eager check: if any domains[j] is empty for j >= idx, return.
        # But domains[idx] is the one we are about to branch on.
        # If domains[idx] is empty, we can only skip.
        
        m = moduli[idx]
        
        # Option 1: Skip m
        # Domains for future are unchanged
        dfs(idx + 1, selected, current_score, domains)
        
        # Option 2: Include m
        if domains[idx]:
            # Get valid residues
            valid_rs = list(domains[idx])
            
            # Heuristic: Order residues by "cost" (restrictiveness on future)
            # Calculate cost for each r
            rs_with_cost = []
            for r in valid_rs:
                cost = 0
                for j in range(idx+1, N):
                    g = gcds[idx][j]
                    if g > 1:
                        # Count residues in domains[j] that conflict
                        # This is O(N * size(domain))
                        # We can optimize this by precomputing counts per modulus/gcd?
                        # For now, simple loop.
                        count = sum(1 for rj in domains[j] if rj % g == r % g)
                        cost += count
                rs_with_cost.append((cost, r))
                
            # Sort by cost ascending
            rs_with_cost.sort()
            
            for cost, r in rs_with_cost:
                # Create new domains
                new_domains = [None] * N
                # Copy previous domains (not needed, they are immutable in this context? No, sets are mutable)
                # We only care about domains[idx+1:]
                # domains[0..idx] are irrelevant for future steps
                for k in range(idx+1, N):
                    new_domains[k] = domains[k] # Shallow copy of reference? 
                    # Wait, we need to modify new_domains[k].
                    # So we must create new sets for the ones we modify.
                    
                # Actually, we only modify domains[j] for j > idx where gcd > 1.
                # Let's just copy the whole list of sets? 
                # N=181. 181 sets. Copying is O(N).
                # Modifying is O(N * m).
                
                # Better: 
                # new_domains = [set(d) for d in domains] # Expensive
                # 
                # Let's just pass the list and create new sets only for modified indices.
                # But the recursive call expects a full list.
                # 
                # Optimization: 
                # We can represent domains as a list of frozensets?
                # Then we can cache? No, domains change.
                
                # Let's just do a deep copy of the domains list for the modified parts.
                # Since we are in a tight loop, let's be careful.
                
                # Alternative:
                # We don't need to copy domains[0..idx].
                # We can pass a tuple or just the relevant part?
                # The function signature uses `domains`.
                
                # Let's try a different representation:
                # domains is a list of lists.
                # We update in place? No, backtracking requires undo.
                
                # Let's use a stack of changes? Too complex.
                
                # Simple approach:
                # new_domains = [None] * N
                # for k in range(idx+1, N):
                #     if gcds[idx][k] > 1:
                #         new_domains[k] = {rj for rj in domains[k] if rj % gcds[idx][k] != r % gcds[idx][k]}
                #     else:
                #         new_domains[k] = domains[k]
                
                # This creates new sets only for affected moduli.
                # Unaffected moduli share the same set object (immutable usage).
                
                new_domains = [None] * N
                for k in range(idx+1, N):
                    g = gcds[idx][k]
                    if g > 1:
                        target = r % g
                        new_domains[k] = {rj for rj in domains[k] if rj % g != target}
                    else:
                        new_domains[k] = domains[k]
                        
                # Check if any new_domains[k] is empty
                # If empty, skip this branch
                valid = True
                for k in range(idx+1, N):
                    if not new_domains[k]:
                        valid = False
                        break
                        
                if valid:
                    dfs(idx + 1, selected + [(m, r)], current_score + 1.0/m, new_domains)

    # Run DFS
    # Initial domains
    init_domains = [set(range(m)) for m in moduli]
    dfs(0, [], 0.0, init_domains)
    
    return [[m, r] for m, r in best_solution]

