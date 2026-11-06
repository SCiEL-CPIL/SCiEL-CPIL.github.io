import time
import random

def construct():
    start_time = time.time()
    timeout = 85.0
    
    # Target: 12 distinct positive integers <= 10000
    # Minimize distinct prime factors of all pairwise sums.
    
    # 1. Try to find a set of 12 integers where all pairwise sums are 7-smooth.
    #    If found, the distinct primes are {2, 3, 5, 7} -> Count 4.
    #    This is likely the optimal solution.
    
    target_primes = [2, 3, 5, 7]
    
    def is_smooth(n, primes):
        for p in primes:
            while n % p == 0:
                n //= p
        return n == 1

    # Precompute smooth numbers up to 200 (max sum for small search space 1..100)
    # If we search up to 200, sum is 400.
    # Let's search b_i in 1..200. Sum up to 400.
    max_search_val = 200
    smooth_nums = set()
    for n in range(2, max_search_val * 2 + 1):
        if is_smooth(n, target_primes):
            smooth_nums.add(n)
            
    def find_clique_7_smooth():
        # Build adjacency list for vertices 1..max_search_val
        adj = {v: set() for v in range(1, max_search_val + 1)}
        for u in range(1, max_search_val + 1):
            for v in range(u + 1, max_search_val + 1):
                if (u + v) in smooth_nums:
                    adj[u].add(v)
                    adj[v].add(u)
                    
        # Backtracking search for clique of size 12
        # Use a simple DFS with pruning.
        # To speed up, order vertices by degree.
        
        def backtrack(clique, candidates):
            if len(clique) == 12:
                return clique
            if not candidates:
                return None
            if time.time() - start_time > timeout - 10:
                return None
                
            # Heuristic: Try vertices with highest degree in the current candidate set
            # This is expensive to calculate every time.
            # Simple heuristic: Just iterate.
            
            # Optimization: If the maximum possible clique size from candidates is < required, return None.
            # Max clique size <= max degree in candidates + 1? Not exactly.
            # Just use a simple bound: if len(candidates) < 12 - len(clique): return None
            
            if len(candidates) < 12 - len(clique):
                return None
                
            for v in candidates:
                new_clique = clique + [v]
                # New candidates: neighbors of v that are in candidates and greater than v (to avoid duplicates)
                # We maintain candidates as a sorted list or set?
                # Let's pass a list and filter.
                
                # To avoid re-visiting, we should ensure we only pick from candidates that are "after" v in some ordering?
                # Standard Bron-Kerbosch uses sets.
                # Here, let's just filter the list.
                
                # Efficiency: 
                # candidates is a list of ints.
                # We need neighbors of v that are in candidates.
                # And we should remove v from candidates for the next recursion?
                # Yes, and also remove any vertices < v if we sorted?
                # Let's sort candidates initially and maintain the property that we only consider vertices > last picked?
                # No, that's for independent sets or specific orderings.
                # For cliques, we just need to ensure we don't pick the same set in different orders.
                # By picking v from candidates, and new_candidates = {u in candidates | u > v and u in adj[v]}, we enforce an ordering.
                
                # Let's sort candidates once.
                # And in backtrack, we iterate through candidates.
                # But we need to ensure we don't pick a vertex that is "smaller" than a previously picked one if we want to avoid permutations.
                # Actually, if we just pick any v from candidates, and remove it, we cover all subsets.
                # To avoid permutations, we can enforce that v is the largest element in the new clique?
                # Or simply: new_candidates = [u for u in candidates if u in adj[v] and u > v]
                # This requires candidates to be sorted.
                
                new_candidates = [u for u in candidates if u in adj[v] and u > v]
                
                result = backtrack(new_clique, new_candidates)
                if result:
                    return result
            return None

        # Initial candidates: 1..max_search_val
        # Sort by degree descending to find large cliques faster?
        # Or just 1..max_search_val.
        # If we use u > v, we need to iterate in increasing order?
        # If we pick v=10, new_candidates are >10.
        # If we pick v=2, new_candidates are >2.
        # This covers all sets {v1, v2, ...} with v1 < v2 < ...
        
        initial_candidates = list(range(1, max_search_val + 1))
        # To find a clique, starting with small numbers is good if they are dense.
        # Let's try.
        
        return backtrack([], initial_candidates)

    best_set = None
    best_score = float('inf')
    
    # 1. Clique Search
    # Run with a strict time limit for this phase
    clique_deadline = start_time + 20.0
    old_timeout = timeout
    timeout = clique_deadline # Temporarily tighten
    
    result = find_clique_7_smooth()
    timeout = old_timeout # Restore
    
    if result:
        # Verify
        valid = True
        for i in range(len(result)):
            for j in range(i+1, len(result)):
                if not is_smooth(result[i]+result[j], target_primes):
                    valid = False
                    break
            if not valid: break
        if valid:
            return result

    # 2. Local Search for general minimum primes
    # If 7-smooth failed, we minimize the count.
    
    def get_primes(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors

    def score_set(A):
        primes = set()
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                primes.update(get_primes(A[i] + A[j]))
        return len(primes)

    # Initial set
    current_A = list(range(1, 13))
    current_score = score_set(current_A)
    best_set = current_A[:]
    best_score = current_score
    
    # Local Search
    # We will use random restarts.
    # To make it faster, we limit the search space to small numbers (1..200)
    # because small numbers tend to have smooth sums.
    
    for restart in range(100):
        if time.time() - start_time > timeout - 5:
            break
            
        if restart == 0:
            A = current_A[:]
        else:
            # Random restart
            A = random.sample(range(1, 201), 12)
            A.sort()
            
        score = score_set(A)
        
        if score < best_score:
            best_score = score
            best_set = A[:]
            
        no_improve = 0
        # Hill climbing
        while no_improve < 200 and time.time() - start_time < timeout - 2:
            k = random.randint(0, 11)
            old_val = A[k]
            improved = False
            
            # Try replacing A[k] with a random value
            for _ in range(20):
                new_val = random.randint(1, 200)
                if new_val in A and new_val != old_val:
                    continue
                if new_val == old_val:
                    continue
                
                A[k] = new_val
                new_score = score_set(A)
                
                if new_score < score:
                    score = new_score
                    improved = True
                    break
                elif new_score == score:
                    # Accept with low probability to escape local minima
                    if random.random() < 0.05:
                        improved = True
                        break
                    else:
                        A[k] = old_val
                else:
                    A[k] = old_val
            
            if improved:
                no_improve = 0
                if score < best_score:
                    best_score = score
                    best_set = A[:]
            else:
                no_improve += 1
                
    return best_set
