import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Try different base bipartite graphs
    # K_{5,11} has 55 edges.
    # K_{4,12} has 48 edges.
    # K_{6,10} has 60 edges.
    
    best_edges = 39 # Seed
    best_graph = None
    
    # We will search in K_{5,11}
    # A = {0,1,2,3,4}, B = {5,6,7,8,9,10,11,12,13,14,15}
    # Actually, let's just use indices 0..15.
    # A = 0..4, B = 5..15
    
    n_a = 5
    n_b = 11
    A = list(range(n_a))
    B = list(range(n_a, n_a + n_b))
    
    # Represent removed edges as bitmasks or sets
    # T[i] is set of neighbors in B removed for A[i]
    
    def count_violations(T):
        # T is list of sets
        # Check all A' (size 4) and B' (size 4)
        # Optimization: Precompute for each B' the intersection sizes with each T[i]
        # But 330 B' is small.
        violations = 0
        # Iterate over all subsets of B of size 4
        # There are 330 such subsets.
        # For each, check if there exists A' (5 choices) such that condition holds.
        # Condition: For a fixed B', if for ALL A' (size 4), there is NO a in A' with |T[a] intersect B'| >= 2, then it's a violation.
        # Wait, the condition for validity is:
        # For EVERY A', B', there EXISTS a in A' with |T[a] intersect B'| >= 2.
        # So a violation is: There EXISTS A', B' such that FOR ALL a in A', |T[a] intersect B'| <= 1.
        
        # Precompute intersections for each B'
        # To speed up, we can just iterate.
        
        from itertools import combinations
        
        # Map B index to 0..10 for local indexing
        # T[i] contains global B indices.
        
        # For each B' (330)
        for b_prime in combinations(B, 4):
            b_set = set(b_prime)
            
            # Check if there is a "bad" A'
            # A' is bad if for all a in A', |T[a] & b_set| <= 1
            # We need to check if ALL A' are bad.
            # If even one A' is GOOD (has some a with >=2), then this B' is safe.
            # So we only have a violation if ALL 5 A' are bad.
            
            all_bad = True
            # There are 5 A' subsets (remove one element from A)
            for i in range(n_a):
                # A' is A \ {i}
                # Check if A' is bad
                a_prime_bad = True
                for j in range(n_a):
                    if j == i: continue
                    inter = len(T[j] & b_set)
                    if inter >= 2:
                        a_prime_bad = False
                        break
                if not a_prime_bad:
                    # This A' is good (has a vertex with >=2)
                    # So this B' is covered by this A'
                    # We don't need to check other A' for this B' to see if it's a violation?
                    # No, the condition is: For EVERY A', B', there exists a...
                    # So if for a specific B', there is ONE A' that is good, that B' is NOT a violation source for that A'.
                    # But we need to ensure that for EVERY pair (A', B'), the condition holds.
                    # So if we find one A' that is good for this B', that specific pair (A', B') is valid.
                    # But we need to check if there is ANY pair (A'', B') that is invalid.
                    # So we must check if there exists an A' that is BAD for this B'.
                    # If there exists an A' that is BAD, then the pair (A', B') is a violation.
                    # So:
                    # Violation exists if: Exists B' such that Exists A' such that A' is bad for B'.
                    
                    # My logic above was: "All A' are bad".
                    # Correct logic: If ANY A' is bad, we have a violation.
                    # So we just need to find if there is an A' that is bad.
                    pass 
            # Re-evaluate logic:
            # Valid Graph: For all A', B', (Exists a in A' s.t. |T[a] & B'| >= 2).
            # Invalid Graph: Exists A', B' s.t. (For all a in A', |T[a] & B'| <= 1).
            
            # So for a fixed B', we check if there is an A' that is "bad".
            # If yes, violation++.
            
            has_bad_a_prime = False
            for i in range(n_a):
                # A' = A \ {i}
                a_prime_bad = True
                for j in range(n_a):
                    if j == i: continue
                    if len(T[j] & b_set) >= 2:
                        a_prime_bad = False
                        break
                if a_prime_bad:
                    has_bad_a_prime = True
                    break
            
            if has_bad_a_prime:
                violations += 1
        
        return violations

    # Local Search
    # Start with T = [set() for _ in range(n_a)]
    # This has many violations.
    # We want to minimize sum(len(T[i])) subject to violations == 0.
    
    # Heuristic: Add elements to T[i] to reduce violations.
    # Or start with a dense T and remove?
    # Start with T[i] = set(B) for all i. Violations = 0. Edges = 0.
    # Remove elements to increase edges, as long as violations stay 0.
    
    T = [set(B) for _ in range(n_a)]
    current_violations = 0
    current_edges = 0 # 5*11 - sum(len(T)) = 0
    
    # Shuffle removal order
    # List of all possible edges (i, b)
    all_edges = [(i, b) for i in range(n_a) for b in B]
    random.shuffle(all_edges)
    
    for i, b in all_edges:
        if time.time() > deadline:
            break
            
        # Try removing b from T[i]
        T[i].remove(b)
        
        # Recalculate violations?
        # This is expensive (330 * 5 * 4 ops).
        # 330 * 20 = 6600 ops per check.
        # 55 edges to try. 55 * 6600 = 363,000 ops. Very fast.
        
        v = count_violations(T)
        
        if v == 0:
            current_edges += 1
        else:
            T[i].add(b)
            current_edges -= 1 # restore state (logically, we didn't add edge)
            # Actually, we just reverted.
            
    # Result
    edges_list = []
    for i in range(n_a):
        for b in B:
            if b not in T[i]:
                edges_list.append([i, b])
    
    if len(edges_list) > best_edges:
        best_edges = len(edges_list)
        best_graph = edges_list
        
    # Try K_{4,12} as well?
    # Maybe K_{5,11} is better.
    
    return best_graph

print(construct())
