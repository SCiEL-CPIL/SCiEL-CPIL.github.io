def construct():
    from itertools import combinations
    from math import gcd
    import random
    import time
    import sys

    n = 128
    nums = list(range(1, n + 1))
    
    # Precompute gcd table
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd(i, j)

    # Precompute forbidden triples and conflict structures
    # conflicts[v][w] = set of u such that {v, w, u} is a forbidden triple
    # We use 0-indexed or 1-indexed? 1-indexed is easier for the final output.
    # conflicts is a dict of dicts or list of lists of sets.
    conflicts = [[set() for _ in range(n + 1)] for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if gcd_table[a][b] == 1:
                continue # Optimization: if gcd is 1, it can't be equal to other gcds unless they are 1 too?
                # Actually, if gcd(a,b)=1, then gcd(a,c) must be 1 and gcd(b,c) must be 1 for the triple to be forbidden.
                # This implies a, b, c are pairwise coprime.
                # Many triples are forbidden if they are pairwise coprime.
            
            g_ab = gcd_table[a][b]
            for c in range(b + 1, n + 1):
                if gcd_table[a][c] == g_ab and gcd_table[b][c] == g_ab:
                    # Triple (a, b, c) is forbidden
                    conflicts[a][b].add(c)
                    conflicts[b][a].add(c)
                    conflicts[a][c].add(b)
                    conflicts[c][a].add(b)
                    conflicts[b][c].add(a)
                    conflicts[c][b].add(a)

    best_size = 0
    best_set = []
    
    def greedy_initial():
        """Find a good initial solution using greedy."""
        S = set()
        C = set(nums)
        # Order by number of conflicts in the full set?
        # Or just random shuffle and greedy pick.
        # Let's try a few random orderings and keep the best.
        max_iter = 20
        best_greedy = set()
        
        for _ in range(max_iter):
            order = nums[:]
            random.shuffle(order)
            curr_S = set()
            curr_C = set(nums)
            for v in order:
                if v not in curr_C:
                    continue
                # Check if v can be added
                valid = True
                for u in curr_S:
                    if u in conflicts[v]: # This is not enough, need pair check
                        pass
                # Correct check:
                # v is valid if no pair (u, w) in curr_S is in conflicts[v][?].
                # Wait, my conflict structure is conflicts[v][u] = set of w.
                # So v is invalid if there exists u in curr_S such that some w in curr_S is in conflicts[v][u].
                # This is O(|S|^2).
                
                # Let's use the pair-based check for safety and simplicity in greedy, 
                # but for B&B we need the fast propagation.
                
                # For greedy, let's just do the O(S^2) check.
                can_add = True
                for u in curr_S:
                    if curr_S.intersection(conflicts[v][u]):
                        can_add = False
                        break
                
                if can_add:
                    curr_S.add(v)
            
            if len(curr_S) > len(best_greedy):
                best_greedy = curr_S
                
        return best_greedy

    # Get initial lower bound
    initial_sol = greedy_initial()
    best_size = len(initial_sol)
    best_set = list(initial_sol)

    # Branch and Bound
    # State: S (set), C (set)
    # To speed up, we represent C as a set.
    
    def branch_and_bound(S, C):
        nonlocal best_size, best_set
        
        if len(S) + len(C) <= best_size:
            return
        
        if not C:
            if len(S) > best_size:
                best_size = len(S)
                best_set = list(S)
            return
        
        # Heuristic for selecting next vertex
        # Pick v in C with maximum "pressure"
        # Pressure = number of candidates in C that would be removed if we add v?
        # Or number of conflicts?
        
        # Let's pick v that maximizes the number of elements in C it conflicts with via S.
        # If we add v, we remove R(v) = union_{u in S} conflicts[v][u] intersect C.
        # We want to prune fast. Adding a v with large R(v) reduces C significantly.
        # Excluding v just removes v.
        # Usually, we branch on the vertex that leads to the most pruning.
        
        # Calculate R(v) for all v in C? That might be expensive.
        # Let's just pick the first one or one with high degree in a heuristic graph.
        
        # Simple heuristic: Pick v with smallest index? No.
        # Pick v with largest len(conflicts[v]) in the original graph?
        
        # Let's try to pick v such that it conflicts with many elements in C via S.
        # If S is empty, R(v) is empty. So we just pick any.
        # If S is not empty, we compute.
        
        # Optimization: If S is empty, we can just pick any.
        # To keep it simple and fast, we'll iterate through C and pick one.
        # A better heuristic: Pick v that has the most neighbors in C in the "conflict graph" induced by S?
        
        # Let's just pick an arbitrary v from C for now, or the one with max conflicts in original graph.
        # Precomputed degrees:
        # degree[v] = sum(len(conflicts[v][u]) for u in range(1, n+1))
        
        # Let's just pick the first element in C to start, or randomize?
        # Randomization helps avoid getting stuck in local optima of the search tree.
        # But B&B is deterministic if order is fixed.
        
        # Let's sort C by a heuristic score and pick the best.
        # Score(v) = number of u in C such that u is in conflicts[v][w] for some w in S.
        # This is the size of R(v).
        
        # If S is empty, Score is 0 for all. Pick arbitrary.
        
        # To avoid O(|C| * |S| * |conflicts|) overhead, we can just pick one.
        # Let's pick the one with the highest static degree.
        
        # Static degree precomputation
        static_degree = [0] * (n + 1)
        for v in range(1, n + 1):
            for u in range(1, n + 1):
                if u != v:
                    static_degree[v] += len(conflicts[v][u])
        
        # Select v with max static_degree from C
        v = max(C, key=lambda x: static_degree[x])
        
        # Branch 1: Exclude v
        C_excl = C - {v}
        branch_and_bound(S, C_excl)
        
        # Branch 2: Include v
        # Check if v is valid to add to S
        # v is valid if no pair in S conflicts with v.
        # i.e., for all u in S, S.intersection(conflicts[v][u]) is empty.
        valid = True
        for u in S:
            if S.intersection(conflicts[v][u]):
                valid = False
                break
        
        if valid:
            S_incl = S | {v}
            # Calculate new C
            # Remove v
            # Remove any u in C such that u is in conflicts[v][w] for some w in S (old S)
            # Note: We use old S for conflicts because v is being added.
            # The new S is S ∪ {v}.
            # The candidates that become invalid are those u in C such that there exists w in S (old)
            # where {v, u, w} is forbidden.
            # This is exactly R(v) = union_{w in S} conflicts[v][w].
            
            to_remove = set()
            for w in S:
                to_remove.update(conflicts[v][w])
            
            C_incl = C_excl - to_remove
            
            branch_and_bound(S_incl, C_incl)
            
    # Start B&B
    # Initial S is empty, C is all nums
    # But we have a good lower bound.
    # We can start with S = initial_sol? No, B&B explores from scratch or extends.
    # If we start with S=empty, it might take long to reach the good size.
    # We can initialize the search with the greedy solution as a "root" of the include branches?
    # No, standard B&B starts from empty.
    
    # To speed up, we can use the greedy solution to prune.
    # If we start with S=empty, C=all, the bound is 128.
    
    # Let's run B&B with a time limit.
    start_time = time.time()
    max_time = 55.0
    
    # We need to handle recursion depth or use iterative?
    # Depth is max 128. Recursion is fine.
    
    # To prevent hitting time limit in the middle of a deep branch,
    # we check time in the loop.
    
    # Since Python recursion limit might be an issue for deep trees (though 128 is small),
    # and time limit is the main constraint.
    
    # We'll wrap the call in a try block or just check time at the start of the function.
    
    def bnb_wrapper(S, C):
        nonlocal best_size, best_set
        if time.time() - start_time > max_time:
            return
        branch_and_bound(S, C)
        
    bnb_wrapper(set(), set(nums))
    
    return best_set
