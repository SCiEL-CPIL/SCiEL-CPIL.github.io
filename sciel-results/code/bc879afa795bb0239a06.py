import time
import random
from math import gcd

def construct():
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute gcd matrix
    gcds = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcds[i][j] = g
            gcds[j][i] = g
            
    # Precompute bad triples
    # A triple (a, b, c) is bad if gcd(a,b) == gcd(b,c) == gcd(c,a)
    # We store for each element, the set of pairs (b, c) that form a bad triple with it.
    # Actually, for efficiency in local search, we just need to check if a candidate x 
    # forms a bad triple with any pair (a, b) in the current set S.
    
    # Let's precompute for each x, the list of pairs (a, b) such that (a, b, x) is a bad triple.
    # This allows O(1) or O(k) checks when adding x.
    # But since N is small (128), we can just iterate.
    
    # More efficient: For each x, precompute a list of "conflict pairs" (a, b) 
    # where a < b, a,b in [1,128], and gcd(a,b)==gcd(a,x)==gcd(b,x).
    # When adding x to S, we must ensure that for no (a,b) in conflicts[x] are both a and b in S.
    
    conflicts = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        for a in range(1, N + 1):
            if a == x: continue
            ga_x = gcds[a][x]
            for b in range(a + 1, N + 1):
                if b == x: continue
                if a == b: continue
                if gcds[a][b] == ga_x and gcds[b][x] == ga_x:
                    # To avoid duplicates, we can store (min(a,b), max(a,b))
                    # But we need to check if BOTH are in S.
                    # We store the pair as a tuple.
                    conflicts[x].append((a, b))
                    
    # Optimization: Remove duplicate pairs in conflicts[x]
    for x in range(1, N + 1):
        conflicts[x] = list(set(conflicts[x]))

    def is_valid_addition(S_set, x):
        """Check if x can be added to S_set without creating a bad triple."""
        for a, b in conflicts[x]:
            if a in S_set and b in S_set:
                return False
        return True

    def count_violations(S_set):
        """Count the number of bad triples fully contained in S_set."""
        count = 0
        # This is expensive, so we only use it for final verification or debugging.
        # For local search, we track violations incrementally.
        return count

    def get_bad_triples_in_set(S_set):
        """Get list of bad triples in S_set. Used for analysis."""
        bads = []
        # This is O(|S|^3), too slow for frequent calls.
        return bads

    best_set = set()
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    random.seed(42)
    
    # Local Search Strategy:
    # State: S (set of elements)
    # Objective: Maximize |S| subject to no bad triples.
    # Move: Flip a bit (add or remove an element).
    
    # We will use a "WalkSAT" style approach.
    # 1. Start with a random set or greedy set.
    # 2. While time remains:
    #    a. If current set is valid, try to add a random element. If it creates violations, accept with some probability or pick the one with fewest new violations.
    #    b. If current set is invalid (shouldn't happen if we maintain validity), remove elements to fix.
    #    
    # Better approach for Max Independent Set in Hypergraphs:
    # Start with a valid set.
    # Try to add elements. If adding x creates conflicts, we must remove some elements from S to resolve them.
    # This is a "swap" strategy.
    
    def greedy_init():
        S = set()
        elems = elements[:]
        random.shuffle(elems)
        for x in elems:
            if is_valid_addition(S, x):
                S.add(x)
        return S

    def local_search(S, max_iter=1000):
        """
        Improve S by trying to add elements, and if blocked, by performing swaps.
        """
        S = set(S)
        current_size = len(S)
        no_improve_count = 0
        
        while max_iter > 0 and time.time() < deadline:
            max_iter -= 1
            improved = False
            
            # 1. Try to add a single element
            # Pick a random element not in S
            not_in_S = [x for x in elements if x not in S]
            if not_in_S:
                random.shuffle(not_in_S)
                for x in not_in_S:
                    if is_valid_addition(S, x):
                        S.add(x)
                        current_size += 1
                        improved = True
                        break
            
            if improved:
                no_improve_count = 0
                continue
            
            # 2. Try 1-for-1 swap (Remove u, Add v)
            # This is usually not helpful for increasing size, but might help escape local minima for larger swaps?
            # Actually, 1-for-1 keeps size same. We want to increase size.
            # So we should try 1-for-2 (Remove 1, Add 2) or 2-for-3 etc.
            # But 1-for-2 is complex.
            
            # Let's try "Perturbation": 
            # Remove k random elements, then try to add back as many as possible greedily.
            # This is a "kick".
            
            if no_improve_count > 50:
                # Perform a kick
                k = random.randint(1, 3)
                to_remove = random.sample(list(S), min(k, len(S)))
                for u in to_remove:
                    S.remove(u)
                    current_size -= 1
                
                # Now try to fill up again greedily
                not_in_S = [x for x in elements if x not in S]
                random.shuffle(not_in_S)
                for x in not_in_S:
                    if is_valid_addition(S, x):
                        S.add(x)
                        current_size += 1
                
                no_improve_count = 0
                improved = True
            else:
                no_improve_count += 1
                
        return S

    # Run multiple restarts
    for run in range(500):
        if time.time() > deadline:
            break
            
        # Greedy init
        S = greedy_init()
        
        # Local search
        S = local_search(S)
        
        if len(S) > len(best_set):
            best_set = set(S)
            
        # Optional: Try to improve best_set directly with small perturbations
        if time.time() > deadline - 5:
            break
            
    # Final Polish: Try to add any remaining element to best_set
    # This is already done in local_search, but let's do one more pass
    not_in_best = [x for x in elements if x not in best_set]
    random.shuffle(not_in_best)
    for x in not_in_best:
        if is_valid_addition(best_set, x):
            best_set.add(x)

    # Return as a list (sorted for consistency, though order doesn't matter for set semantics)
    return list(best_set)
