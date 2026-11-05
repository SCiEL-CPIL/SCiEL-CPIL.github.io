import sys
import random

def construct():
    n = 192
    candidates = list(range(n, 0, -1)) # Descending order: large numbers first
    
    best_set = []
    
    def is_compatible(x, S):
        """
        Check if x can be added to S.
        S is a list of integers.
        Condition 1: x does not divide sum of any 3 distinct elements in S.
        Condition 2: No y in S divides (x + a + b) for any distinct a, b in S \ {y}.
        """
        S_len = len(S)
        
        # Condition 1: x divides sum of 3 others in S
        if S_len >= 3:
            # Optimization: If x is large, sums are small.
            # We can just iterate.
            for i in range(S_len):
                a = S[i]
                for j in range(i + 1, S_len):
                    b = S[j]
                    for k in range(j + 1, S_len):
                        c = S[k]
                        if (a + b + c) % x == 0:
                            return False
        
        # Condition 2: y in S divides x + a + b
        # a, b are distinct from y and from each other, and from x (x is not in S yet)
        for i in range(S_len):
            y = S[i]
            # We need to pick 2 distinct elements a, b from S \ {y}
            # Such that (x + a + b) % y == 0
            # This implies x + a + b = m * y
            # a + b = m * y - x
            
            # Iterate over pairs in S excluding y
            # To optimize, we can precompute sums? No, S changes.
            # But S_len is small (< 50).
            others = [v for v in S if v != y]
            if len(others) < 2:
                continue
            
            # Check all pairs in others
            for a_idx in range(len(others)):
                a = others[a_idx]
                for b_idx in range(a_idx + 1, len(others)):
                    b = others[b_idx]
                    if (x + a + b) % y == 0:
                        return False
        
        return True

    def greedy_upper_bound(S, remaining_candidates):
        """
        Greedily add from remaining_candidates to S to estimate max size.
        Returns the size of the resulting set.
        """
        S_copy = S[:]
        for x in remaining_candidates:
            if is_compatible(x, S_copy):
                S_copy.append(x)
        return len(S_copy)

    def search(idx, current_set):
        nonlocal best_set
        
        # Pruning 1: Simple count
        if len(current_set) + (len(candidates) - idx) <= len(best_set):
            return
            
        # Pruning 2: Greedy bound
        # This is expensive. Only do it if we are close to the best?
        # Or do it at shallow depths?
        # Let's do a cheap bound first: count compatible candidates.
        compat_count = 0
        remaining = candidates[idx:]
        for x in remaining:
            if is_compatible(x, current_set):
                compat_count += 1
                if len(current_set) + compat_count <= len(best_set):
                    break
        if len(current_set) + compat_count <= len(best_set):
            return
            
        # If idx is out of range, update best
        if idx == len(candidates):
            if len(current_set) > len(best_set):
                best_set = current_set[:]
            return
        
        x = candidates[idx]
        
        # Option 1: Include x
        if is_compatible(x, current_set):
            current_set.append(x)
            search(idx + 1, current_set)
            current_set.pop()
            
        # Option 2: Exclude x
        search(idx + 1, current_set)

    # Initial Greedy to get a good lower bound
    # Try to build a set greedily
    initial_set = []
    for x in candidates:
        if is_compatible(x, initial_set):
            initial_set.append(x)
    best_set = initial_set[:]
    
    # Run B&B
    # To handle time limits, we might need to limit recursion depth or use a timeout.
    # But 192 is small enough if pruning is good.
    # The "compat_count" pruning is O(N * CostCheck).
    # CostCheck is O(K^3) where K is current set size.
    # If K=40, CostCheck ~ 64k.
    # N=192. 192 * 64k = 12M ops per node.
    # This is too slow for many nodes.
    
    # We need a faster is_compatible or better pruning.
    
    # Let's optimize is_compatible.
    # Precompute for each y, the set of pairs (a,b) such that a+b is a multiple of y?
    # No, x changes.
    
    # Let's try a different search order or heuristic.
    # The current approach might be too slow.
    # Let's try to improve the greedy initial solution first.
    
    # Randomized Greedy
    best_greedy = initial_set[:]
    for _ in range(10):
        random.shuffle(candidates)
        S = []
        for x in candidates:
            if is_compatible(x, S):
                S.append(x)
        if len(S) > len(best_greedy):
            best_greedy = S[:]
            
    best_set = best_greedy[:]
    
    # Now B&B with a time limit check?
    # Python doesn't have easy time limits in recursion without signals.
    # We can count nodes and stop if too many.
    
    node_count = 0
    MAX_NODES = 50000 # Adjust based on speed
    
    def search_limited(idx, current_set):
        nonlocal best_set, node_count
        node_count += 1
        if node_count > MAX_NODES:
            return
            
        if len(current_set) + (len(candidates) - idx) <= len(best_set):
            return
            
        # Fast prune: check if any remaining is compatible
        # If none are compatible, we can't add more.
        # But we might have already added some.
        
        if idx == len(candidates):
            if len(current_set) > len(best_set):
                best_set = current_set[:]
            return
            
        x = candidates[idx]
        
        # Include x
        if is_compatible(x, current_set):
            current_set.append(x)
            search_limited(idx + 1, current_set)
            current_set.pop()
            
        # Exclude x
        search_limited(idx + 1, current_set)

    search_limited(0, [])
    
    return best_set
