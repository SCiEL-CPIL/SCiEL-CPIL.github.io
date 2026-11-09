"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    n = 256
    
    def is_feasible(S):
        """Check if set S satisfies the property: for all a<b<=c in S, a does not divide b+c."""
        S_sorted = sorted(S)
        k = len(S_sorted)
        for i in range(k):
            a = S_sorted[i]
            for j in range(i+1, k):
                b = S_sorted[j]
                for k2 in range(j, k):
                    c = S_sorted[k2]
                    if (b + c) % a == 0:
                        return False
        return True

    def can_add(S_set, x):
        """
        Check if adding x to set S_set keeps it valid.
        S_set is a Python set of integers.
        Only checks triples involving x, which is sufficient if S_set is already valid.
        """
        if x in S_set:
            return True
        
        S_list = sorted(S_set)
        
        # Case 1: x is the smallest element (a=x)
        # Need b, c in S_set such that x < b <= c and x divides (b+c)
        for b in S_list:
            if b <= x:
                continue
            for c in S_list:
                if c < b:
                    continue
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is the middle element (b=x)
        # Need a, c in S_set such that a < x <= c and a divides (x+c)
        # Note: a must be in S_set, a < x. c must be in S_set, c >= x.
        for a in S_list:
            if a >= x:
                continue
            for c in S_list:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # Case 3: x is the largest element (c=x)
        # Need a, b in S_set such that a < b <= x and a divides (b+x)
        # Note: a, b in S_set, a < b <= x.
        for a in S_list:
            for b in S_list:
                if b <= a:
                    continue
                if b > x:
                    continue
                if (b + x) % a == 0:
                    return False
        
        return True

    def greedy_add(S_set, candidates, order):
        """
        Greedily add elements from candidates in the specified order to S_set.
        Returns the modified set.
        """
        S = set(S_set)
        for x in order:
            if time.time() > deadline - 2:
                break
            if x in S:
                continue
            if can_add(S, x):
                S.add(x)
        return S

    def local_search_add(S_set, max_iter=50):
        """
        Try to add any missing element that fits.
        Repeats until no more elements can be added or time limit is reached.
        """
        S = set(S_set)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for x in range(1, n + 1):
                if time.time() > deadline - 2:
                    break
                if x in S:
                    continue
                if can_add(S, x):
                    S.add(x)
                    improved = True
        return S

    # Strategy 1: Start with upper tail [171, 256]
    # This is known to be feasible.
    # Size = 256 - 171 + 1 = 86
    
    base_set = set(range(171, n + 1))
    
    # Try different orders of adding remaining elements (1 to 170)
    candidates = list(range(1, 171))
    
    orders = [
        candidates,              # Ascending
        list(reversed(candidates)), # Descending
    ]
    
    # Also try a "larger first" heuristic within the lower range?
    # The descending order is already "larger first" relative to the range 1-170.
    
    best_set = base_set
    best_size = len(base_set)
    
    for order in orders:
        if time.time() > deadline - 10:
            break
        S = set(base_set)
        S = greedy_add(S, candidates, order)
        # Perform a final local search to see if any other elements can be added
        S = local_search_add(S, max_iter=10)
        
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
    
    # Try starting from a slightly different base?
    # [170, 256] was found infeasible in comments.
    # Let's try [172, 256] + greedy?
    # Or maybe just stick to the robust [171, 256] base which is proven feasible.
    
    # Let's try one more strategy: Greedy from scratch (small to large)
    # This might find a different structure.
    if time.time() < deadline - 15:
        S_greedy = set()
        for x in range(1, n + 1):
            if time.time() > deadline - 5:
                break
            if can_add(S_greedy, x):
                S_greedy.add(x)
        S_greedy = local_search_add(S_greedy, max_iter=10)
        if len(S_greedy) > best_size:
            best_set = S_greedy
            best_size = len(S_greedy)
            
    # Let's try Greedy from scratch (large to small)
    if time.time() < deadline - 15:
        S_greedy_large = set()
        for x in range(n, 0, -1):
            if time.time() > deadline - 5:
                break
            if can_add(S_greedy_large, x):
                S_greedy_large.add(x)
        S_greedy_large = local_search_add(S_greedy_large, max_iter=10)
        if len(S_greedy_large) > best_size:
            best_set = S_greedy_large
            best_size = len(S_greedy_large)

    # Final verification
    if not is_feasible(best_set):
        # Fallback to known feasible set
        best_set = set(range(171, n + 1))
        
    if not best_set:
        best_set = set(range(171, n + 1))
        
    return sorted(best_set)
# EVOLVE-BLOCK-END
