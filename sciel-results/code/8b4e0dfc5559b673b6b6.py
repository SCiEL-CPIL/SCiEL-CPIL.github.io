"""Rewritten program: greedy + local search for max 3-uniform constraint set."""

def construct():
    import time
    import random
    from math import isqrt
    
    deadline = time.time() + 85  # safety margin before 90s kill
    
    N = 256
    candidates = list(range(2, N + 1))
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    def valid_set(S):
        """Check if S satisfies the constraint: for all distinct a,b,c in S, a ∤ b*c."""
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(n):
                if j == i:
                    continue
                b = S_list[j]
                for k in range(n):
                    if k == i or k == j:
                        continue
                    c = S_list[k]
                    if (b * c) % a == 0:
                        return False
        return True
    
    def can_add(S_set, x):
        """Check if adding x to S_set keeps the constraint valid."""
        S_list = list(S_set)
        for i in range(len(S_list)):
            a = S_list[i]
            # Check a ∤ x * c for all c in S_set, c != a, c != x
            for j in range(len(S_list)):
                if j == i:
                    continue
                c = S_list[j]
                if (x * c) % a == 0:
                    return False
            # Check x ∤ a * c for all c in S_set, c != a, c != x
            for j in range(len(S_list)):
                if j == i:
                    continue
                c = S_list[j]
                if (a * c) % x == 0:
                    return False
        # Also need to check: for any two existing elements b,c, b ∤ x*c and c ∤ x*b
        # This is already covered above by the first two loops
        return True
    
    # Start with all primes
    primes = [a for a in range(2, N + 1) if is_prime(a)]
    best_set = set(primes)
    best_size = len(primes)
    
    # Greedy: try to add composites in various orders
    composites = [a for a in range(2, N + 1) if not is_prime(a)]
    
    # Try multiple random orderings
    for trial in range(50):
        if time.time() > deadline:
            break
        
        order = composites[:]
        random.shuffle(order)
        
        current = set(primes)
        
        for x in order:
            if x in current:
                continue
            if can_add(current, x):
                current.add(x)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current.copy()
    
    # Local search: try 1-for-1 swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_list = list(best_set)
        for i in range(len(best_list)):
            if time.time() > deadline:
                break
            a = best_list[i]
            for x in range(2, N + 1):
                if x in best_set:
                    continue
                # Try swapping a for x
                trial_set = set(best_set)
                trial_set.discard(a)
                trial_set.add(x)
                if can_add(trial_set - {x}, x):
                    # Full validation
                    if valid_set(trial_set):
                        if len(trial_set) > len(best_set):
                            best_set = trial_set
                            best_size = len(best_set)
                            improved = True
                            break
            if improved:
                break
    
    return list(best_set)
