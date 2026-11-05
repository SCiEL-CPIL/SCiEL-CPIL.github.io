import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 192
    candidates = list(range(1, n + 1))
    
    # Precompute: for each x, which triples (a,b,c) have (a+b+c) % x == 0
    # But that's expensive. Instead, check on the fly but efficiently.
    
    def is_valid_set(S):
        """Check if set S is valid."""
        S_set = set(S)
        for x in S:
            # Check x doesn't divide sum of any 3 distinct others
            others = [v for v in S if v != x]
            if len(others) >= 3:
                for a, b, c in combinations(others, 3):
                    if (a + b + c) % x == 0:
                        return False
        return True
    
    def can_add_fast(x, S, S_set):
        """Check if x can be added to S."""
        # 1. x does not divide sum of any 3 distinct others in S
        if len(S) >= 3:
            for a, b, c in combinations(S, 3):
                if (a + b + c) % x == 0:
                    return False
        
        # 2. No a in S divides x + b + c for distinct b, c in S \ {a}
        if len(S) >= 2:
            for a in S:
                others = [v for v in S if v != a]
                if len(others) < 2:
                    continue
                for b, c in combinations(others, 2):
                    if (x + b + c) % a == 0:
                        return False
        return True
    
    # Greedy construction with multiple strategies
    def greedy(order):
        S = []
        S_set = set()
        for x in order:
            if time.time() > deadline:
                break
            if x in S_set:
                continue
            if can_add_fast(x, S, S_set):
                S.append(x)
                S_set.add(x)
        return S
    
    # Local search: try to swap elements to improve
    def local_search(S, max_iterations=500):
        S = list(S)
        S_set = set(S)
        improved = True
        iterations = 0
        while improved and iterations < max_iterations and time.time() < deadline - 5:
            improved = False
            iterations += 1
            # Try removing one element and adding another
            for i, x in enumerate(S):
                if time.time() > deadline - 5:
                    break
                # Try swapping x with some y not in S
                for y in candidates:
                    if y in S_set:
                        continue
                    # Check if S - {x} + {y} is valid
                    S_new = S[:i] + S[i+1:]
                    S_new_set = S_set - {x}
                    
                    # Check if y can be added to S_new
                    if can_add_fast(y, S_new, S_new_set):
                        # Also need to verify the full set is valid (should be if can_add is correct)
                        S_new.append(y)
                        S_new_set.add(y)
                        S = S_new
                        S_set = S_new_set
                        improved = True
                        break
                if improved:
                    break
        return S
    
    best_set = []
    best_size = 0
    
    # Strategy 1: Descending order (large numbers first)
    order = list(range(n, 0, -1))
    S = greedy(order)
    S = local_search(S)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]
    
    # Strategy 2: Ascending
    order = list(range(1, n + 1))
    S = greedy(order)
    S = local_search(S)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]
    
    # Strategy 3: Multiple random orders
    random.seed(42)
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
        
        # Biased random: favor larger numbers
        if attempt % 3 == 0:
            # Pure random
            order = candidates[:]
            random.shuffle(order)
        elif attempt % 3 == 1:
            # Large first, shuffled
            large = [x for x in candidates if x > n // 2]
            small = [x for x in candidates if x <= n // 2]
            random.shuffle(large)
            random.shuffle(small)
            order = large + small
        else:
            # Weighted: higher numbers more likely early
            weights = [x ** 2 for x in candidates]
            order = candidates[:]
            random.shuffle(order)
        
        S = greedy(order)
        S = local_search(S, max_iterations=200)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
            if best_size == 48:
                break
    
    # Additional local search passes on best solution
    if time.time() < deadline - 5 and len(best_set) < 48:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i, x in enumerate(best_set):
                for y in candidates:
                    if y in set(best_set):
                        continue
                    S_new = best_set[:i] + best_set[i+1:] + [y]
                    if is_valid_set(S_new):
                        best_set = S_new
                        improved = True
                        break
                if improved:
                    break
    
    return best_set

