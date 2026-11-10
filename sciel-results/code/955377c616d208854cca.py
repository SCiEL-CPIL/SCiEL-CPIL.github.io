# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations

def construct():
    n = 192
    start_time = time.time()
    time_limit = 55.0  # Leave 5s for cleanup/return
    
    # Precompute for fast checks
    # We will use a list of selected elements S.
    
    def is_valid_addition(a, S):
        """
        Check if adding a to S maintains the constraint.
        S is a list of integers.
        """
        S_set = set(S)
        
        # 1. Check constraint for a:
        # No 3 distinct elements in S sum to a multiple of a.
        if a == 1:
            if len(S) >= 3:
                return False
        else:
            # Optimization: If min sum of 3 in S > 6*a, then no multiple can be formed?
            # Min sum is sum of 3 smallest. Max sum is sum of 3 largest.
            # Multiples are a, 2a, ..., floor(max_sum/a)*a.
            if len(S) >= 3:
                # Use combinations for small S
                # To speed up, we can break early
                # Sort S for min/max sum check?
                # For now, just iterate.
                for b, c, d in combinations(S, 3):
                    if (b + c + d) % a == 0:
                        return False
        
        # 2. Check constraint for existing b in S:
        # Adding a creates new triples (a, x, y) where x, y in S \ {b}.
        # We need b | (a + x + y).
        for b in S:
            if b == 1:
                # 1 divides everything.
                # We need 3 distinct OTHER elements for b.
                # The triple is (a, x, y).
                # Are a, x, y distinct from b? Yes, a not in S, x,y in S\{b}.
                # Are a, x, y distinct from each other?
                # a is new, so a != x, a != y.
                # x, y are distinct.
                # So if there are at least 2 elements in S \ {b}, we have a conflict.
                if len(S) - 1 >= 2:
                    return False
            else:
                # Check pairs x, y in S \ {b}
                # We need (a + x + y) % b == 0
                # x + y == -a mod b
                # Optimization: Precompute residues?
                # For now, iterate.
                S_without_b = [v for v in S if v != b]
                # If len(S_without_b) < 2, no pair exists.
                if len(S_without_b) < 2:
                    continue
                
                # Vectorized check for this b
                # Create array of S_without_b
                arr = np.array(S_without_b)
                # We want to check if there exist i < j such that (a + arr[i] + arr[j]) % b == 0
                # This is equivalent to arr[i] + arr[j] == -a % b
                # Let target = (-a) % b
                # We can use a set of residues?
                # Since b is small (<192), we can use a boolean array of size b.
                residues = np.zeros(b, dtype=bool)
                # Wait, we need distinct x, y.
                # If we just check existence, we might pick the same element twice if residue count is 1?
                # No, if we iterate pairs, it's safe.
                # But vectorized pair check is tricky.
                # Let's just use the loop for correctness, it's fast enough for |S|~50.
                found = False
                for i in range(len(S_without_b)):
                    for j in range(i + 1, len(S_without_b)):
                        if (a + S_without_b[i] + S_without_b[j]) % b == 0:
                            found = True
                            break
                    if found:
                        break
                if found:
                    return False
        
        return True

    # Greedy Initialization
    def greedy_solve(order):
        S = []
        for a in order:
            if is_valid_addition(a, S):
                S.append(a)
        return S

    # Local Search Improvement
    def local_search(S):
        S = S[:]
        improved = True
        while improved:
            improved = False
            # Try to add any missing element
            for a in range(1, n + 1):
                if a not in S:
                    if is_valid_addition(a, S):
                        S.append(a)
                        improved = True
                        break
            if improved:
                continue
            
            # Try to swap: remove one, add two?
            # Or just remove one that allows adding two others.
            # Simplified: Try removing each element and see if we can add more.
            for i in range(len(S)):
                a = S[i]
                S_temp = S[:i] + S[i+1:]
                # Try to fill S_temp greedily
                added = 0
                for b in range(n, 0, -1):
                    if b not in S_temp:
                        if is_valid_addition(b, S_temp):
                            S_temp.append(b)
                            added += 1
                            if added > 1:
                                break
                if len(S_temp) > len(S):
                    S = S_temp
                    improved = True
                    break
        return S

    best_S = []
    best_size = 0
    
    # Try multiple orderings
    orders = [
        list(range(n, 0, -1)),
        list(range(1, n + 1)),
        list(range(n, 0, -1, 2)), # Odds first
        list(range(n-1, 0, -2)), # Evens first
    ]
    
    # Add some random orders
    random.seed(42)
    for _ in range(10):
        ord_list = list(range(1, n + 1))
        random.shuffle(ord_list)
        orders.append(ord_list)
    
    for order in orders:
        if time.time() - start_time > time_limit:
            break
        S = greedy_solve(order)
        S = local_search(S)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S[:]
            
    # If time remains, try a simple backtracking to see if we can beat best_size
    # This is a heuristic B&B
    def backtrack_search(idx, S, current_size):
        nonlocal best_S, best_size
        if time.time() - start_time > time_limit:
            return
        
        # Pruning
        if current_size + (n - idx + 1) <= best_size:
            return
        
        if idx == 0:
            if current_size > best_size:
                best_size = current_size
                best_S = S[:]
            return
        
        a = idx
        # Option 1: Include a
        if is_valid_addition(a, S):
            S.append(a)
            backtrack_search(idx - 1, S, current_size + 1)
            S.pop()
        
        # Option 2: Exclude a
        backtrack_search(idx - 1, S, current_size)

    # Run a limited backtracking from the best greedy solution?
    # Or just run a fresh one if time permits?
    # The greedy solution is likely close to optimal.
    # We can try to "repair" or "expand" best_S.
    
    # Let's try to add elements to best_S in a specific order if not already done.
    # The local_search already tries to add any element.
    
    return best_S
# EVOLVE-BLOCK-END
