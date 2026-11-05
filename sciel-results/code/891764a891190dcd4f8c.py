"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
from math import gcd
from itertools import combinations

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    numbers = list(range(1, N + 1))
    
    # Precompute compatibility: compatible[a][b] = lcm(a,b) <= N
    compatible = [[False] * (N + 1) for _ in range(N + 1)]
    for a in numbers:
        for b in numbers:
            compatible[a][b] = (a * b // gcd(a, b)) <= N
    
    def validate(S):
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                if S_list[i] * S_list[j] // gcd(S_list[i], S_list[j]) > N:
                    return False
        return True
    
    def try_add(S, a):
        return all(compatible[a][b] for b in S)
    
    def greedy(order, S_initial=None):
        S = set(S_initial) if S_initial is not None else set()
        for a in order:
            if a not in S and all(compatible[a][b] for b in S):
                S.add(a)
        return S
    
    def local_search_add(S):
        S = set(S)
        improved = True
        while improved:
            improved = False
            for a in numbers:
                if a not in S and all(compatible[a][b] for b in S):
                    S.add(a)
                    improved = True
                    break
            if time.time() > deadline:
                break
        return S
    
    # Maximum Closure approach:
    # Key insight: The optimal set should be "downward closed" in some sense.
    # If a is in the set and d divides a, then d should also be compatible with all elements.
    # We can model this as a maximum weight closure problem.
    
    # For each number a, compute its "divisor closure" - all divisors of a
    divisor_closure = {}
    for a in numbers:
        divs = []
        for d in range(1, a + 1):
            if a % d == 0:
                divs.append(d)
        divisor_closure[a] = set(divs)
    
    # Precompute for each number a, the set of numbers incompatible with a
    incompatible = [set() for _ in range(N + 1)]
    for a in numbers:
        for b in numbers:
            if not compatible[a][b]:
                incompatible[a].add(b)
    
    # Strategy: Build a closure-based set
    # Start with a candidate "root" set, then close under divisors
    # and check compatibility
    
    def closure_search(initial_roots):
        """Build a set by taking initial roots and closing under divisors,
        then pruning to maintain pairwise compatibility."""
        S = set(initial_roots)
        
        # Close under divisors: if a is in S, add all divisors of a
        changed = True
        while changed:
            changed = False
            for a in list(S):
                for d in divisor_closure[a]:
                    if d not in S:
                        S.add(d)
                        changed = True
        
        # Now prune: remove elements that cause conflicts
        # Use a greedy pruning strategy
        S_list = sorted(S, key=lambda x: len(incompatible[x] & S), reverse=True)
        result = set()
        for a in S_list:
            if all(compatible[a][b] for b in result):
                result.add(a)
        
        # Try to add back any removed elements
        for a in S - result:
            if all(compatible[a][b] for b in result):
                result.add(a)
        
        return result
    
    best_S = set()
    
    # Strategy 1: Greedy with various orderings
    def conflict_degree(a):
        return sum(1 for b in numbers if not compatible[a][b])
    
    order_deg = sorted(numbers, key=conflict_degree)
    S1 = greedy(order_deg)
    S1 = local_search_add(S1)
    if len(S1) > len(best_S):
        best_S = S1
    
    # Order by value ascending
    S2 = greedy(numbers)
    S2 = local_search_add(S2)
    if len(S2) > len(best_S):
        best_S = S2
    
    # Strategy 2: Closure-based search with different root sets
    # Try roots based on numbers with fewest conflicts
    sorted_by_conflicts = sorted(numbers, key=conflict_degree)
    
    # Try different sizes of root sets
    for root_size in [1, 2, 3, 5, 8, 12]:
        if time.time() > deadline:
            break
        roots = sorted_by_conflicts[:root_size]
        S_closure = closure_search(roots)
        S_closure = local_search_add(S_closure)
        if len(S_closure) > len(best_S):
            best_S = S_closure
    
    # Strategy 3: Randomized closure search
    random.seed(42)
    for i in range(100):
        if time.time() > deadline:
            break
        # Random root set
        root_size = random.randint(1, 10)
        roots = random.sample(numbers, root_size)
        S_closure = closure_search(roots)
        S_closure = local_search_add(S_closure)
        if len(S_closure) > len(best_S):
            best_S = S_closure
    
    # Strategy 4: Advanced local search with swap operations
    def advanced_local_search(S):
        S = set(S)
        improved = True
        while improved:
            improved = False
            # Phase 1: Add any compatible element
            for a in numbers:
                if a not in S and all(compatible[a][b] for b in S):
                    S.add(a)
                    improved = True
                    break
            if time.time() > deadline:
                break
            
            # Phase 2: Try swapping one element for another to enable adding a third
            if not improved:
                for a in list(S):
                    for b in numbers:
                        if b in S:
                            continue
                        S_temp = (S - {a}) | {b}
                        # Check if S_temp is valid
                        if not all(compatible[x][y] for x in S_temp for y in S_temp):
                            continue
                        # Now try to add a third element c
                        for c in numbers:
                            if c in S_temp:
                                continue
                            if all(compatible[c][x] for x in S_temp):
                                S = S_temp | {c}
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            if time.time() > deadline:
                break
        return S
    
    # Apply advanced local search to best_S
    best_S = advanced_local_search(best_S)
    
    # Final polish: try to add any remaining numbers
    best_S = local_search_add(best_S)
    
    # Verify
    if not validate(best_S):
        # Fallback to a known valid set
        return list(range(1, 23))
    
    if not best_S:
        return [1]
        
    return sorted(best_S)

# EVOLVE-BLOCK-END
