"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 70.0  # safety margin before 90s wall-clock limit
    N = 192
    K = 3
    
    # Precompute conflict relationships
    # conflict[a] = set of triples (b,c,d) with b,c,d distinct from a and each other, a | (b+c+d)
    # But for efficiency, we'll check on the fly with caching
    
    # Better approach: precompute for each pair (a, s) whether s is divisible by a
    # where s is a sum of 3 distinct elements not including a
    
    # Let's use a different strategy:
    # For each element a, and for each subset S of other elements, check if any 3-element subset of S sums to a multiple of a.
    
    # Precompute: for each a in 1..192, and for each possible sum s (3 to 3*192=576), 
    # does a divide s? We can precompute divisibility.
    
    # Actually, let's think about this differently.
    # The constraint is: for selected set S, for every a in S, there is no 3-element subset {b,c,d} of S\{a} such that a | (b+c+d).
    
    # Strategy: Greedy with local search
    # 1. Start with a greedy construction (try different orderings)
    # 2. Try to add elements that were rejected
    # 3. Try swaps: remove one element, add another
    # 4. Repeat until no improvement or time runs out
    
    def is_valid_addition(a, selected_set):
        """Check if adding a to selected_set maintains validity."""
        # For the new element a: check that no 3 other elements sum to a multiple of a
        others = [x for x in selected_set if x != a]
        for b, c, d in combinations(others, 3):
            if (b + c + d) % a == 0:
                return False
        
        # For each existing element e in selected_set: check that no 3 elements from (selected_set - {e}) including a sum to a multiple of e
        # This means: for each e in selected_set, and for each pair (b,c) in selected_set - {e, a}, check if a+b+c is divisible by e
        # Also check triples that include a and two others
        for e in selected_set:
            if e == a:
                continue
            # Triples from (selected_set - {e}) that include a: a, b, c where b,c in selected_set - {e, a}
            rest = [x for x in selected_set if x != e and x != a]
            for b, c in combinations(rest, 2):
                if (a + b + c) % e == 0:
                    return False
            # Triples from (selected_set - {e}) that don't include a: already checked before adding a
            # But we need to verify they were valid before. Since selected_set was valid, this is fine.
        return True
    
    def is_valid_set(S):
        """Check if set S is fully valid."""
        S_list = list(S)
        for a in S_list:
            others = [x for x in S_list if x != a]
            for b, c, d in combinations(others, 3):
                if (b + c + d) % a == 0:
                    return False
        return True
    
    def greedy(order):
        """Greedy construction with given order of candidates."""
        selected = set()
        for a in order:
            if is_valid_addition(a, selected):
                selected.add(a)
        return selected
    
    # Try multiple greedy orderings
    best_set = set()
    
    # Ordering 1: descending (original)
    order1 = list(range(N, 0, -1))
    s1 = greedy(order1)
    if len(s1) > len(best_set):
        best_set = s1
    
    # Ordering 2: ascending
    order2 = list(range(1, N + 1))
    s2 = greedy(order2)
    if len(s2) > len(best_set):
        best_set = s2
    
    # Ordering 3: random
    random.seed(42)
    for trial in range(50):
        if time.time() > deadline - 5:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        s = greedy(order)
        if len(s) > len(best_set):
            best_set = s
    
    # Local search: try to improve by adding rejected elements or swapping
    def try_add(best_set, candidates):
        """Try to add any candidate to best_set."""
        for a in candidates:
            if a not in best_set:
                if is_valid_addition(a, best_set):
                    best_set.add(a)
                    return True
        return False
    
    def try_swap(best_set, candidates):
        """Try to swap: remove one element, add another."""
        S_list = list(best_set)
        for i in range(len(S_list)):
            for a in candidates:
                if a in best_set:
                    continue
                new_set = best_set.copy()
                new_set.remove(S_list[i])
                new_set.add(a)
                if is_valid_set(new_set):
                    best_set = new_set
                    return True
        return False
    
    # Iterative local search
    all_candidates = set(range(1, N + 1))
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        # Try adding
        rejected = [x for x in all_candidates if x not in best_set]
        if try_add(best_set, rejected):
            improved = True
        # Try swapping
        if try_swap(best_set, rejected):
            improved = True
    
    # More aggressive local search: try removing one and adding two
    if time.time() < deadline - 5:
        S_list = list(best_set)
        rejected = [x for x in all_candidates if x not in best_set]
        for i in range(len(S_list)):
            if time.time() > deadline - 5:
                break
            for j in range(len(rejected)):
                if time.time() > deadline - 5:
                    break
                a1 = rejected[j]
                for k in range(j+1, len(rejected)):
                    a2 = rejected[k]
                    new_set = best_set.copy()
                    new_set.remove(S_list[i])
                    new_set.add(a1)
                    new_set.add(a2)
                    if is_valid_set(new_set):
                        best_set = new_set
                        improved = True
                        break
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
