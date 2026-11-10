"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = 128
    numbers = list(range(1, n + 1))
    
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            l = _lcm(i, j)
            lcm_table[i][j] = l
            lcm_table[j][i] = l
            
    # Precompute bad triples and conflict pairs
    conflict_pairs = [set() for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            L_ab = lcm_table[a][b]
            for z in range(1, n + 1):
                if z == a or z == b:
                    continue
                if lcm_table[a][z] == L_ab and lcm_table[b][z] == L_ab:
                    # {a,b,z} is a bad triple
                    p1 = (min(a,b), max(a,b))
                    p2 = (min(a,z), max(a,z))
                    p3 = (min(b,z), max(b,z))
                    conflict_pairs[z].add(p1)
                    conflict_pairs[b].add(p2)
                    conflict_pairs[a].add(p3)

    def can_add(x, current_set):
        """Check if x can be added to current_set without forming a bad triple."""
        lst = sorted(current_set)
        k = len(lst)
        for i in range(k):
            a = lst[i]
            for j in range(i + 1, k):
                b = lst[j]
                if (a, b) in conflict_pairs[x]:
                    return False
        return True

    def is_valid_set(s):
        """Check if a set is valid (no bad triples)."""
        lst = sorted(s)
        k = len(lst)
        for i in range(k):
            a = lst[i]
            for j in range(i + 1, k):
                b = lst[j]
                for l in range(j + 1, k):
                    c = lst[l]
                    if (a, b) in conflict_pairs[c]:
                        return False
        return True

    def branch_and_bound(candidates, current_set, best_size):
        """
        Branch and bound to find maximum independent set.
        candidates: list of elements that can potentially be added
        current_set: the set we've built so far
        best_size: current best size found
        Returns the best set found.
        """
        nonlocal best_set, best_size_global
        
        if time.time() > deadline - 2:
            return
        
        # Bound: if current + remaining can't beat best, prune
        if len(current_set) + len(candidates) <= best_size_global:
            return
        
        if not candidates:
            if len(current_set) > best_size_global:
                best_size_global = len(current_set)
                best_set = set(current_set)
            return
        
        # Choose a candidate to branch on
        # Heuristic: pick element with fewest conflicts in candidates
        x = candidates[0]
        
        # Option 1: Include x
        if can_add(x, current_set):
            new_current = current_set | {x}
            # Remove x and any elements that conflict with x
            new_candidates = []
            for y in candidates[1:]:
                # Check if y conflicts with x (i.e., there exists z in new_current such that {x,y,z} is bad)
                # Actually, we need to check if y can coexist with all elements in new_current
                # Simplified: just filter out elements that would form a bad triple with x and some element in current_set
                can_coexist = True
                for z in current_set:
                    if (min(x,y), max(x,y)) in conflict_pairs[z]:
                        can_coexist = False
                        break
                if can_coexist:
                    new_candidates.append(y)
            
            branch_and_bound(new_candidates, new_current, best_size_global)
        
        # Option 2: Exclude x
        branch_and_bound(candidates[1:], current_set, best_size_global)

    def find_max_independent_set(order, time_limit=10):
        """
        Find maximum independent set using branch and bound.
        order: list of elements to consider
        """
        global best_set, best_size_global
        best_set = set()
        best_size_global = 0
        
        # Start with a good initial solution
        initial = set()
        for x in order:
            if time.time() > deadline - 2:
                break
            if can_add(x, initial):
                initial.add(x)
        
        best_set = set(initial)
        best_size_global = len(initial)
        
        # Branch and bound
        candidates = list(order)
        branch_and_bound(candidates, set(), best_size_global)
        
        return best_set

    random.seed(42)
    best_set = set()
    best_size_global = 0
    
    # Candidate 1: Primes + 1
    primes = [x for x in range(2, n + 1) if all(x % i != 0 for i in range(2, int(x**0.5) + 1))]
    candidate1 = set(primes + [1])
    if is_valid_set(candidate1):
        if len(candidate1) > len(best_set):
            best_set = candidate1
            best_size_global = len(candidate1)

    # Candidate 2: Range [65, 128]
    candidate2 = set(range(65, n + 1))
    if is_valid_set(candidate2):
        if len(candidate2) > len(best_set):
            best_set = candidate2
            best_size_global = len(candidate2)

    # Use branch and bound with different orderings
    # Ordering 1: Sort by conflict count (least conflicted first)
    conflict_counts = [len(conflict_pairs[x]) for x in range(n + 1)]
    order1 = sorted(range(1, n + 1), key=lambda x: conflict_counts[x])
    
    if time.time() < deadline - 15:
        s1 = find_max_independent_set(order1)
        if len(s1) > len(best_set):
            best_set = s1
            best_size_global = len(best_set)

    # Ordering 2: Random shuffle
    order2 = list(range(1, n + 1))
    random.shuffle(order2)
    
    if time.time() < deadline - 10:
        s2 = find_max_independent_set(order2)
        if len(s2) > len(best_set):
            best_set = s2
            best_size_global = len(best_set)

    # Ordering 3: Reverse order (larger numbers first)
    order3 = list(range(n, 0, -1))
    
    if time.time() < deadline - 5:
        s3 = find_max_independent_set(order3)
        if len(s3) > len(best_set):
            best_set = s3
            best_size_global = len(best_set)

    # Final verification and return
    if not best_set:
        best_set = {1}
    
    # Verify the final set is valid
    if not is_valid_set(best_set):
        # Fallback to a known valid set
        best_set = {1}
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
