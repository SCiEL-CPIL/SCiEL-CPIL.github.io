"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s kill
    
    n = 128
    numbers = list(range(1, n + 1))
    
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            l = _lcm(i, j)
            lcm_table[i][j] = l
            lcm_table[j][i] = l
            
    # Precompute conflicts: for each x, the set of pairs (a, b) with a < b
    # such that {a, b, x} is a bad triple (LCM(a,b)==LCM(a,x)==LCM(b,x))
    # We store these as a set of tuples (a, b) for fast lookup.
    conflicts = [set() for _ in range(n + 1)]
    for x in range(1, n + 1):
        for a in range(1, n + 1):
            if a == x:
                continue
            for b in range(a + 1, n + 1):
                if b == x:
                    continue
                # Check if (a, b, x) is a bad triple
                if lcm_table[a][b] == lcm_table[a][x] == lcm_table[b][x]:
                    # Store the pair (min(a,b), max(a,b)) in conflicts[x]
                    if a < b:
                        conflicts[x].add((a, b))
                    else:
                        conflicts[x].add((b, a))

    def can_add(x, current_set):
        """Check if x can be added to current_set without forming a bad triple."""
        # We need to check if any pair (a, b) in current_set is in conflicts[x]
        # To make this fast, we can just iterate over all pairs in current_set
        # and check membership in the set conflicts[x].
        # current_set is a set of integers.
        # Number of pairs is k*(k-1)/2.
        # For k=60, that's ~1800 checks. Set lookup is O(1).
        lst = list(current_set)
        k = len(lst)
        for i in range(k):
            a = lst[i]
            for j in range(i + 1, k):
                b = lst[j]
                if a < b:
                    pair = (a, b)
                else:
                    pair = (b, a)
                if pair in conflicts[x]:
                    return False
        return True

    def greedy_add(order):
        """Greedy: add elements in given order if valid."""
        selected = set()
        for x in order:
            if time.time() > deadline:
                break
            if can_add(x, selected):
                selected.add(x)
        return selected

    def local_search(initial_set):
        """Try to improve the set by swapping elements."""
        current_set = set(initial_set)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to remove one element and add another
            # We iterate over elements in the set to remove
            # and elements not in the set to add
            # To keep it fast, we limit the number of attempts per iteration
            attempts = 0
            for out_elem in list(current_set):
                if time.time() > deadline:
                    break
                for in_elem in range(1, n + 1):
                    if in_elem in current_set:
                        continue
                    # Trial: remove out_elem, add in_elem
                    # Check if in_elem can be added to (current_set - {out_elem})
                    trial_base = current_set - {out_elem}
                    if can_add(in_elem, trial_base):
                        current_set = trial_base | {in_elem}
                        improved = True
                        break
                if improved:
                    break
            # Also try to just add an element if possible
            if not improved and time.time() < deadline:
                for in_elem in range(1, n + 1):
                    if in_elem in current_set:
                        continue
                    if can_add(in_elem, current_set):
                        current_set.add(in_elem)
                        improved = True
                        break
        return current_set

    random.seed(42)
    best_set = set()
    
    # Run multiple random restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
        random_order = list(range(1, n + 1))
        random.shuffle(random_order)
        s = greedy_add(random_order)
        # Run local search on the result
        s_improved = local_search(s)
        if len(s_improved) > len(best_set):
            best_set = s_improved
            
    # Final check and return
    if not best_set:
        best_set = {1}
        
    return sorted(best_set)
# EVOLVE-BLOCK-END
