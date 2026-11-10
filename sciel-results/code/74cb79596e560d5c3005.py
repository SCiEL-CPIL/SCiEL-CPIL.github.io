"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

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
            
    # Precompute bad triples
    # bad_triples[x] = set of pairs (a,b) such that {a,b,x} is a bad triple
    # We store as: for each x, which pairs in the rest of the set are forbidden
    # Actually, let's store: conflict_pairs[x] = set of (a,b) such that {a,b,x} is bad
    # Then can_add(x, S) is: check if any pair in S is in conflict_pairs[x]
    
    conflict_pairs = [set() for _ in range(n + 1)]
    
    # Iterate over all triples? 128^3 is ~2 million. Fast enough.
    # Or iterate over pairs and find witnesses.
    # Pair (a,b) has witness z if lcm(a,b)==lcm(a,z)==lcm(b,z).
    # Let's use the pair-witness approach as it's O(N^3) worst case but sparse.
    
    # For each pair (a,b), find all z
    # This is O(N^3) if done naively.
    # N=128, N^3 = 2M. Python can do 2M ops in ~0.2s.
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            L_ab = lcm_table[a][b]
            for z in range(1, n + 1):
                if z == a or z == b:
                    continue
                if lcm_table[a][z] == L_ab and lcm_table[b][z] == L_ab:
                    # {a,b,z} is a bad triple
                    # Add (a,b) to conflict_pairs[z]
                    # Add (a,z) to conflict_pairs[b]
                    # Add (b,z) to conflict_pairs[a]
                    # Note: pairs should be sorted (min, max)
                    p1 = (min(a,b), max(a,b))
                    p2 = (min(a,z), max(a,z))
                    p3 = (min(b,z), max(b,z))
                    conflict_pairs[z].add(p1)
                    conflict_pairs[b].add(p2)
                    conflict_pairs[a].add(p3)

    def can_add(x, current_set):
        """Check if x can be added to current_set without forming a bad triple."""
        # Check if any pair in current_set is in conflict_pairs[x]
        lst = sorted(current_set)
        k = len(lst)
        for i in range(k):
            a = lst[i]
            for j in range(i + 1, k):
                b = lst[j]
                if (a, b) in conflict_pairs[x]:
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

    def local_search(initial_set, max_iters=500):
        """Local search with swaps and kicks."""
        current_set = set(initial_set)
        best_set = set(current_set)
        improved = True
        iter_count = 0
        
        while improved and time.time() < deadline - 5 and iter_count < max_iters:
            improved = False
            iter_count += 1
            
            # 1. Try to add an element
            added = False
            for in_elem in range(1, n + 1):
                if in_elem in current_set:
                    continue
                if can_add(in_elem, current_set):
                    current_set.add(in_elem)
                    improved = True
                    added = True
                    break
            
            if added:
                if len(current_set) > len(best_set):
                    best_set = set(current_set)
                continue
            
            # 2. Try swap: remove one, add another
            swapped = False
            for out_elem in list(current_set):
                if time.time() > deadline - 5:
                    break
                trial_base = current_set - {out_elem}
                for in_elem in range(1, n + 1):
                    if in_elem in current_set:
                        continue
                    if can_add(in_elem, trial_base):
                        # Check if this swap actually improves size?
                        # Swap keeps size same. But might allow future additions.
                        # We accept swaps that maintain size but change composition.
                        # To avoid loops, we only accept if it leads to a state we haven't seen?
                        # Or just accept if it's "better" in some heuristic?
                        # Simple: accept swap if it allows adding another element immediately?
                        # Or just accept randomly?
                        # Let's accept if it results in a set that is different.
                        # To prevent oscillation, we can keep a history.
                        # For simplicity, we accept the swap and break.
                        current_set = trial_base | {in_elem}
                        improved = True
                        swapped = True
                        break
                if swapped:
                    break
            
            if swapped:
                if len(current_set) > len(best_set):
                    best_set = set(current_set)
                continue

            # 3. Kick: remove 2 random elements, try to add 2 new ones
            if len(current_set) >= 2:
                removed = random.sample(list(current_set), 2)
                trial_base = current_set - set(removed)
                # Try to add any element that fits
                added_any = False
                for in_elem in range(1, n + 1):
                    if in_elem in current_set:
                        continue
                    if can_add(in_elem, trial_base):
                        trial_base.add(in_elem)
                        added_any = True
                        break
                if added_any:
                    # Try to add one more
                    for in_elem in range(1, n + 1):
                        if in_elem in trial_base:
                            continue
                        if can_add(in_elem, trial_base):
                            trial_base.add(in_elem)
                            break
                    if len(trial_base) >= len(current_set):
                        current_set = trial_base
                        improved = True
                        if len(current_set) > len(best_set):
                            best_set = set(current_set)

        return best_set

    random.seed(42)
    best_set = set()
    
    # Candidate 1: Primes + 1
    primes = [x for x in range(2, n + 1) if all(x % i != 0 for i in range(2, int(x**0.5) + 1))]
    candidate1 = set(primes + [1])
    if len(candidate1) > len(best_set):
        best_set = candidate1

    # Candidate 2: Range [65, 128]
    candidate2 = set(range(65, n + 1))
    # Check validity of candidate2
    valid = True
    lst2 = sorted(candidate2)
    for i in range(len(lst2)):
        if not valid: break
        a = lst2[i]
        for j in range(i + 1, len(lst2)):
            if not valid: break
            b = lst2[j]
            for k in range(j + 1, len(lst2)):
                c = lst2[k]
                if (a, b) in conflict_pairs[c]:
                    valid = False
                    break
    if valid and len(candidate2) > len(best_set):
        best_set = candidate2

    # Candidate 3: Greedy with random restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        if i % 2 == 0:
            order = list(range(1, n + 1))
            random.shuffle(order)
        else:
            # Sort by conflict count (least conflicted first)
            conflict_counts = [len(conflict_pairs[x]) for x in range(n + 1)]
            order = sorted(range(1, n + 1), key=lambda x: conflict_counts[x])
        
        s = greedy_add(order)
        s_improved = local_search(s, max_iters=100)
        if len(s_improved) > len(best_set):
            best_set = s_improved

    # Final local search on the best set found
    if time.time() < deadline - 5:
        best_set = local_search(best_set, max_iters=200)
        
    # Verify and return
    if not best_set:
        best_set = {1}
        
    return sorted(best_set)
# EVOLVE-BLOCK-END
