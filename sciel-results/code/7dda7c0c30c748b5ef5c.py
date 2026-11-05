import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 256
    
    # Precompute for each a in [1, n] a set of "bad sums" S such that a | S.
    # Sums range from 2 to 2n.
    # bad_sums[a] = {s in [2, 2n] such that s % a == 0}
    bad_sums = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        for s in range(2*a, 2*n + 1, a):
            bad_sums[a].add(s)
    
    def is_valid_addition(x, S):
        """Check if adding x to S keeps the set valid."""
        if not S:
            return True

        # We need to check all triples involving x.
        # The condition: for all a < b <= c in S_new, a does not divide b+c.
        # S_new = S ∪ {x}.
        # Violations must involve x.
        
        # Case 1: x is 'a'.
        # We need b, c in S such that x < b <= c and x | (b+c).
        # Let S_gt = [y for y in S if y > x]
        # For all b, c in S_gt with b <= c, check if (b+c) % x == 0.
        
        # Case 2: x is 'b'.
        # We need a, c in S such that a < x <= c and a | (x+c).
        # Note: c can be x? No, c >= b = x. If c=x, then b=c=x.
        # The condition is a < b <= c. If b=x, c=x, then a < x.
        # Check: a | (x+x) = 2x.
        # If c > x, check a | (x+c).
        
        # Case 3: x is 'c'.
        # We need a, b in S such that a < b <= x and a | (b+x).
        
        # Optimization:
        # Pre-sort S? S is small (<256). Sorting each time is O(N log N) ~ 2000 ops.
        # We can just iterate.
        
        # To speed up Case 1, we can iterate b in S_gt, and for each b, iterate c in S_gt where c>=b.
        # But we need to be careful not to miss pairs.
        
        # Let's filter S into three lists: less, equal (none, x not in S), greater.
        less = []
        greater = []
        for y in S:
            if y < x:
                less.append(y)
            elif y > x:
                greater.append(y)
            # y == x is impossible as x is not in S
        
        # Case 1: x is a
        # b, c in greater, b <= c
        # If len(greater) < 2, no pairs.
        if len(greater) >= 2:
            g_sorted = sorted(greater)
            # Optimization: use precomputed bad_sums?
            # We check if (b+c) in bad_sums[x].
            for i in range(len(g_sorted)):
                b = g_sorted[i]
                for j in range(i, len(g_sorted)):
                    c = g_sorted[j]
                    if (b + c) % x == 0:
                        return False
        
        # Case 2: x is b
        # a in less, c in S such that c >= x.
        # c can be x (if we consider b=c=x) or c in greater.
        # Wait, if c=x, then b=x, c=x. This requires x to be in the set?
        # The set is S ∪ {x}. So x is in the set.
        # So pairs (b,c) can be (x,x).
        # Condition: a < x <= c.
        # If c=x, then a < x. Check a | (x+x).
        # If c > x, then a < x. Check a | (x+c).
        
        for a in less:
            if (2 * x) % a == 0:
                return False
            for c in greater:
                if (x + c) % a == 0:
                    return False
        
        # Case 3: x is c
        # a, b in S such that a < b <= x.
        # So a, b in less (since b <= x and b != x? No, b can be x? No, b is in S, x not in S. So b < x).
        # So a, b in less, a < b.
        # Check a | (b+x).
        
        if len(less) >= 2:
            l_sorted = sorted(less)
            for i in range(len(l_sorted)):
                a = l_sorted[i]
                for j in range(i + 1, len(l_sorted)):
                    b = l_sorted[j]
                    # b > a is guaranteed by sorting and i < j
                    if (b + x) % a == 0:
                        return False
        
        return True

    def is_valid_set(S):
        """Check if the entire set S is valid."""
        S_sorted = sorted(S)
        m = len(S_sorted)
        for i in range(m):
            a = S_sorted[i]
            for j in range(i + 1, m):
                b = S_sorted[j]
                for k in range(j, m):
                    c = S_sorted[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def find_conflict(S):
        """Find a conflicting triple (a, b, c) in S, or return None."""
        S_sorted = sorted(S)
        m = len(S_sorted)
        for i in range(m):
            a = S_sorted[i]
            for j in range(i + 1, m):
                b = S_sorted[j]
                for k in range(j, m):
                    c = S_sorted[k]
                    if (b + c) % a == 0:
                        return (a, b, c)
        return None

    # Strategy:
    # 1. Start with a strong baseline: Upper tail [129..256] or similar.
    #    The seed [192..256] is size 65.
    #    Let's try [129..256] (size 128). It might have conflicts.
    #    We will prune it.
    
    best_set = []
    best_size = 0

    # Initial construction: Greedy from large to small
    S_greedy = []
    for x in range(n, 0, -1):
        if time.time() > deadline:
            break
        if is_valid_addition(x, S_greedy):
            S_greedy.append(x)
    
    if len(S_greedy) > best_size:
        best_size = len(S_greedy)
        best_set = S_greedy[:]

    # Improvement 1: Start with dense upper tail, prune, then add.
    # Try different starting thresholds for the upper tail.
    # e.g., start with [128..256], [140..256], etc.
    
    for start_val in [128, 140, 150, 160, 180, 200]:
        if time.time() > deadline:
            break
        
        S_try = list(range(start_val, n + 1))
        
        # Prune S_try to make it valid.
        # We can do this by iterating through elements and removing those that cause conflicts.
        # To be efficient, we can check validity of the whole set, and if invalid,
        # find a conflicting triple and remove one element from it.
        # Or simpler: Rebuild it greedily from the upper tail?
        # Actually, `is_valid_addition` is fast.
        # Let's try to keep as many as possible.
        # We can sort S_try in descending order and try to add them one by one to an empty set?
        # That's just the greedy from large to small again, but restricted to [start_val..n].
        # That might be better than pruning a potentially invalid large set.
        
        # Let's try a "local search" on the upper tail.
        # Start with S_upper = [start_val..n].
        # If invalid, find a conflict and remove one element.
        # Repeat until valid.
        
        S_pruned = list(S_try)
        changed = True
        while changed:
            changed = False
            # Check if valid
            if is_valid_set(S_pruned):
                break
            
            # Find a conflict
            S_sorted = sorted(S_pruned)
            conflict_found = False
            for i in range(len(S_sorted)):
                a = S_sorted[i]
                for j in range(i + 1, len(S_sorted)):
                    b = S_sorted[j]
                    for k in range(j, len(S_sorted)):
                        c = S_sorted[k]
                        if (b + c) % a == 0:
                            # Conflict found: (a, b, c)
                            # Remove one of them. Which one?
                            # Heuristic: Remove the one that "causes" the most conflicts?
                            # Or just remove 'a' (smallest) as it divides many sums?
                            # Or remove 'c' (largest) as it's least likely to be needed?
                            # Let's try removing 'a' first.
                            S_pruned.remove(a)
                            changed = True
                            conflict_found = True
                            break
                    if conflict_found:
                        break
                if conflict_found:
                    break
            
            if time.time() > deadline:
                break
        
        # Now try to add smaller elements to S_pruned
        S_targeted = S_pruned[:]
        for x in range(1, start_val):
            if time.time() > deadline:
                break
            if x in S_targeted:
                continue
            if is_valid_addition(x, S_targeted):
                S_targeted.append(x)
        
        if len(S_targeted) > best_size:
            best_size = len(S_targeted)
            best_set = S_targeted[:]

    # Random restarts with local improvement
    for attempt in range(30):
        if time.time() > deadline:
            break
        
        # Randomize the order of candidates
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        S_try = []
        for x in candidates:
            if time.time() > deadline:
                break
            if is_valid_addition(x, S_try):
                S_try.append(x)
        
        # Local improvement: try to add any remaining numbers
        S_improved = S_try[:]
        for x in range(1, n + 1):
            if time.time() > deadline:
                break
            if x in S_improved:
                continue
            if is_valid_addition(x, S_improved):
                S_improved.append(x)
        
        if len(S_improved) > best_size:
            best_size = len(S_improved)
            best_set = S_improved[:]

    # Final polish: Try to add any remaining numbers to best_set
    S_final = best_set[:]
    for x in range(1, n + 1):
        if time.time() > deadline:
            break
        if x in S_final:
            continue
        if is_valid_addition(x, S_final):
            S_final.append(x)
    
    if len(S_final) > best_size:
        best_size = len(S_final)
        best_set = S_final[:]

    return best_set
# EVOLVE-BLOCK-END
