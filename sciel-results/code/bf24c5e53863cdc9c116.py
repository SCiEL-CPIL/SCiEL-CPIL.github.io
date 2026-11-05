import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    n = 256

    # Precompute for speed: for each a, which sums (b+c) are multiples of a?
    # But sums range up to 512. We can precompute for each a a set of bad sums.
    # Actually, we can just compute (b+c) % a on the fly; it's fast enough.

    def is_valid_addition(x, S):
        """Check if adding x to S keeps the set valid."""
        if not S:
            return True

        # Convert to set for O(1) membership if needed, but we use list here for iteration
        # S is a list. Let's assume S is unsorted.
        
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
            # Sort greater to make b <= c easy? Or just double loop with check.
            # Double loop is O(K^2). K <= 256. 256^2 = 65k.
            # 256 insertions * 65k = 16M ops. Acceptable.
            # We can optimize by sorting 'greater' once.
            g_sorted = sorted(greater)
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
        
        # So for each a in less:
        # 1. Check if (2*x) % a == 0. (Corresponds to b=x, c=x)
        # 2. For each c in greater, check if (x+c) % a == 0.
        
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
        
        if is_valid_set(S_pruned):
            # Now try to add elements from [1, start_val-1]
            for x in range(1, start_val):
                if time.time() > deadline:
                    break
                if is_valid_addition(x, S_pruned):
                    S_pruned.append(x)
            
            if len(S_pruned) > best_size:
                best_size = len(S_pruned)
                best_set = S_pruned[:]

    # Improvement 2: Local Search / Swapping
    # Try to improve best_set by:
    # 1. Adding any remaining element.
    # 2. Removing one element and adding two others (if net gain is +1).
    
    # First, pass to add any possible element to best_set
    S_improve = best_set[:]
    for x in range(1, n + 1):
        if time.time() > deadline:
            break
        if x not in S_improve:
            if is_valid_addition(x, S_improve):
                S_improve.append(x)
    
    if len(S_improve) > best_size:
        best_size = len(S_improve)
        best_set = S_improve[:]

    # Second, try random restarts with different greedy orders
    # This is often effective for combinatorial problems.
    random.seed(42)
    
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
        
        # Random shuffle of candidates
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        S_rand = []
        for x in candidates:
            if time.time() > deadline - 2:
                break
            if is_valid_addition(x, S_rand):
                S_rand.append(x)
        
        if len(S_rand) > best_size:
            best_size = len(S_rand)
            best_set = S_rand[:]
            
            # Try to add more to this new best
            for x in range(1, n + 1):
                if time.time() > deadline - 2:
                    break
                if x not in best_set:
                    if is_valid_addition(x, best_set):
                        best_set.append(x)
            
            if len(best_set) > best_size:
                best_size = len(best_set)

    # Third, Local Search: Swap
    # Try to remove one element from best_set and see if we can add two.
    # This is complex to implement efficiently.
    # Simpler: Try removing one element, then try to add any two.
    # If we can add two, net gain +1.
    
    # Given time constraints, we might skip complex swaps if we already have a good set.
    # Let's do a limited version:
    # For each element in best_set, try removing it.
    # Then try to add the two largest available elements that fit.
    
    # Actually, a better local search:
    # For each pair of elements (u, v) not in S, try to add them if we remove one element w from S.
    # This is O(N^3) roughly. 256^3 is too big.
    
    # Let's stick to the greedy + random restarts + targeted add.
    # The "prune upper tail" strategy is the main new addition from the inspiration.
    
    return best_set
