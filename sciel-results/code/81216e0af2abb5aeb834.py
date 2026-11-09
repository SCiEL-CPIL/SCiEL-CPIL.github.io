import time
import random

def construct():
    N = 1000
    max_val = 1000
    deadline = time.time() + 110  # Safety margin for 120s limit
    
    # Data structures
    S = []  # List of chosen integers
    in_S = [False] * (max_val + 1)  # in_S[i] is True if i is in S
    # counts[s] = number of pairs (a,b) in S with a<=b and a+b=s
    # Max sum is 2000
    counts = [0] * (2 * max_val + 1)
    
    # Helper to add a number to the set and update counts
    # Returns True if successful, False if it would violate B2[2]
    def try_add(x):
        # Check 2*x
        if counts[2 * x] >= 2:
            return False
        # Check x + y for y in S
        for y in S:
            if y == x:
                continue
            s = x + y
            if counts[s] >= 2:
                return False
        # If we get here, it's valid. Actually, we should check 2*x again if x is already in S? 
        # No, x is unused.
        # Perform addition
        S.append(x)
        in_S[x] = True
        counts[2 * x] += 1
        for y in S:
            if y != x:
                counts[x + y] += 1
        return True

    # Helper to remove a number
    def try_remove(x):
        S.remove(x)
        in_S[x] = False
        counts[2 * x] -= 1
        for y in S:
            if y != x:
                counts[x + y] -= 1

    # Helper to check if a swap (remove a, add c) is valid
    # Does not modify state, just checks
    def is_valid_swap(a, c):
        # Sums affected:
        # Removed: 2*a, a+y (y in S, y!=a)
        # Added: 2*c, c+y (y in S\{a})
        
        # We need to check if any resulting count > 2
        
        # 1. Check sums involving c (new)
        # 2*c
        # Current count of 2*c in S: counts[2*c]
        # Does a contribute to 2*c? 
        # If 2*a == 2*c -> a==c (impossible, a in S, c not in S)
        # If a + y == 2*c? We need to subtract contributions from a.
        
        # Let's simulate the counts for the affected sums.
        # Affected sums are:
        # - 2*a
        # - a + y for y in S, y != a
        # - 2*c
        # - c + y for y in S, y != a
        
        # We can just check the max possible count for these sums.
        
        # Sum K = 2*c
        # New count = counts[K] - (1 if a contributes to K in S else 0) + 1
        # a contributes to K if:
        #   2*a == K (a == c, false)
        #   a + y == K for some y in S, y != a.
        #   y = K - a = 2*c - a.
        #   If y in S and y != a and y != c (c not in S), then a contributes.
        
        # This logic is getting messy. 
        # Alternative: Just do the swap on a copy of counts? 
        # Copying a list of 2001 ints is fast.
        # Then check max? No, just check the affected sums.
        
        # Let's use a temporary dictionary for deltas.
        deltas = {}
        
        # Remove a
        deltas[2*a] = deltas.get(2*a, 0) - 1
        for y in S:
            if y != a:
                s = a + y
                deltas[s] = deltas.get(s, 0) - 1
        
        # Add c
        deltas[2*c] = deltas.get(2*c, 0) + 1
        for y in S:
            if y != a:
                s = c + y
                deltas[s] = deltas.get(s, 0) + 1
        
        # Check all affected sums
        for s, d in deltas.items():
            new_count = counts[s] + d
            if new_count > 2:
                return False
            if new_count < 0:
                return False # Should not happen
        
        return True

    # 1. Greedy Initialization
    for i in range(1, max_val + 1):
        if try_add(i):
            pass
        # Greedy just skips if invalid. It doesn't backtrack.

    # 2. Local Search
    # We will try to add random unused numbers.
    # If it fails, we try to swap an existing number with the candidate.
    
    # To speed up random selection of unused numbers, we can keep a list of unused numbers.
    # But maintaining it is overhead. 
    # Random picking from 1..1000 and checking in_S is O(1).
    
    while time.time() < deadline:
        c = random.randint(1, max_val)
        if in_S[c]:
            continue
            
        # Try to add c
        if try_add(c):
            continue
            
        # If add failed, try to find a swap
        # We need to identify why it failed.
        # The failure is due to some sum K having count >= 2 in S, and c adding to it.
        # Or 2*c having count >= 2.
        
        # Let's just try random swaps with elements in S.
        # To be more targeted, we can pick a random element a from S.
        # But picking a "good" a is hard without checking.
        # Let's try a few random swaps.
        
        # Optimization: Only try swaps if S is not empty
        if not S:
            continue
            
        # Try up to 5 random swaps per failed addition
        swapped = False
        for _ in range(5):
            a = random.choice(S)
            if is_valid_swap(a, c):
                # Perform swap
                try_remove(a)
                try_add(c)
                swapped = True
                break
        
        # If we didn't swap, we just discard c for now.
        # The loop continues.

    return S
