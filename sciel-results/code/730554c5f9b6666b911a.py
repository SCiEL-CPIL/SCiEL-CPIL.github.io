import time
import random
import numpy as np

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
        # If we get here, it's valid.
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
    
    # Probabilistic greedy: select candidate with probability proportional to
    # number of underutilized sums in its neighborhood
    def select_probabilistic_candidate():
        unused = [i for i in range(1, max_val + 1) if not in_S[i]]
        if not unused:
            return None
        
        # For each unused candidate, compute a score based on how many
        # sums in its neighborhood are underutilized (count < 2)
        scores = []
        for x in unused:
            # Sums affected: 2*x and x+y for y in S
            # Count how many of these sums have count < 2
            score = 0
            if counts[2 * x] < 2:
                score += 1
            for y in S:
                if y != x:
                    s = x + y
                    if counts[s] < 2:
                        score += 1
            scores.append(score)
        
        # Convert scores to probabilities (add small epsilon to avoid zero)
        eps = 1e-9
        scores = [s + eps for s in scores]
        total = sum(scores)
        probs = [s / total for s in scores]
        
        # Sample from distribution
        r = random.random() * total
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return unused[i]
        return unused[-1]
    
    # Greedy initialization with probabilistic selection
    for _ in range(max_val):
        if time.time() >= deadline:
            break
        c = select_probabilistic_candidate()
        if c is None:
            break
        if try_add(c):
            pass

    # Track best solution
    best_S = list(S)
    best_size = len(S)
    
    # Local search phase with probabilistic candidate selection
    while time.time() < deadline:
        # Select candidate using probabilistic method
        c = select_probabilistic_candidate()
        if c is None:
            break
        
        # Try to add c
        if try_add(c):
            # Improvement
            if len(S) > best_size:
                best_size = len(S)
                best_S = list(S)
            continue
            
        # If add failed, try to find a swap
        if not S:
            continue
            
        # Try up to 20 random swaps per failed addition
        swapped = False
        for _ in range(20):
            a = random.choice(S)
            if is_valid_swap(a, c):
                # Perform swap
                try_remove(a)
                try_add(c)
                swapped = True
                break
        
        # Occasionally, try a "remove" move to escape local minima
        if random.random() < 0.03 and S:
            a = random.choice(S)
            try_remove(a)

    # Final verification and return
    # Verify the best solution is valid
    def verify_set(S_set):
        cnt = [0] * (2 * max_val + 1)
        for i, a in enumerate(S_set):
            for j in range(i, len(S_set)):
                b = S_set[j]
                cnt[a + b] += 1
                if cnt[a + b] > 2:
                    return False
        return True
    
    if verify_set(best_S):
        return best_S
    
    # Fallback: rebuild from scratch if verification fails
    S.clear()
    in_S = [False] * (max_val + 1)
    counts = [0] * (2 * max_val + 1)
    
    for _ in range(max_val):
        if time.time() >= deadline:
            break
        c = select_probabilistic_candidate()
        if c is None:
            break
        if try_add(c):
            pass
    
    return S
