import time
import random

def construct():
    start = time.time()
    deadline = start + 85.0
    n = 256

    def _valid(S):
        """Check the property for sorted set S. Returns True if valid."""
        lst = sorted(S)
        m = len(lst)
        for i in range(m):
            a = lst[i]
            for j in range(i + 1, m):
                b = lst[j]
                for k in range(j, m):
                    c = lst[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def can_add(a, S):
        """Check if element a can be added to set S without violating property."""
        elements = sorted(S)
        m = len(elements)
        for j in range(m):
            b = elements[j]
            if b <= a:
                continue
            for k in range(j, m):
                c = elements[k]
                if (b + c) % a == 0:
                    return False
        return True

    def greedy_descending():
        """Standard greedy: add largest first if valid."""
        S = set()
        for a in range(n, 0, -1):
            if can_add(a, S):
                S.add(a)
        return S

    def local_search_improve(S, max_time=5.0):
        """
        Local search:
        1. Try to add any valid element (size +1).
        2. Try to swap: remove b, add a (size 0) if it leads to a state where a new element can be added.
           Simplified: Random swaps to escape local minima, then greedy fill.
        """
        S = set(S)
        best_size = len(S)
        start_ls = time.time()
        
        # Phase 1: Greedy fill
        for a in range(n, 0, -1):
            if time.time() > start_ls + max_time:
                break
            if a not in S and can_add(a, S):
                S.add(a)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
        else:
            best_S = S

        # Phase 2: Randomized Swaps to explore
        # We try to replace a large element with a smaller one, hoping to unlock more additions.
        # This is risky, so we keep the best set found.
        for _ in range(1000):
            if time.time() > start_ls + max_time:
                break
            
            S_trial = set(S)
            # Pick a random element to remove (prefer large ones? No, prefer ones that cause conflicts?)
            # Just pick random
            if not S_trial:
                break
            to_remove = random.choice(list(S_trial))
            S_trial.remove(to_remove)
            
            # Try to add to_remove back? No.
            # Try to add a random smaller element?
            # Or just greedy fill from scratch on S_trial?
            # Greedy fill is fast.
            for a in range(n, 0, -1):
                if a not in S_trial and can_add(a, S_trial):
                    S_trial.add(a)
            
            if len(S_trial) > best_size:
                best_size = len(S_trial)
                best_S = S_trial
                S = S_trial
                break # Restart search from new best

        return best_S

    # Initial Solution
    S_best = greedy_descending()
    best_size = len(S_best)
    
    # Try multiple random restarts with noise
    # The greedy is deterministic. To vary it, we can randomly skip valid elements?
    # Or randomly remove elements and re-greedy.
    
    while time.time() < deadline - 5:
        # Random Perturbation
        S_perturbed = set(S_best)
        # Remove 1-3 random elements
        for _ in range(random.randint(1, 3)):
            if S_perturbed:
                S_perturbed.remove(random.choice(list(S_perturbed)))
        
        # Re-greedy fill
        for a in range(n, 0, -1):
            if a not in S_perturbed and can_add(a, S_perturbed):
                S_perturbed.add(a)
        
        if len(S_perturbed) > best_size:
            best_size = len(S_perturbed)
            S_best = S_perturbed
            # If we improved, do a local search on it
            S_best = local_search_improve(S_best, max_time=5.0)
            best_size = len(S_best)
            
    # Final Verification
    if not _valid(S_best):
        # Fallback
        S_best = greedy_descending()
        if not _valid(S_best):
             # Extreme fallback
             S_best = set(range(171, n+1))

    return sorted(S_best)
