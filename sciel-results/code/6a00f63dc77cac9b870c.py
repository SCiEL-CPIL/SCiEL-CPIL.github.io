import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 256
    # Precompute reciprocals as floats for fast comparison
    recip = [0.0] + [1.0 / i for i in range(1, n + 1)]
    
    # in_set: list of booleans, index 0 unused
    in_set = [False] * (n + 1)
    
    def can_add(x, in_set):
        """
        Check if x can be added to the set represented by in_set.
        Returns True if no 3-AP is formed.
        """
        # We need to check if x forms a 3-AP with any two elements a, b in S.
        # The 3-AP is (u, v, w) with u + w = 2v.
        # x can be u, v, or w.
        
        # Case 1: x is the middle term v.
        # Then u + w = 2x.
        # For any u in S, w = 2x - u must NOT be in S (and w != x).
        # Also u != x (since x not in S yet).
        for u in range(1, n + 1):
            if in_set[u]:
                w = 2 * x - u
                if w != x and 1 <= w <= n and in_set[w]:
                    return False
        
        # Case 2: x is an endpoint (u or w).
        # Let x = u. Then x + w = 2v => w = 2v - x.
        # For any v in S, w = 2v - x must NOT be in S (and w != x).
        # Note: v cannot be x because x not in S.
        for v in range(1, n + 1):
            if in_set[v]:
                w = 2 * v - x
                if w != x and 1 <= w <= n and in_set[w]:
                    return False
                    
        return True

    def greedy_fill(in_set):
        """
        Add all elements that can be added, in increasing order of x (decreasing 1/x).
        Returns the number of elements added.
        """
        added = 0
        changed = True
        while changed:
            changed = False
            for x in range(1, n + 1):
                if not in_set[x]:
                    if can_add(x, in_set):
                        in_set[x] = True
                        added += 1
                        changed = True
        return added

    def get_sum(in_set):
        """Calculate sum of reciprocals."""
        s = 0.0
        for i in range(1, n + 1):
            if in_set[i]:
                s += recip[i]
        return s

    # 1. Greedy Initialization
    # Start with empty set, add 1, 2, 3... if possible.
    # This prioritizes small numbers (high weight).
    for x in range(1, n + 1):
        if can_add(x, in_set):
            in_set[x] = True
            
    # Ensure maximality
    greedy_fill(in_set)
    
    best_in_set = in_set[:]
    best_sum = get_sum(best_in_set)
    
    # 2. Local Search
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        current_sum = best_sum
        
        # Strategy A: Remove one element, then greedy fill.
        # This allows removing a "blocking" element that prevents multiple high-value elements.
        for r in range(1, n + 1):
            if not best_in_set[r]:
                continue
            if time.time() > deadline - 5.0:
                break
                
            trial = best_in_set[:]
            trial[r] = False
            greedy_fill(trial)
            new_sum = get_sum(trial)
            
            if new_sum > current_sum + 1e-12:
                best_in_set = trial
                best_sum = new_sum
                current_sum = new_sum
                improved = True
                # Restart the loop to re-evaluate with the new set
                break
        
        # Strategy B: Swap one element with another, then greedy fill.
        if not improved and time.time() < deadline - 10.0:
            for r in range(1, n + 1):
                if not best_in_set[r]:
                    continue
                if time.time() > deadline - 10.0:
                    break
                
                for a in range(1, n + 1):
                    if best_in_set[a] or a == r:
                        continue
                    
                    trial = best_in_set[:]
                    trial[r] = False
                    # Check if a can be added to the set without r
                    if can_add(a, trial):
                        trial[a] = True
                        greedy_fill(trial)
                        new_sum = get_sum(trial)
                        
                        if new_sum > current_sum + 1e-12:
                            best_in_set = trial
                            best_sum = new_sum
                            current_sum = new_sum
                            improved = True
                            break
                if improved:
                    break

    # 3. Randomized Perturbation (Simulated Annealing-like or Random Restarts)
    # Try to escape local optima by randomly removing a few elements and refilling.
    random.seed(42)
    
    while time.time() < deadline - 10.0:
        # Create a perturbed copy
        trial = best_in_set[:]
        
        # Remove 2 to 4 random elements
        num_removals = random.randint(2, 4)
        candidates = [i for i in range(1, n + 1) if trial[i]]
        
        for _ in range(num_removals):
            if not candidates:
                break
            idx = random.randint(0, len(candidates) - 1)
            rem_val = candidates[idx]
            trial[rem_val] = False
            candidates.pop(idx)
            
        # Greedy fill
        greedy_fill(trial)
        
        new_sum = get_sum(trial)
        
        if new_sum > best_sum + 1e-12:
            best_in_set = trial
            best_sum = new_sum
            
    # Final greedy fill to ensure no single element can be added
    greedy_fill(best_in_set)
    
    # Extract result
    result = [i for i in range(1, n + 1) if best_in_set[i]]
    
    return result
