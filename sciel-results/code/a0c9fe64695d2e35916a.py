"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin before 60s limit
    
    n = 256
    
    # Precompute all 3-APs (a, b, c) such that a < b < c and 2b = a + c
    # We need to ensure that for any selected set, no such triple is fully contained.
    # Instead of checking all triples every time, we can precompute for each pair (a, c)
    # the middle b = (a+c)/2.
    # Or better: for each potential move (add/remove x), check if it creates a 3-AP.
    
    # Let's use a bitmask or a boolean array for the current set.
    # To make SA efficient, we need fast violation checks.
    # A 3-AP is (x-d, x, x+d) or (x, x+d, x+2d).
    # If we add x, it forms a 3-AP if:
    # 1. There exists d such that x-d in S and x+d in S (x is middle)
    # 2. There exists d such that x+d in S and x+2d in S (x is start)
    # 3. There exists d such that x-d in S and x-2d in S (x is end)
    
    # To speed this up, we can precompute for each x the set of pairs that would conflict.
    # However, dynamic updates are complex.
    
    # Alternative: Represent set as a set of integers.
    # Check validity of adding x:
    # For each y in S:
    #   if x > y: check if x + (x-y) in S  (y, x, x+(x-y))
    #   if x < y: check if y - (y-x) in S  (y-(y-x), y, x) -> let z = y-x, check y-z in S? No.
    #   Let's stick to the standard:
    #   A triple (a, b, c) is a 3-AP if a+c=2b.
    #   If we add x, it can be:
    #   - The middle: a = x-d, c = x+d. Need (x-d) in S and (x+d) in S.
    #   - The start: a = x, c = x+2d. Need (x+2d) in S.
    #   - The end: c = x, a = x-2d. Need (x-2d) in S.
    
    # This is O(|S|) per addition check. |S| is small (~30-40 for n=256).
    # So O(40) operations per move check. Very fast.
    
    def is_valid_addition(x, S):
        # S is a set of integers
        for d in range(1, n+1):
            # x is middle
            if (x - d) in S and (x + d) in S:
                return False
            # x is start (a=x, b=x+d, c=x+2d)
            if (x + 2*d) in S:
                return False
            # x is end (a=x-2d, b=x-d, c=x)
            if (x - 2*d) in S:
                return False
        return True

    def get_score(S):
        # Sum of reciprocals
        # Using float for speed during SA, exact Fraction for final result
        return sum(1.0 / x for x in S)

    # Initial solution: Greedy (same as seed) to ensure a good starting point
    S = set()
    for x in range(1, n + 1):
        if is_valid_addition(x, S):
            S.add(x)
    
    best_S = set(S)
    best_score = get_score(best_S)
    
    # Simulated Annealing
    current_S = set(S)
    current_score = get_score(current_S)
    
    T = 1.0
    cooling_rate = 0.995
    
    while time.time() - start_time < timeout:
        if T < 1e-4:
            break
            
        # Randomly pick an element to toggle
        x = random.randint(1, n)
        
        if x in current_S:
            # Remove x
            new_S = current_S - {x}
            new_score = current_score - (1.0 / x)
            delta = new_score - current_score
            # Always accept removals if they improve score? 
            # No, we want to maximize score, so removals decrease score.
            # We only accept if delta > 0 (improvement) or with probability exp(delta/T).
            # Since delta is negative for removals, we rarely accept them unless T is high.
            # This allows escaping local maxima where removing a "bad" element allows adding better ones later.
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                current_S = new_S
                current_score = new_score
                if current_score > best_score:
                    best_S = set(current_S)
                    best_score = current_score
        else:
            # Add x
            if is_valid_addition(x, current_S):
                new_S = current_S | {x}
                new_score = current_score + (1.0 / x)
                delta = new_score - current_score
                # delta is positive
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    current_S = new_S
                    current_score = new_score
                    if current_score > best_score:
                        best_S = set(current_S)
                        best_score = current_score
            # If invalid, we cannot add it. Do we accept the "move" as a no-op?
            # Usually in SA, invalid moves are rejected or treated as large penalty.
            # Here, we just skip the move (state doesn't change).
        
        T *= cooling_rate
        
        # Optional: Restart if stuck?
        # For now, simple geometric cooling.

    # Return the best found set
    # The task requires returning a nonempty subset.
    # The score is the floating serialization.
    # The evaluator calculates the score. We just return the list/set.
    return list(best_S)

# EVOLVE-BLOCK-END
