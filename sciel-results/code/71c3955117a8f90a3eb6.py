"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from itertools import combinations

def construct():
    n = 192
    best_set = []
    
    # Helper to get divisors of a number
    def get_divisors(num):
        divs = []
        i = 1
        while i * i <= num:
            if num % i == 0:
                divs.append(i)
                if i != num // i:
                    divs.append(num // i)
            i += 1
        return divs

    def check_valid_addition(candidate, selected):
        """
        Check if adding 'candidate' to 'selected' (list of ints) maintains validity.
        Constraints:
        1. candidate does not divide sum of any 3 distinct others in selected.
        2. No a in selected divides sum of candidate + 2 distinct others in selected (excluding a).
        """
        # Check 1: candidate | (x+y+z) for x,y,z in selected, distinct
        if len(selected) >= 3:
            for x, y, z in combinations(selected, 3):
                if (x + y + z) % candidate == 0:
                    return False
        
        # Check 2: a | (candidate + x + y) for a in selected, x,y in selected \ {a}
        if len(selected) >= 2:
            # Precompute pairs sums? No, just iterate.
            # Optimize: For each pair (x,y) in selected, compute V = candidate + x + y.
            # Check if any a in selected \ {x,y} divides V.
            selected_set = set(selected)
            for x, y in combinations(selected, 2):
                v = candidate + x + y
                # Find divisors of v
                divs = get_divisors(v)
                for a in divs:
                    if a in selected_set and a != x and a != y:
                        return False
        
        return True

    # Strategy 1: Descending order (Large numbers first)
    selected = []
    for c in range(n, 0, -1):
        if check_valid_addition(c, selected):
            selected.append(c)
    best_set = max(best_set, selected, key=len)
    
    # Strategy 2: Random restarts with bias towards larger numbers
    # We will run multiple random shuffles.
    # To improve efficiency, we limit the number of restarts based on time, but 100 is safe.
    random.seed(42)
    
    for _ in range(100):
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        # Bias: Sort by a random key that favors larger numbers?
        # Or just shuffle. The check is fast enough.
        
        selected = []
        for c in candidates:
            if check_valid_addition(c, selected):
                selected.append(c)
        
        if len(selected) > len(best_set):
            best_set = selected
            # Early stopping if we hit theoretical max? Max is 48.
            if len(best_set) == 48:
                break

    return best_set

# EVOLVE-BLOCK-END
