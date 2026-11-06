"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from itertools import combinations
import time
import random

def _is_valid_addition(selected, x):
    """Check if adding x to selected maintains the constraint.
    For each element a in selected ∪ {x}, no 3 distinct OTHER elements sum to a multiple of a.
    """
    candidate = selected + [x]
    n = len(candidate)
    # Precompute all triples' sums for efficiency
    for i, a in enumerate(candidate):
        others = [candidate[j] for j in range(n) if j != i]
        for combo in combinations(others, 3):
            if sum(combo) % a == 0:
                return False
    return True

def _greedy(order):
    """Greedy construction following a given order of candidates."""
    selected = []
    for x in order:
        if _is_valid_addition(selected, x):
            selected.append(x)
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin before 90s kill
    
    # Try multiple different orderings to find better solutions
    best = []
    
    # Strategy 1: Descending order (original)
    best = _greedy(range(192, 0, -1))
    if time.time() > deadline:
        return best
    
    # Strategy 2: Try various heuristic orderings
    rng = random.Random(42)
    
    # Strategy 3: Random restarts with different seeds
    for seed in range(50):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        # Mix of large numbers first with some randomness
        candidates = list(range(192, 0, -1))
        # Partially shuffle: keep top 50 mostly in order, shuffle the rest
        top = candidates[:50]
        rest = candidates[50:]
        rng.shuffle(rest)
        order = top + rest
        result = _greedy(order)
        if len(result) > len(best):
            best = result
    
    # Strategy 4: Try ascending then descending hybrid
    if time.time() < deadline:
        for seed in range(30):
            if time.time() > deadline:
                break
            rng = random.Random(1000 + seed)
            candidates = list(range(192, 0, -1))
            # Weight toward larger numbers
            weighted = []
            for c in candidates:
                weight = c / 192.0
                for _ in range(int(weight * 3) + 1):
                    weighted.append(c)
            # Remove duplicates preserving order
            seen = set()
            unique_weighted = []
            for w in weighted:
                if w not in seen:
                    seen.add(w)
                    unique_weighted.append(w)
            rng.shuffle(unique_weighted)
            result = _greedy(unique_weighted)
            if len(result) > len(best):
                best = result
    
    # Strategy 5: Try to add more elements by trying different insertion points
    if time.time() < deadline:
        # For the best solution, try adding each excluded element
        best_set = set(best)
        for x in range(192, 0, -1):
            if x in best_set:
                continue
            if time.time() > deadline:
                break
            if _is_valid_addition(best, x):
                best.append(x)
                best_set.add(x)
    
    return best
# EVOLVE-BLOCK-END
