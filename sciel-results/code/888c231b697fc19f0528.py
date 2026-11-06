"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import sys
from itertools import combinations

def construct():
    n = 192
    k = 3
    
    # Precompute conflict information using modular arithmetic
    # For each a, precompute for each pair (x, y) the residue class of z that would make a | (x+y+z)
    # z ≡ -(x+y) (mod a)
    
    # Strategy: Greedy independent set on a hypergraph
    # We'll use an efficient incremental check
    
    def can_add(selected_set, candidate):
        """Check if candidate can be added to selected_set without violating constraints."""
        selected_list = list(selected_set)
        
        # Check 1: For each existing 'a' in selected, does 'a' divide sum of any 3 distinct others including candidate?
        # The 3 others must be distinct from 'a'. If candidate is one of them, we need 2 others from selected (excluding a and candidate).
        for a in selected_list:
            if a == candidate:
                continue
            # Need to check: does a divide (x + y + candidate) for any pair (x,y) from selected, x != y, x != a, y != a, x != candidate, y != candidate
            for i, x in enumerate(selected_list):
                if x == a or x == candidate:
                    continue
                for j, y in enumerate(selected_list):
                    if j <= i:
                        continue
                    if y == a or y == candidate or y == x:
                        continue
                    if (x + y + candidate) % a == 0:
                        return False
        
        # Check 2: Does candidate divide sum of any 3 distinct others from selected?
        for triple in combinations(selected_list, 3):
            if (triple[0] + triple[1] + triple[2]) % candidate == 0:
                return False
        
        return True
    
    # Try multiple greedy strategies with different ordering heuristics
    best = []
    
    # Strategy 1: Descending order (largest first) - avoids small numbers that divide many sums
    selected = set()
    for x in range(n, 0, -1):
        if can_add(selected, x):
            selected.add(x)
    best = max(best, list(selected), key=len)
    
    # Strategy 2: Ascending order
    selected = set()
    for x in range(1, n + 1):
        if can_add(selected, x):
            selected.add(x)
    best = max(best, list(selected), key=len)
    
    # Strategy 3: Random restarts with different seeds
    random.seed(42)
    for trial in range(100):
        order = list(range(1, n + 1))
        random.shuffle(order)
        selected = set()
        for x in order:
            if can_add(selected, x):
                selected.add(x)
        best = max(best, list(selected), key=len)
    
    # Strategy 4: Prioritize numbers in (n/2, n] which have fewer divisors
    selected = set()
    upper_half = list(range(n // 2 + 1, n + 1))
    lower_half = list(range(1, n // 2 + 1))
    random.shuffle(upper_half)
    random.shuffle(lower_half)
    for x in upper_half + lower_half:
        if can_add(selected, x):
            selected.add(x)
    best = max(best, list(selected), key=len)
    
    # Strategy 5: Multiple random restarts with bias toward larger numbers
    for trial in range(100):
        # Create a weighted random order favoring larger numbers
        weights = [i for i in range(1, n + 1)]  # weight proportional to value
        order = []
        remaining = list(range(1, n + 1))
        while remaining:
            total_weight = sum(remaining)
            r = random.random() * total_weight
            cumulative = 0
            for idx, val in enumerate(remaining):
                cumulative += val
                if cumulative >= r:
                    order.append(val)
                    remaining.pop(idx)
                    break
        selected = set()
        for x in order:
            if can_add(selected, x):
                selected.add(x)
        best = max(best, list(selected), key=len)
    
    return best

# EVOLVE-BLOCK-END
