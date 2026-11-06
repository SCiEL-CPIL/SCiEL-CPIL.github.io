"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def _can_add(S, x):
    """Check if x can be added to set S without creating a 3-AP."""
    for a in S:
        # x is middle: a, x, b => b = 2x - a
        b = 2 * x - a
        if b != a and b != x and b in S:
            return False
        # x is last: b, a, x => b = 2a - x
        b = 2 * a - x
        if b != a and b != x and b in S:
            return False
    return True

def _reciprocal_sum(selected):
    return sum(1.0 / x for x in selected)

def _greedy_ascending():
    S = set()
    for x in range(1, 257):
        if _can_add(S, x):
            S.add(x)
    return S

def _local_search(selected, deadline):
    """Local search with add, swap, and kick operations."""
    S = set(selected)
    best_S = set(S)
    best_sum = _reciprocal_sum(S)
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Add any missing element that fits
        for x in range(1, 257):
            if x not in S and _can_add(S, x):
                S.add(x)
                improved = True
                new_sum = _reciprocal_sum(S)
                if new_sum > best_sum:
                    best_sum = new_sum
                    best_S = set(S)
        
        # Phase 2: Try swaps - remove y, add x where x < y (so 1/x > 1/y)
        swap_improved = True
        while swap_improved and time.time() < deadline:
            swap_improved = False
            best_gain = 0
            best_x = None
            best_y = None
            for x in range(1, 257):
                if x in S:
                    continue
                for y in list(S):
                    if y <= x:
                        continue
                    gain = 1.0 / x - 1.0 / y
                    if gain <= best_gain:
                        continue
                    # Check if (S - {y}) U {x} is valid
                    S_test = S - {y}
                    if _can_add(S_test, x):
                        best_gain = gain
                        best_x = x
                        best_y = y
            if best_x is not None:
                S.discard(best_y)
                S.add(best_x)
                swap_improved = True
                new_sum = _reciprocal_sum(S)
                if new_sum > best_sum:
                    best_sum = new_sum
                    best_S = set(S)
                improved = True
        
        # Phase 3: If no improvement, try a kick
        if not improved:
            # Random kick: remove 3-7 random elements, then re-add greedily
            if len(S) > 15:
                kick_size = random.randint(3, 7)
                to_remove = random.sample(list(S), min(kick_size, len(S) - 10))
                for r in to_remove:
                    S.discard(r)
                # Re-add greedily
                for x in range(1, 257):
                    if x not in S and _can_add(S, x):
                        S.add(x)
                new_sum = _reciprocal_sum(S)
                if new_sum > best_sum:
                    best_sum = new_sum
                    best_S = set(S)
    
    return best_S

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Start with greedy ascending
    initial = _greedy_ascending()
    
    # Apply local search
    result = _local_search(initial, deadline)
    
    return list(result)
# EVOLVE-BLOCK-END
