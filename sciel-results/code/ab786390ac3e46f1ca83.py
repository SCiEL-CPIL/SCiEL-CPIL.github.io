"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _precompute_triples():
    """
    Precompute all forbidden triples.
    Returns a dict: triple_map[x] = list of (u, v) such that {x, u, v} is a forbidden triple.
    """
    triple_map = {i: [] for i in range(1, 181)}
    for b in range(1, 181):
        for c in range(b + 1, 181):
            prod = b * c
            ssum = b + c
            if prod % ssum == 0:
                a = prod // ssum
                if a != b and a != c:
                    # Triple is {a, b, c}
                    triple_map[a].append((b, c))
                    triple_map[b].append((a, c))
                    triple_map[c].append((a, b))
    return triple_map

_TRIPLE_MAP = _precompute_triples()

def _is_valid_addition(x, selected):
    """Check if adding x to selected creates a forbidden triple."""
    for u, v in _TRIPLE_MAP[x]:
        if u in selected and v in selected:
            return False
    return True

def _greedy_construct(order):
    """Greedy: add elements in the given order if no triple is formed."""
    selected = set()
    for x in order:
        if _is_valid_addition(x, selected):
            selected.add(x)
    return sorted(selected)

def _local_search(initial, time_budget=10.0):
    """Local search: try to add elements, and if stuck, try swaps (remove 1, add 1+)."""
    current = set(initial)
    best_size = len(current)
    best_set = set(current)
    start = time.time()
    
    # Phase 1: Greedy Add
    improved = True
    while improved and time.time() - start < time_budget:
        improved = False
        for x in range(1, 181):
            if x in current:
                continue
            if _is_valid_addition(x, current):
                current.add(x)
                improved = True
                break # Restart to check for more additions
        
    if len(current) > best_size:
        best_size = len(current)
        best_set = set(current)
        
    # Phase 2: Swap Search (Remove 1, Try to Add 2)
    # This helps escape local minima where adding 1 is blocked, but removing 1 allows adding 2.
    for _ in range(50): # Limit swap attempts
        if time.time() - start > time_budget:
            break
        # Pick a random element to remove? Or one that blocks many?
        # For simplicity, try removing each element in current (or a subset)
        # To be fast, we'll just try a few random removals
        candidates_to_remove = list(current)
        random.shuffle(candidates_to_remove)
        
        for x in candidates_to_remove[:10]: # Try top 10 random removals
            if time.time() - start > time_budget:
                break
            trial = current - {x}
            # Try to add as many elements as possible to trial
            added = 0
            for y in range(1, 181):
                if y in trial:
                    continue
                if _is_valid_addition(y, trial):
                    trial.add(y)
                    added += 1
            if added > 1: # Net gain
                current = trial
                if len(current) > best_size:
                    best_size = len(current)
                    best_set = set(current)
                break # Restart swap loop with new current

    return sorted(best_set)

def construct():
    deadline = time.time() + 85.0  # 5s safety margin
    
    best = None
    best_size = 0
    
    # Strategy 1: Greedy Top-Down
    order_top = list(range(180, 0, -1))
    candidate = _greedy_construct(order_top)
    best = candidate
    best_size = len(candidate)
    
    # Strategy 2: Greedy Bottom-Up
    order_bot = list(range(1, 181))
    candidate = _greedy_construct(order_bot)
    if len(candidate) > best_size:
        best = candidate
        best_size = len(candidate)
        
    # Strategy 3: Randomized Greedy
    rng = random.Random(42)
    for attempt in range(500):
        if time.time() > deadline - 5.0:
            break
        elements = list(range(1, 181))
        rng.shuffle(elements)
        selected = set()
        for x in elements:
            if _is_valid_addition(x, selected):
                selected.add(x)
        if len(selected) > best_size:
            best = sorted(selected)
            best_size = len(selected)
            
    # Strategy 4: Local Search on Best
    if time.time() < deadline - 5.0:
        improved = _local_search(best, time_budget=min(deadline - time.time() - 2.0, 20.0))
        if len(improved) > best_size:
            best = improved
            best_size = len(improved)

    if best is None or len(best) == 0:
        return list(range(91, 181))
    
    return best

# EVOLVE-BLOCK-END
