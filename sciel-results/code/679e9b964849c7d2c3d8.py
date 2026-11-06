"""Rewritten program: structured search for large sets without subset averages."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

MAX_N = 512
TARGET_SIZE = 48
DEADLINE_SEC = 85.0  # safety margin under 90s wall-clock


def _is_mean_conflict(a, others):
    """Check if a is the exact arithmetic mean of some subset of others (size >= 2).
    Returns True if conflict exists.
    """
    n = len(others)
    if n < 2:
        return False
    
    # For each subset size k >= 2, check if any subset sums to k*a
    # Use dynamic programming: dp[s] = bitmask where bit k is set if a subset of size k sums to s
    max_sum = sum(others)
    if max_sum < 2 * a:
        return False
    if max_sum > 48 * 512:  # sanity
        max_sum = 48 * 512
    
    # dp[s] is a bitmask where bit k is set if a subset of size k sums to s
    dp = [0] * (max_sum + 1)
    dp[0] = 1  # size 0, sum 0
    
    for x in others:
        # iterate sums in reverse to avoid using x twice
        for s in range(max_sum - x, -1, -1):
            if dp[s]:
                dp[s + x] |= (dp[s] << 1)
    
    # Check if for any k >= 2, bit k is set in dp[k*a]
    for k in range(2, n + 1):
        target_sum = k * a
        if target_sum <= max_sum and (dp[target_sum] >> k) & 1:
            return True
    return False


def _is_valid_set(selected):
    """Check if no element in selected is the mean of any subset of >= 2 other elements."""
    for i, a in enumerate(selected):
        others = [x for j, x in enumerate(selected) if j != i]
        if _is_mean_conflict(a, others):
            return False
    return True


def _is_valid_addition(new_elem, selected):
    """Check if adding new_elem to selected keeps the set valid.
    Need to check:
    1. new_elem is not the mean of any subset of >= 2 elements from selected
    2. No existing element becomes the mean of a subset involving new_elem
    """
    # Check 1: new_elem not mean of subset of selected
    if _is_mean_conflict(new_elem, selected):
        return False
    # Check 2: for each existing element a, check if a is mean of subset of (selected - {a}) + {new_elem}
    for i, a in enumerate(selected):
        others = [x for j, x in enumerate(selected) if j != i]
        combined_others = others + [new_elem]
        if _is_mean_conflict(a, combined_others):
            return False
    return True


def _greedy_construct(order=None):
    """Greedy construction: add elements in given order (or natural order) if they don't create conflicts."""
    if order is None:
        order = list(range(1, MAX_N + 1))
    selected = []
    for x in order:
        if _is_valid_addition(x, selected):
            selected.append(x)
        if len(selected) >= TARGET_SIZE:
            break
    return selected


def _random_greedy(seed):
    """Randomized greedy: shuffle candidates, add if valid."""
    rng = random.Random(seed)
    candidates = list(range(1, MAX_N + 1))
    rng.shuffle(candidates)
    return _greedy_construct(candidates)


def _local_search(selected, seed, time_limit):
    """Try to improve selected by swapping elements."""
    rng = random.Random(seed)
    start = time.time()
    best = list(selected)
    best_size = len(best)
    best_set = set(best)
    
    for _ in range(2000):
        if time.time() - start > time_limit:
            break
        
        if len(best) < TARGET_SIZE:
            # Try adding a random candidate
            candidates = [x for x in range(1, MAX_N + 1) if x not in best_set]
            if not candidates:
                break
            x = rng.choice(candidates)
            if _is_valid_addition(x, best):
                best.append(x)
                best_set.add(x)
                best_size = len(best)
        else:
            # Try swapping
            i = rng.randrange(len(best))
            removed = best[i]
            candidates = [x for x in range(1, MAX_N + 1) if x not in best_set and x != removed]
            if not candidates:
                break
            x = rng.choice(candidates)
            temp = [y for y in best if y != removed]
            if _is_valid_addition(x, temp):
                best = temp + [x]
                best_set = set(best)
    return best


def _conflict_graph_approach(time_limit):
    """Build a conflict graph and find a large independent set.
    Two elements conflict if one is the mean of a subset involving the other.
    Actually, we need: no element is the mean of a subset of >= 2 OTHER elements.
    So conflicts are not just pairwise - they involve subsets.
    This approach may not directly apply. Instead, use a different strategy.
    """
    # Alternative: use a simulated annealing approach
    rng = random.Random(int(time.time() * 1000) % 100000)
    start = time.time()
    
    # Start with a random valid set
    best = _random_greedy(0)
    best_set = set(best)
    best_size = len(best)
    
    # Simulated annealing: try to add elements, remove if conflicts arise
    current = list(best)
    current_set = set(current)
    current_size = len(current)
    temperature = 50.0
    cooling = 0.99
    
    while time.time() - start < time_limit and current_size < TARGET_SIZE:
        # Try to add a random element
        candidates = [x for x in range(1, MAX_N + 1) if x not in current_set]
        if not candidates:
            break
        x = rng.choice(candidates)
        
        if _is_valid_addition(x, current):
            current.append(x)
            current_set.add(x)
            current_size = len(current)
            if current_size > best_size:
                best = list(current)
                best_set = set(current)
                best_size = current_size
        else:
            # Try a swap: remove a random element, add x
            if current:
                i = rng.randrange(len(current))
                removed = current[i]
                temp = [y for y in current if y != removed]
                if _is_valid_addition(x, temp):
                    current = temp + [x]
                    current_set = set(current)
                    current_size = len(current)
                    if current_size > best_size:
                        best = list(current)
                        best_set = set(current)
                        best_size = current_size
        
        temperature *= cooling
    
    return best


def construct():
    start_time = time.time()
    
    # Phase 1: Greedy construction in natural order
    best = _greedy_construct()
    best_size = len(best)
    best_set = set(best)
    
    # Phase 2: Try multiple random greedy approaches
    rng = random.Random(42)
    for seed in range(30):
        if time.time() - start_time > DEADLINE_SEC - 15:
            break
        candidate = _random_greedy(seed)
        if len(candidate) > best_size:
            best = candidate
            best_size = len(best)
            best_set = set(best)
    
    # Phase 3: Local search to try to reach 48
    if best_size < TARGET_SIZE:
        remaining_time = DEADLINE_SEC - (time.time() - start_time)
        if remaining_time > 5:
            improved = _local_search(best, 123, min(remaining_time - 3, 30))
            if len(improved) > best_size:
                best = improved
                best_size = len(best)
                best_set = set(best)
    
    # Phase 4: Simulated annealing approach
    if best_size < TARGET_SIZE:
        remaining_time = DEADLINE_SEC - (time.time() - start_time)
        if remaining_time > 10:
            improved = _conflict_graph_approach(min(remaining_time - 3, 25))
            if len(improved) > best_size:
                best = improved
                best_size = len(best)
                best_set = set(best)
    
    # Final verification
    if not _is_valid_set(best):
        # Fallback: rebuild greedily with full verification
        best = _greedy_construct()
    
    return best
# EVOLVE-BLOCK-END
