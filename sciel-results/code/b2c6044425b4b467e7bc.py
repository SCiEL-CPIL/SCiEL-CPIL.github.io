import time
import random
from itertools import combinations

DEADLINE = 85.0  # safety margin from 90s kill

def _subset_sums(S):
    """Return set of all non-empty subset sums of S."""
    sums = {0}
    for x in S:
        sums |= {s + x for s in sums}
    sums.discard(0)
    return sums

def _is_valid(S):
    """Check if S satisfies all constraints."""
    n = len(S)
    if n == 0:
        return True
    if len(set(S)) != n:
        return False
    for x in S:
        if not (1 <= x <= 256):
            return False
    if not all(isinstance(x, int) for x in S):
        return False
    
    sums = _subset_sums(S)
    if len(sums) != (1 << n) - 1:
        return False  # collision detected
    
    # Check divisibility
    s_list = sorted(sums)
    is_sum = set(s_list)
    for s in s_list:
        for k in range(2, 256 * n // s + 1):
            if s * k in is_sum:
                return False
    return True

def _try_add(S, x):
    """Try to add x to S, check validity incrementally."""
    if x in S:
        return None
    if not (1 <= x <= 256):
        return None
    
    # Get current sums
    sums = _subset_sums(S)
    
    # New sums would be sums + {x + s for s in sums}
    new_sums = set(sums)
    for s in sums:
        new_sums.add(s + x)
    
    # Check for collision (excluding 0)
    if len(new_sums - {0}) != len(sums) + (1 << len(S)) - 1:
        return None
    
    # Check divisibility
    ns_list = sorted(new_sums - {0})
    for s in ns_list:
        for k in range(2, 256 * (len(S)+1) // s + 1):
            if s * k in new_sums:
                return None
    
    return S + [x]

def _greedy_grow(start, deadline):
    """Grow a set greedily."""
    S = list(start)
    current_sums = _subset_sums(S)
    max_sum = max(current_sums) if current_sums else 0
    
    while time.time() < deadline:
        best_x = None
        best_S = None
        candidates = [x for x in range(1, 257) if x not in S]
        random.shuffle(candidates)
        
        for x in candidates:
            new_S = _try_add(S, x)
            if new_S is not None:
                new_sums = _subset_sums(new_S)
                new_max = max(new_sums)
                if best_S is None or new_max < max_sum:
                    best_x = x
                    best_S = new_S
                    if new_max < max_sum:
                        current_sums = new_sums
                        max_sum = new_max
        
        if best_S is None:
            break
        
        S = best_S
        current_sums = _subset_sums(S)
        max_sum = max(current_sums)
    
    return S

def _simulated_annealing(deadline):
    """Use simulated annealing to find larger sets."""
    # Start from a known good set
    best_S = [31, 32, 34]
    current_S = list(best_S)
    current_sums = _subset_sums(current_S)
    
    T = 10.0
    cooling = 0.995
    
    while time.time() < deadline and T > 0.1:
        # Pick a random move
        move_type = random.random()
        if move_type < 0.4:
            # Add a random element
            candidates = [x for x in range(1, 257) if x not in current_S]
            if candidates:
                x = random.choice(candidates)
                new_S = _try_add(current_S, x)
                if new_S is not None:
                    if len(new_S) > len(best_S):
                        best_S = new_S
                    new_sums = _subset_sums(new_S)
                    # Accept with some probability
                    if len(new_S) > len(current_S) or random.random() < min(1.0, T):
                        current_S = new_S
                        current_sums = new_sums
        elif move_type < 0.7:
            # Remove a random element
            if len(current_S) > 2:
                idx = random.randint(0, len(current_S) - 1)
                new_S = current_S[:idx] + current_S[idx+1:]
                new_sums = _subset_sums(new_S)
                # Always accept removal if it keeps valid
                if _is_valid(new_S):
                    if len(new_S) > len(best_S):
                        best_S = new_S
                    if random.random() < min(1.0, T * 0.5):
                        current_S = new_S
                        current_sums = new_sums
        else:
            # Swap: remove one, add another
            if len(current_S) >= 2:
                idx = random.randint(0, len(current_S) - 1)
                old_x = current_S[idx]
                new_S = current_S[:idx] + current_S[idx+1:]
                candidates = [x for x in range(1, 257) if x not in new_S]
                if candidates:
                    x = random.choice(candidates)
                    new_S = _try_add(new_S, x)
                    if new_S is not None:
                        if len(new_S) > len(best_S):
                            best_S = new_S
                        new_sums = _subset_sums(new_S)
                        if len(new_S) >= len(current_S) or random.random() < min(1.0, T):
                            current_S = new_S
                            current_sums = new_sums
        
        T *= cooling
    
    return best_S

def _backtrack_search(max_size, deadline):
    """Backtracking search for sets of given size."""
    found = None
    # Try to build a set of max_size
    # Use a smarter ordering: start with large numbers
    candidates = list(range(256, 0, -1))
    
    def backtrack(S, idx, depth):
        nonlocal found
        if found is not None or time.time() > deadline:
            return
        
        if depth == max_size:
            if _is_valid(S):
                found = S
                return
        
        # Limit branching
        for i in range(idx, min(idx + 30, len(candidates))):
            x = candidates[i]
            new_S = _try_add(S, x)
            if new_S is not None:
                backtrack(new_S, i + 1, depth + 1)
                if found is not None:
                    return
    
    backtrack([], 0, 0)
    return found

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Start with known valid set
    best_set = [31, 32, 34]
    
    # Phase 1: Greedy growth from multiple seeds
    seeds = [
        [200, 220, 240],
        [150, 180, 210],
        [100, 150, 200],
        [50, 100, 200],
        [31, 32, 34],
        [100, 101, 102],
        [128, 192, 256],
    ]
    
    for seed in seeds:
        if time.time() > deadline - 20:
            break
        S = _greedy_grow(seed, min(time.time() + 5, deadline - 10))
        if len(S) > len(best_set):
            best_set = S
    
    # Phase 2: Simulated annealing from best
    if time.time() < deadline - 15:
        S = _simulated_annealing(deadline - 10)
        if len(S) > len(best_set):
            best_set = S
    
    # Phase 3: Backtracking for larger sizes
    for target_size in range(len(best_set) + 1, 16):
        if time.time() > deadline - 5:
            break
        S = _backtrack_search(target_size, deadline - 3)
        if S is not None:
            best_set = S
            break
    
    # Final validation
    if _is_valid(best_set):
        return list(best_set)
    else:
        # Fallback to known good set
        return [31, 32, 34]
