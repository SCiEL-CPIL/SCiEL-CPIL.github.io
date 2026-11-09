"""Find maximum cardinality dissociated reciprocal subset of [1,22].

Structural approach:
- Represent reciprocal sums as exact integers via LCM(1..22).
- Use a fast collision check: adding element x to set S is valid iff
  no existing subset sum s satisfies (s + w_x) in S.
- Search via randomized greedy construction + local search (add/swap moves)
  with simulated-annealing-style acceptance to escape local optima.
"""
import time
import random

# LCM of 1..22
_LCM = 232792560

def _weights(subset):
    return [_LCM // x for x in subset]

def _check_add(sums_set, v):
    """Check if adding weight v to the set of subset sums creates a collision.
    Returns True if valid (no collision)."""
    for s in sums_set:
        if (s + v) in sums_set:
            return False
    return True

def _build_sums_set(subset):
    """Build the set of all subset sums for the given subset."""
    sums = {0}
    for x in subset:
        v = _LCM // x
        sums |= {s + v for s in sums}
    return sums

def _is_dissociated(subset):
    """Check if all reciprocal subset sums are distinct."""
    if not subset:
        return True
    sums = {0}
    for x in subset:
        v = _LCM // x
        new_sums = {s + v for s in sums}
        # Collision if any new sum already exists
        if new_sums & sums:
            return False
        sums |= new_sums
    return True

def _greedy_build(order, deadline):
    """Build a dissociated set greedily following the given order."""
    current = []
    sums = {0}
    for x in order:
        if time.time() > deadline:
            break
        v = _LCM // x
        if _check_add(sums, v):
            current.append(x)
            sums |= {s + v for s in sums}
    return current, sums

def _local_search_add(subset, sums, candidates, deadline):
    """Try to add elements from candidates to subset. Returns updated subset, sums, size."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in candidates:
            if x in subset:
                continue
            v = _LCM // x
            if _check_add(sums, v):
                subset.append(x)
                sums |= {s + v for s in sums}
                improved = True
                break
    return subset, sums

def _swap_improve(subset, sums, universe, deadline, max_swaps=500):
    """Try swap moves: remove one element, add another. Returns best found."""
    best_subset = subset[:]
    best_sums = sums
    best_size = len(subset)
    
    for _ in range(max_swaps):
        if time.time() > deadline:
            break
        
        # Try all possible swaps
        for i, out_elem in enumerate(subset):
            if time.time() > deadline:
                break
            for in_elem in universe:
                if in_elem in subset:
                    continue
                
                # Build new set by removing out_elem and adding in_elem
                new_subset = subset[:i] + subset[i+1:] + [in_elem]
                
                # Check if valid
                if _is_dissociated(new_subset):
                    new_sums = _build_sums_set(new_subset)
                    if len(new_subset) > best_size:
                        best_subset = new_subset
                        best_sums = new_sums
                        best_size = len(new_subset)
                        # Restart with improved set
                        subset = new_subset
                        sums = new_sums
                        break
            else:
                continue
            break
    
    return best_subset, best_sums, best_size

def construct():
    deadline = time.time() + 82.0  # Safety margin before 90s kill
    
    universe = list(range(1, 23))
    
    # Start with a known good seed
    best_subset = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_sums = _build_sums_set(best_subset)
    best_size = len(best_subset)
    
    # Verify seed
    if not _is_dissociated(best_subset):
        # Fallback: find a valid set from scratch
        for size in range(9, 0, -1):
            import itertools
            found = False
            for combo in itertools.combinations(range(1, 23), size):
                if _is_dissociated(list(combo)):
                    best_subset = list(combo)
                    best_sums = _build_sums_set(best_subset)
                    best_size = size
                    found = True
                    break
            if found:
                break
    
    # Strategy 1: Greedy with multiple random orderings
    random.seed(42)
    
    for attempt in range(300):
        if time.time() > deadline:
            break
        
        # Try different orderings
        shuffled = universe[:]
        random.shuffle(shuffled)
        
        # Sometimes bias toward larger elements (smaller weights)
        if attempt % 3 == 0:
            shuffled.sort(key=lambda x: -x)  # Descending: larger numbers first
        elif attempt % 3 == 1:
            shuffled.sort(key=lambda x: x)   # Ascending: smaller numbers first
        # else: already shuffled
        
        current, cur_sums = _greedy_build(shuffled, deadline)
        
        # Local search: try to add more elements
        current, cur_sums = _local_search_add(current, cur_sums, universe, deadline)
        
        if len(current) > best_size:
            best_subset = current[:]
            best_sums = cur_sums
            best_size = len(current)
    
    # Strategy 2: Simulated annealing-style local search on best_subset
    if time.time() < deadline:
        # Temperature-based acceptance
        temperature = 1.0
        cooling = 0.95
        
        current_subset = best_subset[:]
        current_sums = best_sums
        current_size = best_size
        
        for step in range(2000):
            if time.time() > deadline:
                break
            
            if not current_subset:
                break
            
            # Random move: add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.3:
                # Add a random element not in set
                available = [x for x in universe if x not in current_subset]
                if not available:
                    continue
                in_elem = random.choice(available)
                new_subset = current_subset + [in_elem]
            elif move_type < 0.5:
                # Remove a random element
                if len(current_subset) <= 1:
                    continue
                out_idx = random.randint(0, len(current_subset) - 1)
                new_subset = current_subset[:out_idx] + current_subset[out_idx+1:]
            else:
                # Swap: remove one, add another
                if not current_subset:
                    continue
                out_idx = random.randint(0, len(current_subset) - 1)
                out_elem = current_subset[out_idx]
                available = [x for x in universe if x not in current_subset]
                if not available:
                    continue
                in_elem = random.choice(available)
                new_subset = current_subset[:out_idx] + current_subset[out_idx+1:] + [in_elem]
            
            # Evaluate new state
            if _is_dissociated(new_subset):
                new_size = len(new_subset)
                # Accept if better, or with probability based on temperature
                if new_size >= current_size:
                    current_subset = new_subset
                    current_sums = _build_sums_set(current_subset)
                    current_size = new_size
                else:
                    # Accept worse with probability exp(-1/temperature)
                    if random.random() < (1.0 / (temperature * 10)):
                        current_subset = new_subset
                        current_sums = _build_sums_set(current_subset)
                        current_size = new_size
            
            # Update best
            if current_size > best_size:
                best_subset = current_subset[:]
                best_sums = current_sums
                best_size = current_size
            
            temperature *= cooling
        
        # Final local search on best
        if time.time() < deadline:
            best_subset, best_sums = _local_search_add(best_subset, best_sums, universe, deadline)
            best_size = len(best_subset)
    
    return best_subset
