"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def _get_sums(elements):
    """Get all non-empty subset sums as a set."""
    n = len(elements)
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        sums.add(s)
    return sums

def _has_divisibility(sums):
    """Fast check: does any pair of distinct sums have divisibility?
    For each sum s, check if any multiple of s (other than s itself) is in sums."""
    sums_list = sorted(sums)
    max_s = sums_list[-1] if sums_list else 0
    # For each s, check multiples
    for s in sums_list:
        if s == 0:
            continue
        for m in range(2 * s, max_s + 1, s):
            if m in sums:
                return True
    return False

def _is_valid(elements):
    """Check if the set satisfies all constraints."""
    if not elements:
        return True
    if len(elements) != len(set(elements)):
        return False
    for x in elements:
        if not isinstance(x, int) or isinstance(x, bool) or not (1 <= x <= 256):
            return False
    sums = _get_sums(elements)
    # Check uniqueness of subset sums (no two subsets give same sum)
    # This is guaranteed if all elements are positive and we check via the set
    # But we need to verify: count of subset sums should be 2^n - 1
    n = len(elements)
    if len(sums) != (1 << n) - 1:
        return False
    # Check no divisibility
    if _has_divisibility(sums):
        return False
    return True

def _try_add(elements, sums, candidate):
    """Try to add candidate to elements. Returns (new_elements, new_sums) or (None, None)."""
    if candidate in elements:
        return None, None
    new_elements = elements + [candidate]
    # New sums = old_sums ∪ (old_sums + candidate) ∪ {candidate}
    shifted = {s + candidate for s in sums}
    # Check for collisions: if any s in sums equals s' + candidate for some s' in sums
    # i.e., if candidate + s' == s for some s, s' in sums
    # This means s - candidate in sums for some s in sums
    for s in sums:
        if s - candidate in sums or s == candidate:
            # Collision or duplicate
            pass
    # More careful: new_sums = sums ∪ {s + candidate for s in sums} ∪ {candidate}
    # We need len(new_sums) == len(sums) + len(sums) + 1 = 2*len(sums)+1
    # But actually new_sums should have exactly 2*(2^n - 1) + 1 = 2^(n+1) - 1 elements
    new_sums = sums | shifted | {candidate}
    expected_size = (1 << (len(elements) + 1)) - 1
    if len(new_sums) != expected_size:
        return None, None
    # Check divisibility
    if _has_divisibility(new_sums):
        return None, None
    return new_elements, new_sums

def _greedy_from(start_elements, rng, deadline):
    """Greedy grow from start_elements."""
    elements = list(start_elements)
    sums = _get_sums(elements) if elements else set()
    candidates = list(range(256, 0, -1))
    rng.shuffle(candidates)
    
    while time.time() < deadline:
        added = False
        for c in candidates:
            if c in elements:
                continue
            ne, ns = _try_add(elements, sums, c)
            if ne is not None:
                elements = ne
                sums = ns
                candidates.remove(c)
                added = True
                break
        if not added:
            break
    return elements

def _anneal(start_elements, rng, deadline, max_iter=5000):
    """Simulated annealing: try to grow by removing one element and adding another."""
    elements = list(start_elements)
    sums = _get_sums(elements) if elements else set()
    best = list(elements)
    best_len = len(elements)
    
    candidates_all = set(range(1, 257))
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        if len(elements) == 0:
            # Add random element
            c = rng.choice(list(candidates_all))
            ne, ns = _try_add(elements, sums, c)
            if ne is not None:
                elements = ne
                sums = ns
            continue
        
        # Choose a random move
        r = rng.random()
        if r < 0.5:
            # Try to add a random candidate
            avail = candidates_all - set(elements)
            if not avail:
                break
            c = rng.choice(list(avail))
            ne, ns = _try_add(elements, sums, c)
            if ne is not None:
                elements = ne
                sums = ns
                if len(elements) > best_len:
                    best = list(elements)
                    best_len = len(elements)
        else:
            # Try to swap: remove one random element, then try to add a different one
            if len(elements) < 2:
                continue
            rem_idx = rng.randrange(len(elements))
            removed = elements[rem_idx]
            new_elements = elements[:rem_idx] + elements[rem_idx+1:]
            new_sums = _get_sums(new_elements) if new_elements else set()
            
            # Now try to add a candidate (including possibly the removed one)
            avail = candidates_all - set(new_elements)
            if avail:
                # Try several candidates
                shuffled_avail = list(avail)
                rng.shuffle(shuffled_avail)
                for c in shuffled_avail[:10]:
                    ne, ns = _try_add(new_elements, new_sums, c)
                    if ne is not None:
                        elements = ne
                        sums = ns
                        if len(elements) > best_len:
                            best = list(elements)
                            best_len = len(elements)
                        break
                else:
                    # If we couldn't add anything, revert
                    pass
            # If we removed and didn't re-add, we have a smaller set; that's ok for SA
            # But let's try to recover by adding back
            if len(elements) < len(new_elements):
                # We lost an element; try to add something
                avail2 = candidates_all - set(elements)
                if avail2:
                    for c in list(avail2)[:5]:
                        ne, ns = _try_add(elements, sums, c)
                        if ne is not None:
                            elements = ne
                            sums = ns
                            break
    
    return best

def _backtrack_search(depth, current_elements, current_sums, deadline, rng):
    """Backtracking with pruning to find a valid set of given depth."""
    if time.time() > deadline:
        return None
    if depth == 0:
        return list(current_elements)
    
    # Try candidates in randomized order
    candidates = list(range(256, 0, -1))
    rng.shuffle(candidates)
    
    for c in candidates:
        if c in current_elements:
            continue
        ne, ns = _try_add(current_elements, current_sums, c)
        if ne is not None:
            result = _backtrack_search(depth - 1, ne, ns, deadline, rng)
            if result is not None:
                return result
    return None

def construct():
    deadline = time.time() + 85  # 85 second budget, 5s margin
    
    best = [31, 32, 34]
    best_len = len(best)
    
    rng = random.Random(42)
    
    # Strategy 1: Multiple greedy restarts with different random seeds
    for trial in range(200):
        if time.time() > deadline:
            break
        rng_t = random.Random(1000 + trial)
        # Start with 1-3 large random elements
        start_size = rng_t.randint(1, 3)
        start = rng_t.sample(range(128, 257), start_size)
        if not _is_valid(start):
            continue
        result = _greedy_from(start, rng_t, deadline)
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    # Strategy 2: Simulated annealing from good starting points
    for trial in range(50):
        if time.time() > deadline:
            break
        rng_a = random.Random(5000 + trial)
        start_size = rng_a.randint(2, 4)
        start = rng_a.sample(range(100, 257), start_size)
        if not _is_valid(start):
            continue
        result = _anneal(start, rng_a, deadline, max_iter=3000)
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    # Strategy 3: Backtracking for smaller depths
    for depth in range(2, 8):
        if time.time() > deadline:
            break
        if depth <= best_len:
            continue
        rng_b = random.Random(9000 + depth)
        result = _backtrack_search(depth, [], set(), deadline, rng_b)
        if result is not None and len(result) > best_len:
            best = result
            best_len = len(result)
    
    # Final validation
    if not _is_valid(best):
        # Fallback
        best = [31, 32, 34]
    
    # Ensure return type is list of Python ints
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
