"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import sys

def _get_sums(elements):
    """Get all non-empty subset sums as a set. Returns None if duplicate sums exist."""
    n = len(elements)
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        if s in sums:
            return None  # duplicate sum
        sums.add(s)
    return sums

def _check_divisibility(sums):
    """Check if any two sums divide each other. Returns True if valid (no divisibility)."""
    if sums is None:
        return False
    sl = sorted(sums)
    for i in range(len(sl)):
        for j in range(i + 1, len(sl)):
            if sl[j] % sl[i] == 0:
                return False
    return True

def _is_valid(elements):
    """Full validation."""
    if not elements:
        return True
    if len(elements) != len(set(elements)):
        return False
    for x in elements:
        if not (1 <= x <= 256):
            return False
    sums = _get_sums(elements)
    if sums is None:
        return False
    return _check_divisibility(sums)

def _try_add(elements, candidate):
    """Try adding candidate to elements. Returns True if valid."""
    new_elements = elements + [candidate]
    if len(new_elements) != len(set(new_elements)):
        return False
    if not (1 <= candidate <= 256):
        return False
    sums = _get_sums(new_elements)
    if sums is None:
        return False
    return _check_divisibility(sums)

def _greedy_grow(start_elements, candidates, rng):
    """Greedy: try to add candidates in random order."""
    current = list(start_elements)
    remaining = [c for c in candidates if c not in current]
    rng.shuffle(remaining)
    
    for cand in remaining:
        if _try_add(current, cand):
            current.append(cand)
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    best = [31, 32, 34]  # fallback seed
    
    # Strategy: use large numbers to make subset sums large and close
    # Try various strategies with random restarts
    
    rng = random.Random(42)
    
    # Strategy 1: Start with large numbers, greedily add
    for trial in range(500):
        if time.time() > deadline:
            break
        
        # Pick a random subset of large numbers to start
        large_nums = list(range(180, 257))
        rng.shuffle(large_nums)
        
        # Try different starting sizes
        start_size = rng.randint(2, 5)
        start = large_nums[:start_size]
        
        # Make sure start is valid
        if not _is_valid(start):
            continue
        
        # Candidates: all numbers not in start
        candidates = [x for x in range(1, 257) if x not in start]
        
        result = _greedy_grow(start, candidates, rng)
        
        if len(result) > len(best):
            best = result
    
    # Strategy 2: Start with specific known-good large numbers
    # Try pairs of large numbers and grow
    for a in range(200, 257, 3):
        if time.time() > deadline:
            break
        for b in range(a + 1, 257, 3):
            if time.time() > deadline:
                break
            start = [a, b]
            if not _is_valid(start):
                continue
            candidates = [x for x in range(1, 257) if x not in start]
            result = _greedy_grow(start, candidates, rng)
            if len(result) > len(best):
                best = result
    
    # Strategy 3: Try to build from scratch with random ordering
    for trial in range(200):
        if time.time() > deadline:
            break
        
        # Random ordering of all numbers, biased toward large
        all_nums = list(range(1, 257))
        # Weight toward large numbers
        weighted = []
        for x in all_nums:
            weight = x / 256.0
            weighted.extend([x] * max(1, int(weight * 10)))
        rng.shuffle(weighted)
        
        current = []
        for cand in weighted:
            if cand in current:
                continue
            if len(current) >= 30:
                break
            if _try_add(current, cand):
                current.append(cand)
        
        if len(current) > len(best):
            best = current
    
    # Strategy 4: Local search - try to improve by swapping
    if len(best) > 0:
        if time.time() < deadline - 5:
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                for i in range(len(best)):
                    if time.time() > deadline - 2:
                        break
                    for x in range(1, 257):
                        if x in best:
                            continue
                        new_set = best[:i] + [x] + best[i+1:]
                        if _is_valid(new_set):
                            # Check if this leads to a better set via greedy
                            candidates = [c for c in range(1, 257) if c not in new_set]
                            result = _greedy_grow(new_set, candidates, rng)
                            if len(result) > len(best):
                                best = result
                                improved = True
                                break
                    if improved:
                        break
    
    # Final validation
    if not _is_valid(best):
        best = [31, 32, 34]
    
    # Ensure we return Python ints in a list
    return [int(x) for x in best]

# EVOLVE-BLOCK-END
