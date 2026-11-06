"""Rewritten program for divisibility-antichain representations by smooth numbers."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    max_target = 128

    # Generate all 2,3,5-smooth numbers up to max_target
    smooth = []
    for n in range(1, max_target + 1):
        x = n
        while x % 2 == 0:
            x //= 2
        while x % 3 == 0:
            x //= 3
        while x % 5 == 0:
            x //= 5
        if x == 1:
            smooth.append(n)

    # Precompute divisibility relations
    # divs[a] = list of smooth numbers that a divides
    n_smooth = len(smooth)
    idx_of = {v: i for i, v in enumerate(smooth)}
    divs = [set() for _ in range(n_smooth)]
    for i, a in enumerate(smooth):
        for j, b in enumerate(smooth):
            if i != j and b % a == 0:
                divs[i].add(j)

    # For a set of indices, check if it's an antichain
    def is_antichain(indices):
        for i in indices:
            for j in indices:
                if i != j and j in divs[i]:
                    return False
        return True

    # DP approach: for each target n, find minimum number of terms
    # We'll use a BFS-like approach from small counts upward
    
    # Precompute all valid antichains of size 1, 2, 3, ... up to some limit
    # For each sum value, store the minimum count and the terms

    # Strategy: enumerate all combinations of smooth numbers that form antichains
    # and record the minimum count for each sum

    best_count = [max_target + 1] * (max_target + 1)
    best_terms = [None] * (max_target + 1)

    # Size 1: each smooth number represents itself
    for i, v in enumerate(smooth):
        if v <= max_target and 1 < best_count[v]:
            best_count[v] = 1
            best_terms[v] = [v]

    # Size 2: pairs of smooth numbers forming antichain
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            a, b = smooth[i], smooth[j]
            s = a + b
            if s > max_target:
                break
            # Check antichain: a doesn't divide b and b doesn't divide a
            if b % a != 0 and a % b != 0:
                if 2 < best_count[s]:
                    best_count[s] = 2
                    best_terms[s] = [a, b]

    # Size 3: triples of smooth numbers forming antichain
    # This is O(n^3) which could be large, but n_smooth is at most ~45
    # Let's be smart: for each pair (i,j) that's an antichain, try adding k
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            a, b = smooth[i], smooth[j]
            if b % a == 0 or a % b == 0:
                continue
            for k in range(j + 1, n_smooth):
                c = smooth[k]
                s = a + b + c
                if s > max_target:
                    break
                # Check c is not divisible by a or b, and doesn't divide a or b
                if c % a == 0 or a % c == 0:
                    continue
                if c % b == 0 or b % c == 0:
                    continue
                if 3 < best_count[s]:
                    best_count[s] = 3
                    best_terms[s] = [a, b, c]

    # Check time
    if time.time() > deadline - 5.0:
        # Fall back to seed-like approach for remaining
        return _fallback_construct(smooth, best_count, best_terms, max_target)

    # Size 4: quadruples
    # This gets expensive. Let's only do it if we have time and if there are gaps
    gaps = [n for n in range(1, max_target + 1) if best_count[n] > 4]
    if gaps:
        # Try to fill gaps with size 4+
        # For efficiency, use recursive search with pruning
        for target in gaps:
            if time.time() > deadline - 10.0:
                break
            found = _search_antichain(smooth, target, 4, max_count=6, 
                                       divs=divs, n_smooth=n_smooth)
            if found:
                best_count[target] = len(found)
                best_terms[target] = found

    # Check for any remaining gaps
    still_gaps = [n for n in range(1, max_target + 1) if best_count[n] > max_target]
    if still_gaps:
        for target in still_gaps:
            if time.time() > deadline - 5.0:
                break
            # Brute force with larger search
            found = _search_antichain(smooth, target, 4, max_count=10,
                                       divs=divs, n_smooth=n_smooth)
            if found:
                best_count[target] = len(found)
                best_terms[target] = found

    # Final fallback for any still-missing
    still_missing = [n for n in range(1, max_target + 1) if best_terms[n] is None]
    if still_missing:
        for target in still_missing:
            if time.time() > deadline - 2.0:
                break
            # Use the seed approach as last resort
            answer = None
            for count in range(4, 11):
                candidates = [x for x in smooth if x <= target]
                for terms in combinations(candidates, count):
                    if sum(terms) == target:
                        if is_antichain([idx_of[t] for t in terms]):
                            answer = list(terms)
                            break
                if answer:
                    break
            if answer:
                best_count[target] = len(answer)
                best_terms[target] = answer
            else:
                # Very last resort: allow any representation (shouldn't happen)
                best_count[target] = 1
                best_terms[target] = [target] if target in idx_of else [1]

    return [best_terms[n] for n in range(1, max_target + 1)]


def _search_antichain(smooth, target, min_count, max_count, divs, n_smooth):
    """Search for an antichain representation of target with min_count to max_count terms."""
    # Use backtracking
    best = None
    
    def backtrack(remaining, min_idx, count, current):
        nonlocal best
        if len(current) > max_count:
            return
        if remaining == 0:
            if len(current) >= min_count and (best is None or len(current) < len(best)):
                best = list(current)
            return
        if remaining < 0:
            return
        
        # Pruning: if we can't reach remaining with remaining slots
        slots_left = max_count - len(current)
        if slots_left <= 0:
            return
        # Minimum possible sum with slots_left terms (smallest smooth numbers)
        # Just use 1 as lower bound per term
        if remaining < slots_left:
            return
        
        for i in range(min_idx, n_smooth):
            v = smooth[i]
            if v > remaining:
                break
            # Check if v is compatible with all in current
            idx = i
            compatible = True
            for j in current:
                if j in divs[idx] or idx in divs[j]:
                    compatible = False
                    break
            if not compatible:
                continue
            current.append(idx)
            backtrack(remaining - v, i + 1, count + 1, current)
            current.pop()
            if best and len(best) == min_count:
                return
    
    backtrack(target, 0, 0, [])
    if best:
        return [smooth[i] for i in best]
    return None


def _fallback_construct(smooth, best_count, best_terms, max_target):
    """Fallback: use seed approach for any missing."""
    for n in range(1, max_target + 1):
        if best_terms[n] is None:
            answer = None
            for count in (3, 2, 1):
                candidates = [x for x in smooth if x <= n]
                for terms in combinations(candidates, count):
                    if sum(terms) == n:
                        # Check antichain
                        ok = True
                        for a, b in combinations(terms, 2):
                            if b % a == 0 or a % b == 0:
                                ok = False
                                break
                        if ok:
                            answer = list(terms)
                            break
                if answer:
                    break
            if answer:
                best_count[n] = len(answer)
                best_terms[n] = answer
            else:
                # Last resort
                best_count[n] = 1
                best_terms[n] = [n] if n in [x for x in smooth] else [1]
    return [best_terms[n] for n in range(1, max_target + 1)]


# EVOLVE-BLOCK-END
