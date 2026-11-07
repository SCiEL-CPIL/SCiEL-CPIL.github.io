"""Cardinality-separated subset sums: model-and-optimize approach.

Parameterize the set as a geometric-like progression, then locally optimize
to maximize size while maintaining the cardinality-separation property.
"""
import time
import random
from itertools import combinations

DEADLINE_SECONDS = 85.0  # safety margin before 90s kill


def _subset_sums_by_cardinality(S):
    """Return list: sums_by_k[k] = set of achievable sums with exactly k elements."""
    n = len(S)
    sums_by_k = [set() for _ in range(n + 1)]
    sums_by_k[0].add(0)
    for x in S:
        # iterate backwards to avoid using x more than once
        for k in range(n, 0, -1):
            if sums_by_k[k - 1]:
                for s in sums_by_k[k - 1]:
                    sums_by_k[k].add(s + x)
    return sums_by_k


def _is_valid(S):
    """Check no two subsets of different cardinalities share a sum."""
    sums_by_k = _subset_sums_by_cardinality(S)
    # For each sum, record which cardinalities achieve it
    sum_to_ks = {}
    for k, sums in enumerate(sums_by_k):
        for s in sums:
            if s in sum_to_ks:
                sum_to_ks[s].add(k)
            else:
                sum_to_ks[s] = {k}
    for s, ks in sum_to_ks.items():
        if len(ks) > 1:
            return False
    return True


def _incremental_check(current, candidate):
    """Check if adding candidate to current preserves validity.
    Uses the existing sums_by_k of current to do a quick check."""
    if not current:
        return True  # single element is always valid
    
    n = len(current)
    sums_by_k = _subset_sums_by_cardinality(current)
    
    # Build the full new sums_by_k
    new_sums_by_k = [set() for _ in range(n + 2)]
    for k in range(n + 1):
        new_sums_by_k[k] = set(sums_by_k[k])
    for k in range(1, n + 2):
        for s in sums_by_k[k - 1]:
            new_sums_by_k[k].add(s + candidate)
    
    # Check: for each sum, all cardinalities that achieve it must be the same
    sum_to_ks = {}
    for k in range(n + 2):
        for s in new_sums_by_k[k]:
            if s in sum_to_ks:
                sum_to_ks[s].add(k)
            else:
                sum_to_ks[s] = {k}
    
    for s, ks in sum_to_ks.items():
        if len(ks) > 1:
            return False
    return True


def _try_add(current, candidate):
    """Try adding candidate; return True if valid."""
    if candidate in current:
        return False
    if not (1 <= candidate <= 256):
        return False
    return _incremental_check(current, candidate)


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE_SECONDS
    
    def time_left():
        return deadline - time.time()
    
    best_set = []
    best_size = 0
    
    # Strategy 1: Greedy with multiple orderings
    # Key insight: to maximize cardinality separation, we want the sums for different
    # cardinalities to be in disjoint ranges. This is easier if elements grow fast.
    
    # Try deterministic ascending first
    for attempt in range(500):
        if time_left() < 1.0:
            break
        
        if attempt == 0:
            candidates = list(range(1, 257))
        elif attempt == 1:
            candidates = list(range(256, 0, -1))
        elif attempt == 2:
            # Start small, then jump
            candidates = list(range(1, 50)) + list(range(100, 257))
        elif attempt == 3:
            # Powers of 2 style
            candidates = [2**i for i in range(8) if 2**i <= 256] + list(range(3, 257))
        else:
            candidates = list(range(1, 257))
            random.shuffle(candidates)
        
        current = []
        for x in candidates:
            if time_left() < 0.5:
                break
            if len(current) >= 60:
                break
            if _try_add(current, x):
                current.append(x)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
        
        # If we've hit a high size, try to improve further
        if best_size >= 10 and time_left() < 30:
            break
    
    # Strategy 2: Local optimization - try to swap or add elements
    if time_left() > 5 and best_size > 0:
        current = best_set[:]
        current_set = set(current)
        
        # Try adding new elements
        for x in range(1, 257):
            if time_left() < 1.0:
                break
            if x in current_set:
                continue
            if _try_add(current, x):
                current.append(x)
                current_set.add(x)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
        
        # Try swapping: remove one element, try adding several others
        if time_left() > 10:
            for i in range(len(current)):
                if time_left() < 2.0:
                    break
                saved = current[i]
                current.pop(i)
                current_set.discard(saved)
                
                added_any = False
                for x in range(1, 257):
                    if time_left() < 1.0:
                        break
                    if x in current_set or x == saved:
                        continue
                    if _try_add(current, x):
                        current.append(x)
                        current_set.add(x)
                        added_any = True
                
                if len(current) > best_size:
                    best_size = len(current)
                    best_set = current[:]
                else:
                    # Restore
                    if not added_any:
                        current.insert(i, saved)
                        current_set.add(saved)
                    # If we did add something but didn't improve, keep the improved version
                    # Actually, if len(current) <= best_size, revert to best_set
                    # Let's just keep current for now and compare at the end
    
    # Strategy 3: If we have room, try a more structured approach
    # Build from scratch with a focus on creating wide gaps between cardinality sums
    if time_left() > 20 and best_size < 15:
        # Use a base-2 like construction but with some flexibility
        # Start with 1, then try to add the smallest valid element each time
        current = []
        for x in range(1, 257):
            if time_left() < 2.0:
                break
            if len(current) >= 60:
                break
            if _try_add(current, x):
                current.append(x)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
    
    # Final validation
    if best_set and not _is_valid(best_set):
        # Fallback: return empty or minimal valid set
        best_set = [1]
    
    # Ensure all elements are Python ints (not bools or numpy ints)
    result = []
    for x in best_set:
        if isinstance(x, bool):
            continue
        if not isinstance(x, int):
            continue
        result.append(x)
    
    return result
