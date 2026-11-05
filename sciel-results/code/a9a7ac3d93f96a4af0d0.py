"""Improved program: integer-scaled backtracking with smarter heuristics to maximize
dissociated reciprocal set cardinality within [1,22]."""
import time
from math import gcd
from functools import lru_cache
import random

# Deadline guard: finish well before 90s wall-clock
_START = time.time()
_DEADLINE = 80.0  # safety margin

def _time_ok():
    return (time.time() - _START) < _DEADLINE


def construct():
    """Return a nonempty subset of [1,22] with all reciprocal subset sums distinct,
    maximizing cardinality."""
    LCM = 232792560
    weights = [LCM // k for k in range(1, 23)]
    indices = list(range(22))
    
    def can_add(subset_sums, w):
        """Check if adding weight w maintains dissociation. Returns new sums or None."""
        for s in subset_sums:
            if (s + w) in subset_sums:
                return None
        return subset_sums | {s + w for s in subset_sums}
    
    def greedy(order, subset_sums, chosen):
        """Greedy add elements from order starting with current state."""
        for i in order:
            if not _time_ok():
                break
            new_sums = can_add(subset_sums, weights[i])
            if new_sums is not None:
                subset_sums = new_sums
                chosen.append(i)
        return subset_sums, chosen
    
    def backtrack(current_sums, chosen, remaining, target_size, depth=0):
        """DFS to build a set of size target_size."""
        if not _time_ok():
            return None
        if len(chosen) == target_size:
            return chosen[:]
        if len(chosen) + len(remaining) < target_size:
            return None
        
        # Sort remaining by weight descending for better pruning
        remaining_sorted = sorted(remaining, key=lambda i: weights[i], reverse=True)
        
        for j, i in enumerate(remaining_sorted):
            if not _time_ok():
                return None
            new_sums = can_add(current_sums, weights[i])
            if new_sums is not None:
                result = backtrack(new_sums, chosen + [i], remaining_sorted[j+1:], target_size, depth+1)
                if result is not None:
                    return result
        return None
    
    best_size = 0
    best_set = None
    best_sums = {0}
    
    # Phase 1: Greedy with many different orderings
    # Heuristic orderings
    orderings = [
        sorted(indices, key=lambda i: weights[i], reverse=True),  # weight desc
        sorted(indices, key=lambda i: weights[i]),                  # weight asc
        sorted(indices, key=lambda i: i+1, reverse=True),           # k desc
        sorted(indices, key=lambda i: i+1),                         # k asc
        sorted(indices, key=lambda i: weights[i] % 1000, reverse=True),  # pseudo-random
    ]
    
    # Add random orderings
    for seed in range(30):
        r = random.Random(seed)
        idx = list(range(22))
        r.shuffle(idx)
        orderings.append(idx)
    
    for order in orderings:
        if not _time_ok():
            break
        subset_sums = {0}
        chosen = []
        subset_sums, chosen = greedy(order, subset_sums, chosen)
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = chosen[:]
            best_sums = subset_sums
    
    # Phase 2: Backtracking to extend best solution
    for target in range(best_size + 1, 14):
        if not _time_ok():
            break
        found = False
        # Try backtracking from empty with different orderings
        for order in [sorted(indices, key=lambda i: weights[i], reverse=True),
                      sorted(indices, key=lambda i: i+1, reverse=True),
                      sorted(indices, key=lambda i: weights[i])]:
            if not _time_ok():
                break
            result = backtrack({0}, [], order, target)
            if result is not None:
                best_size = target
                best_set = result
                found = True
                break
        if found:
            # Recompute best_sums for the new best_set
            best_sums = {0}
            for i in best_set:
                best_sums = can_add(best_sums, weights[i])
    
    # Phase 3: Local search - try to swap elements to extend
    if best_set and _time_ok():
        in_set = set(best_set)
        not_in_set = [i for i in indices if i not in in_set]
        
        # Try adding one more element by removing one
        for add_i in not_in_set:
            if not _time_ok():
                break
            for rem_i in best_set:
                if not _time_ok():
                    break
                candidate = [i for i in best_set if i != rem_i] + [add_i]
                # Check if candidate is dissociated
                sums = {0}
                valid = True
                for i in candidate:
                    new_sums = can_add(sums, weights[i])
                    if new_sums is None:
                        valid = False
                        break
                    sums = new_sums
                if valid and len(candidate) > best_size:
                    best_size = len(candidate)
                    best_set = candidate
                    break
            if not _time_ok():
                break
    
    # Phase 4: More aggressive backtracking with time budget
    if _time_ok() and best_size < 13:
        # Try to find size best_size+1 with more thorough search
        target = best_size + 1
        found = False
        for order in orderings[:10]:
            if not _time_ok():
                break
            result = backtrack({0}, [], order, target)
            if result is not None:
                best_size = target
                best_set = result
                found = True
                break
    
    # Final verification
    def verify_dissociated(selected_indices):
        w = [weights[i] for i in selected_indices]
        n = len(w)
        sums = set()
        for mask in range(1 << n):
            s = 0
            for b in range(n):
                if mask & (1 << b):
                    s += w[b]
            if s in sums:
                return False
            sums.add(s)
        return True
    
    if best_set and not verify_dissociated(best_set):
        best_set = [0, 1, 2, 4, 6, 10, 12, 16, 18]
    
    result = sorted([i + 1 for i in best_set])
    if not result:
        result = [1]
    
    return result
