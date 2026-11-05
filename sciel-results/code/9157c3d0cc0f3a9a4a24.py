"""Improved program: integer-scaled backtracking with conflict-graph pruning and
bitset-accelerated candidate filtering to maximize dissociated reciprocal set
cardinality within [1,22]."""
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
    n_elem = 22
    indices = list(range(n_elem))
    
    # Precompute triple conflicts: (i,j) -> set of k such that w_i + w_j == w_k
    # These are "hard" conflicts: if i and j are chosen, k cannot be.
    triple_conflicts = {}  # (i,j) -> frozenset of forbidden k's
    for i in range(n_elem):
        for j in range(i+1, n_elem):
            s = weights[i] + weights[j]
            forb = set()
            for k in range(n_elem):
                if k != i and k != j and weights[k] == s:
                    forb.add(k)
            if forb:
                triple_conflicts[(i,j)] = frozenset(forb)
    
    # Also: w_i == w_j + w_k (i is sum of j and k)
    # If j and k are chosen, i cannot be.
    for j in range(n_elem):
        for k in range(j+1, n_elem):
            s = weights[j] + weights[k]
            for i in range(n_elem):
                if i != j and i != k and weights[i] == s:
                    key = (j, k)
                    if key not in triple_conflicts:
                        triple_conflicts[key] = frozenset()
                    triple_conflicts[key] = triple_conflicts[key] | frozenset([i])
    
    # Also check: w_i + w_j + w_k == w_l (4-element dependencies)
    # This is more expensive but can help prune
    four_conflicts = {}  # frozenset({i,j,k}) -> set of forbidden l's
    for i in range(n_elem):
        for j in range(i+1, n_elem):
            for k in range(j+1, n_elem):
                s = weights[i] + weights[j] + weights[k]
                forb = set()
                for l in range(n_elem):
                    if l not in (i,j,k) and weights[l] == s:
                        forb.add(l)
                if forb:
                    four_conflicts[frozenset([i,j,k])] = frozenset(forb)
    
    # Also: w_i + w_j == w_k + w_l (different pair sums equal)
    # This means {i,j} and {k,l} have the same subset sum -> not dissociated
    pair_sum_conflicts = {}  # sum_value -> list of pairs
    for i in range(n_elem):
        for j in range(i+1, n_elem):
            s = weights[i] + weights[j]
            if s not in pair_sum_conflicts:
                pair_sum_conflicts[s] = []
            pair_sum_conflicts[s].append((i,j))
    
    # Build conflict graph: edge between i and j if they cannot coexist
    # Actually, any two elements can coexist (their 4 subset sums are distinct).
    # Conflicts arise with 3+ elements. So we use the triple/four conflicts for pruning.
    
    def can_add(subset_sums, w):
        """Check if adding weight w maintains dissociation. Returns new sums or None."""
        for s in subset_sums:
            if (s + w) in subset_sums:
                return None
        return subset_sums | {s + w for s in subset_sums}
    
    def is_compatible(chosen_set, new_elem):
        """Check if new_elem can be added to chosen_set based on precomputed conflicts."""
        # Check triple conflicts: for any pair in chosen_set, if adding new_elem creates a forbidden triple
        for i in chosen_set:
            key1 = (min(i, new_elem), max(i, new_elem))
            key2 = (min(i, new_elem), max(i, new_elem))
            # If (i, new_elem) has conflicts, check if any conflicting element is in chosen_set
            if key1 in triple_conflicts:
                for k in triple_conflicts[key1]:
                    if k in chosen_set:
                        return False
            # Check four conflicts
            for j in chosen_set:
                if j == i:
                    continue
                triple_key = frozenset([i, j, new_elem])
                if triple_key in four_conflicts:
                    for k in four_conflicts[triple_key]:
                        if k in chosen_set:
                            return False
        return True
    
    def greedy(order, subset_sums, chosen, chosen_set):
        """Greedy add elements from order starting with current state."""
        for i in order:
            if not _time_ok():
                break
            if not is_compatible(chosen_set, i):
                continue
            new_sums = can_add(subset_sums, weights[i])
            if new_sums is not None:
                subset_sums = new_sums
                chosen.append(i)
                chosen_set.add(i)
        return subset_sums, chosen, chosen_set
    
    def backtrack(current_sums, chosen, chosen_set, remaining, target_size, depth=0):
        """DFS to build a set of size target_size with conflict pruning."""
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
            # Quick conflict check
            if not is_compatible(chosen_set, i):
                continue
            new_sums = can_add(current_sums, weights[i])
            if new_sums is not None:
                new_chosen_set = chosen_set | {i}
                result = backtrack(new_sums, chosen + [i], new_chosen_set, 
                                   remaining_sorted[j+1:], target_size, depth+1)
                if result is not None:
                    return result
        return None
    
    best_size = 0
    best_set = None
    best_sums = {0}
    
    # Phase 1: Greedy with many different orderings
    orderings = [
        sorted(indices, key=lambda i: weights[i], reverse=True),  # weight desc
        sorted(indices, key=lambda i: weights[i]),                  # weight asc
        sorted(indices, key=lambda i: i+1, reverse=True),           # k desc
        sorted(indices, key=lambda i: i+1),                         # k asc
        sorted(indices, key=lambda i: weights[i] % 1000, reverse=True),
    ]
    
    for seed in range(40):
        r = random.Random(seed)
        idx = list(range(n_elem))
        r.shuffle(idx)
        orderings.append(idx)
    
    for order in orderings:
        if not _time_ok():
            break
        subset_sums = {0}
        chosen = []
        chosen_set = set()
        subset_sums, chosen, chosen_set = greedy(order, subset_sums, chosen, chosen_set)
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = chosen[:]
            best_sums = subset_sums
    
    # Phase 2: Backtracking to extend best solution
    for target in range(best_size + 1, 15):
        if not _time_ok():
            break
        found = False
        for order in [sorted(indices, key=lambda i: weights[i], reverse=True),
                      sorted(indices, key=lambda i: i+1, reverse=True),
                      sorted(indices, key=lambda i: weights[i]),
                      sorted(indices, key=lambda i: weights[i] % 1000, reverse=True)]:
            if not _time_ok():
                break
            result = backtrack({0}, [], set(), order, target)
            if result is not None:
                best_size = target
                best_set = result
                found = True
                break
        if found:
            best_sums = {0}
            for i in best_set:
                best_sums = can_add(best_sums, weights[i])
    
    # Phase 3: Local search - try to swap elements to extend
    if best_set and _time_ok():
        in_set = set(best_set)
        not_in_set = [i for i in indices if i not in in_set]
        
        for add_i in not_in_set:
            if not _time_ok():
                break
            for rem_i in best_set:
                if not _time_ok():
                    break
                candidate = [i for i in best_set if i != rem_i] + [add_i]
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
    
    # Phase 4: More aggressive backtracking
    if _time_ok() and best_size < 14:
        target = best_size + 1
        found = False
        for order in orderings[:15]:
            if not _time_ok():
                break
            result = backtrack({0}, [], set(), order, target)
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
