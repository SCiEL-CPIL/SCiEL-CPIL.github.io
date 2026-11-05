"""Construct a small inclusion-maximal Sidon set in [1,256]."""
import time
import random

N = 256

# EVOLVE-BLOCK-START

def _is_valid_sidon(selected):
    """Check if selected is a valid Sidon set."""
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            s = selected[i] + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected):
    """Check if selected is inclusion-maximal in [1, N]."""
    sel_set = set(selected)
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            sums.add(selected[i] + selected[j])
    
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Check if adding x would break Sidon property
        can_add = True
        for a in selected:
            s = x + a
            if s in sums:
                can_add = False
                break
        if can_add and 2 * x in sums:
            can_add = False
        if can_add:
            new_sums = set()
            for a in selected:
                s = x + a
                if s in new_sums or s in sums:
                    can_add = False
                    break
                new_sums.add(s)
            if can_add:
                if 2 * x in new_sums or 2 * x in sums:
                    can_add = False
        if can_add:
            return False
    return True

def _greedy_construct(order):
    """Greedy construction given a specific order of elements to consider."""
    selected = []
    sums = set()
    for x in order:
        added = set()
        conflict = False
        for a in selected:
            s = x + a
            if s in sums or s in added:
                conflict = True
                break
            added.add(s)
        if not conflict:
            s = 2 * x
            if s in sums or s in added:
                conflict = True
            else:
                added.add(s)
        if not conflict:
            selected.append(x)
            sums.update(added)
    return selected

def _prune_to_minimal_maximal(S, deadline):
    """Try to remove elements while maintaining maximality."""
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(len(S) - 1, -1, -1):
            if time.time() > deadline:
                break
            new_S = S[:i] + S[i+1:]
            if not new_S:
                continue
            if _is_maximal(new_S):
                S = new_S
                improved = True
    return S

def _local_search_swap(S, deadline):
    """Try to improve by removing one element and adding another."""
    S = list(S)
    improved = True
    max_iterations = 50
    iteration = 0
    while improved and iteration < max_iterations:
        if time.time() > deadline:
            break
        improved = False
        iteration += 1
        for i in range(len(S)):
            if time.time() > deadline:
                break
            S_without = S[:i] + S[i+1:]
            if not S_without:
                continue
            sums_without = set()
            for a in range(len(S_without)):
                for b in range(a, len(S_without)):
                    sums_without.add(S_without[a] + S_without[b])
            S_without_set = set(S_without)
            # Try to find an element x not in S_without that can be added
            # and results in a maximal set
            for x in range(1, N + 1):
                if x in S_without_set:
                    continue
                # Check if x can be added to S_without
                ok = True
                if 2 * x in sums_without:
                    ok = False
                else:
                    for a in S_without:
                        if x + a in sums_without:
                            ok = False
                            break
                if not ok:
                    continue
                new_S = sorted(S_without + [x])
                # Check if new_S is maximal
                if _is_maximal(new_S):
                    S = new_S
                    improved = True
                    break
    return S

def construct():
    """Return an inclusion-maximal Sidon set in [1,256] with small cardinality."""
    deadline = time.time() + 55.0  # safety margin before 60s limit
    
    best_size = N + 1
    best_set = None
    
    # Phase 1: Ascending greedy baseline
    S0 = _greedy_construct(range(1, N + 1))
    S0 = _prune_to_minimal_maximal(S0, deadline)
    if len(S0) < best_size:
        best_size = len(S0)
        best_set = list(S0)
    
    # Phase 2: Randomized greedy constructions with pruning
    rng = random.Random(42)
    
    for trial in range(5000):
        if time.time() > deadline:
            break
        
        # Random permutation
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        S = _greedy_construct(order)
        S = _prune_to_minimal_maximal(S, deadline)
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = list(S)
        
        # Early termination if we find a very small set
        if best_size <= 15:
            break
    
    # Phase 3: Local search from best set
    if best_set is not None and time.time() < deadline - 5:
        S = _local_search_swap(best_set, deadline)
        if len(S) < best_size:
            best_size = len(S)
            best_set = list(S)
    
    # Final verification and repair
    if best_set is None:
        S0 = _greedy_construct(range(1, N + 1))
        S0 = _prune_to_minimal_maximal(S0, deadline)
        best_set = S0
    
    # Ensure it's a valid Sidon set
    S = list(best_set)
    S_sorted = sorted(S)
    if not _is_valid_sidon(S_sorted):
        # Fall back to ascending greedy
        S0 = _greedy_construct(range(1, N + 1))
        S0 = _prune_to_minimal_maximal(S0, deadline)
        return S0
    
    # Check maximality and repair if needed
    if not _is_maximal(S_sorted):
        changed = True
        while changed:
            changed = False
            sums = set()
            for i in range(len(S_sorted)):
                for j in range(i, len(S_sorted)):
                    sums.add(S_sorted[i] + S_sorted[j])
            S_set = set(S_sorted)
            for x in range(1, N + 1):
                if x in S_set:
                    continue
                ok = True
                if 2 * x in sums:
                    ok = False
                else:
                    for a in S_sorted:
                        if x + a in sums:
                            ok = False
                            break
                if ok:
                    S_sorted.append(x)
                    S_sorted.sort()
                    S_set.add(x)
                    changed = True
                    break
        best_set = S_sorted
    
    return best_set

# EVOLVE-BLOCK-END
