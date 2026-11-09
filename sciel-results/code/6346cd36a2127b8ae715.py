"""Structural change: covering-based approach for minimal maximal Sidon sets in [1,256]."""
import time
import random

N = 256
DEADLINE = 55.0  # safety margin under 60s wall-clock

# EVOLVE-BLOCK-START

def _build_sums(S):
    """Return set of all unordered sums a+b with a<=b from S."""
    sums = set()
    n = len(S)
    for i in range(n):
        a = S[i]
        for j in range(i, n):
            sums.add(a + S[j])
    return sums

def _get_addable(S, sums):
    """Return list of elements y not in S that can be added to preserve Sidon."""
    S_set = set(S)
    addable = []
    for y in range(1, N + 1):
        if y in S_set:
            continue
        ok = True
        for a in S:
            if (y + a) in sums:
                ok = False
                break
        if ok and (2 * y) in sums:
            ok = False
        if ok:
            addable.append(y)
    return addable

def _is_maximal(S, sums=None):
    """Check if S is maximal Sidon."""
    if sums is None:
        sums = _build_sums(S)
    return len(_get_addable(S, sums)) == 0

def _greedy_maximal_sidon():
    """Build a maximal Sidon set greedily."""
    selected = []
    sums = set()
    for x in range(1, N + 1):
        ok = True
        for a in selected:
            if (x + a) in sums:
                ok = False
                break
        if ok and (2 * x) in sums:
            ok = False
        if ok:
            selected.append(x)
            sums.update([x + a for a in selected[:-1]] + [2 * x])
    return selected

def _make_maximal(S, t_start):
    """Greedily add elements to make S maximal. Returns (S, sums)."""
    S = list(S)
    sums = _build_sums(S)
    S_set = set(S)
    while time.time() < t_start + DEADLINE - 2.0:
        addable = _get_addable(S, sums)
        if not addable:
            break
        # Add the element that blocks the most other potential additions
        best_y = None
        best_block = -1
        for y in addable:
            # Count how many other addable elements y would block
            block_count = 0
            for z in addable:
                if z == y:
                    continue
                # Check if adding y would block z
                # z can be added if for all a in S, z+a not in sums, and 2z not in sums
                # After adding y, new sums include y+a for a in S, and 2y
                # z is blocked if (z+a) in (sums ∪ {y+a for a in S} ∪ {2y}) for some a, or 2z in that set
                new_sums = sums | {y + a for a in S} | {2 * y}
                blocked = False
                for a in S:
                    if (z + a) in new_sums:
                        blocked = True
                        break
                if not blocked and (2 * z) in new_sums:
                    blocked = True
                if blocked:
                    block_count += 1
            if block_count > best_block:
                best_block = block_count
                best_y = y
        if best_y is None:
            break
        S.append(best_y)
        S.sort()
        sums.update([best_y + a for a in S[:-1]] + [2 * best_y])
        S_set.add(best_y)
    return S, sums

def _remove_and_remaximize(S, sums, t_start):
    """Try to remove one element from S and re-maximize. Return (new_S, new_sums) or None."""
    n = len(S)
    if n <= 1:
        return None
    
    # Try removing each element, prefer those whose removal allows re-maximization with few additions
    candidates = []
    for i in range(n):
        if time.time() > t_start + DEADLINE - 2.0:
            break
        S_trial = [x for j, x in enumerate(S) if j != i]
        sums_trial = _build_sums(S_trial)
        addable = _get_addable(S_trial, sums_trial)
        # We want to minimize: (number of additions needed) - 1 (since we removed one)
        # If addable is empty, we've reduced size by 1 with no additions
        candidates.append((len(addable), i, S_trial, sums_trial, addable))
    
    # Sort by number of additions needed
    candidates.sort(key=lambda x: x[0])
    
    for add_count, i, S_trial, sums_trial, addable in candidates:
        if time.time() > t_start + DEADLINE - 2.0:
            break
        if add_count == 0:
            # Direct removal works!
            return S_trial, sums_trial
        else:
            # Try to add elements to make it maximal
            S_new = list(S_trial)
            sums_new = set(sums_trial)
            S_set_new = set(S_new)
            remaining_addable = list(addable)
            for _ in range(add_count + 1):  # Try up to add_count+1 additions
                if not remaining_addable:
                    break
                if time.time() > t_start + DEADLINE - 2.0:
                    break
                # Greedily pick the most blocking element
                best_y = None
                best_block = -1
                for y in remaining_addable:
                    new_sums = sums_new | {y + a for a in S_new} | {2 * y}
                    block_count = 0
                    for z in remaining_addable:
                        if z == y:
                            continue
                        blocked = False
                        for a in S_new:
                            if (z + a) in new_sums:
                                blocked = True
                                break
                        if not blocked and (2 * z) in new_sums:
                            blocked = True
                        if blocked:
                            block_count += 1
                    if block_count > best_block:
                        best_block = block_count
                        best_y = y
                if best_y is None:
                    break
                S_new.append(best_y)
                S_new.sort()
                sums_new.update([best_y + a for a in S_new[:-1]] + [2 * best_y])
                S_set_new.add(best_y)
                remaining_addable = [z for z in remaining_addable if z != best_y]
            
            # Check if we achieved maximality with size < original
            if len(S_new) < n and _is_maximal(S_new, sums_new):
                return S_new, sums_new
    
    return None

def construct():
    """
    Construct a minimal-size maximal Sidon set in [1,256].
    Uses covering-based approach: build maximal set, then iteratively remove and re-maximize.
    """
    t_start = time.time()
    
    # Start with greedy maximal Sidon set
    S = _greedy_maximal_sidon()
    sums = _build_sums(S)
    best_S = list(S)
    best_sums = sums
    
    # Iteratively try to reduce size
    while time.time() < t_start + DEADLINE - 2.0:
        result = _remove_and_remaximize(S, sums, t_start)
        if result is None:
            # No improvement possible; try perturbation
            if len(S) > 2 and random.random() < 0.5:
                # Randomly remove 1-2 elements and re-maximize
                k = random.randint(1, min(2, len(S) - 1))
                S_pert = list(S)
                random.shuffle(S_pert)
                S_pert = S_pert[k:]
                if S_pert:
                    S_pert, sums_pert = _make_maximal(S_pert, t_start)
                    if len(S_pert) < len(best_S):
                        S = S_pert
                        sums = sums_pert
                        best_S = list(S)
                        best_sums = sums
                    elif len(S_pert) < len(S):
                        S = S_pert
                        sums = sums_pert
            else:
                break
        else:
            S, sums = result
            if len(S) < len(best_S):
                best_S = list(S)
                best_sums = sums
    
    # Verify the best result
    S = best_S
    sums = best_sums
    
    # Check Sidon property
    n = len(S)
    seen_sums = set()
    valid = True
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in seen_sums:
                valid = False
                break
            seen_sums.add(s)
        if not valid:
            break
    
    if not valid:
        # Fallback to greedy
        return _greedy_maximal_sidon()
    
    # Check maximality
    if not _is_maximal(S, sums):
        # Re-maximize
        S, sums = _make_maximal(S, t_start)
    
    return S

# EVOLVE-BLOCK-END
