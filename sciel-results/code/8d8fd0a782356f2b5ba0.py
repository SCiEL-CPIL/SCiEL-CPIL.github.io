"""Optimized program for small maximal Sidon sets in [1,256] via greedy set-cover."""
import time
import random
import sys

N = 256
_DEADLINE = 55.0  # safety margin before 60s kill


def _is_sidon(S):
    """Check if S is a Sidon set (all a+b for a<=b distinct)."""
    sums = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True


def _is_maximal(S, N):
    """Check if S is inclusion-maximal Sidon in [1,N]."""
    S_set = set(S)
    sums = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            sums.add(S[i] + S[j])

    for x in range(1, N + 1):
        if x in S_set:
            continue
        # Check if adding x preserves Sidon
        new_sums = [x + a for a in S] + [2 * x]
        # Check new_sums against existing sums
        for s in new_sums:
            if s in sums:
                break
        else:
            # Check new_sums are distinct among themselves
            if len(set(new_sums)) == len(new_sums):
                return False  # Can extend, not maximal
    return True


def _sidon_and_maximal(S, N):
    """Combined check: S must be nonempty, Sidon, and inclusion-maximal."""
    if not S:
        return False
    if not _is_sidon(S):
        return False
    if not _is_maximal(S, N):
        return False
    return True


def _greedy_set_cover_construct(seed=0):
    """
    Greedy set-cover style construction:
    At each step, pick the element x that blocks the most other unselected elements
    from being added to a Sidon set, while maintaining maximality potential.
    """
    rng = random.Random(seed)
    selected = []
    sums = set()
    selected_set = set()
    
    def count_blocked_by_element(x, S, sums_set):
        """Count how many elements would become blocked if we add x to S."""
        S_list = list(S)
        new_sums = [x + a for a in S_list] + [2 * x]
        # Check if x can be added at all
        for s in new_sums:
            if s in sums_set:
                return -1  # Cannot add x
        if len(set(new_sums)) != len(new_sums):
            return -1
        
        # Compute which elements would be blocked after adding x
        new_sums_set = sums_set | set(new_sums)
        new_S = S_list + [x]
        blocked_count = 0
        for y in range(1, N + 1):
            if y in S or y == x:
                continue
            # Check if y can be added to new_S
            y_new_sums = [y + a for a in new_S] + [2 * y]
            can_add_y = True
            for s in y_new_sums:
                if s in new_sums_set:
                    can_add_y = False
                    break
            if can_add_y:
                if len(set(y_new_sums)) != len(y_new_sums):
                    can_add_y = False
            if not can_add_y:
                blocked_count += 1
        return blocked_count
    
    # Greedy: at each step, pick element that blocks the most
    while True:
        best_x = None
        best_block_count = -1
        candidates = [x for x in range(1, N + 1) if x not in selected_set]
        
        if not candidates:
            break
        
        # Sample a subset for speed if too many candidates
        if len(candidates) > 50:
            rng.shuffle(candidates)
            candidates = candidates[:50]
        
        for x in candidates:
            bc = count_blocked_by_element(x, selected, sums)
            if bc > best_block_count:
                best_block_count = bc
                best_x = x
        
        if best_x is None or best_block_count < 0:
            break
        
        # Add best_x
        selected.append(best_x)
        selected_set.add(best_x)
        new_sums = [best_x + a for a in selected[:-1]] + [2 * best_x]
        sums.update(new_sums)
    
    selected.sort()
    return selected


def _greedy_forward_construct(seed=0):
    """Standard greedy forward construction with randomization."""
    rng = random.Random(seed)
    selected = []
    sums = set()
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    
    for x in candidates:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    
    selected.sort()
    return selected


def _try_minimize_by_removal(S, N, deadline=None):
    """Try to remove elements while keeping Sidon + maximal."""
    S = list(S)
    improved = True
    while improved:
        improved = False
        if deadline is not None and time.time() > deadline:
            break
        for i in range(len(S)):
            candidate = S[:i] + S[i+1:]
            if _sidon_and_maximal(candidate, N):
                S = candidate
                improved = True
                break
    return S


def _local_search(S, N, deadline=None):
    """Local search: try swaps and removals to reduce size while maintaining maximality."""
    S = list(S)
    improved = True
    while improved:
        improved = False
        if deadline is not None and time.time() > deadline:
            break
        
        # Try removing each element
        for i in range(len(S)):
            candidate = S[:i] + S[i+1:]
            if _sidon_and_maximal(candidate, N):
                S = candidate
                improved = True
                break
        if improved:
            continue
        
        # Try swapping: remove one, add one
        S_set = set(S)
        for i in range(len(S)):
            remaining = S[:i] + S[i+1:]
            rem_set = set(remaining)
            rem_sums = set()
            n_rem = len(remaining)
            for a in range(n_rem):
                for b in range(a, n_rem):
                    rem_sums.add(remaining[a] + remaining[b])
            
            for x in range(1, N + 1):
                if x in rem_set:
                    continue
                new_sums = [x + a for a in remaining] + [2 * x]
                if len(set(new_sums)) == len(new_sums) and not rem_sums.intersection(set(new_sums)):
                    candidate = remaining + [x]
                    candidate.sort()
                    if _sidon_and_maximal(candidate, N) and len(candidate) < len(S):
                        S = candidate
                        improved = True
                        break
            if improved:
                break
    return S


def _greedy_with_tie_breaking(seed=0, use_random_tie=False):
    """
    Greedy set-cover with better tie-breaking.
    When multiple candidates have the same block count, pick randomly among them
    to explore different parts of the search space.
    """
    rng = random.Random(seed)
    selected = []
    sums = set()
    selected_set = set()
    
    def count_blocked_by_element(x, S, sums_set):
        """Count how many elements would become blocked if we add x to S."""
        S_list = list(S)
        new_sums = [x + a for a in S_list] + [2 * x]
        for s in new_sums:
            if s in sums_set:
                return -1
        if len(set(new_sums)) != len(new_sums):
            return -1
        
        new_sums_set = sums_set | set(new_sums)
        new_S = S_list + [x]
        blocked_count = 0
        for y in range(1, N + 1):
            if y in S or y == x:
                continue
            y_new_sums = [y + a for a in new_S] + [2 * y]
            can_add_y = True
            for s in y_new_sums:
                if s in new_sums_set:
                    can_add_y = False
                    break
            if can_add_y:
                if len(set(y_new_sums)) != len(y_new_sums):
                    can_add_y = False
            if not can_add_y:
                blocked_count += 1
        return blocked_count
    
    while True:
        candidates = [x for x in range(1, N + 1) if x not in selected_set]
        
        if not candidates:
            break
        
        if len(candidates) > 50:
            rng.shuffle(candidates)
            candidates = candidates[:50]
        
        # Compute block counts for all candidates
        scores = []
        for x in candidates:
            bc = count_blocked_by_element(x, selected, sums)
            if bc >= 0:
                scores.append((bc, x))
        
        if not scores:
            break
        
        # Find max block count
        max_bc = max(s[0] for s in scores)
        
        if use_random_tie:
            # Pick randomly among all candidates with max block count
            max_candidates = [x for bc, x in scores if bc == max_bc]
            best_x = rng.choice(max_candidates)
        else:
            # Pick the first one with max block count (deterministic)
            best_x = None
            for bc, x in scores:
                if bc == max_bc:
                    best_x = x
                    break
        
        if best_x is None:
            break
        
        selected.append(best_x)
        selected_set.add(best_x)
        new_sums = [best_x + a for a in selected[:-1]] + [2 * best_x]
        sums.update(new_sums)
    
    selected.sort()
    return selected


def construct():
    start_time = time.time()
    best_set = None
    best_size = N + 1

    def elapsed():
        return time.time() - start_time

    # Phase 1: Greedy set-cover constructions with different seeds
    for seed in range(200):
        if elapsed() > _DEADLINE - 15:
            break
        S = _greedy_set_cover_construct(seed)
        if S and len(S) < best_size:
            S_min = _try_minimize_by_removal(S, N, deadline=_DEADLINE - 10)
            if _sidon_and_maximal(S_min, N) and len(S_min) < best_size:
                best_size = len(S_min)
                best_set = S_min

    # Phase 2: Random forward greedy with minimization
    for seed in range(3000):
        if elapsed() > _DEADLINE - 8:
            break
        S = _greedy_forward_construct(seed)
        if len(S) < best_size:
            S_min = _try_minimize_by_removal(S, N, deadline=_DEADLINE - 5)
            if _sidon_and_maximal(S_min, N) and len(S_min) < best_size:
                best_size = len(S_min)
                best_set = S_min

    # Phase 3: Greedy with random tie-breaking (new strategy)
    for seed in range(500):
        if elapsed() > _DEADLINE - 5:
            break
        S = _greedy_with_tie_breaking(seed, use_random_tie=True)
        if S and len(S) < best_size:
            S_min = _try_minimize_by_removal(S, N, deadline=_DEADLINE - 3)
            if _sidon_and_maximal(S_min, N) and len(S_min) < best_size:
                best_size = len(S_min)
                best_set = S_min

    # Phase 4: Local search from best found
    if best_set is not None:
        if elapsed() < _DEADLINE - 3:
            S_ls = _local_search(best_set, N, deadline=_DEADLINE - 1)
            if _sidon_and_maximal(S_ls, N) and len(S_ls) < best_size:
                best_set = S_ls
                best_size = len(S_ls)

    # Fallback: deterministic greedy
    if best_set is None:
        S = _greedy_forward_construct(0)
        S_min = _try_minimize_by_removal(S, N)
        if _sidon_and_maximal(S_min, N):
            best_set = S_min
        else:
            best_set = S

    return best_set
