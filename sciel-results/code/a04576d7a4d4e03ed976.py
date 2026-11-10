"""Optimized program for small maximal Sidon sets in [1,256]."""
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


def _get_unblocked(S, N):
    """
    Returns the set of x in [1, N] \\ S that can be added to S while keeping it Sidon.
    If this set is empty, S is maximal.
    """
    S_set = set(S)
    sums = set()
    for i, a in enumerate(S):
        for j in range(i, len(S)):
            sums.add(S[i] + S[j])
    
    unblocked = []
    for x in range(1, N + 1):
        if x in S_set:
            continue
        blocked = False
        if 2*x in sums:
            blocked = True
        else:
            for s in S:
                if x + s in sums:
                    blocked = True
                    break
        if not blocked:
            unblocked.append(x)
    return unblocked


def _is_maximal(S, N):
    """Check if S is inclusion-maximal Sidon in [1,N]."""
    if not S:
        return False
    if not _is_sidon(S):
        return False
    return len(_get_unblocked(S, N)) == 0


def _greedy_set_cover_construct(N, seed, deadline=None):
    """
    Greedy set-cover approach: build a Sidon set by always choosing the element
    that blocks the most currently-unblocked elements. This directly constructs
    a small maximal Sidon set.
    
    The idea: we want a set S such that every element in [1,N] is either in S
    or blocked by S. We build S incrementally, at each step choosing the element
    that maximizes the number of newly-blocked elements.
    """
    rng = random.Random(seed)
    S = []
    S_set = set()
    sums = set()
    
    # Start with no elements; all elements are "unblocked"
    # At each step, pick the element x not in S that:
    # 1. Can be added to S (keeps Sidon property)
    # 2. Blocks the most elements that are currently unblocked
    
    # We'll track which elements are "covered" (either in S or blocked by S)
    # An element x is blocked by S if adding x to S would violate Sidon
    
    # For efficiency, we'll compute unblocked dynamically
    
    # To break ties randomly, we shuffle candidate ordering
    all_elements = list(range(1, N + 1))
    
    # We'll iterate: at each step, find the best element to add
    # An element x is "addable" if S ∪ {x} is Sidon
    # When we add x, it blocks some elements y (those where y+x or 2y creates collision)
    
    # Simpler approach: just greedily add elements that are addable,
    # prioritizing those that block the most other addable elements
    
    # First, let's get the initial unblocked set (all elements)
    # Then iteratively:
    #   - For each addable element x, compute how many other addable elements it would block
    #   - Pick the one that blocks the most
    #   - Add it to S
    #   - Update unblocked set
    #   - Repeat until no addable elements remain (maximal)
    
    # But this is expensive. Let's use a simpler heuristic:
    # Just add elements in random order, but skip those that don't block much
    
    # Actually, let's try a different approach: 
    # Build S by adding elements that are "critical" - i.e., they block many elements
    
    # For now, let's use a randomized greedy: shuffle elements, add if possible,
    # but with a bias toward elements that seem to block more
    
    # Simpler: just do randomized greedy with multiple restarts
    rng.shuffle(all_elements)
    
    for x in all_elements:
        if deadline is not None and time.time() > deadline:
            break
        # Check if x can be added
        new_sums = [x + a for a in S] + [2 * x]
        new_sums_set = set(new_sums)
        if len(new_sums_set) == len(new_sums) and not sums.intersection(new_sums_set):
            S.append(x)
            S_set.add(x)
            sums.update(new_sums_set)
    
    S.sort()
    return S


def _greedy_block_priority_construct(N, seed, deadline=None):
    """
    More sophisticated greedy: at each step, choose the addable element that
    blocks the most currently-addable elements.
    """
    rng = random.Random(seed)
    S = []
    S_set = set()
    sums = set()
    
    # Get initial unblocked
    unblocked = list(range(1, N + 1))
    
    while unblocked and (deadline is None or time.time() < deadline):
        # For each candidate in unblocked, check if it's addable and count how many
        # other unblocked elements it would block
        best_x = None
        best_blocked_count = -1
        best_blocked_set = None
        
        # Sample a subset for speed if unblocked is large
        if len(unblocked) > 50:
            candidates = rng.sample(unblocked, 50)
        else:
            candidates = unblocked
        
        for x in candidates:
            # Check if x is addable
            new_sums = [x + a for a in S] + [2 * x]
            new_sums_set = set(new_sums)
            if len(new_sums_set) != len(new_sums) or sums.intersection(new_sums_set):
                continue
            
            # Count how many other unblocked elements x would block
            blocked_count = 0
            blocked_elements = set()
            for y in unblocked:
                if y == x:
                    continue
                # Check if y would be blocked by adding x
                # y is blocked if 2y in (sums ∪ new_sums) OR y+s in (sums ∪ new_sums) for some s in S∪{x}
                # Actually, y is blocked if S ∪ {x, y} is not Sidon
                # Simpler: y is blocked if adding y to S∪{x} violates Sidon
                # That happens if 2y in sums∪new_sums OR y+s in sums∪new_sums for some s in S∪{x}
                
                all_sums = sums.union(new_sums_set)
                blocked = False
                if 2*y in all_sums:
                    blocked = True
                else:
                    for s in S:
                        if y + s in all_sums:
                            blocked = True
                            break
                    if not blocked and (y + x) in all_sums:
                        blocked = True
                
                if blocked:
                    blocked_count += 1
                    blocked_elements.add(y)
            
            if blocked_count > best_blocked_count:
                best_blocked_count = blocked_count
                best_x = x
                best_blocked_set = blocked_elements
        
        if best_x is None:
            # No addable elements found in candidates; try full unblocked if small
            if len(unblocked) <= 50:
                break
            # Try full unblocked
            for x in unblocked:
                new_sums = [x + a for a in S] + [2 * x]
                new_sums_set = set(new_sums)
                if len(new_sums_set) != len(new_sums) or sums.intersection(new_sums_set):
                    continue
                S.append(x)
                S_set.add(x)
                sums.update(new_sums_set)
                break
            if best_x is None:
                break
        else:
            S.append(best_x)
            S_set.add(best_x)
            new_sums = [best_x + a for a in S[:-1]] + [2 * best_x]
            sums.update(new_sums)
            # Remove blocked elements from unblocked
            if best_blocked_set:
                unblocked = [x for x in unblocked if x != best_x and x not in best_blocked_set]
            else:
                unblocked = [x for x in unblocked if x != best_x]
    
    S.sort()
    return S


def _local_search_improve(S, N, deadline=None):
    """
    Local search: try to remove elements from S while maintaining maximality.
    If removing an element keeps S maximal, remove it.
    """
    S = list(S)
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        for i in range(len(S)):
            S_new = S[:i] + S[i+1:]
            if not S_new:
                continue
            if _is_maximal(S_new, N):
                S = S_new
                improved = True
                break
    return S


def construct():
    start_time = time.time()
    best_set = None
    best_size = N + 1
    
    def elapsed():
        return time.time() - start_time
    
    # Phase 1: Greedy set-cover construction with multiple seeds
    for seed in range(500):
        if elapsed() > _DEADLINE - 15:
            break
        S = _greedy_block_priority_construct(N, seed, deadline=_DEADLINE - 10)
        if S and len(S) < best_size:
            # Try to improve with local search
            if elapsed() < _DEADLINE - 5:
                S_improved = _local_search_improve(S, N, deadline=_DEADLINE - 5)
                if len(S_improved) < len(S):
                    S = S_improved
            if len(S) < best_size:
                best_set = S
                best_size = len(S)
    
    # Phase 2: If we have a good set, try more aggressive local search
    if best_set is not None and elapsed() < _DEADLINE - 3:
        # Try removal-based local search more aggressively
        for _ in range(10):
            if elapsed() > _DEADLINE - 3:
                break
            S_new = _local_search_improve(best_set, N, deadline=_DEADLINE - 3)
            if len(S_new) < len(best_set):
                best_set = S_new
                best_size = len(S_new)
    
    # Fallback: if no maximal set found, use a simple greedy
    if best_set is None:
        candidates = list(range(1, N + 1))
        random.seed(42)
        random.shuffle(candidates)
        S = []
        sums = set()
        for x in candidates:
            new_sums = [x + a for a in S] + [2 * x]
            if len(set(new_sums)) == len(new_sums) and not sums.intersection(set(new_sums)):
                S.append(x)
                sums.update(new_sums)
        S.sort()
        best_set = S
        best_size = len(S)
    
    return best_set
