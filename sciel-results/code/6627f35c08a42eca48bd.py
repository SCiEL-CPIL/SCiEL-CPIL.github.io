"""
Minimum-cardinality inclusion-maximal Sidon set in [1, 256].

Structural change: formulate as a minimum vertex cover problem on a
"sum-collision hypergraph".  Vertices are the 256 elements.  A hyperedge
is a subset of elements that must NOT all be present simultaneously,
because their pairwise sums would collide.  We want a Sidon set S that
is inclusion-maximal: every element outside S is "covered" (blocked) by S.

The key insight for minimality: we want to pick as few elements as
possible such that every other element is blocked.  This is a set-cover
problem where each chosen element "covers" (blocks) a set of other
elements.

Algorithm:
1. Precompute for each element x, the set of elements y that x would
   block (i.e., adding both x and y to a Sidon set would create a
   collision, assuming the rest of the set is empty or compatible).
2. Greedy set cover: repeatedly pick the element that blocks the most
   currently-unblocked elements, add it to S, and update the blocked set.
3. Local search: try removing elements from S while maintaining
   maximality.

The blocking relation is dynamic (depends on the current sums in S),
so we compute it on the fly during the greedy phase.
"""
import time
import random
import sys

N = 256
_DEADLINE = 55.0


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
    if not S:
        return False
    if not _is_sidon(S):
        return False
    S_set = set(S)
    sums = set()
    for i, a in enumerate(S):
        for j in range(i, len(S)):
            sums.add(S[i] + S[j])
    for x in range(1, N + 1):
        if x in S_set:
            continue
        # Check if x can be added
        blocked = False
        if 2 * x in sums:
            blocked = True
        else:
            for s in S:
                if x + s in sums:
                    blocked = True
                    break
        if not blocked:
            return False
    return True


def _greedy_set_cover(N, seed, deadline=None):
    """
    Greedy set-cover: at each step, pick the addable element that
    blocks the most currently-unblocked elements.
    """
    rng = random.Random(seed)
    S = []
    S_set = set()
    sums = set()
    
    # unblocked: elements not in S that are not yet blocked
    unblocked = set(range(1, N + 1))
    
    while unblocked and (deadline is None or time.time() < deadline):
        # Find the addable element that blocks the most unblocked elements
        best_x = None
        best_block_count = -1
        best_blocked_set = set()
        
        # To speed up, sample if unblocked is large
        if len(unblocked) > 60:
            candidates = rng.sample(list(unblocked), 60)
        else:
            candidates = list(unblocked)
        
        for x in candidates:
            # Check if x is addable to S
            new_sums = set()
            ok = True
            for a in S:
                s = x + a
                if s in sums or s in new_sums:
                    ok = False
                    break
                new_sums.add(s)
            if not ok:
                continue
            s2 = 2 * x
            if s2 in sums or s2 in new_sums:
                continue
            new_sums.add(s2)
            
            # Count how many unblocked elements x would block
            # An element y (y != x, y not in S) is blocked by adding x if:
            #   2y in sums ∪ new_sums, or y+s in sums ∪ new_sums for some s in S ∪ {x}
            all_sums = sums | new_sums
            blocked_count = 0
            blocked_elems = set()
            
            for y in unblocked:
                if y == x:
                    continue
                # Check if y is blocked
                if 2 * y in all_sums:
                    blocked_count += 1
                    blocked_elems.add(y)
                    continue
                y_blocked = False
                for s in S:
                    if y + s in all_sums:
                        y_blocked = True
                        break
                if not y_blocked and (y + x) in all_sums:
                    y_blocked = True
                if y_blocked:
                    blocked_count += 1
                    blocked_elems.add(y)
            
            if blocked_count > best_block_count:
                best_block_count = blocked_count
                best_x = x
                best_blocked_set = blocked_elems
        
        if best_x is None:
            # No addable element found in candidates; try all unblocked
            if len(unblocked) <= 60:
                break
            # Try a few more
            found = False
            for x in unblocked:
                new_sums = set()
                ok = True
                for a in S:
                    s = x + a
                    if s in sums or s in new_sums:
                        ok = False
                        break
                    new_sums.add(s)
                if not ok:
                    continue
                s2 = 2 * x
                if s2 in sums or s2 in new_sums:
                    continue
                new_sums.add(s2)
                
                all_sums = sums | new_sums
                blocked_elems = set()
                for y in unblocked:
                    if y == x:
                        continue
                    if 2 * y in all_sums:
                        blocked_elems.add(y)
                        continue
                    y_blocked = False
                    for s in S:
                        if y + s in all_sums:
                            y_blocked = True
                            break
                    if not y_blocked and (y + x) in all_sums:
                        y_blocked = True
                    if y_blocked:
                        blocked_elems.add(y)
                
                S.append(x)
                S_set.add(x)
                sums |= new_sums
                unblocked.discard(x)
                unblocked -= blocked_elems
                found = True
                break
            if not found:
                break
        else:
            S.append(best_x)
            S_set.add(best_x)
            new_sums = set()
            for a in S[:-1]:
                new_sums.add(best_x + a)
            new_sums.add(2 * best_x)
            sums |= new_sums
            unblocked.discard(best_x)
            unblocked -= best_blocked_set
    
    S.sort()
    return S


def _local_search_remove(S, N, deadline=None):
    """
    Try to remove elements from S while maintaining maximality.
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


def _swap_improve(S, N, deadline=None):
    """
    Try swap moves: replace one element with another to reduce size or
    enable further removals.
    """
    S = list(S)
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        for i in range(len(S)):
            for j in range(1, N + 1):
                if j in S:
                    continue
                S_new = S[:i] + [j] + S[i+1:]
                S_new.sort()
                if _is_maximal(S_new, N):
                    # Check if this enables a removal
                    S_trial = list(S_new)
                    for k in range(len(S_trial)):
                        S_trial2 = S_trial[:k] + S_trial[k+1:]
                        if S_trial2 and _is_maximal(S_trial2, N):
                            S = S_trial2
                            improved = True
                            break
                    if improved:
                        break
            if improved:
                break
    return S


def construct():
    start_time = time.time()
    best_set = None
    best_size = N + 1
    
    def elapsed():
        return time.time() - start_time
    
    # Phase 1: Greedy set-cover with many seeds
    num_seeds = 300
    for seed in range(num_seeds):
        if elapsed() > _DEADLINE - 12:
            break
        S = _greedy_set_cover(N, seed, deadline=_DEADLINE - 10)
        if S and len(S) < best_size:
            if elapsed() < _DEADLINE - 5:
                S_imp = _local_search_remove(S, N, deadline=_DEADLINE - 5)
                if len(S_imp) < len(S):
                    S = S_imp
            if len(S) < best_size:
                best_set = S
                best_size = len(S)
    
    # Phase 2: More aggressive local search on best
    if best_set is not None and elapsed() < _DEADLINE - 3:
        for _ in range(20):
            if elapsed() > _DEADLINE - 3:
                break
            S_new = _local_search_remove(best_set, N, deadline=_DEADLINE - 3)
            if len(S_new) < len(best_set):
                best_set = S_new
                best_size = len(S_new)
    
    # Phase 3: Swap improvement
    if best_set is not None and elapsed() < _DEADLINE - 2:
        S_new = _swap_improve(best_set, N, deadline=_DEADLINE - 2)
        if len(S_new) < len(best_set):
            best_set = S_new
            best_size = len(S_new)
    
    # Fallback
    if best_set is None:
        S = []
        sums = set()
        rng = random.Random(42)
        elems = list(range(1, N + 1))
        rng.shuffle(elems)
        for x in elems:
            new_sums = set()
            ok = True
            for a in S:
                s = x + a
                if s in sums or s in new_sums:
                    ok = False
                    break
                new_sums.add(s)
            if not ok:
                continue
            if 2 * x in sums or 2 * x in new_sums:
                continue
            new_sums.add(2 * x)
            S.append(x)
            sums |= new_sums
        S.sort()
        best_set = S
    
    return best_set
