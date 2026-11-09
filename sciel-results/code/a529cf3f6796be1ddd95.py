"""Structural change: minimum independent dominating set via conflict-graph set cover for minimal maximal Sidon sets in [1,256]."""
import time
import random
import math

N = 256
DEADLINE = 55.0  # safety margin under 60s wall-clock

# EVOLVE-BLOCK-START

def _can_coexist(x, y, sums):
    """Check if y can be added to a Sidon set with existing sums."""
    for a in sums:
        pass  # placeholder - use the efficient version below
    
def _check_addable(y, S, sums):
    """Check if y can be added to Sidon set S with given sums."""
    for a in S:
        if (y + a) in sums:
            return False
    if (2 * y) in sums:
        return False
    return True

def _build_sums(S):
    """Return set of all unordered sums a+b with a<=b from S."""
    sums = set()
    n = len(S)
    for i in range(n):
        a = S[i]
        for j in range(i, n):
            sums.add(a + S[j])
    return sums

def _is_sidon(S):
    """Check if S is a valid Sidon set."""
    sums = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(S):
    """Check if S is a maximal Sidon set in [1,N]."""
    sums = _build_sums(S)
    S_set = set(S)
    for y in range(1, N + 1):
        if y in S_set:
            continue
        if _check_addable(y, S, sums):
            return False
    return True

def _dominates(y, S, sums):
    """
    Return the set of elements z (not in S) that y blocks,
    i.e., z cannot be added to S even after adding y.
    This is the 'coverage' of y in the set-cover sense.
    """
    S_set = set(S)
    blocked = set()
    new_sums = sums | {y + a for a in S} | {2 * y}
    for z in range(1, N + 1):
        if z in S_set or z == y:
            continue
        # Check if z is blocked by adding y
        is_blocked = False
        for a in S:
            if (z + a) in new_sums:
                is_blocked = True
                break
        if not is_blocked and (2 * z) in new_sums:
            is_blocked = True
        if is_blocked:
            blocked.add(z)
    return blocked

def _greedy_maximal_sidon():
    """Build a maximal Sidon set greedily (baseline)."""
    selected = []
    sums = set()
    for x in range(1, N + 1):
        if _check_addable(x, selected, sums):
            selected.append(x)
            sums.update([x + a for a in selected[:-1]] + [2 * x])
    return selected

def _greedy_set_cover_for_maximal_sidon():
    """
    Use a set-cover / dominating-set heuristic:
    1. Start with an empty Sidon set.
    2. At each step, pick the element that covers (blocks) the most uncovered elements,
       while maintaining the Sidon property.
    3. Continue until all elements are either in S or blocked.
    
    This is a greedy approximation to the minimum independent dominating set.
    """
    S = []
    sums = set()
    S_set = set()
    
    # Track which elements are still "uncovered" (can still be added)
    uncovered = set(range(1, N + 1))
    
    while uncovered:
        if time.time() > DEADLINE - 3.0:
            break
        
        best_y = None
        best_coverage = -1
        best_block_set = set()
        
        for y in sorted(uncovered):
            if not _check_addable(y, S, sums):
                continue
            
            # Compute coverage: how many uncovered elements would be blocked
            coverage = 0
            for z in uncovered:
                if z == y:
                    continue
                # Would z be blocked if we add y?
                new_sums = sums | {y + a for a in S} | {2 * y}
                blocked = False
                for a in S:
                    if (z + a) in new_sums:
                        blocked = True
                        break
                if not blocked and (2 * z) in new_sums:
                    blocked = True
                if blocked:
                    coverage += 1
            
            if coverage > best_coverage:
                best_coverage = coverage
                best_y = y
                best_block_set = set()
        
        if best_y is None:
            break
        
        S.append(best_y)
        S_set.add(best_y)
        sums.update([best_y + a for a in S[:-1]] + [2 * best_y])
        
        # Update uncovered: remove y and any elements now blocked
        new_uncovered = set()
        for z in uncovered:
            if z == best_y:
                continue
            if not _check_addable(z, S, sums):
                continue
            new_uncovered.add(z)
        uncovered = new_uncovered
    
    return S

def _local_search_reduce(S, t_start):
    """
    Try to reduce the size of a maximal Sidon set by removing elements
    and re-adding optimally. Uses a set-cover perspective.
    """
    S = list(S)
    best_S = list(S)
    best_size = len(S)
    
    for iteration in range(50):
        if time.time() > t_start + DEADLINE - 3.0:
            break
        
        improved = False
        
        # Try removing each element
        for i in range(len(S)):
            if time.time() > t_start + DEADLINE - 3.0:
                break
            
            S_trial = S[:i] + S[i+1:]
            if not S_trial:
                continue
            
            sums_trial = _build_sums(S_trial)
            
            # Check if it's still maximal
            if _is_maximal(S_trial):
                S = S_trial
                best_S = list(S)
                best_size = len(S)
                improved = True
                break
            
            # Try to re-maximize with minimal additions
            addable = []
            S_set_trial = set(S_trial)
            for y in range(1, N + 1):
                if y in S_set_trial:
                    continue
                if _check_addable(y, S_trial, sums_trial):
                    addable.append(y)
            
            if not addable:
                continue
            
            # Greedily add elements to minimize final size
            S_new = list(S_trial)
            sums_new = set(sums_trial)
            S_set_new = set(S_new)
            remaining = list(addable)
            
            while remaining:
                if time.time() > t_start + DEADLINE - 3.0:
                    break
                
                best_y = None
                best_cov = -1
                for y in remaining:
                    if not _check_addable(y, S_new, sums_new):
                        continue
                    cov = 0
                    new_sums = sums_new | {y + a for a in S_new} | {2 * y}
                    for z in remaining:
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
                            cov += 1
                    if cov > best_cov:
                        best_cov = cov
                        best_y = y
                
                if best_y is None:
                    break
                
                S_new.append(best_y)
                S_set_new.add(best_y)
                sums_new.update([best_y + a for a in S_new[:-1]] + [2 * best_y])
                remaining = [z for z in remaining if z != best_y]
            
            if len(S_new) < len(S) and _is_maximal(S_new):
                S = S_new
                best_S = list(S)
                best_size = len(S)
                improved = True
                break
        
        if not improved:
            break
    
    return best_S

def _randomized_restarts(t_start, initial_S):
    """
    Multiple randomized restarts with different orderings to escape local optima.
    """
    best_S = list(initial_S)
    best_size = len(initial_S)
    
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > t_start + DEADLINE - 3.0:
            break
        
        # Randomize the order for greedy construction
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        S = []
        sums = set()
        for x in order:
            if time.time() > t_start + DEADLINE - 3.0:
                break
            if _check_addable(x, S, sums):
                S.append(x)
                sums.update([x + a for a in S[:-1]] + [2 * x])
        
        # Make maximal
        S_set = set(S)
        while True:
            if time.time() > t_start + DEADLINE - 3.0:
                break
            addable = [y for y in range(1, N+1) if y not in S_set and _check_addable(y, S, sums)]
            if not addable:
                break
            # Pick most blocking
            best_y = None
            best_cov = -1
            for y in addable:
                new_sums = sums | {y + a for a in S} | {2 * y}
                cov = 0
                for z in addable:
                    if z == y:
                        continue
                    blocked = False
                    for a in S:
                        if (z + a) in new_sums:
                            blocked = True
                            break
                    if not blocked and (2 * z) in new_sums:
                        blocked = True
                    if blocked:
                        cov += 1
                if cov > best_cov:
                    best_cov = cov
                    best_y = y
            if best_y is None:
                break
            S.append(best_y)
            S_set.add(best_y)
            sums.update([best_y + a for a in S[:-1]] + [2 * best_y])
        
        # Local search to reduce
        S = _local_search_reduce(S, t_start)
        
        if len(S) < best_size:
            best_size = len(S)
            best_S = list(S)
    
    return best_S

def construct():
    """
    Construct a minimal-size maximal Sidon set in [1,256].
    Uses minimum independent dominating set / set-cover formulation with
    greedy approximation and randomized restarts.
    """
    t_start = time.time()
    
    # Phase 1: Greedy set-cover based construction
    S1 = _greedy_set_cover_for_maximal_sidon()
    if not S1:
        S1 = _greedy_maximal_sidon()
    
    # Phase 2: Local search to reduce
    S2 = _local_search_reduce(S1, t_start)
    
    # Phase 3: Randomized restarts
    S3 = _randomized_restarts(t_start, S2)
    
    # Take the best
    candidates = [S1, S2, S3]
    best_S = min(candidates, key=len)
    
    # Verify and fix if needed
    if not _is_sidon(best_S):
        best_S = _greedy_maximal_sidon()
    if not _is_maximal(best_S):
        # Re-maximize
        sums = _build_sums(best_S)
        S_set = set(best_S)
        while True:
            if time.time() > t_start + DEADLINE - 1.0:
                break
            addable = [y for y in range(1, N+1) if y not in S_set and _check_addable(y, best_S, sums)]
            if not addable:
                break
            best_y = addable[0]
            best_S.append(best_y)
            S_set.add(best_y)
            sums.update([best_y + a for a in best_S[:-1]] + [2 * best_y])
    
    return best_S

# EVOLVE-BLOCK-END
