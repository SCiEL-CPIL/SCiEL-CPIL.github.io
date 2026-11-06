import random
import time

N = 256


def _compute_sums(sel):
    """Compute set of all pairwise sums a+b with a<=b."""
    sums = set()
    n = len(sel)
    for i in range(n):
        a = sel[i]
        for j in range(i, n):
            sums.add(a + sel[j])
    return sums


def _is_maximal_fast(sel, sums):
    """Check if sel is inclusion-maximal in [1, N]."""
    sel_set = set(sel)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Check if x can be added
        if 2 * x in sums:
            continue
        blocked = False
        for a in sel:
            if (x + a) in sums:
                blocked = True
                break
        if not blocked:
            return False
    return True


def _blocking_score(x, sel, sums, candidates):
    """Count how many candidates x would block if added."""
    if not candidates:
        return 0
    blocked = 0
    for y in candidates:
        if y == x:
            continue
        # y is blocked if x+y is in sums, or 2*x in sums (but that blocks x itself)
        # Actually, y is blocked if adding x makes y non-addable
        # y is non-addable if: 2*y in (sums ∪ {2*x}) OR ∃a in (sel ∪ {x}): y+a in (sums ∪ {2*x, x+a for a in sel})
        # Simplified: y is blocked if:
        #   2*y in sums OR 2*y == 2*x (i.e., y==x, skip)
        #   OR x+y in sums (new sum from x)
        #   OR ∃a in sel: y+a in sums (already in sums, so y was already blocked)
        #   OR ∃a in sel: y+a == 2*x (i.e., y = 2*x - a)
        #   OR ∃a in sel: y+a == x+a' for some a' (i.e., y = a' + x - a, but this is complex)
        
        # Simpler: check if y can be added to sel ∪ {x}
        # y can be added if: 2*y not in (sums ∪ {2*x}) AND for all a in (sel ∪ {x}): y+a not in (sums ∪ {2*x, x+a for a in sel})
        
        # Efficient check:
        can_add = True
        if 2 * y == 2 * x:  # y == x, skip
            continue
        if 2 * y in sums:
            can_add = False
        elif 2 * y == 2 * x:
            can_add = False
        else:
            # Check x+y
            if x + y in sums or x + y == 2 * x:
                can_add = False
            else:
                # Check y+a for a in sel
                for a in sel:
                    if y + a in sums:
                        can_add = False
                        break
                    if y + a == 2 * x:
                        can_add = False
                        break
                    if y + a == x + a:  # y == x, already handled
                        can_add = False
                        break
        if not can_add:
            blocked += 1
    return blocked


def _greedy_with_blocking(order, sel_init=None, sums_init=None):
    """Greedy Sidon construction prioritizing high-blocking elements."""
    if sel_init is None:
        sel = []
        sums = set()
    else:
        sel = sel_init[:]
        sums = sums_init.copy()
    
    sel_set = set(sel)
    candidates = [x for x in order if x not in sel_set]
    
    while candidates:
        # Find the candidate with maximum blocking score
        best_x = None
        best_score = -1
        for x in candidates:
            # Check if x can be added
            can_add = True
            if 2 * x in sums:
                can_add = False
            else:
                for a in sel:
                    if x + a in sums:
                        can_add = False
                        break
            
            if not can_add:
                candidates.remove(x)
                continue
            
            score = _blocking_score(x, sel, sums, candidates)
            if score > best_score:
                best_score = score
                best_x = x
        
        if best_x is None:
            break
        
        # Add best_x
        sel.append(best_x)
        sums.add(2 * best_x)
        for a in sel:
            sums.add(best_x + a)
        candidates.remove(best_x)
    
    return sel, sums


def _try_remove(sel, sums):
    """Try to remove an element while maintaining maximality."""
    n = len(sel)
    for idx in range(n):
        x = sel[idx]
        new_sel = sel[:idx] + sel[idx+1:]
        new_sums = _compute_sums(new_sel)
        if _is_maximal_fast(new_sel, new_sums):
            return new_sel, new_sums, x
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 25.0  # Safety margin before 30s kill

    best_sel = None
    best_size = N + 1

    # Phase 1: Ascending and descending greedy
    for order in [list(range(1, N + 1)), list(range(N, 0, -1))]:
        if time.time() > deadline:
            break
        sel, sums = _greedy_with_blocking(order)
        if _is_maximal_fast(sel, sums) and len(sel) < best_size:
            best_size = len(sel)
            best_sel = sel[:]

    # Phase 2: Randomized greedy with blocking priority
    rng = random.Random(42)
    trials = 0
    while time.time() < deadline - 3:
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        sel, sums = _greedy_with_blocking(order)
        
        if _is_maximal_fast(sel, sums) and len(sel) < best_size:
            best_size = len(sel)
            best_sel = sel[:]
            
            # Local search: try to remove elements
            if time.time() < deadline - 1:
                improved = True
                while improved:
                    if time.time() > deadline - 1:
                        break
                    improved = False
                    result = _try_remove(sel, sums)
                    if result is not None:
                        sel, sums, _ = result
                        improved = True
                        if len(sel) < best_size:
                            best_size = len(sel)
                            best_sel = sel[:]
        
        trials += 1
    
    # Phase 3: If we have a good candidate, do more aggressive local search
    if best_sel is not None and time.time() < deadline - 2:
        sel = best_sel[:]
        sums = _compute_sums(sel)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for attempt in range(100):
                if time.time() > deadline - 2:
                    break
                result = _try_remove(sel, sums)
                if result is not None:
                    sel, sums, _ = result
                    improved = True
                    if len(sel) < best_size:
                        best_size = len(sel)
                        best_sel = sel[:]
    
    if best_sel is None:
        # Fallback: simple ascending greedy
        sel = []
        sums = set()
        for x in range(1, N + 1):
            if 2 * x not in sums:
                can_add = True
                for a in sel:
                    if x + a in sums:
                        can_add = False
                        break
                if can_add:
                    sel.append(x)
                    sums.add(2 * x)
                    for a in sel:
                        sums.add(x + a)
        best_sel = sel

    return best_sel
