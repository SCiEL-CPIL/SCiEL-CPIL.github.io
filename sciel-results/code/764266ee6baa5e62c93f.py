import random
import time

N = 256
MAX_SUM = 2 * N


def _can_add(x, sel, sums):
    """Check if x can be added to sel (Sidon property preserved)."""
    for a in sel:
        if a == x:
            return False
    for a in sel:
        if (x + a) in sums:
            return False
    if (2 * x) in sums:
        return False
    return True


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
    """Fast maximality check: try every missing element."""
    sel_set = set(sel)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        ok = True
        for a in sel:
            if (x + a) in sums:
                ok = False
                break
        if ok and (2 * x) in sums:
            ok = False
        if ok:
            return False
    return True


def _greedy_construct(order):
    """Greedy Sidon construction following given order."""
    sel = []
    sums = set()
    for x in order:
        if _can_add(x, sel, sums):
            sel.append(x)
            for a in sel:
                sums.add(x + a)
            sums.add(2 * x)
    return sel, sums


def _greedy_with_blocking(order):
    """Greedy that prioritizes elements that block many others."""
    sel = []
    sums = set()
    sel_set = set()
    
    # Precompute for each x, how many elements it would block if added
    # An element y is "blocked" by x if y+x is in sums or 2*y is in sums
    # Actually, we want to pick elements that, once added, prevent many others from being added
    
    remaining = list(order)
    while remaining:
        if not sel:
            # Pick first element
            x = remaining[0]
            remaining = remaining[1:]
        else:
            # Score each candidate by how many remaining elements it blocks
            best_x = None
            best_score = -1
            for x in remaining:
                if _can_add(x, sel, sums):
                    # Count how many other remaining elements would be blocked
                    score = 0
                    for y in remaining:
                        if y == x:
                            continue
                        if (x + y) in sums or (2 * y) in sums:
                            score += 1
                    # Also count how many elements x itself blocks via its own sums
                    if score > best_score:
                        best_score = score
                        best_x = x
            
            if best_x is None:
                # No element can be added, break
                break
            
            x = best_x
            remaining.remove(x)
        
        sel.append(x)
        sel_set.add(x)
        for a in sel:
            sums.add(x + a)
        sums.add(2 * x)
    
    return sel, sums


def _try_remove_fast(sel, sums):
    """Try to remove an element while maintaining maximality. Returns (new_sel, new_sums) or None."""
    n = len(sel)
    sel_set = set(sel)
    
    # Precompute which sums are unique to each element
    # For element x at index i, sums involving x are: x+sel[j] for j<=i and 2*x
    # When we remove x, these sums disappear
    
    for idx in range(n):
        x = sel[idx]
        new_sel = [a for i, a in enumerate(sel) if i != idx]
        
        # Compute new sums efficiently
        new_sums = set()
        nn = len(new_sel)
        for i in range(nn):
            a = new_sel[i]
            for j in range(i, nn):
                new_sums.add(a + new_sel[j])
        
        # Check maximality of new_sel
        new_sel_set = set(new_sel)
        maximal = True
        for y in range(1, N + 1):
            if y in new_sel_set:
                continue
            can = True
            for a in new_sel:
                if (y + a) in new_sums:
                    can = False
                    break
            if can and (2 * y) in new_sums:
                can = False
            if can:
                maximal = False
                break
        
        if maximal:
            return new_sel, new_sums
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill

    best_sel = None
    best_size = N + 1

    # Phase 1: Standard greedy with different orderings
    for trial in range(1500):
        if time.time() > deadline:
            break

        if trial == 0:
            order = list(range(1, N + 1))
        elif trial == 1:
            order = list(range(N, 0, -1))
        else:
            order = list(range(1, N + 1))
            random.shuffle(order)

        sel, sums = _greedy_construct(order)

        if len(sel) < best_size:
            best_size = len(sel)
            best_sel = sel[:]
            best_sums = sums

        # Local search: try to remove elements
        if len(sel) < best_size:
            improved = True
            while improved:
                if time.time() > deadline:
                    break
                improved = False
                result = _try_remove_fast(sel, sums)
                if result is not None:
                    sel, sums = result

            if len(sel) < best_size:
                best_size = len(sel)
                best_sel = sel[:]
                best_sums = sums

    # Phase 2: Blocking-priority greedy
    for trial in range(800):
        if time.time() > deadline:
            break

        order = list(range(1, N + 1))
        random.shuffle(order)

        sel, sums = _greedy_with_blocking(order)

        if len(sel) < best_size:
            best_size = len(sel)
            best_sel = sel[:]
            best_sums = sums

        # Local search
        if len(sel) < best_size:
            improved = True
            while improved:
                if time.time() > deadline:
                    break
                improved = False
                result = _try_remove_fast(sel, sums)
                if result is not None:
                    sel, sums = result

            if len(sel) < best_size:
                best_size = len(sel)
                best_sel = sel[:]
                best_sums = sums

    # Phase 3: Random skip greedy
    for trial in range(2000):
        if time.time() > deadline:
            break

        order = list(range(1, N + 1))
        random.shuffle(order)

        sel = []
        sums = set()
        for x in order:
            if random.random() < 0.05:
                continue
            if _can_add(x, sel, sums):
                sel.append(x)
                for a in sel:
                    sums.add(x + a)
                sums.add(2 * x)

        if len(sel) < best_size:
            best_size = len(sel)
            best_sel = sel[:]
            best_sums = sums

        # Local search
        if len(sel) < best_size:
            improved = True
            while improved:
                if time.time() > deadline:
                    break
                improved = False
                result = _try_remove_fast(sel, sums)
                if result is not None:
                    sel, sums = result

            if len(sel) < best_size:
                best_size = len(sel)
                best_sel = sel[:]
                best_sums = sums

    # Phase 4: Final polish - try more removals on best solution
    if best_sel is not None and best_size < N:
        sel = best_sel[:]
        sums = _compute_sums(sel)
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            result = _try_remove_fast(sel, sums)
            if result is not None:
                sel, sums = result
                best_size = len(sel)
                best_sel = sel[:]
                best_sums = sums

    if best_sel is None:
        # Fallback: simple ascending greedy
        sel = []
        sums = set()
        for x in range(1, N + 1):
            if _can_add(x, sel, sums):
                sel.append(x)
                for a in sel:
                    sums.add(x + a)
                sums.add(2 * x)
        best_sel = sel

    return best_sel
