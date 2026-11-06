from math import gcd
import time
import random

_START = time.time()
_DEADLINE = 145.0  # Safety margin under 150s


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _pair_ok(m1, r1, m2, r2):
    g = gcd(m1, m2)
    return (r1 % g) != (r2 % g)


def _is_valid(classes):
    n = len(classes)
    for i in range(n):
        m1, r1 = classes[i]
        for j in range(i + 1, n):
            m2, r2 = classes[j]
            if not _pair_ok(m1, r1, m2, r2):
                return False
    return True


def _score(classes):
    return sum(1.0 / m for m, _ in classes)


def _find_residue(classes, m):
    """Return a residue r for modulus m that is disjoint from all existing, else None."""
    # Optimization: Precompute the required residues modulo gcd(m, mi)
    # For each existing class (mi, ri), r must NOT be congruent to ri mod gcd(m, mi)
    # This is a system of congruence constraints.
    # Simple approach: iterate r from 0 to m-1 and check all.
    # Since m <= 200, this is fast enough if done carefully.
    
    # We can speed this up by checking constraints incrementally or using a bitset if m is small,
    # but m is up to 200, so a simple loop is O(n*m). With n~180, m~200, that's 36k ops.
    # In Python, 36k ops is ~1ms. If we do this 180 times, that's 180ms. Acceptable.
    
    for r in range(m):
        ok = True
        for (mi, ri) in classes:
            g = gcd(m, mi)
            if (r % g) == (ri % g):
                ok = False
                break
        if ok:
            return r
    return None


def _greedy_build(order, classes=None):
    """Build a solution by processing moduli in the given order."""
    if classes is None:
        classes = []
    used = set()
    for m in order:
        if m in used:
            continue
        r = _find_residue(classes, m)
        if r is not None:
            classes.append([m, r])
            used.add(m)
    return classes


def _add_pass(classes):
    """Add every missing modulus that has a valid residue. Returns True if anything added."""
    added = False
    used = set(m for m, _ in classes)
    for m in range(20, 201):
        if m in used:
            continue
        r = _find_residue(classes, m)
        if r is not None:
            classes.append([m, r])
            used.add(m)
            added = True
    return added


def _sa_pass(classes, budget, rng):
    """
    Simulated Annealing with a focus on improving the set of moduli.
    Moves:
    1. Reassign residue of an existing class.
    2. Add a missing class if possible.
    3. Remove a class if it blocks too many others (less common, but helps escape local minima).
    """
    t0 = time.time()
    n = len(classes)
    best = [list(c) for c in classes]
    best_score = _score(best)
    current = [list(c) for c in classes]
    current_score = _score(current)

    # Precompute gcd matrix for speed
    gcd_mat = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g = gcd(current[i][0], current[j][0])
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    T0 = 0.05
    T_min = 1e-6
    # Run for the entire budget
    while _time_left() > 5 and (time.time() - t0) < budget:
        # Geometric cooling
        T = max(T_min, T0 * (0.9999 ** (int((time.time() - t0) * 1000))))
        
        # Pick a random move type
        move_type = rng.random()
        
        if move_type < 0.7:
            # Reassign residue of a random class
            i = rng.randrange(n)
            mi = current[i][0]
            old_r = current[i][1]
            new_r = rng.randrange(mi)
            if new_r == old_r:
                continue
            
            # Check validity
            ok = True
            for j in range(n):
                if j == i:
                    continue
                g = gcd_mat[i][j]
                if (new_r % g) == (current[j][1] % g):
                    ok = False
                    break
            
            if not ok:
                continue
            
            current[i][1] = new_r
            
            # Try to add missing moduli
            added = []
            used = set(m for m, _ in current)
            for m in range(20, 201):
                if m in used:
                    continue
                r = _find_residue(current, m)
                if r is not None:
                    current.append([m, r])
                    added.append([m, r])
            
            new_score = _score(current)
            delta = new_score - current_score
            
            # Acceptance
            if delta >= 0 or rng.random() < pow(2.718281828, delta / T):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best = [list(c) for c in current]
                    # Update gcd matrix if we added elements
                    if added:
                        n = len(current)
                        gcd_mat = [[0] * n for _ in range(n)]
                        for ii in range(n):
                            for jj in range(ii + 1, n):
                                g = gcd(current[ii][0], current[jj][0])
                                gcd_mat[ii][jj] = g
                                gcd_mat[jj][ii] = g
            else:
                # Revert
                for a in reversed(added):
                    current.pop()
                current[i][1] = old_r

        elif move_type < 0.9:
            # Try to add a specific missing modulus by reassigning a random existing one
            used = set(m for m, _ in current)
            missing = [m for m in range(20, 201) if m not in used]
            if not missing:
                continue
            m_target = rng.choice(missing)
            i = rng.randrange(n)
            mi = current[i][0]
            old_r = current[i][1]
            
            # Try a few random residues for i to see if it helps add m_target
            for _ in range(5):
                new_r = rng.randrange(mi)
                if new_r == old_r:
                    continue
                ok = True
                for j in range(n):
                    if j == i:
                        continue
                    g = gcd_mat[i][j]
                    if (new_r % g) == (current[j][1] % g):
                        ok = False
                        break
                if not ok:
                    continue
                
                current[i][1] = new_r
                r_target = _find_residue(current, m_target)
                if r_target is not None:
                    current.append([m_target, r_target])
                    new_score = _score(current)
                    delta = new_score - current_score
                    if delta >= 0 or rng.random() < pow(2.718281828, delta / T):
                        current_score = new_score
                        if new_score > best_score:
                            best_score = new_score
                            best = [list(c) for c in current]
                        n = len(current)
                        gcd_mat = [[0] * n for _ in range(n)]
                        for ii in range(n):
                            for jj in range(ii + 1, n):
                                g = gcd(current[ii][0], current[jj][0])
                                gcd_mat[ii][jj] = g
                                gcd_mat[jj][ii] = g
                        break
                    else:
                        current.pop()
                        current[i][1] = old_r
                else:
                    current[i][1] = old_r

        else:
            # Remove a random class (exploration)
            if n > 50:
                i = rng.randrange(n)
                removed = current.pop(i)
                current_score = _score(current)
                n -= 1
                # Rebuild gcd matrix
                gcd_mat = [[0] * n for _ in range(n)]
                for ii in range(n):
                    for jj in range(ii + 1, n):
                        g = gcd(current[ii][0], current[jj][0])
                        gcd_mat[ii][jj] = g
                        gcd_mat[jj][ii] = g
                # Add back if it was a bad move? No, just let SA decide.
                # But we should try to add it back or other moduli.
                added = []
                used = set(m for m, _ in current)
                for m in range(20, 201):
                    if m in used:
                        continue
                    r = _find_residue(current, m)
                    if r is not None:
                        current.append([m, r])
                        added.append([m, r])
                new_score = _score(current)
                delta = new_score - current_score
                if delta >= 0 or rng.random() < pow(2.718281828, delta / T):
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best = [list(c) for c in current]
                    n = len(current)
                    gcd_mat = [[0] * n for _ in range(n)]
                    for ii in range(n):
                        for jj in range(ii + 1, n):
                            g = gcd(current[ii][0], current[jj][0])
                            gcd_mat[ii][jj] = g
                            gcd_mat[jj][ii] = g
                else:
                    # Revert removal
                    for a in reversed(added):
                        current.pop()
                    current.insert(i, removed)
                    current_score = _score(current)
                    n += 1
                    gcd_mat = [[0] * n for _ in range(n)]
                    for ii in range(n):
                        for jj in range(ii + 1, n):
                            g = gcd(current[ii][0], current[jj][0])
                            gcd_mat[ii][jj] = g
                            gcd_mat[jj][ii] = g

    return best


def _polish_pass(classes, budget, rng):
    """
    Deterministic polish: try to improve by reassigning residues of small moduli.
    """
    t0 = time.time()
    best = [list(c) for c in classes]
    best_score = _score(best)
    n = len(best)
    
    # Sort indices by modulus (smallest first)
    indices = sorted(range(n), key=lambda i: best[i][0])
    
    improved = True
    while improved and _time_left() > 5 and (time.time() - t0) < budget:
        improved = False
        # Re-sort if size changed
        indices = sorted(range(len(best)), key=lambda i: best[i][0])
        
        for idx in indices:
            if _time_left() < 5 or (time.time() - t0) > budget:
                break
            mi = best[idx][0]
            old_r = best[idx][1]
            
            for r in range(mi):
                if _time_left() < 5 or (time.time() - t0) > budget:
                    break
                if r == old_r:
                    continue
                
                # Check validity
                ok = True
                for j in range(len(best)):
                    if j == idx:
                        continue
                    mj, rj = best[j]
                    if not _pair_ok(mi, r, mj, rj):
                        ok = False
                        break
                
                if not ok:
                    continue
                
                best[idx][1] = r
                
                # Try to add missing moduli
                used = set(m for m, _ in best)
                added = []
                for m in range(20, 201):
                    if m in used:
                        continue
                    r2 = _find_residue(best, m)
                    if r2 is not None:
                        best.append([m, r2])
                        added.append([m, r2])
                
                new_score = _score(best)
                if new_score > best_score:
                    best_score = new_score
                    improved = True
                    break
                else:
                    # Revert
                    for a in reversed(added):
                        best.pop()
                    best[idx][1] = old_r
            
            if improved:
                break # Restart the outer loop
    
    return best


def construct():
    rng = random.Random(20240517)
    best = None
    best_score = -1.0

    def consider(classes):
        nonlocal best, best_score
        if classes and _is_valid(classes):
            s = _score(classes)
            if s > best_score:
                best_score = s
                best = [list(c) for c in classes]

    # Strategy 1: Greedy on all moduli
    if _time_left() > 10:
        all_moduli = list(range(20, 201))
        c1 = _greedy_build(all_moduli)
        c1 = _add_pass(c1)
        consider(c1)

    # Strategy 2: SA on the best solution
    if best is not None and _time_left() > 30:
        c2 = [list(c) for c in best]
        c2 = _sa_pass(c2, min(60.0, max(10.0, _time_left() - 20)), rng)
        consider(c2)

    # Strategy 3: Polish
    if best is not None and _time_left() > 20:
        c3 = [list(c) for c in best]
        c3 = _polish_pass(c3, min(30.0, max(5.0, _time_left() - 10)), rng)
        consider(c3)

    # Strategy 4: Another SA round if time permits
    if best is not None and _time_left() > 20:
        c4 = [list(c) for c in best]
        c4 = _sa_pass(c4, min(20.0, max(5.0, _time_left() - 10)), rng)
        consider(c4)
        
        if _time_left() > 10:
            c4 = _polish_pass(c4, min(5.0, max(1.0, _time_left() - 5)), rng)
            consider(c4)

    # Fallback
    if best is None:
        best = [[20 * j, j - 1] for j in range(1, 11)]

    return best
