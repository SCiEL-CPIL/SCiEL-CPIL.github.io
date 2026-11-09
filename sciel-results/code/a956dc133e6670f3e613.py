import time
import random
import itertools

_START = time.time()
_DEADLINE = 80.0  # safety margin before 90s kill

def _sq_nonzero_count(coefs):
    n = len(coefs)
    cnt = 0
    for i in range(n):
        a = coefs[i]
        if a == 0:
            continue
        for j in range(i, n):
            v = a * a if i == j else 2 * a * coefs[j]
            if v != 0:
                cnt += 1
                break
    return cnt

def _valid(coefs):
    if len(coefs) > 41:
        return False
    nz = 0
    for c in coefs:
        if not isinstance(c, int):
            return False
        if c < -64 or c > 64:
            return False
        if c != 0:
            nz += 1
    if nz != 12:
        return False
    if coefs[-1] == 0:
        return False
    return True

def _polish(best, support, rng):
    """Local hill-climbing polish: try changing one coefficient at a time."""
    n = len(support)
    hi = max(support)
    best_local = best[1][:] if best[1] else None
    if best_local is None:
        return best
    best_score = best[0]
    
    # Try multiple polish passes
    for _pass in range(3):
        if time.time() > _DEADLINE:
            break
        improved = True
        while improved and time.time() < _DEADLINE:
            improved = False
            for idx, d in enumerate(support):
                if time.time() > _DEADLINE:
                    break
                old_val = best_local[d]
                # Try various new values
                candidates = []
                for v in [-64,-32,-16,-8,-4,-3,-2,-1,1,2,3,4,8,16,32,64]:
                    if v != old_val:
                        candidates.append(v)
                # Also try small perturbations
                for delta in [-3,-2,-1,1,2,3]:
                    v = old_val + delta
                    if -64 <= v <= 64 and v != 0 and v != old_val:
                        candidates.append(v)
                rng.shuffle(candidates)
                for new_val in candidates[:8]:
                    if time.time() > _DEADLINE:
                        break
                    best_local[d] = new_val
                    if not _valid(best_local):
                        best_local[d] = old_val
                        continue
                    sc = _sq_nonzero_count(best_local)
                    if sc < best_score:
                        best_score = sc
                        best[0] = sc
                        best[1] = best_local[:]
                        improved = True
                    else:
                        best_local[d] = old_val
    
    return best

def _try_place(best, support, rng):
    # assign coefficients to the given support, optimize to reduce square nz
    n = len(support)
    best_local = None
    best_score = 10**9
    hi = max(support)
    
    # More diverse initial coefficient values
    val_pool = [-4, -3, -2, -1, 1, 2, 3, 4]
    
    for _ in range(300):
        if time.time() > _DEADLINE:
            break
        coefs = [0] * 41
        for k, d in enumerate(support):
            c = rng.choice(val_pool)
            coefs[d] = c
        # ensure final (highest degree) nonzero
        if coefs[hi] == 0:
            coefs[hi] = 1
        sc = _sq_nonzero_count(coefs)
        if sc < best_score:
            best_score = sc
            best_local = coefs[:]
        
        # Local perturbation on current best
        if best_local is not None:
            cand = best_local[:]
            d = support[rng.randrange(n)]
            old = cand[d]
            new = old + rng.choice([-3, -2, -1, 1, 2, 3])
            if -64 <= new <= 64 and new != 0:
                cand[d] = new
            # Occasionally swap support positions
            if rng.random() < 0.15:
                d2 = support[rng.randrange(n)]
                if d2 != hi and cand[d2] != 0:
                    cand[d2] = 0
                    for dd in support:
                        if cand[dd] == 0 and dd != hi:
                            cand[dd] = rng.choice([-1, 1])
                            break
            if _valid(cand):
                sc2 = _sq_nonzero_count(cand)
                if sc2 < best_score:
                    best_score = sc2
                    best_local = cand[:]
    
    if best_local is not None and _valid(best_local):
        sc = _sq_nonzero_count(best_local)
        if sc < best[0]:
            best[0] = sc
            best[1] = best_local[:]
            # Polish
            _polish(best, support, rng)
    return best

def _make_palindromic_support():
    """Create a palindromic support around degree 20."""
    # Pick 6 pairs + possibly center
    left_degrees = sorted(random.sample(range(0, 20), 6))
    right_degrees = [40 - d for d in left_degrees]
    support = sorted(set(left_degrees + right_degrees))
    # Ensure we have exactly 12 and degree 40 present
    if 40 not in support:
        support = support[:11] + [40]
        support = sorted(set(support))
    while len(support) < 12:
        support.append(40 - support[0])
        support = sorted(set(support))
    if len(support) > 12:
        support = sorted(support[-12:])
    # Ensure degree 40 is present
    if 40 not in support:
        support = support[:11] + [40]
        support = sorted(set(support))
    return support[:12]

def _make_symmetric_support(rng):
    """Create a support with symmetry that encourages cancellation."""
    # Use pairs (d, 40-d) to get palindromic structure
    pairs = [(d, 40-d) for d in range(0, 21) if d != 40-d]
    rng.shuffle(pairs)
    support = set()
    for d1, d2 in pairs:
        if len(support) >= 12:
            break
        support.add(d1)
        support.add(d2)
    # Ensure 40 is present
    if 40 not in support:
        support = set(list(support)[:11] + [40])
    support = sorted(support)
    if len(support) != 12:
        # Pad or trim
        if len(support) > 12:
            support = sorted(support[-12:])
        else:
            for d in range(40, -1, -1):
                if d not in support:
                    support.append(d)
                    support = sorted(support)
                if len(support) == 12:
                    break
    return support

def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 80.0
    rng = random.Random(12345)
    best = [10**9, None]

    # 1) Palindromic/symmetric supports - these encourage cancellation
    for t in range(40):
        if time.time() > _DEADLINE:
            break
        sup = _make_symmetric_support(rng)
        _try_place(best, sup, rng)
    
    # 2) Arithmetic progressions with various gaps
    for start in range(0, 10):
        for gap in range(2, 14):
            if time.time() > _DEADLINE:
                break
            sup = []
            d = start
            while d <= 40 and len(sup) < 12:
                sup.append(d)
                d += gap
            if len(sup) < 12:
                continue
            if 40 not in sup:
                sup = sup[:-1] + [40]
                sup = sorted(set(sup))
                if len(sup) != 12:
                    continue
            _try_place(best, sup, rng)
    
    # 3) Randomized supports
    def make_support():
        pool = list(range(0, 40))
        rng.shuffle(pool)
        chosen = sorted(pool[:11]) + [40]
        return chosen
    
    for t in range(30):
        if time.time() > _DEADLINE:
            break
        sup = make_support()
        _try_place(best, sup, rng)
    
    # 4) Final polish on the best result
    if best[1] is not None:
        support = [i for i in range(41) if best[1][i] != 0]
        _polish(best, support, rng)

    if best[1] is None:
        # fallback: a valid 12-term polynomial
        coefs = [0] * 41
        for i in range(12):
            coefs[i] = 1
        return coefs

    return best[1]
