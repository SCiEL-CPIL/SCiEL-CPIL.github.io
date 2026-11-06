import time
import random
import numpy as np

BASE = 51975

# All divisors of 51975 = 3^3 * 5^2 * 7 * 11, greater than 1
_DIVS = []
for a in range(4):          # 3^a
    for b in range(3):      # 5^b
        for c in range(2):  # 7^c
            for d in range(2):  # 11^d
                v = (3 ** a) * (5 ** b) * (7 ** c) * (11 ** d)
                if v > 1:
                    _DIVS.append(v)
_DIVS.sort()
N = len(_DIVS)

# Precompute: for each divisor index i, a list of numpy boolean arrays (one per residue)
# _MASK[i][r] = boolean array of length BASE, True where residue r mod m covers that position
_MASK = []
for m in _DIVS:
    lst = []
    for r in range(m):
        arr = np.zeros(BASE, dtype=np.bool_)
        arr[r::m] = True
        lst.append(arr)
    _MASK.append(lst)


def _score(covered):
    # number of residues covered by no class
    return int(np.count_nonzero(~covered))


def _greedy_dynamic(rng=None, deadline=None, max_classes=None):
    """Dynamic greedy set cover: at each step, pick the (divisor, residue) class
    that covers the most currently-uncovered positions.
    Returns (classes, covered_array, score)."""
    covered = np.zeros(BASE, dtype=np.bool_)
    classes = []
    used_divs = set()
    
    if max_classes is None:
        max_classes = N
    
    for step in range(max_classes):
        if deadline is not None and time.time() > deadline:
            break
        
        best_gain = 0
        best_i = -1
        best_r = -1
        
        for i in range(N):
            if i in used_divs:
                continue
            m = _DIVS[i]
            masks = _MASK[i]
            for r in range(m):
                mask = masks[r]
                gain = int(np.count_nonzero(mask & ~covered))
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_r = r
        
        if best_gain <= 0:
            break
        
        used_divs.add(best_i)
        m = _DIVS[best_i]
        classes.append([m, best_r])
        covered |= _MASK[best_i][best_r]
    
    score = _score(covered)
    return classes, covered, score


def _greedy_with_order(order, deadline=None):
    """Greedy following a fixed order of divisor indices.
    Returns (classes, covered, score)."""
    covered = np.zeros(BASE, dtype=np.bool_)
    classes = []
    
    for i in order:
        if deadline is not None and time.time() > deadline:
            break
        m = _DIVS[i]
        masks = _MASK[i]
        best_r = 0
        best_c = -1
        for r in range(m):
            c = int(np.count_nonzero(masks[r] & ~covered))
            if c > best_c:
                best_c = c
                best_r = r
        classes.append([m, best_r])
        covered |= masks[best_r]
    
    score = _score(covered)
    return classes, covered, score


def _local_swap_improve(classes, covered, rng, deadline):
    """Try to improve by swapping residues of existing classes.
    Returns (new_classes, new_covered, new_score)."""
    best_classes = [c[:] for c in classes]
    best_covered = covered.copy()
    best_score = _score(covered)
    
    for _ in range(30):
        if deadline is not None and time.time() > deadline:
            break
        if not best_classes:
            break
        
        ci = rng.randrange(len(best_classes))
        m, old_r = best_classes[ci]
        div_idx = _DIVS.index(m)
        
        tried = set()
        for _ in range(15):
            new_r = rng.randrange(m)
            if new_r in tried or new_r == old_r:
                continue
            tried.add(new_r)
            
            # Recompute covered with swap
            new_covered = np.zeros(BASE, dtype=np.bool_)
            for c in best_classes:
                mi = _DIVS.index(c[0])
                ri = c[1]
                new_covered |= _MASK[mi][ri]
            new_covered &= ~_MASK[div_idx][old_r]
            new_covered |= _MASK[div_idx][new_r]
            
            new_score = _score(new_covered)
            if new_score < best_score:
                best_score = new_score
                best_classes[ci] = [m, new_r]
                best_covered = new_covered
                if best_score == 0:
                    return best_classes, best_covered, best_score
                break
    
    return best_classes, best_covered, best_score


def construct():
    deadline = time.time() + 110.0
    best_classes = None
    best_score = BASE + 1
    best_covered = None
    
    rng = random.Random(42)
    
    # Phase 1: Dynamic greedy (no fixed order)
    classes, covered, s = _greedy_dynamic(rng=rng, deadline=deadline - 30)
    if s < best_score:
        best_score = s
        best_classes = [c[:] for c in classes]
        best_covered = covered.copy()
    if best_score == 0:
        return best_classes
    
    # Phase 2: Local search / swap improvement
    if best_classes is not None:
        best_classes, best_covered, best_score = _local_swap_improve(
            best_classes, best_covered, rng, deadline - 20
        )
        if best_score == 0:
            return best_classes
    
    # Phase 3: Try different fixed orders
    orders = []
    orders.append(list(range(N - 1, -1, -1)))  # large first
    orders.append(list(range(N)))              # small first
    for _ in range(5):
        o = list(range(N))
        rng.shuffle(o)
        orders.append(o)
    
    for order in orders:
        if time.time() > deadline - 10:
            break
        classes, covered, s = _greedy_with_order(order, deadline=deadline - 10)
        if s < best_score:
            best_score = s
            best_classes = [c[:] for c in classes]
            best_covered = covered.copy()
            if best_score == 0:
                return best_classes
        if best_classes is not None and time.time() < deadline - 5:
            best_classes, best_covered, best_score = _local_swap_improve(
                best_classes, best_covered, rng, deadline - 5
            )
            if best_score == 0:
                return best_classes
    
    # Phase 4: More dynamic greedy with different random seeds
    while time.time() < deadline - 3:
        rng2 = random.Random(rng.randrange(10**9))
        classes, covered, s = _greedy_dynamic(rng=rng2, deadline=deadline - 3)
        if s < best_score:
            best_score = s
            best_classes = [c[:] for c in classes]
            best_covered = covered.copy()
            if best_score == 0:
                return best_classes
        if best_classes is not None and time.time() < deadline - 1:
            best_classes, best_covered, best_score = _local_swap_improve(
                best_classes, best_covered, rng, deadline - 1
            )
            if best_score == 0:
                return best_classes
    
    return best_classes
