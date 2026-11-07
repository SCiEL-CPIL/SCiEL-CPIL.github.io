import numpy as np
import time
import random
import math

N = 128
T0 = time.time()
DEADLINE = T0 + 80  # safety margin before 90s kill

def _compute_forbidden():
    """Precompute all forbidden 6-tuples {a,b,c,a+b,a+c,b+c} with distinct a<b<c and sums in 1..N."""
    forbidden = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            ab = a + b
            if ab > N:
                break
            for c in range(b + 1, N + 1):
                ac = a + c
                if ac > N:
                    break
                bc = b + c
                if bc > N:
                    break
                # All six values must be distinct
                vals = [a, b, c, ab, ac, bc]
                if len(set(vals)) == 6:
                    forbidden.append((a, b, c, ab, ac, bc))
    return forbidden

FORBIDDEN = _compute_forbidden()

# Build conflict graph: for each value v, list of forbidden tuples containing v
# and the other 5 values in each tuple
from collections import defaultdict
conflicts = defaultdict(list)
for tup in FORBIDDEN:
    for v in tup:
        others = [x for x in tup if x != v]
        conflicts[v].append(others)

def _is_valid(mask):
    """True iff no forbidden 6-tuple is fully contained in mask."""
    for tup in FORBIDDEN:
        if mask[tup[0]] and mask[tup[1]] and mask[tup[2]] and mask[tup[3]] and mask[tup[4]] and mask[tup[5]]:
            return False
    return True

def _is_valid_fast(mask, elem):
    """Check if adding elem to mask keeps it valid. Returns True if valid."""
    for others in conflicts[elem]:
        if all(mask[o] for o in others):
            return False
    return True

def _conflict_count(mask, elem):
    """Count how many forbidden tuples would be violated by adding elem to mask."""
    count = 0
    for others in conflicts[elem]:
        if all(mask[o] for o in others):
            count += 1
    return count

def _greedy_add(mask):
    """Add elements one by one, always choosing the one with fewest conflicts."""
    changed = True
    while changed:
        changed = False
        best_elem = -1
        best_conflicts = float('inf')
        for v in range(1, N + 1):
            if mask[v]:
                continue
            c = _conflict_count(mask, v)
            if c < best_conflicts:
                best_conflicts = c
                best_elem = v
            elif c == best_conflicts and v < best_elem:
                best_elem = v
        if best_elem > 0 and best_conflicts == 0:
            mask[best_elem] = True
            changed = True
    return mask

def _greedy_add_with_removal(mask):
    """Add elements, and if no zero-conflict element exists, try removing one blocker."""
    changed = True
    while changed:
        changed = False
        # First try zero-conflict additions
        best_elem = -1
        best_conflicts = float('inf')
        for v in range(1, N + 1):
            if mask[v]:
                continue
            c = _conflict_count(mask, v)
            if c < best_conflicts:
                best_conflicts = c
                best_elem = v
            elif c == best_conflicts and v < best_elem:
                best_elem = v
        if best_elem > 0 and best_conflicts == 0:
            mask[best_elem] = True
            changed = True
            continue
        # No zero-conflict element: try to remove a blocker to enable addition
        if best_elem > 0 and best_conflicts > 0:
            # Find blockers for best_elem
            blockers = set()
            for others in conflicts[best_elem]:
                if all(mask[o] for o in others):
                    for o in others:
                        blockers.add(o)
            # Try removing each blocker
            for b in blockers:
                mask2 = mask.copy()
                mask2[b] = False
                if _is_valid_fast(mask2, best_elem):
                    mask2[best_elem] = True
                    if mask2.sum() > mask.sum():
                        mask[:] = mask2
                        changed = True
                        break
    return mask

def _initial(seed):
    """Create initial mask using upper half + greedy fill."""
    rng = random.Random(seed)
    mask = np.zeros(N + 1, dtype=bool)
    mask[N // 2 + 1:] = True
    small = list(range(1, N // 2 + 1))
    rng.shuffle(small)
    for v in small:
        if _is_valid_fast(mask, v):
            mask[v] = True
    _greedy_add(mask)
    return mask

def _sa(mask, deadline, seed):
    """Simulated annealing with smart moves."""
    rng = random.Random(seed)
    cur = mask.copy()
    cur_size = int(cur.sum())
    best = cur.copy()
    best_size = cur_size
    T = 8.0
    while time.time() < deadline:
        in_set = np.nonzero(cur)[0]
        out_set = np.nonzero(~cur)[0]
        out_set = out_set[out_set >= 1]
        if len(in_set) == 0 or len(out_set) == 0:
            break
        r = rng.random()
        new = None
        delta = 0
        if r < 0.35:
            # Add with minimal conflicts
            best_c = float('inf')
            best_v = -1
            for v in out_set:
                c = _conflict_count(cur, int(v))
                if c < best_c:
                    best_c = c
                    best_v = int(v)
                elif c == best_c and int(v) < best_v:
                    best_v = int(v)
            if best_v > 0:
                new = cur.copy()
                new[best_v] = True
                delta = 1
                if not _is_valid(new):
                    new = None
        elif r < 0.60:
            # Smart add: insert x; if invalid, evict one blocker
            x = int(rng.choice(out_set))
            new = cur.copy()
            new[x] = True
            if _is_valid(new):
                delta = 1
            else:
                found = False
                blockers = set()
                for others in conflicts[x]:
                    if all(new[o] for o in others):
                        for o in others:
                            blockers.add(int(o))
                lst = list(blockers)
                rng.shuffle(lst)
                for y in lst[:50]:
                    new2 = new.copy()
                    new2[y] = False
                    if _is_valid(new2):
                        new = new2
                        delta = 0
                        found = True
                        break
                if not found:
                    new = None
        elif r < 0.80:
            # Remove random element
            x = int(rng.choice(in_set))
            new = cur.copy()
            new[x] = False
            delta = -1
        else:
            # Swap
            x = int(rng.choice(out_set))
            y = int(rng.choice(in_set))
            new = cur.copy()
            new[int(x)] = True
            new[int(y)] = False
            delta = 0
            if not _is_valid(new):
                new = None
        if new is None:
            continue
        if delta >= 0 or rng.random() < math.exp(-delta / T):
            cur = new
            cur_size += delta
            if cur_size > best_size:
                best_size = cur_size
                best = cur.copy()
        T = max(0.5, T * 0.999)
    return best, best_size

def _fallback():
    m = np.zeros(N + 1, dtype=bool)
    m[N // 2 + 1:] = True
    return [int(v) for v in np.nonzero(m)[0]]

def construct():
    try:
        t0 = time.time()
        deadline = T0 + 80
        best = None
        best_size = 0
        for seed in range(40):
            if time.time() > deadline - 10:
                break
            m = _initial(seed)
            cur_size = int(m.sum())
            if cur_size > best_size:
                best_size = cur_size
                best = m.copy()
            sub = min(deadline - 8, time.time() + 18)
            m2, sz = _sa(m, sub, 2000 + seed)
            _greedy_add_with_removal(m2)
            sz = int(m2.sum())
            if sz > best_size:
                best_size = sz
                best = m2.copy()
        if best is None:
            best = _initial(0)
        _greedy_add_with_removal(best)
        res = [int(v) for v in np.nonzero(best)[0]]
        chk = np.zeros(N + 1, dtype=bool)
        for v in res:
            chk[v] = True
        if not _is_valid(chk):
            return _fallback()
        return res
    except Exception:
        return _fallback()
