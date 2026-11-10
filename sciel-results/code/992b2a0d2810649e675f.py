"""Fresh design: simulated annealing with numpy-vectorized 4-AP checks,
multi-start, and a time-budget guard.  Bitmask representation of the subset
of [1,256].  Objective: maximize #3-APs subject to no 4-AP."""

import time
import random
import numpy as np

N = 256
DEADLINE = 85.0  # safety margin under 90 s

# Precompute all 3-APs (a, a+d, a+2d) with d>=1, 1<=a, a+2d<=256
# and all 4-APs (a, a+d, a+2d, a+3d) with d>=1, 1<=a, a+3d<=256
def _build_aps():
    aps3 = []
    for a in range(1, N + 1):
        maxd = (N - a) // 2
        for d in range(1, maxd + 1):
            aps3.append((a - 1, a + d - 1, a + 2 * d - 1))
    aps4 = []
    for a in range(1, N + 1):
        maxd = (N - a) // 3
        for d in range(1, maxd + 1):
            aps4.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    return aps3, aps4

_APS3, _APS4 = _build_aps()
_APS3_ARR = np.array(_APS3, dtype=np.int32)  # shape (M3, 3)
_APS4_ARR = np.array(_APS4, dtype=np.int32)  # shape (M4, 4)

def _count_3aps(mask):
    """mask: numpy bool array of length N.  Return number of 3-APs fully in mask."""
    if not mask.any():
        return 0
    # mask[aps3] gives shape (M3,3); all(axis=1)
    return int(np.all(mask[_APS3_ARR], axis=1).sum())

def _has_4ap(mask):
    """Return True if mask contains any 4-AP."""
    if mask.sum() < 4:
        return False
    return bool(np.any(np.all(mask[_APS4_ARR], axis=1)))

def _violating_4aps(mask):
    """Return list of indices into _APS4_ARR that are violated."""
    if mask.sum() < 4:
        return []
    viol = np.all(mask[_APS4_ARR], axis=1)
    return np.where(viol)[0].tolist()

def _greedy_construct(rng, bias=0.0):
    """Greedy: add elements in random order if no 4-AP is created."""
    mask = np.zeros(N, dtype=bool)
    order = list(range(N))
    rng.shuffle(order)
    for idx in order:
        mask[idx] = True
        if _has_4ap(mask):
            mask[idx] = False
    return mask

def _remove_to_fix(mask, violating):
    """Given mask and list of violating 4-AP indices, remove one element
    per violation (the element in the 4-AP that is in the mask).  Choose
    the element that appears in the most violations to minimize removals."""
    if not violating:
        return mask
    # Count how many violations each element participates in
    from collections import Counter
    cnt = Counter()
    for vi in violating:
        ap = _APS4_ARR[vi]
        for e in ap:
            if mask[e]:
                cnt[e] += 1
    # Remove the most-violating element
    if cnt:
        worst = max(cnt, key=cnt.get)
        mask = mask.copy()
        mask[worst] = False
    return mask

def _simulate(mask, rng, t_start):
    """Simulated annealing on the mask.  Moves: flip a random element.
    Accept if no 4-AP and 3-AP count improves; with probability exp(-delta/T)
    accept worsening if still 4-AP-free.  Periodically repair 4-APs."""
    T = 5.0
    T_min = 0.01
    cooling = 0.9995
    cur_3 = _count_3aps(mask)
    best_mask = mask.copy()
    best_3 = cur_3
    step = 0
    while time.time() - t_start < DEADLINE - 2.0:
        # Pick a random element to flip
        idx = rng.randrange(N)
        new_mask = mask.copy()
        new_mask[idx] = not new_mask[idx]
        if _has_4ap(new_mask):
            # Repair: try to fix by removing one element from each violation
            viol = _violating_4aps(new_mask)
            if viol:
                fixed = new_mask
                for _ in range(4):
                    v = _violating_4aps(fixed)
                    if not v:
                        break
                    # Remove the element in the first violation that's in mask
                    ap = _APS4_ARR[v[0]]
                    cand = [e for e in ap if fixed[e]]
                    if not cand:
                        break
                    # Pick the one that appears in most violations
                    from collections import Counter
                    cnt = Counter()
                    for vi in v:
                        for e in _APS4_ARR[vi]:
                            if fixed[e]:
                                cnt[e] += 1
                    worst = max(cand, key=lambda e: cnt.get(e, 0))
                    fixed[worst] = False
                new_mask = fixed
            else:
                continue
        new_3 = _count_3aps(new_mask)
        delta = new_3 - cur_3
        if delta >= 0:
            mask = new_mask
            cur_3 = new_3
            if cur_3 > best_3:
                best_3 = cur_3
                best_mask = mask.copy()
        else:
            # Accept with probability exp(delta / T)
            if T > T_min and rng.random() < np.exp(delta / T):
                mask = new_mask
                cur_3 = new_3
        T = max(T_min, T * cooling)
        step += 1
        if step % 2000 == 0 and T <= T_min:
            break
    return best_mask, best_3

def construct():
    t_start = time.time()
    rng = random.Random(12345)
    best_mask = np.zeros(N, dtype=bool)
    best_score = -1

    # Multiple restarts with different strategies
    n_restarts = 12
    for r in range(n_restarts):
        if time.time() - t_start > DEADLINE - 5.0:
            break
        if r == 0:
            # Greedy start
            mask = _greedy_construct(rng)
        elif r == 1:
            # Start from a structured set: multiples of 2 (even numbers) - no 4-AP? 
            # Actually evens: 2,4,6,8 -> 4-AP with d=2. So not safe.
            # Use a base-3-like construction: numbers with only 0,1 in base 3? 
            # For now, use random mask with density ~0.3
            mask = np.zeros(N, dtype=bool)
            density = 0.25 + 0.1 * r
            for i in range(N):
                if rng.random() < density:
                    mask[i] = True
            # Fix 4-APs
            for _ in range(20):
                v = _violating_4aps(mask)
                if not v:
                    break
                ap = _APS4_ARR[v[0]]
                cand = [e for e in ap if mask[e]]
                if cand:
                    mask[rng.choice(cand)] = False
        else:
            mask = _greedy_construct(rng)
        
        score = _count_3aps(mask)
        if score > best_score:
            best_score = score
            best_mask = mask.copy()
        
        # Run simulated annealing if time permits
        if time.time() - t_start < DEADLINE - 8.0:
            mask2, score2 = _simulate(mask, rng, t_start)
            if score2 > best_score:
                best_score = score2
                best_mask = mask2.copy()
    
    # Final verification
    if _has_4ap(best_mask):
        # Emergency fix
        for _ in range(50):
            v = _violating_4aps(best_mask)
            if not v:
                break
            ap = _APS4_ARR[v[0]]
            cand = [e for e in ap if best_mask[e]]
            if cand:
                best_mask[rng.choice(cand)] = False
            else:
                break
    
    result = [i + 1 for i in range(N) if best_mask[i]]
    if not result:
        result = [1]
    return sorted(result)

if __name__ == "__main__":
    res = construct()
    print(f"Cardinality: {len(res)}, 3-APs: {_count_3aps(np.array([x-1 in res for x in range(1,257)]))}")
