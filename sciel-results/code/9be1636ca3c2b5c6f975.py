"""
Combined: precomputed partner tables for 3-AP/4-AP + greedy seed +
strict local search + simulated annealing with periodic polishing.
Structural idea from inspiration 1 (precomputation + SA), refinement
machinery from current program (bitmask ops, deadline guards).
"""
import time
import random
import math

N = 256
DEADLINE = 55.0  # safety margin under 90s kill

# Precompute partner tables
# ap3_partners[x] (x in 0..N-1) = list of (y, z) such that {x+1, y+1, z+1} form a 3-AP
# ap4_partners[x] = list of (y, z, w) such that {x+1, y+1, z+1, w+1} form a 4-AP
ap3_partners = [[] for _ in range(N)]
ap4_partners = [[] for _ in range(N)]

def _precompute():
    # 3-APs: a, a+d, a+2d  (1-indexed a)
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            ia, ib, ic = a - 1, b - 1, c - 1
            ap3_partners[ia].append((ib, ic))
            ap3_partners[ib].append((ia, ic))
            ap3_partners[ic].append((ia, ib))
    # 4-APs: a, a+d, a+2d, a+3d
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            ia, ib, ic, ie = a - 1, b - 1, c - 1, e - 1
            ap4_partners[ia].append((ib, ic, ie))
            ap4_partners[ib].append((ia, ic, ie))
            ap4_partners[ic].append((ia, ib, ie))
            ap4_partners[ie].append((ia, ib, ic))

_precompute()

def _has_4ap(mask, x):
    """Check if adding element x (0-indexed) to mask creates a 4-AP involving x."""
    for y, z, w in ap4_partners[x]:
        if (mask >> y) & 1 and (mask >> z) & 1 and (mask >> w) & 1:
            return True
    return False

def _delta_add(mask, x):
    """Number of new 3-APs created by adding element x (0-indexed) to mask (x not in mask)."""
    cnt = 0
    for y, z in ap3_partners[x]:
        if (mask >> y) & 1 and (mask >> z) & 1:
            cnt += 1
    return cnt

def _delta_remove(mask, x):
    """Number of 3-APs destroyed by removing element x (0-indexed) from mask (x in mask)."""
    cnt = 0
    for y, z in ap3_partners[x]:
        if (mask >> y) & 1 and (mask >> z) & 1:
            cnt += 1
    return cnt

def _count_3aps(mask):
    """Total number of distinct 3-APs in mask."""
    total = 0
    for x in range(N):
        if (mask >> x) & 1:
            total += _delta_add(mask ^ (1 << x), x)
    return total // 3

def _present_list(mask):
    return [i for i in range(N) if (mask >> i) & 1]

def _absent_list(mask):
    return [i for i in range(N) if not (mask >> i) & 1]

def _greedy_seed():
    """Greedy: add elements in order if no 4-AP is created."""
    mask = 0
    for x in range(N):
        if not _has_4ap(mask, x):
            mask |= (1 << x)
    return mask

def _strict_local_search(mask, score, start_time):
    """
    Strict hill climbing: try additions, then swaps (remove+add).
    Returns (mask, score).
    """
    improved = True
    while improved:
        if time.time() - start_time > DEADLINE:
            break
        improved = False
        present = _present_list(mask)
        absent = _absent_list(mask)

        # 1. Try single additions
        for x in absent:
            if _has_4ap(mask, x):
                continue
            delta = _delta_add(mask, x)
            if delta > 0:
                mask |= (1 << x)
                score += delta
                improved = True
                break

        if improved:
            continue

        # 2. Try swaps: remove x (present), add y (absent)
        present = _present_list(mask)
        absent = _absent_list(mask)
        random.shuffle(present)
        random.shuffle(absent)

        found_swap = False
        for x in present:
            if time.time() - start_time > DEADLINE:
                break
            mask_wo = mask ^ (1 << x)
            loss = _delta_remove(mask, x)
            for y in absent:
                if _has_4ap(mask_wo, y):
                    continue
                gain = _delta_add(mask_wo, y)
                delta = gain - loss
                if delta > 0:
                    mask = mask_wo | (1 << y)
                    score += delta
                    improved = True
                    found_swap = True
                    break
            if found_swap:
                break

    return mask, score

def _simulated_annealing(mask, score, start_time, duration=2.0):
    """
    Simulated Annealing to escape local optima.
    Move types: add, swap, remove.
    """
    T0 = 8.0
    Tend = 0.1
    call_start = time.time()
    end_time = call_start + duration

    while time.time() < end_time and time.time() - start_time < DEADLINE:
        elapsed = time.time() - call_start
        progress = min(1.0, elapsed / duration)
        T = T0 * math.exp(-8.0 * progress)
        if T < Tend:
            T = Tend

        present = _present_list(mask)
        absent = _absent_list(mask)

        if not present:
            break

        r = random.random()

        if r < 0.45 and absent:
            # Add a random absent element
            x = random.choice(absent)
            if _has_4ap(mask, x):
                continue
            delta = _delta_add(mask, x)
            if delta > 0 or random.random() < math.exp(delta / T):
                mask |= (1 << x)
                score += delta

        elif r < 0.80 and present and absent:
            # Swap: remove random present, add random absent
            rem = random.choice(present)
            add = random.choice(absent)
            mask_wo = mask ^ (1 << rem)
            if _has_4ap(mask_wo, add):
                continue
            loss = _delta_remove(mask, rem)
            gain = _delta_add(mask_wo, add)
            delta = gain - loss
            if delta > 0 or random.random() < math.exp(delta / T):
                mask = mask_wo | (1 << add)
                score += delta

        else:
            # Remove a random present element
            if len(present) > 1:
                rem = random.choice(present)
                loss = _delta_remove(mask, rem)
                delta = -loss
                if delta > 0 or random.random() < math.exp(delta / T):
                    mask ^= (1 << rem)
                    score += delta

    return mask, score

def construct():
    start = time.time()
    random.seed(12345)

    # Initial solution: greedy seed
    mask = _greedy_seed()
    score = _count_3aps(mask)

    best_mask = mask
    best_score = score

    # Phase 1: Strict local search
    mask, score = _strict_local_search(mask, score, start)
    if score > best_score:
        best_score = score
        best_mask = mask

    # Phase 2: SA with periodic polishing
    sa_burst = 2.0
    while time.time() - start < DEADLINE:
        mask, score = _simulated_annealing(mask, score, start, duration=sa_burst)
        mask, score = _strict_local_search(mask, score, start)
        if score > best_score:
            best_score = score
            best_mask = mask

        # If we've been stuck, do a perturbation
        if time.time() - start < DEADLINE:
            present = _present_list(mask)
            if len(present) > 2:
                # Remove 2-3 random elements to perturb
                for _ in range(random.randint(2, 3)):
                    if len(_present_list(mask)) > 1:
                        p = _present_list(mask)
                        rem = random.choice(p)
                        mask ^= (1 << rem)
                        score -= _delta_remove(mask | (1 << rem), rem)
                # Polish after perturbation
                mask, score = _strict_local_search(mask, score, start)
                if score > best_score:
                    best_score = score
                    best_mask = mask

    # Convert best_mask to sorted list (1-indexed)
    result = [x + 1 for x in range(N) if (best_mask >> x) & 1]
    return sorted(result)
