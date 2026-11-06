"""Bitmask precomputation + simulated-annealing local search.

Strategy:
  * Represent the candidate subset as a 256-bit integer (bit i ↔ element i+1).
  * Precompute, for every element, the partner-tuples for every 3-AP and 4-AP
    it participates in. This turns 4-AP validity checks and 3-AP delta
    evaluations into tight loops over small partner lists with bit tests.
  * Build a maximal 4-AP-free seed greedily.
  * Run a simulated-annealing local search over valid single-bit flips and
    swaps. SA accepts negative moves with a decaying temperature, which lets
    the search escape local optima that strict hill-climbing gets stuck in.
  * Track the best mask seen and return its sorted elements.
"""
import time
import random
import math

N = 256
DEADLINE = 55.0  # safety margin under 90s kill

# Precomputation: for each index x (0..N-1), store partner tuples.
ap3_partners = [[] for _ in range(N)]
ap4_partners = [[] for _ in range(N)]


def _precompute():
    # 3-APs: a, a+d, a+2d (1-indexed), stored as 0-indexed
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
    """True if mask ∪ {x} contains a nonconstant 4-AP involving x."""
    for y, z, w in ap4_partners[x]:
        if (mask >> y) & 1 and (mask >> z) & 1 and (mask >> w) & 1:
            return True
    return False


def _delta_add(mask, x):
    """Number of new 3-APs created by adding x to mask."""
    cnt = 0
    for y, z in ap3_partners[x]:
        if (mask >> y) & 1 and (mask >> z) & 1:
            cnt += 1
    return cnt


def _delta_remove(mask, x):
    """Number of 3-APs destroyed by removing x from mask (x must be in mask)."""
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


def _greedy_seed():
    mask = 0
    for x in range(N):
        if not _has_4ap(mask, x):
            mask |= (1 << x)
    return mask


def _present_list(mask):
    return [i for i in range(N) if (mask >> i) & 1]


def _absent_list(mask):
    return [i for i in range(N) if not (mask >> i) & 1]


def _local_search(mask, score, start_time):
    """SA-driven local search with single flips and swaps."""
    T0 = 5.0
    Tend = 0.05
    # Cool over the remaining budget
    total_budget = DEADLINE - (start_time - start_time)  # placeholder
    # We'll cool based on elapsed within this call
    call_start = time.time()
    call_budget = max(1.0, DEADLINE - (time.time() - start_time))
    alpha = math.log(T0 / Tend) / call_budget if call_budget > 0 else 0.0

    while time.time() - start_time < DEADLINE:
        elapsed_in_call = time.time() - call_start
        T = T0 * math.exp(-alpha * elapsed_in_call) if alpha > 0 else Tend
        if T < Tend:
            T = Tend

        present = _present_list(mask)
        absent = _absent_list(mask)
        if not present:
            break

        improved_this_round = False
        # Try a batch of random moves
        batch = 200
        for _ in range(batch):
            if time.time() - start_time > DEADLINE:
                return mask, score

            r = random.random()
            if r < 0.5 and absent:
                # Single flip: try adding a random absent element
                x = random.choice(absent)
                if _has_4ap(mask, x):
                    continue
                delta = _delta_add(mask, x)
                if delta >= 0:
                    new_mask = mask | (1 << x)
                    new_score = score + delta
                    if delta > 0 or random.random() < math.exp(delta / T):
                        mask, score = new_mask, new_score
                        improved_this_round = True
            elif r < 0.8 and present and absent:
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
                    improved_this_round = True
            else:
                # Single removal (to allow re-addition later)
                if len(present) > 1:
                    rem = random.choice(present)
                    loss = _delta_remove(mask, rem)
                    if loss < 0 or random.random() < math.exp(-loss / T):
                        mask ^= (1 << rem)
                        score -= loss
                        improved_this_round = True

        # If no improvement and T is low, try a strict improvement pass
        if not improved_this_round and T < 0.5:
            # Try all absent for strict add
            found = False
            for x in absent:
                if _has_4ap(mask, x):
                    continue
                delta = _delta_add(mask, x)
                if delta > 0:
                    mask |= (1 << x)
                    score += delta
                    found = True
                    break
            if found:
                continue
            # Try all swaps for strict improvement (sampled)
            present2 = _present_list(mask)
            absent2 = _absent_list(mask)
            if present2 and absent2:
                random.shuffle(present2)
                random.shuffle(absent2)
                found = False
                for rem in present2[:50]:
                    mask_wo = mask ^ (1 << rem)
                    loss = _delta_remove(mask, rem)
                    for add in absent2[:50]:
                        if _has_4ap(mask_wo, add):
                            continue
                        gain = _delta_add(mask_wo, add)
                        if gain - loss > 0:
                            mask = mask_wo | (1 << add)
                            score += gain - loss
                            found = True
                            break
                    if found:
                        break
            if not found:
                break  # local optimum

    return mask, score


def construct():
    start = time.time()
    random.seed(12345)

    mask = _greedy_seed()
    score = _count_3aps(mask)
    best_mask = mask
    best_score = score

    # Phase 1: strict local search to quickly reach a good local optimum
    mask, score = _local_search(mask, score, start)
    if score > best_score:
        best_score = score
        best_mask = mask

    # Phase 2: SA with restarts
    while time.time() - start < DEADLINE:
        # Perturb: remove a few random elements
        elems = _present_list(mask)
        if not elems:
            break
        k = min(3, max(1, len(elems) // 10))
        to_remove = random.sample(elems, k)
        for rem in to_remove:
            score -= _delta_remove(mask, rem)
            mask ^= (1 << rem)

        if not _present_list(mask):
            mask = _greedy_seed()
            score = _count_3aps(mask)

        mask, score = _local_search(mask, score, start)
        if score > best_score:
            best_score = score
            best_mask = mask

        # Occasionally restart from fresh greedy
        if random.random() < 0.2:
            fresh = _greedy_seed()
            fresh_score = _count_3aps(fresh)
            if fresh_score > score:
                mask = fresh
                score = fresh_score
                if score > best_score:
                    best_score = score
                    best_mask = mask

    # Convert best_mask to sorted list of elements (1-indexed)
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
    return sorted(result)
