"""
4-AP-free set construction: combine delta-based pair-swap local search
with a multi-stage pipeline (Cantor, greedy variants, SA, pair-swap, fill).
Enhanced with weighted greedy, improved pair-swap efficiency, and additional
SA restarts.
"""
import time
import random
import math

N = 512

# ---------------------------------------------------------------------------
# Precompute 4-AP structure
# ---------------------------------------------------------------------------

def _build_4aps():
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    return aps

_ALL_APS = _build_4aps()
_NUM_APS = len(_ALL_APS)

# For each element x, list of (o1, o2, o3) such that x,o1,o2,o3 form a 4-AP
def _build_others():
    others = [[] for _ in range(N + 1)]
    for (a, b, c, e) in _ALL_APS:
        others[a].append((b, c, e))
        others[b].append((a, c, e))
        others[c].append((a, b, e))
        others[e].append((a, b, c))
    return others

_OTHERS = _build_others()

# For each element x, precompute the bitmask of the other 3 elements in each AP
# This allows faster delta computation using bitwise AND
def _build_other_masks():
    masks = [[] for _ in range(N + 1)]
    for (a, b, c, e) in _ALL_APS:
        masks[a].append((1 << b) | (1 << c) | (1 << e))
        masks[b].append((1 << a) | (1 << c) | (1 << e))
        masks[c].append((1 << a) | (1 << b) | (1 << e))
        masks[e].append((1 << a) | (1 << b) | (1 << c))
    return masks

_OTHER_MASKS = _build_other_masks()

# ---------------------------------------------------------------------------
# Violation helpers (O(degree) delta)
# ---------------------------------------------------------------------------

def _count_violations(bits):
    v = 0
    for (a, b, c, e) in _ALL_APS:
        if ((bits >> a) & 1) and ((bits >> b) & 1) and ((bits >> c) & 1) and ((bits >> e) & 1):
            v += 1
    return v

def _delta_add(bits, x):
    c = 0
    for mask in _OTHER_MASKS[x]:
        if (bits & mask) == mask:
            c += 1
    return c

def _delta_remove(bits, x):
    c = 0
    for mask in _OTHER_MASKS[x]:
        if (bits & mask) == mask:
            c += 1
    return c

# ---------------------------------------------------------------------------
# Initial constructions
# ---------------------------------------------------------------------------

def _cantor_bits():
    bits = 0
    for x in range(1, N + 1):
        t = x
        ok = True
        while t > 0:
            if t % 3 == 2:
                ok = False
                break
            t //= 3
        if ok:
            bits |= (1 << x)
    return bits

def _greedy_order(bits, order):
    for x in order:
        if not ((bits >> x) & 1):
            if _delta_add(bits, x) == 0:
                bits |= (1 << x)
    return bits

def _greedy_forward():
    return _greedy_order(0, range(1, N + 1))

def _greedy_reverse():
    return _greedy_order(0, range(N, 0, -1))

def _greedy_random(rng):
    order = list(range(1, N + 1))
    rng.shuffle(order)
    return _greedy_order(0, order)

def _greedy_weighted(rng):
    """
    Weighted greedy: at each step, choose among safe elements the one with
    the fewest future conflicts (i.e., smallest degree in the remaining
    conflict graph). This tends to produce larger sets.
    """
    bits = 0
    remaining = list(range(1, N + 1))
    while remaining:
        # Find all safe elements
        safe = []
        for x in remaining:
            if _delta_add(bits, x) == 0:
                safe.append(x)
        if not safe:
            break
        # Among safe elements, pick the one with minimum conflict count
        # (conflict count = number of 4-APs where x and 2 others are in set)
        best_x = None
        best_conf = float('inf')
        for x in safe:
            conf = 0
            for (o1, o2, o3) in _OTHERS[x]:
                cnt = 0
                if (bits >> o1) & 1: cnt += 1
                if (bits >> o2) & 1: cnt += 1
                if (bits >> o3) & 1: cnt += 1
                if cnt >= 2:
                    conf += 1
            if conf < best_conf:
                best_conf = conf
                best_x = x
        bits |= (1 << best_x)
        remaining.remove(best_x)
    return bits

# ---------------------------------------------------------------------------
# Simulated Annealing (single add/remove moves)
# ---------------------------------------------------------------------------

def _simulated_annealing(start_bits, deadline, rng):
    cur = start_bits
    cur_size = bin(cur).count('1')
    cur_v = _count_violations(cur)

    best_bits = cur
    best_size = cur_size

    P = 5.0
    T = 2.0
    T_min = 0.05
    alpha = 0.995

    while time.time() < deadline:
        if T < T_min:
            break
        x = rng.randint(1, N)
        in_set = bool((cur >> x) & 1)

        if in_set:
            dv = -_delta_remove(cur, x)
            ds = -1
        else:
            dv = _delta_add(cur, x)
            ds = 1

        dc = -ds + P * dv
        if dc < 0 or rng.random() < math.exp(-dc / T):
            cur ^= (1 << x)
            cur_size += ds
            cur_v += dv
            if cur_v == 0 and cur_size > best_size:
                best_bits = cur
                best_size = cur_size
        T *= alpha
    return best_bits, best_size

# ---------------------------------------------------------------------------
# Pair-swap local search (remove one, add one)
# ---------------------------------------------------------------------------

def _repair_violations(bits, deadline):
    """Remove elements to eliminate all 4-AP violations, then fill back in."""
    cur = bits
    cur_v = _count_violations(cur)
    
    # Iteratively remove one element from each violation
    while cur_v > 0 and time.time() < deadline:
        # Find a violation
        found = False
        for (a, b, c, e) in _ALL_APS:
            if ((cur >> a) & 1) and ((cur >> b) & 1) and ((cur >> c) & 1) and ((cur >> e) & 1):
                # Remove the element with the most conflicts in this AP
                # Try removing each and see which reduces violations most
                best_rem = None
                best_new_v = cur_v + 1
                for x in (a, b, c, e):
                    bits_no_x = cur & ~(1 << x)
                    new_v = _count_violations(bits_no_x)
                    if new_v < best_new_v:
                        best_new_v = new_v
                        best_rem = x
                if best_rem is not None:
                    cur &= ~(1 << best_rem)
                    cur_v = best_new_v
                    found = True
                    break
        if not found:
            break
    
    # Now fill back in safe elements
    cur = _greedy_fill(cur, deadline)
    return cur

# ---------------------------------------------------------------------------
# Greedy fill (add any safe element)
# ---------------------------------------------------------------------------

def _greedy_fill(bits, deadline):
    """Aggressive greedy fill: keep adding safe elements until no more can be added."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if not ((bits >> x) & 1) and _delta_add(bits, x) == 0:
                bits |= (1 << x)
                changed = True
    return bits

def _local_improve(bits, deadline, rng):
    """Local improvement: try swaps and single additions/removals to improve size while maintaining 4-AP-free."""
    cur = bits
    cur_size = bin(cur).count('1')
    cur_v = _count_violations(cur)
    
    best_bits = cur
    best_size = cur_size
    
    in_list = [i for i in range(1, N + 1) if (cur >> i) & 1]
    out_list = [i for i in range(1, N + 1) if not ((cur >> i) & 1)]
    
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Try a swap: remove x from in_list, add y from out_list
        if in_list and out_list:
            x = rng.choice(in_list)
            y = rng.choice(out_list)
            if x != y:
                bits_no_x = cur & ~(1 << x)
                # Check if adding y to bits_no_x creates violations
                if _delta_add(bits_no_x, y) == 0:
                    new_cur = bits_no_x | (1 << y)
                    new_size = cur_size  # same size
                    # This is a valid swap if it doesn't create violations
                    # But we want to improve, so this is only useful if it enables future additions
                    # For now, accept neutral swaps with some probability
                    if rng.random() < 0.3:
                        cur = new_cur
                        in_list.remove(x)
                        out_list.remove(y)
                        in_list.append(y)
                        out_list.append(x)
                        no_improve += 1
                        continue
        
        # Try adding a safe element
        added = False
        for y in out_list:
            if _delta_add(cur, y) == 0:
                cur |= (1 << y)
                cur_size += 1
                out_list.remove(y)
                in_list.append(y)
                no_improve = 0
                if cur_size > best_size:
                    best_bits = cur
                    best_size = cur_size
                added = True
                break
        
        if not added:
            no_improve += 1
    
    # Final fill
    if time.time() < deadline:
        cur = _greedy_fill(cur, deadline)
        cur_size = bin(cur).count('1')
        if cur_size > best_size:
            best_bits = cur
            best_size = cur_size
    
    return best_bits, best_size

# ---------------------------------------------------------------------------
# Multi-restart SA
# ---------------------------------------------------------------------------

def _multi_sa(start_bits, deadline, rng, n_restarts=3):
    best_bits = start_bits
    best_size = bin(start_bits).count('1')
    
    if n_restarts <= 1:
        sb, ss = _simulated_annealing(start_bits, deadline, rng)
        if ss > best_size:
            best_bits = sb
            best_size = ss
        return best_bits, best_size
    
    time_per_restart = (deadline - time.time()) / n_restarts
    for i in range(n_restarts):
        rd = time.time() + time_per_restart
        if rd > deadline:
            rd = deadline
        if time.time() >= deadline - 2:
            break
        sb, ss = _simulated_annealing(best_bits, rd, rng)
        if ss > best_size:
            best_bits = sb
            best_size = ss
    return best_bits, best_size

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def construct():
    deadline = time.time() + 55  # safety margin under 60s
    rng = random.Random(7)

    # --- Stage 1: diverse initial seeds ---
    seeds = [_cantor_bits(), _greedy_forward(), _greedy_reverse()]
    for _ in range(6):
        seeds.append(_greedy_random(rng))
    # Add weighted greedy seeds
    for _ in range(2):
        seeds.append(_greedy_weighted(rng))

    best_bits = 0
    best_size = 0

    for seed in seeds:
        if time.time() > deadline - 15:
            break
        filled = _greedy_fill(seed, deadline - 15)
        sz = bin(filled).count('1')
        if sz > best_size:
            best_bits = filled
            best_size = sz

    # --- Stage 2: Repair + local improvement on best seed ---
    if time.time() < deadline - 25:
        # First, repair any violations
        repaired = _repair_violations(best_bits, deadline - 25)
        repaired_size = bin(repaired).count('1')
        if repaired_size > best_size:
            best_bits = repaired
            best_size = repaired_size
        
        # Then, local improvement
        li_bits, li_sz = _local_improve(best_bits, deadline - 25, rng)
        if li_sz > best_size:
            best_bits = li_bits
            best_size = li_sz

    # --- Stage 3: Additional SA restarts ---
    if time.time() < deadline - 15:
        sa_bits, sa_sz = _multi_sa(best_bits, deadline - 15, rng, n_restarts=2)
        if sa_sz > best_size:
            best_bits = sa_bits
            best_size = sa_sz

    # --- Stage 4: Final repair and fill ---
    if time.time() < deadline - 3:
        final_repaired = _repair_violations(best_bits, deadline - 3)
        final_size = bin(final_repaired).count('1')
        if final_size > best_size:
            best_bits = final_repaired
            best_size = final_size
        else:
            best_bits = _greedy_fill(best_bits, deadline - 3)

    # --- Verification & fallback ---
    if _count_violations(best_bits) != 0:
        # Try to repair
        repaired = _repair_violations(best_bits, time.time() + 1)
        if _count_violations(repaired) == 0:
            best_bits = repaired
        else:
            cb = _cantor_bits()
            best_bits = cb

    best_set = {i for i in range(1, N + 1) if (best_bits >> i) & 1}
    return sorted(best_set)
