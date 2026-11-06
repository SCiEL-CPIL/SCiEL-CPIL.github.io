"""Optimized: greedy with marginal-gain insertion + local 2-opt improvement + restarts."""
import math
import time
import random

# EVOLVE-BLOCK-START

_GCD_CACHE = {}
def _gcd(a, b):
    key = (a, b) if a <= b else (b, a)
    v = _GCD_CACHE.get(key)
    if v is None:
        v = math.gcd(a, b)
        _GCD_CACHE[key] = v
    return v

def _is_compatible(m, r, m2, r2):
    g = _gcd(m, m2)
    return (r - r2) % g != 0

def _conflict_list(m, r, chosen):
    """Return list of indices in chosen that conflict with (m, r)."""
    out = []
    for i, (m2, r2) in enumerate(chosen):
        if (r - r2) % _gcd(m, m2) == 0:
            out.append(i)
    return out

def _score(chosen):
    return sum(1.0 / m for m, r in chosen)

def _greedy_construct(order, deadline):
    """Greedy with marginal-gain insertion. `order` is list of moduli to try."""
    chosen = []  # list of [m, r]
    cur_score = 0.0
    for m in order:
        if time.time() > deadline:
            break
        # find residue r minimizing (sum of 1/m_c over conflicts) - 1/m, i.e. minimizing conflict loss
        best_r = None
        best_net = -float('inf')  # net gain = 1/m - conflict_loss
        inv_m = 1.0 / m
        for r in range(m):
            loss = 0.0
            for (m2, r2) in chosen:
                if (r - r2) % _gcd(m, m2) == 0:
                    loss += 1.0 / m2
            net = inv_m - loss
            if net > best_net:
                best_net = net
                best_r = r
        if best_net > 0:
            # insert, removing conflicts
            new_chosen = []
            for (m2, r2) in chosen:
                if (best_r - r2) % _gcd(m, m2) == 0:
                    cur_score -= 1.0 / m2
                else:
                    new_chosen.append([m2, r2])
            new_chosen.append([m, best_r])
            cur_score += inv_m
            chosen = new_chosen
    return chosen

def _local_improve(chosen, deadline):
    """2-opt: try replacing each chosen class with a better (m', r') not in chosen."""
    chosen = [list(c) for c in chosen]
    cur_score = _score(chosen)
    improved = True
    # set of chosen moduli
    chosen_moduli = set(m for m, r in chosen)
    while improved and time.time() < deadline:
        improved = False
        # try to add a new modulus (not chosen) that improves net
        for m in range(20, 201):
            if time.time() > deadline:
                break
            if m in chosen_moduli:
                continue
            inv_m = 1.0 / m
            best_r = None
            best_net = -float('inf')
            for r in range(m):
                loss = 0.0
                for (m2, r2) in chosen:
                    if (r - r2) % _gcd(m, m2) == 0:
                        loss += 1.0 / m2
                net = inv_m - loss
                if net > best_net:
                    best_net = net
                    best_r = r
            if best_net > 1e-12:
                new_chosen = []
                for (m2, r2) in chosen:
                    if (best_r - r2) % _gcd(m, m2) == 0:
                        cur_score -= 1.0 / m2
                    else:
                        new_chosen.append([m2, r2])
                new_chosen.append([m, best_r])
                cur_score += inv_m
                chosen = new_chosen
                chosen_moduli = set(m2 for m2, r2 in chosen)
                improved = True
    return chosen, cur_score

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin under 150 s wall-clock
    rng = random.Random(12345)
    best = None
    best_score = -1.0

    # Restart 1: ascending moduli (smallest first)
    c = _greedy_construct(list(range(20, 201)), deadline)
    c, s = _local_improve(c, deadline)
    if s > best_score:
        best_score = s
        best = c

    # Restart 2: descending moduli
    if time.time() < deadline:
        c = _greedy_construct(list(range(200, 19, -1)), deadline)
        c, s = _local_improve(c, deadline)
        if s > best_score:
            best_score = s
            best = c

    # Restart 3: random order
    if time.time() < deadline:
        order = list(range(20, 201))
        rng.shuffle(order)
        c = _greedy_construct(order, deadline)
        c, s = _local_improve(c, deadline)
        if s > best_score:
            best_score = s
            best = c

    # More randomized restarts if time remains
    while time.time() < deadline - 2.0:
        order = list(range(20, 201))
        rng.shuffle(order)
        c = _greedy_construct(order, deadline)
        c, s = _local_improve(c, deadline)
        if s > best_score:
            best_score = s
            best = c

    return best
# EVOLVE-BLOCK-END
