"""Reciprocal weight of 3-progression-free sets; bitmask local search."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

_N = 256
_DEADLINE = 78.0          # safety margin before 90s kill
_RNG_SEED = 123456789

def _recip_sum(sel):
    s = Fraction(0, 1)
    for x in sel:
        s += Fraction(1, x)
    return s

def _has_3ap(sel):
    s = set(sel)
    for a in sel:
        for b in sel:
            if a != b and (2 * b - a) in s:
                return True
    return False

def _build_masks():
    masks = [0] * (_N + 1)
    for x in range(1, _N + 1):
        m = 0
        for d in range(1, _N + 1):
            a = x - d
            if a >= 1:
                m |= 1 << (a - 1)
            c = x + d
            if c <= _N:
                m |= 1 << (c - 1)
        masks[x] = m
    return masks

def _greedy(masks, rng):
    cur = 0
    for x in range(1, _N + 1):
        if (masks[x] & cur) == 0:
            cur |= 1 << (x - 1)
    return cur

def _local_search(masks, cur, rng, deadline):
    n = _N
    best = cur
    best_score = _recip_sum([i + 1 for i in range(n) if (best >> i) & 1])
    cur_score = best_score
    last_improve = time.time()

    def try_state(cand, sc):
        nonlocal best, best_score, cur, cur_score, last_improve
        if sc > cur_score:
            cur = cand
            cur_score = sc
            if sc > best_score:
                best = cand
                best_score = sc
                last_improve = time.time()

    # Phase 1: single adds and swaps
    while time.time() < deadline:
        improved = False
        order = list(range(1, n + 1))
        rng.shuffle(order)
        for x in order:
            if time.time() > deadline:
                break
            if (cur >> (x - 1)) & 1:
                continue
            if (masks[x] & cur) == 0:
                cand = cur | (1 << (x - 1))
                try_state(cand, cur_score + Fraction(1, x))
                improved = True
        # swaps: remove y, add x
        for x in order:
            if time.time() > deadline:
                break
            if (cur >> (x - 1)) & 1:
                continue
            # candidates y whose removal frees x
            for y in range(1, n + 1):
                if not ((cur >> (y - 1)) & 1):
                    continue
                if (masks[x] & (cur & ~(1 << (y - 1)))) == 0:
                    cand = (cur & ~(1 << (y - 1))) | (1 << (x - 1))
                    sc = cur_score - Fraction(1, y) + Fraction(1, x)
                    try_state(cand, sc)
                    improved = True
        if not improved:
            # 2-move moves: add two
            for x in order:
                if time.time() > deadline:
                    break
                if (cur >> (x - 1)) & 1:
                    continue
                if (masks[x] & cur) != 0:
                    continue
                for z in order:
                    if z <= x:
                        continue
                    if (cur >> (z - 1)) & 1:
                        continue
                    if (masks[z] & (cur | (1 << (x - 1)))) != 0:
                        continue
                    cand = cur | (1 << (x - 1)) | (1 << (z - 1))
                    sc = cur_score + Fraction(1, x) + Fraction(1, z)
                    try_state(cand, sc)
                    improved = True
            if not improved:
                break

    return best, best_score

def construct():
    start = time.time()
    rng = random.Random(_RNG_SEED)
    masks = _build_masks()

    cur = _greedy(masks, rng)
    best = cur
    best_score = _recip_sum([i + 1 for i in range(_N) if (best >> i) & 1])

    # restarts with perturbation
    while time.time() < _DEADLINE:
        # perturb: random flip of a few bits
        pert = best
        for _ in range(rng.randint(1, 4)):
            y = rng.randint(1, _N)
            pert ^= 1 << (y - 1)
        # repair: drop any element that creates a 3AP
        fixed = 0
        for x in range(1, _N + 1):
            if (pert >> (x - 1)) & 1 and (masks[x] & (pert & ~(1 << (x - 1)))) != 0:
                pert &= ~(1 << (x - 1))
                fixed += 1
        if fixed == 0 and pert == best:
            continue
        cur = pert
        sc = _recip_sum([i + 1 for i in range(_N) if (cur >> i) & 1])
        cur2, sc2 = _local_search(masks, cur, rng, _DEADLINE)
        if sc2 > best_score:
            best = cur2
            best_score = sc2
        if time.time() > _DEADLINE:
            break

    # final validity check (defensive)
    sel = [i + 1 for i in range(_N) if (best >> i) & 1]
    if not sel or _has_3ap(sel):
        sel = [i + 1 for i in range(_N) if (cur >> i) & 1]
        if not sel or _has_3ap(sel):
            # fallback greedy
            g = _greedy(masks, rng)
            sel = [i + 1 for i in range(_N) if (g >> i) & 1]
    return sel
# EVOLVE-BLOCK-END
