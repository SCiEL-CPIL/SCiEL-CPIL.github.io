"""Covering 1..L by prime residue classes with reciprocal budget 1."""
import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
import sys

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

def _covered_prefix_len(classes, cap=100000):
    if not classes:
        return 0
    covered = bytearray(cap + 1)
    for p, r in classes:
        for x in range(r if r >= 1 else p, cap + 1, p):
            covered[x] = 1
    L = 0
    for i in range(1, cap + 1):
        if covered[i]:
            L = i
        else:
            break
    return L

def _recip_sum(classes):
    s = Fraction(0, 1)
    for p, _ in classes:
        s += Fraction(1, p)
    return s

def construct():
    t0 = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill
    MAXL = 100000
    primes = _primes_upto(1000)
    pset = set(primes)
    pidx = {p: i for i, p in enumerate(primes)}
    recip = {p: Fraction(1, p) for p in primes}

    # ---- Initial construction: greedy covering to build a large base L ----
    # We want a strong starting point. Use a randomized greedy that adds
    # (p,r) maximizing marginal prefix extension, subject to budget.
    def greedy_init(seed):
        random.seed(seed)
        covered = bytearray(MAXL + 1)
        L = 0
        budget = Fraction(1, 1)
        used = {}  # p -> r
        # Precompute nothing; do incremental marginal evaluation.
        # Candidate selection: for each prime, best residue that extends L.
        # To keep it fast, iterate primes in random order and pick residue.
        order = primes[:]
        random.shuffle(order)
        # We'll repeatedly add the best marginal (p,r).
        for _ in range(len(primes)):
            best_gain = 0
            best_p = None
            best_r = None
            # evaluate each unused prime's best residue
            for p in order:
                if p in used:
                    continue
                if budget < recip[p]:
                    continue
                # find residue that maximizes new covered prefix
                # covered prefix currently L. We need to cover L+1.. to extend.
                # Try residues; the extension depends on which numbers become covered.
                # Cheap heuristic: choose r to cover L+1 if possible (r = (L+1) % p)
                r = (L + 1) % p
                # simulate adding this class on a copy? Too slow. Instead approximate:
                # count how many of the next K numbers get covered.
                # We'll do a bounded scan.
                K = 2000
                gain = 0
                for x in range(L + 1, min(L + K, MAXL) + 1):
                    if covered[x]:
                        continue
                    if (x - r) % p == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
                    best_r = r
            if best_p is None or best_gain == 0:
                break
            # commit
            r = best_r
            p = best_p
            for x in range(r if r >= 1 else p, MAXL + 1, p):
                covered[x] = 1
            used[p] = r
            budget -= recip[p]
            # recompute L
            newL = 0
            for i in range(1, MAXL + 1):
                if covered[i]:
                    newL = i
                else:
                    break
            L = newL
        return used, covered, L, budget

    # Try several seeds, keep best
    best_used = None
    best_covered = None
    best_L = 0
    best_budget = Fraction(1, 1)
    for seed in range(12):
        if time.time() > DEADLINE:
            break
        used, covered, L, budget = greedy_init(seed)
        if L > best_L:
            best_L = L
            best_used = used
            best_covered = covered
            best_budget = budget

    if best_used is None:
        # fallback to seed-like program
        primes13 = [p for p in primes if p >= 13]
        return [[p, (i + 1) % p] for i, p in enumerate(primes13)]

    # ---- Simulated annealing with incremental coverage ----
    # State: mapping p->r for used primes; plus we may add/remove primes.
    # Maintain covered array and L incrementally is hard on remove.
    # Strategy: represent state as dict; recompute covered from scratch when
    # needed (remove), but for add/residue-change do incremental.
    # We'll keep a full covered array and recompute L via scan.
    state = dict(best_used)
    covered = bytearray(best_covered)
    L = best_L
    budget = best_budget
    best_state = dict(state)
    best_L_sa = L

    def recompute_full():
        nonlocal covered, L
        covered = bytearray(MAXL + 1)
        for p, r in state.items():
            for x in range(r if r >= 1 else p, MAXL + 1, p):
                covered[x] = 1
        L = 0
        for i in range(1, MAXL + 1):
            if covered[i]:
                L = i
            else:
                break

    def add_prime(p, r):
        nonlocal covered, L, budget
        if p in state:
            return False
        if budget < recip[p]:
            return False
        for x in range(r if r >= 1 else p, MAXL + 1, p):
            covered[x] = 1
        state[p] = r
        budget -= recip[p]
        # update L
        newL = L
        for i in range(L + 1, MAXL + 1):
            if covered[i]:
                newL = i
            else:
                break
        L = newL
        return True

    def remove_prime(p):
        nonlocal covered, L, budget
        if p not in state:
            return False
        r = state[p]
        # recompute covered from scratch (remove is expensive)
        del state[p]
        budget += recip[p]
        recompute_full()
        return True

    def change_residue(p, new_r):
        nonlocal covered, L
        if p not in state:
            return False
        old_r = state[p]
        if old_r == new_r:
            return False
        # recompute full (simpler and safe)
        state[p] = new_r
        recompute_full()
        return True

    def score():
        return L

    # Annealing schedule
    T = 1.0
    T_min = 1e-4
    alpha = 0.9995
    rng = random.Random(12345)

    # Precompute gap positions for move generation (numbers not covered)
    def gap_positions():
        gaps = []
        for i in range(1, MAXL + 1):
            if not covered[i]:
                gaps.append(i)
        return gaps

    iter_count = 0
    while time.time() < DEADLINE:
        T = max(T_min, T * alpha)
        # Generate a move
        move_type = rng.random()
        if move_type < 0.5:
            # change residue of a used prime, focused on covering a gap
            gaps = gap_positions()
            if gaps:
                g = rng.choice(gaps[:5000])
                # pick a used prime that could cover g
                candidates = [p for p in state if (g % p) != state[p]]
                if candidates:
                    p = rng.choice(candidates)
                    new_r = g % p
                    old_L = L
                    change_residue(p, new_r)
                    delta = L - old_L
                    if delta < 0 and rng.random() > (0.1 ** (delta / 50 + 1)):
                        # revert
                        change_residue(p, state[p])  # state already changed; revert to old
                        # we need old_r; recompute not tracked. Instead recompute full.
                        recompute_full()
                else:
                    # random residue change
                    p = rng.choice(list(state.keys()))
                    old_r = state[p]
                    new_r = rng.randrange(p)
                    old_L = L
                    change_residue(p, new_r)
                    if L < old_L and rng.random() > (0.1 ** ((old_L - L) / 50 + 1)):
                        change_residue(p, old_r)
                        recompute_full()
            else:
                # no gaps; try random changes
                p = rng.choice(list(state.keys()))
                old_r = state[p]
                new_r = rng.randrange(p)
                change_residue(p, new_r)
                if L < old_r and rng.random() > 0.5:
                    change_residue(p, old_r)
                    recompute_full()
        elif move_type < 0.75:
            # add a new prime to cover a gap
            gaps = gap_positions()
            if gaps:
                g = rng.choice(gaps[:5000])
                unused = [p for p in primes if p not in state]
                # only primes affordable
                unused = [p for p in unused if budget >= recip[p]]
                if unused:
                    p = rng.choice(unused)
                    r = g % p
                    old_L = L
                    if add_prime(p, r):
                        if L < old_L and rng.random() > 0.5:
                            remove_prime(p)
                    else:
                        pass
            else:
                # try to add a prime that extends beyond L
                unused = [p for p in primes if p not in state and budget >= recip[p]]
                if unused:
                    p = rng.choice(unused)
                    r = rng.randrange(p)
                    add_prime(p, r)
        else:
            # remove a prime (to free budget) - low probability
            if len(state) > 5:
                p = rng.choice(list(state.keys()))
                old_L = L
                remove_prime(p)
                if L < old_L and rng.random() > 0.5:
                    # re-add with old residue? we lost it. Just recompute.
                    # To keep simple, skip revert; recompute full already done.
                    pass

        iter_count += 1
        if L > best_L_sa:
            best_L_sa = L
            best_state = dict(state)
            # occasionally reset state to best to avoid drift
            if rng.random() < 0.01:
                state.clear()
                state.update(best_state)
                recompute_full()

    # Return best found
    if best_L_sa > best_L:
        result = [[p, r] for p, r in best_state.items()]
        return result
    else:
        result = [[p, r] for p, r in best_used.items()]
        return result
# EVOLVE-BLOCK-END
