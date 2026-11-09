"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _get_primes(limit):
    """Return list of primes up to limit."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]

def _compute_coverage(assignment, L_max=20000):
    """Compute largest L s.t. 1..L all covered by assignment {p: r}."""
    if not assignment:
        return 0
    covered = bytearray(L_max + 1)
    for p, r in assignment.items():
        start = r if r > 0 else p
        for n in range(start, L_max + 1, p):
            covered[n] = 1
    L = 0
    for i in range(1, L_max + 1):
        if covered[i]:
            L = i
        else:
            break
    return L

def _sum_recip(assignment):
    """Exact sum of 1/p over keys of assignment."""
    s = Fraction(0)
    for p in assignment:
        s += Fraction(1, p)
    return s

def construct():
    t0 = time.time()
    deadline = t0 + 110

    primes = _get_primes(1000)

    # Initial: seed-style assignment
    init_primes = [p for p in primes if p >= 13]
    best_assignment = {}
    for i, p in enumerate(init_primes):
        best_assignment[p] = (i + 1) % p

    # Enforce budget
    budget = _sum_recip(best_assignment)
    if budget > Fraction(1):
        for p in sorted(best_assignment.keys(), reverse=True):
            if budget <= Fraction(1):
                break
            del best_assignment[p]
            budget -= Fraction(1, p)

    best_L = _compute_coverage(best_assignment)

    import random
    rng = random.Random(42)

    while time.time() < deadline:
        # Pick a move
        move = rng.choice(['residue', 'swap', 'add', 'remove', 'optimize'])

        if move == 'residue' and best_assignment:
            p = rng.choice(list(best_assignment.keys()))
            new_r = rng.randint(0, p - 1)
            if new_r == best_assignment[p]:
                continue
            trial = dict(best_assignment)
            trial[p] = new_r
            L = _compute_coverage(trial)
            if L > best_L:
                best_assignment = trial
                best_L = L

        elif move == 'swap' and best_assignment:
            p_out = rng.choice(list(best_assignment.keys()))
            rem = Fraction(1) - _sum_recip(best_assignment)
            cands = [p for p in primes if p not in best_assignment and Fraction(1, p) <= rem]
            if cands:
                p_in = rng.choice(cands)
                r_in = rng.randint(0, p_in - 1)
                trial = dict(best_assignment)
                del trial[p_out]
                trial[p_in] = r_in
                L = _compute_coverage(trial)
                if L > best_L:
                    best_assignment = trial
                    best_L = L

        elif move == 'add':
            cur = _sum_recip(best_assignment)
            rem = Fraction(1) - cur
            cands = [p for p in primes if p not in best_assignment and Fraction(1, p) <= rem]
            if cands:
                p_in = rng.choice(cands)
                r_in = rng.randint(0, p_in - 1)
                trial = dict(best_assignment)
                trial[p_in] = r_in
                L = _compute_coverage(trial)
                if L > best_L:
                    best_assignment = trial
                    best_L = L

        elif move == 'remove' and len(best_assignment) > 2:
            p_out = rng.choice(list(best_assignment.keys()))
            trial = dict(best_assignment)
            del trial[p_out]
            L = _compute_coverage(trial)
            if L > best_L:
                best_assignment = trial
                best_L = L

        elif move == 'optimize' and best_assignment:
            # Hill-climb residues
            changed = True
            while changed and time.time() < deadline:
                changed = False
                for p in list(best_assignment.keys()):
                    old_L = best_L
                    best_r = best_assignment[p]
                    for r in range(p):
                        if r == best_assignment[p]:
                            continue
                        trial = dict(best_assignment)
                        trial[p] = r
                        L = _compute_coverage(trial)
                        if L > old_L:
                            old_L = L
                            best_r = r
                    if best_r != best_assignment[p]:
                        best_assignment[p] = best_r
                        best_L = old_L
                        changed = True

    return [[p, r] for p, r in best_assignment.items()]
# EVOLVE-BLOCK-END
