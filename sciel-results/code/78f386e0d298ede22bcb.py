"""Rewritten solver: bitset DP validity + simulated annealing to maximize
cardinality of a subset of [1,100] with no subset summing to 200."""
import time
import random
import math

# EVOLVE-BLOCK-START
_U = list(range(1, 101))
_USET = frozenset(_U)
_BIT200 = 1 << 200


def _valid(kept_iter):
    """True iff no subset of kept_iter sums to 200. Bitset DP with early exit."""
    r = 1  # bit 0 set
    for x in kept_iter:
        r |= (r << x)
        if r & _BIT200:
            return False
    return True


def _greedy():
    """Fast greedy: descending, keep if it does not create sum 200."""
    kept = []
    r = 1
    for x in range(100, 0, -1):
        nr = r | (r << x)
        if not (nr & _BIT200):
            kept.append(x)
            r = nr
    return set(kept)


def _repair(rng):
    """Start from full set, remove random elements until valid."""
    kept = set(_U)
    while not _valid(kept):
        x = rng.choice(tuple(kept))
        kept.discard(x)
    return kept


def _add_backs(kept, R, rng=None):
    """Repeatedly add back removed elements in (random) order while staying valid."""
    improved = True
    while improved:
        improved = False
        order = list(R)
        if rng is not None:
            rng.shuffle(order)
        for x in order:
            if x in kept:
                continue
            kept.add(x)
            R.discard(x)
            if _valid(kept):
                improved = True
            else:
                kept.discard(x)
                R.add(x)
    return kept, R


def _sa(kept, R, best, rng, deadline):
    """Simulated annealing on the removed set. Mutates kept/R in place,
    updates best (a set) whenever a larger valid kept is found."""
    T = 1.0
    while time.time() < deadline:
        roll = rng.random()
        if roll < 0.5 and R:
            # ADD_BACK: remove a random element from R (cost -1 if valid)
            x = rng.choice(tuple(R))
            kept.add(x)
            R.discard(x)
            if _valid(kept):
                if len(kept) > len(best):
                    best.clear()
                    best.update(kept)
            else:
                kept.discard(x)
                R.add(x)
        elif roll < 0.9 and R and kept:
            # SWAP: exchange one removed with one kept (cost unchanged)
            x = rng.choice(tuple(R))     # removed
            y = rng.choice(tuple(kept))  # kept
            kept.add(x)
            R.discard(x)
            kept.discard(y)
            R.add(y)
            if _valid(kept):
                if len(kept) > len(best):
                    best.clear()
                    best.update(kept)
            else:
                # revert
                kept.discard(x)
                R.add(x)
                kept.add(y)
                R.discard(y)
        else:
            # ADD_TO_R: drop a kept element (cost +1), accept with prob exp(1/T)
            if kept:
                y = rng.choice(tuple(kept))
                kept.discard(y)
                R.add(y)
                if rng.random() < math.exp(1.0 / T):
                    # accept the worse state
                    pass
                else:
                    kept.add(y)
                    R.discard(y)
        # cool
        T = max(0.004, T * 0.99998)
    return best


def construct():
    t0 = time.time()
    deadline = t0 + 82.0  # safety margin before the 90s kill
    rng = random.Random(20240517)

    best = set()

    # --- Initialization 1: greedy ---
    kept = _greedy()
    R = _USET - kept
    kept, R = _add_backs(kept, R, rng)
    if len(kept) > len(best):
        best.update(kept)

    # --- Simulated annealing from greedy start ---
    if time.time() < deadline:
        _sa(kept, R, best, rng, deadline)
    # polish
    R = _USET - kept
    kept, R = _add_backs(kept, R, rng)
    if len(kept) > len(best):
        best.update(kept)

    # --- Initialization 2: repair-from-full, if time remains ---
    if time.time() < deadline - 2.0:
        kept2 = _repair(rng)
        R2 = _USET - kept2
        kept2, R2 = _add_backs(kept2, R2, rng)
        if len(kept2) > len(best):
            best.update(kept2)
        if time.time() < deadline:
            _sa(kept2, R2, best, rng, deadline)
        R2 = _USET - kept2
        kept2, R2 = _add_backs(kept2, R2, rng)
        if len(kept2) > len(best):
            best.update(kept2)

    # --- Initialization 3: another repair with a fresh rng, if time remains ---
    if time.time() < deadline - 2.0:
        rng3 = random.Random(987654321)
        kept3 = _repair(rng3)
        R3 = _USET - kept3
        kept3, R3 = _add_backs(kept3, R3, rng3)
        if len(kept3) > len(best):
            best.update(kept3)
        if time.time() < deadline:
            _sa(kept3, R3, best, rng3, deadline)
        R3 = _USET - kept3
        kept3, R3 = _add_backs(kept3, R3, rng3)
        if len(kept3) > len(best):
            best.update(kept3)

    # --- Final safety: ensure validity and nonemptiness ---
    if not best or not _valid(best):
        kept = _greedy()
        R = _USET - kept
        kept, R = _add_backs(kept, R, rng)
        best = set(kept)

    if not best:
        best = {100}

    # Final polish add-backs on the best set itself
    Rb = _USET - best
    best, _ = _add_backs(best, Rb, rng)

    return list(best)
# EVOLVE-BLOCK-END
