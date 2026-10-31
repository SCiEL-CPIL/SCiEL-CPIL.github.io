"""Rewritten program: Legendre-seed + greedy sign-flip local search for C3'."""
import time
import numpy as np

N = 400
_DEADLINE = 150.0 - 10.0  # safety margin before 150 s wall-clock kill


def _autoconv_max_abs(x):
    """Maximum absolute value of the full linear autoconvolution of x."""
    # full autoconvolution length 2N-1
    ac = np.correlate(x, x, mode='full')
    return float(np.max(np.abs(ac)))


def _integral_sq(x):
    """Squared integral of the step function (sum of squares of heights)."""
    return float(np.dot(x, x))


def _score(x):
    """C3' = max |autoconv| / (integral)^2.  Lower is better."""
    den = _integral_sq(x)
    if den <= 0.0:
        return float('inf')
    return _autoconv_max_abs(x) / (den * den)


def _legendre_sequence(p, n):
    """
    Deterministic balanced binary pattern from the Legendre symbol of a
    Mersenne prime p = 2^q - 1 (q prime).  Map +1/-1 to +1/-1 step heights.
    We tile the p-periodic Legendre sequence to length n, then balance the
    sign so the sum is as close to zero as possible (reduces the DC term that
    inflates the autoconvolution peak at lag 0).
    """
    # smallest Mersenne prime with prime exponent: p=7 (q=3)
    if p is None:
        p = 7
    # Legendre symbol via quadratic-residue test
    base = np.zeros(p, dtype=np.int8)
    for a in range(1, p):
        base[a] = 1 if pow(a, (p - 1) // 2, p) == 1 else -1
    base[0] = 0
    # tile to length n
    reps = (n + p - 1) // p
    seq = np.tile(base, reps)[:n].astype(np.float64)
    # balance: flip the minority sign so sum is minimized in magnitude
    s = seq.sum()
    if s > 0:
        # too many +1s; flip some +1s to -1s (the zeros become -1)
        seq[seq == 0] = -1
    elif s < 0:
        seq[seq == 0] = 1
    else:
        # already balanced; make zeros +1 for determinism
        seq[seq == 0] = 1
    return seq


def _greedy_flip(x, deadline):
    """
    Greedy sign-flip local search.  Precompute the full autoconvolution,
    then for each candidate flip compute the delta in max|ac| efficiently
    by updating only the affected lags.  Accept flips that reduce the score.
    Stops when no improving flip is found or the deadline is reached.
    """
    n = len(x)
    x = x.copy().astype(np.float64)
    ac = np.correlate(x, x, mode='full')  # length 2n-1
    best_score = _score(x)
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                break
            old = x[i]
            new = -old
            # Flipping x[i] changes ac at lags where i and j=i+lag (or j-i=lag)
            # For full correlate, ac[k] = sum_j x[j]*x[j+k-n+1] (offset)
            # Simpler: recompute affected lags by direct formula.
            # ac_full[t] for t in [0, 2n-2] equals sum over j of x[j]*x[j-(t-(n-1))]
            # i.e., ac_full[t] = sum_j x[j] * x[j - d] where d = t - (n-1)
            # Flipping x[i] changes terms where j=i or j-d=i (i.e., j=i+d).
            # So affected t values: those where d = t-(n-1) satisfies i in [0,n-1]
            # and (j=i or j=i+d) is in [0,n-1].
            # Easiest robust approach: recompute ac from scratch each flip?  Too slow.
            # Instead: ac[t] = sum_j x[j]*x[j-d].  Flip x[i]:
            #   term with j=i: x[i]*x[i-d]  ->  new: (-x[i])*x[i-d] = -x[i]*x[i-d]
            #   term with j=i+d: x[i+d]*x[i] -> new: x[i+d]*(-x[i]) = -x[i+d]*x[i]
            # So delta at lag d (t = d + n - 1):
            #   if 0<=i-d<n: ac[t] -= 2*x[i]*x[i-d]
            #   if 0<=i+d<n: ac[t] -= 2*x[i]*x[i+d]   (same t if d != 0)
            # For d=0 (t=n-1): both terms are the same x[i]*x[i], so ac[t] -= 4*x[i]^2? 
            #   Actually when d=0, j=i and j=i+d=i are the same term, counted once.
            #   ac[n-1] = sum_j x[j]^2.  Flip x[i]: new = old - 2*x[i]^2.
            # Let's handle generally:
            d_list = []
            # d such that i-d in [0,n-1] => d in [i-n+1, i]
            # d such that i+d in [0,n-1] => d in [-i, n-1-i]
            # Union of d values:
            d_min = max(-(n - 1), -(n - 1))  # = -(n-1)
            d_max = min(n - 1, n - 1)
            # Actually d ranges over all integers where either index is valid.
            # d from -(n-1) to (n-1).
            # For each d, t = d + (n-1).
            # We'll just iterate over the at most 2n-1 lags?  That's O(n) per flip, O(n^2) per pass.
            # n=400, n^2=160k, times a few passes = fine.
            for d in range(-(n - 1), n):
                t = d + (n - 1)
                delta = 0.0
                # contribution from j=i: x[i]*x[i-d]
                if 0 <= i - d < n:
                    delta += 2.0 * x[i] * x[i - d]
                # contribution from j=i+d: x[i+d]*x[i]
                if 0 <= i + d < n and d != 0:
                    delta += 2.0 * x[i] * x[i + d]
                # if d==0, the two terms are the same, already counted once above? 
                # No: when d==0, i-d=i and i+d=i, both conditions true.
                # But they represent the SAME product x[i]*x[i], so we should count it once.
                # My formula adds 2*x[i]*x[i] from first, and 2*x[i]*x[i] from second (since d!=0 false).
                # Wait, d!=0 is false, so second is skipped. Good.
                # So for d=0: delta = 2*x[i]^2.  But flipping x[i] changes x[i]^2 to (-x[i])^2 = x[i]^2.
                # So ac[n-1] should NOT change!  But my delta says it changes by 2*x[i]^2.
                # Error: ac[n-1] = sum_j x[j]^2.  Flipping x[i] leaves x[i]^2 unchanged.
                # So delta at d=0 should be 0.
                # Let me reconsider: ac[t] = sum_j x[j] * x[j-d].
                # When we flip x[i], the terms involving x[i] are:
                #   j=i: x[i]*x[i-d]
                #   j such that j-d=i => j=i+d: x[i+d]*x[i]
                # The NEW ac[t] = OLD ac[t] - x[i]*x[i-d] - x[i+d]*x[i] + (-x[i])*x[i-d] + x[i+d]*(-x[i])
                #   = OLD ac[t] - 2*x[i]*x[i-d] - 2*x[i]*x[i+d]
                # (with the understanding that if i-d is out of range, that term is 0; same for i+d)
                # For d=0: -2*x[i]*x[i] - 2*x[i]*x[i] = -4*x[i]^2.  But x[i]^2 doesn't change!
                # The issue: when d=0, j=i and j=i+d=i are the SAME j, so the term x[i]*x[i] appears
                # only ONCE in the sum, not twice.  So the correct delta for d=0 is -2*x[i]^2? 
                # No: sum_j x[j]^2.  Term j=i is x[i]^2.  After flip, it's still x[i]^2.  Delta=0.
                # The formula -2*x[i]*x[i-d] - 2*x[i]*x[i+d] double-counts when d=0.
                # Fix: if d==0, delta = 0.
                if d == 0:
                    delta = 0.0
                if delta != 0.0:
                    new_ac_t = ac[t] - delta
                    # Check if this reduces max|ac|
                    old_max = np.max(np.abs(ac))
                    # We need to know if new_ac_t could be the new max.
                    # Simple check: if |new_ac_t| < old_max or it's not the argmax, likely fine.
                    # For safety, just accept if |new_ac_t| <= old_max.
                    if abs(new_ac_t) <= old_max + 1e-12:
                        # Apply flip
                        x[i] = new
                        ac[t] = new_ac_t
                        improved = True
                        break  # restart scan from i=0
            # end d loop
            if improved:
                break
        # end i loop
    return x


def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE

    # Initial pattern: Legendre symbol of Mersenne prime p=7, tiled to length N
    x = _legendre_sequence(7, N)

    # Quick local search
    x = _greedy_flip(x, deadline)

    # If time remains, try a few random restarts with small perturbations
    best_x = x.copy()
    best_s = _score(x)
    if time.time() < deadline - 2.0:
        rng = np.random.default_rng(42)
        for restart in range(20):
            if time.time() >= deadline - 2.0:
                break
            # perturb: flip a random 10% of signs
            xp = best_x.copy()
            n_flip = max(1, N // 10)
            idx = rng.choice(N, size=n_flip, replace=False)
            xp[idx] *= -1
            xp = _greedy_flip(xp, deadline)
            s = _score(xp)
            if s < best_s:
                best_s = s
                best_x = xp.copy()

    return best_x
