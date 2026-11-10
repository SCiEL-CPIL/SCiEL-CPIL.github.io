"""Rewritten: bottleneck-driven local search with NumPy coverage evaluation."""
import time
import numpy as np

# --- primes up to 1000 (168 primes) ---
def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = [False] * len(sieve[i * i::i])
    return np.array([i for i, v in enumerate(sieve) if v], dtype=np.int64)

PRIMES = _primes_upto(1000)
NPRIMES = len(PRIMES)

# --- fast coverage evaluation over 1..L ---
# returns (score, first_fail) where score = largest L with all 1..L covered >=2
def evaluate(res):
    """res: array of 168 residues. Returns (L, first_fail)."""
    # work over 1..5000; count how many congruences hit each integer
    L = 5000
    x = np.arange(1, L + 1, dtype=np.int64)          # shape (L,)
    counts = np.zeros(L, dtype=np.int16)
    for r, p in zip(res, PRIMES):
        # integers in 1..L with x ≡ r (mod p)
        # first such integer >= 1:
        first = r if r >= 1 else r + p
        if first > L:
            continue
        idx = (first - 1) // p * p + (first - 1)  # index in x (x[0]=1)
        # actually x[i] = i+1; we want i+1 ≡ r mod p  => i ≡ r-1 mod p
        # i in [0, L-1] with i ≡ (r-1) mod p
        rem = (r - 1) % p
        if rem == 0:
            i0 = 0
        else:
            i0 = p - rem
        if i0 >= L:
            continue
        counts[i0::p] += 1
    # find first index where counts < 2
    bad = np.where(counts < 2)[0]
    if len(bad) == 0:
        return L, -1
    first_fail = bad[0] + 1  # integer value
    return first_fail - 1, first_fail

# --- bottleneck-driven local search ---
def construct():
    deadline = time.time() + 135  # safety margin under 150s
    rng = np.random.default_rng(12345)

    # initial: the seed idea (covers 1..84)
    res = np.array([((i + 2) // 2) % p for i, p in enumerate(PRIMES)], dtype=np.int64)
    best_res = res.copy()
    best_L, _ = evaluate(best_res)

    # warm start with a random perturbation pass
    def perturb(res_in, n_mut):
        r = res_in.copy()
        idx = rng.integers(0, NPRIMES, size=n_mut)
        for i in idx:
            r[i] = rng.integers(0, PRIMES[i])
        return r

    # --- main loop: bottleneck-driven ---
    # At each step, find first_fail. The primes that can cover integer x are
    # those p > x (since p <= x can only cover x if x mod p == r, but we can
    # only change r, and if p <= x then x mod p is fixed; changing r_p to
    # x mod p WOULD cover x). Actually any prime can cover x by setting
    # r_p = x % p. But small primes p <= x: setting r_p = x%p covers x but
    # the same r_p also covers x-p, x-2p, ... which might already be covered.
    # The "cheap" primes to change for covering x are those where the residue
    # change has minimal side effects. We'll just pick among primes p > x/2
    # (so x is the only multiple in 1..x region) plus some random ones.
    while time.time() < deadline:
        L_cur, first_fail = evaluate(res)
        if L_cur > best_L:
            best_L = L_cur
            best_res = res.copy()
            if best_L >= 1000:
                break
        if first_fail == -1:
            break

        # choose primes to mutate: focus on those that can cover first_fail
        # and have limited side effects (large p)
        x = first_fail
        # primes p > x: changing r_p to x%p covers x and only affects
        # numbers x + kp, x - kp; within 1..x only x itself (since p > x)
        large = np.where(PRIMES > x)[0]
        # primes x/2 < p <= x: x and x-p (two points)
        med = np.where((PRIMES > x // 2) & (PRIMES <= x))[0]
        # small primes: more side effects
        small = np.where(PRIMES <= x // 2)[0]

        # build candidate mutation targets
        candidates = []
        if len(large) > 0:
            candidates.append(('large', large))
        if len(med) > 0:
            candidates.append(('med', med))
        if len(small) > 0:
            candidates.append(('small', small))

        # do several mutation attempts per loop
        n_attempts = 8
        for _ in range(n_attempts):
            # pick a category (bias toward large)
            if candidates:
                # weighted: large 0.5, med 0.3, small 0.2
                weights = []
                for name, arr in candidates:
                    if name == 'large':
                        weights.append(0.5)
                    elif name == 'med':
                        weights.append(0.3)
                    else:
                        weights.append(0.2)
                w = np.array(weights, dtype=float)
                w /= w.sum()
                cat = rng.choice(len(candidates), p=w)
                name, arr = candidates[cat]
                i = arr[rng.integers(0, len(arr))]
            else:
                i = rng.integers(0, NPRIMES)

            # set residue to cover first_fail
            new_res = res.copy()
            new_res[i] = x % PRIMES[i]

            L_new, ff_new = evaluate(new_res)
            if L_new > L_cur:
                res = new_res
            elif rng.random() < 0.05:
                # occasional random exploration
                res = perturb(res, 1)

        # periodic full re-evaluation of best
        if best_L >= 1000:
            break

    return best_res.tolist()
