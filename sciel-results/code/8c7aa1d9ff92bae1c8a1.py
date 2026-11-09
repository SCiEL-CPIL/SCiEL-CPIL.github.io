import time
import numpy as np

N = 51975  # 3^3 * 5^2 * 7 * 11, odd


def _divisors(n):
    ds = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            ds.append(i)
            if i != n // i:
                ds.append(n // i)
        i += 1
    return ds


# All odd divisors > 1 of N (N is odd, so every divisor is odd).
MODS = sorted(d for d in _divisors(N) if d > 1)
K = {m: N // m for m in MODS}  # fiber size for modulus m
NUM_MODS = len(MODS)

# Precompute bitset masks: mask[m][r] has bits set for positions x where x ≡ r (mod m)
# Using Python integers as arbitrary-precision bitsets
_MASKS = {}
for m in MODS:
    masks_for_m = []
    for r in range(m):
        # Build bitset: positions r, r+m, r+2m, ...
        mask = 0
        for pos in range(r, N, m):
            mask |= (1 << pos)
        masks_for_m.append(mask)
    _MASKS[m] = masks_for_m

# Precompute for fast CD: for each modulus m, list of (r, mask) pairs
_MASK_LISTS = {m: [(r, _MASKS[m][r]) for r in range(m)] for m in MODS}


def _count_bits(x):
    """Count set bits in integer x."""
    return bin(x).count('1')


def _uncovered_count(cov_bits):
    """Count uncovered positions given coverage bitset."""
    # Total positions N, covered = popcount(cov_bits), uncovered = N - covered
    return N - _count_bits(cov_bits)


def _cd(assign, max_passes=50):
    """Coordinate descent using bitsets. assign: dict m -> r"""
    for _ in range(max_passes):
        improved = False
        for m in MODS:
            # Compute coverage from all other moduli
            cov = 0
            for m2 in MODS:
                if m2 != m:
                    cov |= _MASKS[m2][assign[m2]]
            # For each residue r mod m, count new positions covered
            best_r = assign[m]
            best_gain = -1
            for r in range(m):
                mask = _MASKS[m][r]
                new_cov = cov | mask
                gain = _count_bits(new_cov) - _count_bits(cov)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            if best_r != assign[m]:
                assign[m] = best_r
                improved = True
        if not improved:
            break
    return assign


def _cd_fast(assign, max_passes=10):
    """Faster CD with fewer passes for SA moves."""
    for _ in range(max_passes):
        improved = False
        for m in MODS:
            cov = 0
            for m2 in MODS:
                if m2 != m:
                    cov |= _MASKS[m2][assign[m2]]
            best_r = assign[m]
            best_gain = -1
            for r in range(m):
                mask = _MASKS[m][r]
                gain = _count_bits(cov | mask) - _count_bits(cov)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            if best_r != assign[m]:
                assign[m] = best_r
                improved = True
        if not improved:
            break
    return assign


def _full_coverage(assign):
    """Compute full coverage bitset."""
    cov = 0
    for m in MODS:
        cov |= _MASKS[m][assign[m]]
    return cov


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s limit
    rng = np.random.default_rng(20240517)

    best_assign = None
    best_unc = N

    def consider(a):
        nonlocal best_assign, best_unc
        cov = _full_coverage(a)
        u = N - _count_bits(cov)
        if u < best_unc:
            best_unc = u
            best_assign = dict(a)
        return u

    # Deterministic baseline: all residues 0, then coordinate descent.
    a0 = {m: 0 for m in MODS}
    _cd(a0, max_passes=100)
    consider(a0)

    # Try structured initializations
    for init_val in [1, 2, 3]:
        a_init = {m: init_val % m for m in MODS}
        _cd(a_init, max_passes=100)
        consider(a_init)

    # Greedy with different orderings
    for order_seed in [1, 2, 3, 4, 5, 6, 7, 8]:
        rng_g = np.random.default_rng(order_seed)
        mods_shuffled = list(MODS)
        rng_g.shuffle(mods_shuffled)
        a_g = {m: 0 for m in MODS}
        # Greedy: process moduli in shuffled order, pick best residue given current coverage
        cov = 0
        for m in mods_shuffled:
            best_r = 0
            best_gain = -1
            for r in range(m):
                mask = _MASKS[m][r]
                gain = _count_bits(cov | mask) - _count_bits(cov)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            a_g[m] = best_r
            cov |= _MASKS[m][best_r]
        _cd(a_g, max_passes=100)
        consider(a_g)

    # Random restarts + coordinate descent.
    while time.time() < deadline - 30.0:
        a = {m: int(rng.integers(0, m)) for m in MODS}
        _cd(a, max_passes=50)
        u = consider(a)
        if u == 0:
            break

    # Simulated annealing on the best solution to escape local minima.
    if best_assign is not None and time.time() < deadline - 5.0:
        a = dict(best_assign)
        cur_u = best_unc
        T = 5.0
        T_min = 0.01
        while time.time() < deadline - 3.0 and T > T_min:
            # Random single-residue perturbation
            m = MODS[int(rng.integers(0, NUM_MODS))]
            a[m] = int(rng.integers(0, m))
            _cd_fast(a, max_passes=5)
            cov = _full_coverage(a)
            u = N - _count_bits(cov)
            if u < cur_u:
                cur_u = u
                if u < best_unc:
                    best_unc = u
                    best_assign = dict(a)
                    if u == 0:
                        break
            else:
                delta_u = u - cur_u
                if delta_u > 0:
                    accept_prob = np.exp(-delta_u / max(T, 1e-9))
                else:
                    accept_prob = 1.0
                if rng.random() < accept_prob:
                    cur_u = u
            T *= 0.999

    # Final polish: run more coordinate descent passes on the best solution
    if best_assign is not None and time.time() < deadline - 1.0:
        a = dict(best_assign)
        _cd(a, max_passes=200)
        consider(a)

    return [[m, int(best_assign[m])] for m in MODS]
