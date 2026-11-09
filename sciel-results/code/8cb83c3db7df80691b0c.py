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
NM = len(MODS)
MODS_LIST = list(MODS)


def _uncovered_from_cnt(cnt):
    """Count uncovered residues from a coverage count array."""
    return int((cnt == 0).sum())


def _build_cnt(assign):
    """Build coverage count array from assignment."""
    cnt = np.zeros(N, dtype=np.int16)
    for m in MODS:
        cnt[assign[m]::m] += 1
    return cnt


def _uncovered(assign):
    """Count uncovered residues using integer accumulation."""
    cnt = _build_cnt(assign)
    return _uncovered_from_cnt(cnt)


def _best_residue_from_cnt(cnt, m, assign):
    """Find best residue for modulus m given coverage from all OTHER moduli.
    We subtract m's current contribution, then find residue maximizing uncovered."""
    # Remove m's contribution
    r_old = assign[m]
    cnt[r_old::m] -= 1
    # Now cnt reflects all other moduli
    # For each residue r, uncovered_in_fiber = number of positions in r::m where cnt==0
    # cnt has shape (N,), fiber for residue r is cnt[r::m] which has K[m] elements
    # uncovered = (cnt[r::m] == 0).sum()
    # We want to maximize this, i.e., minimize covered = K[m] - uncovered
    # Equivalent: minimize (cnt[r::m] > 0).sum()
    # Reshape: cnt.reshape(K[m], m) — row i, col j is cnt[i*m + j]
    # Fiber for residue j is column j: cnt[j::m] = cnt.reshape(K[m], m)[:, j]
    # covered_in_fiber_j = (cnt.reshape(K[m], m)[:, j] > 0).sum(axis=0)
    # We want to minimize covered, i.e., maximize uncovered
    covered = (cnt.reshape(K[m], m) > 0).sum(axis=0)
    best_r = int(np.argmin(covered))
    # Restore m's contribution
    cnt[r_old::m] += 1
    return best_r


def _cd(assign, max_passes=200):
    """Coordinate descent with incremental coverage tracking."""
    cnt = _build_cnt(assign)
    for _ in range(max_passes):
        improved = False
        for m in MODS:
            br = _best_residue_from_cnt(cnt, m, assign)
            if br != assign[m]:
                # Update cnt: remove old, add new
                cnt[assign[m]::m] -= 1
                assign[m] = br
                cnt[br::m] += 1
                improved = True
        if not improved:
            break
    return assign


def _cd_limited(assign, max_passes=3):
    """Limited coordinate descent for faster SA moves."""
    cnt = _build_cnt(assign)
    for _ in range(max_passes):
        improved = False
        for m in MODS:
            br = _best_residue_from_cnt(cnt, m, assign)
            if br != assign[m]:
                cnt[assign[m]::m] -= 1
                assign[m] = br
                cnt[br::m] += 1
                improved = True
        if not improved:
            break
    return assign


def _sa_step(assign, cnt, rng, mode):
    """Perform one SA perturbation step, updating cnt incrementally.
    Returns the new uncovered count."""
    if mode == 'swap' and NM >= 2:
        i, j = rng.integers(0, NM, size=2)
        if i == j:
            j = (j + 1) % NM
        mi, mj = MODS_LIST[i], MODS_LIST[j]
        ri, rj = assign[mi], assign[mj]
        # Try swap if valid
        if rj < mj and ri < mi:
            # Remove old contributions
            cnt[ri::mi] -= 1
            cnt[rj::mj] -= 1
            # Swap
            assign[mi] = rj
            assign[mj] = ri
            # Add new contributions
            cnt[rj::mi] += 1
            cnt[ri::mj] += 1
        else:
            # Random new residues
            new_ri = int(rng.integers(0, mi))
            new_rj = int(rng.integers(0, mj))
            cnt[ri::mi] -= 1
            cnt[rj::mj] -= 1
            assign[mi] = new_ri
            assign[mj] = new_rj
            cnt[new_ri::mi] += 1
            cnt[new_rj::mj] += 1
    else:
        # Single residue perturbation
        m = MODS_LIST[int(rng.integers(0, NM))]
        old_r = assign[m]
        new_r = int(rng.integers(0, m))
        if new_r != old_r:
            cnt[old_r::m] -= 1
            assign[m] = new_r
            cnt[new_r::m] += 1
    return _uncovered_from_cnt(cnt)


def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin under 150s limit
    rng = np.random.default_rng(20240517)

    best_assign = None
    best_unc = N

    def consider(a):
        nonlocal best_assign, best_unc
        u = _uncovered(a)
        if u < best_unc:
            best_unc = u
            best_assign = dict(a)
        return u

    # Deterministic baseline: all residues 0, then coordinate descent.
    a0 = {m: 0 for m in MODS}
    _cd(a0, max_passes=300)
    consider(a0)

    # Try greedy with different orderings
    for order_seed in [1, 2, 3, 4, 5, 6, 7, 8]:
        rng_g = np.random.default_rng(order_seed)
        mods_shuffled = list(MODS)
        rng_g.shuffle(mods_shuffled)
        a_g = {m: 0 for m in MODS}
        # Greedy: process moduli in shuffled order, pick best residue given current coverage
        for m in mods_shuffled:
            cov = np.zeros(N, dtype=bool)
            for m2 in MODS:
                if m2 != m:
                    cov[a_g[m2]::m2] = True
            unc = ~cov
            cnt_g = unc.reshape(K[m], m).sum(axis=0)
            a_g[m] = int(np.argmax(cnt_g))
        _cd(a_g, max_passes=200)
        consider(a_g)

    # Random restarts + coordinate descent.
    while time.time() < deadline - 60.0:
        a = {m: int(rng.integers(0, m)) for m in MODS}
        _cd(a, max_passes=200)
        u = consider(a)
        if u == 0:
            break

    # Simulated annealing on the best solution to escape local minima.
    if best_assign is not None and time.time() < deadline - 10.0:
        a = dict(best_assign)
        cnt = _build_cnt(a)
        cur_u = _uncovered_from_cnt(cnt)
        T = 10.0
        T_min = 0.005
        step = 0
        while time.time() < deadline - 5.0 and T > T_min:
            step += 1
            mode = 'swap' if (step % 4 == 0 and NM >= 2) else 'single'
            u = _sa_step(a, cnt, rng, mode)
            if u < cur_u:
                cur_u = u
                if u < best_unc:
                    best_unc = u
                    best_assign = dict(a)
                    if u == 0:
                        break
            else:
                delta = u - cur_u
                if rng.random() < np.exp(-delta / max(T, 1e-9)):
                    cur_u = u
            T *= 0.9985

        # Final polish: full CD on best
        if best_assign is not None and time.time() < deadline - 1.0:
            a_final = dict(best_assign)
            _cd(a_final, max_passes=500)
            consider(a_final)

    return [[m, int(best_assign[m])] for m in MODS]
