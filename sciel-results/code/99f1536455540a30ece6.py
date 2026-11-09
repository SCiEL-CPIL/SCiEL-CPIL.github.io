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


MODS = sorted(d for d in _divisors(N) if d > 1)
K = {m: N // m for m in MODS}
MODS_LIST = list(MODS)
N_MODS = len(MODS_LIST)

# Precompute position arrays: pos[m][r] = all x in [0,N) with x % m == r
# Stored as 2D arrays (m, km) so pos[m][r] gives the km positions
_pos_cache = {}
def _get_pos(m):
    if m not in _pos_cache:
        km = K[m]
        base = np.arange(km) * m
        _pos_cache[m] = base[None, :] + np.arange(m)[:, None]
    return _pos_cache[m]

# Precompute all
for m in MODS_LIST:
    _get_pos(m)

# Also precompute flat versions for faster indexing when needed
_pos_flat = {m: _pos_cache[m].reshape(-1) for m in MODS_LIST}


def _build_count(assign):
    cnt = np.zeros(N, dtype=np.int32)
    for m in MODS_LIST:
        r = assign[m]
        cnt[_pos_cache[m][r]] += 1
    return cnt


def _uncovered_from_count(cnt):
    return int((cnt == 0).sum())


def _cd(assign, max_passes=300, deadline=None):
    """Coordinate descent: for each modulus, pick residue that covers most uncovered points."""
    cnt = _build_count(assign)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for m in MODS_LIST:
            old_r = assign[m]
            cnt[_pos_cache[m][old_r]] -= 1
            unc = (cnt == 0)
            km = K[m]
            fiber_counts = unc.reshape(km, m).sum(axis=0)
            br = int(np.argmax(fiber_counts))
            cnt[_pos_cache[m][br]] += 1
            if br != old_r:
                assign[m] = br
                improved = True
        if not improved:
            break
    return assign


def _cd_with_coverage(assign, max_passes=300, deadline=None):
    """Coordinate descent that also tracks coverage for efficiency."""
    cnt = _build_count(assign)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for m in MODS_LIST:
            old_r = assign[m]
            cnt[_pos_cache[m][old_r]] -= 1
            unc = (cnt == 0)
            km = K[m]
            fiber_counts = unc.reshape(km, m).sum(axis=0)
            br = int(np.argmax(fiber_counts))
            cnt[_pos_cache[m][br]] += 1
            if br != old_r:
                assign[m] = br
                improved = True
        if not improved:
            break
    return assign


def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    rng = np.random.default_rng(20240517)

    best_assign = None
    best_unc = N

    def consider(a):
        nonlocal best_assign, best_unc
        u = _uncovered_from_count(_build_count(a))
        if u < best_unc:
            best_unc = u
            best_assign = dict(a)
        return u

    # Deterministic baseline: all residues 0, then CD
    a0 = {m: 0 for m in MODS_LIST}
    _cd(a0, max_passes=500, deadline=deadline)
    consider(a0)

    # Greedy with different orderings
    orderings = [
        MODS_LIST[::-1],  # largest first
        MODS_LIST,         # smallest first
    ]
    for _ in range(8):
        orderings.append(list(rng.choice(MODS_LIST, size=N_MODS, replace=False)))

    for order in orderings:
        if time.time() > deadline - 35.0:
            break
        a_g = {m: 0 for m in MODS_LIST}
        cov = np.zeros(N, dtype=bool)
        for m in order:
            unc = ~cov
            km = K[m]
            fiber_counts = unc.reshape(km, m).sum(axis=0)
            a_g[m] = int(np.argmax(fiber_counts))
            cov[_pos_cache[m][a_g[m]]] = True
        _cd(a_g, max_passes=300, deadline=deadline)
        consider(a_g)

    # Random restarts + CD with varied strategies
    while time.time() < deadline - 45.0:
        strategy = rng.integers(0, 3)
        if strategy == 0:
            # Pure random
            a = {m: int(rng.integers(0, m)) for m in MODS_LIST}
        elif strategy == 1:
            # Random with bias toward smaller residues
            a = {m: int(rng.integers(0, max(1, m // 2))) for m in MODS_LIST}
        else:
            # Start from best and perturb
            if best_assign is not None:
                a = dict(best_assign)
                for _ in range(rng.integers(1, 4)):
                    m = MODS_LIST[int(rng.integers(0, N_MODS))]
                    a[m] = int(rng.integers(0, m))
            else:
                a = {m: int(rng.integers(0, m)) for m in MODS_LIST}
        
        _cd(a, max_passes=250, deadline=deadline)
        u = consider(a)
        if u == 0:
            break

    # Simulated annealing on the best solution
    if best_assign is not None and time.time() < deadline - 8.0:
        a = dict(best_assign)
        cur_u = best_unc
        cnt = _build_count(a)
        T = 5.0
        T_min = 0.01
        sa_deadline = deadline - 5.0
        
        while time.time() < sa_deadline and T > T_min:
            m = MODS_LIST[int(rng.integers(0, N_MODS))]
            old_r = a[m]
            new_r = int(rng.integers(0, m))
            if new_r == old_r:
                continue
            cnt[_pos_cache[m][old_r]] -= 1
            cnt[_pos_cache[m][new_r]] += 1
            u = _uncovered_from_count(cnt)
            if u < cur_u:
                cur_u = u
                a[m] = new_r
                if u < best_unc:
                    best_unc = u
                    best_assign = dict(a)
                    if u == 0:
                        break
            else:
                delta = u - cur_u
                if delta == 0 or rng.random() < np.exp(-delta / max(T, 1e-9)):
                    cur_u = u
                    a[m] = new_r
                else:
                    cnt[_pos_cache[m][new_r]] -= 1
                    cnt[_pos_cache[m][old_r]] += 1
            T *= 0.9995

        # Final polish with more aggressive CD
        if best_assign is not None and time.time() < deadline - 2.0:
            a_final = dict(best_assign)
            _cd(a_final, max_passes=800, deadline=deadline)
            consider(a_final)

    return [[m, int(best_assign[m])] for m in MODS_LIST]
