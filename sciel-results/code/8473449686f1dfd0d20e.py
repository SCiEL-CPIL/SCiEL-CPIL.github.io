"""Sums and differences of finite sets: maximize 1 + log(|U-U|/|U+U|)/log(2*maxU+1)."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from numpy.fft import fft, ifft

_START = time.time()

def _time_left(deadline):
    return deadline - time.time()

def _next_pow2(n):
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()

def _sumset_size_fft(U):
    max_u = max(U)
    sum_len = 2 * max_u + 1
    fft_len = _next_pow2(2 * sum_len)
    ind = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        ind[u] = 1.0
    f = fft(ind)
    conv = ifft(f * f).real
    return int(np.sum(conv[:sum_len] > 1e-6)), max_u

def _diffset_size_fft(U):
    max_u = max(U)
    diff_len = 2 * max_u + 1
    fft_len = _next_pow2(2 * diff_len)
    vec = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        vec[u + max_u] = 1.0
    f = fft(vec)
    res = ifft(f * np.conj(f)).real
    return int(np.sum(np.abs(res[:diff_len]) > 1e-6)), max_u

def _score_set_fft(U):
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    ss, _ = _sumset_size_fft(U)
    ds, _ = _diffset_size_fft(U)
    if ss == 0:
        return 0.0
    ratio = ds / ss
    if ratio <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def _score_set_exact(U):
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    ss = set()
    ds = set()
    for i in range(n):
        for j in range(n):
            ss.add(U[i] + U[j])
            ds.add(U[i] - U[j])
    ratio = len(ds) / len(ss)
    if ratio <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def _score_set(U):
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    if n <= 28:
        return _score_set_exact(U)
    return _score_set_fft(U)

def _scale(U, k):
    """Scale set by k: preserves |U+U|,|U-U|, multiplies maxU by k."""
    return sorted([u * k for u in U])

def _bose_sidon(q):
    """Bose Sidon set from finite field GF(q^2) with q odd prime power."""
    if q < 3:
        return [0]
    # Build field GF(q^2) = GF(q)[x]/(irreducible). Find irreducible quadratic.
    # Represent elements as (a, b) -> a + b*w, w^2 = c (c non-square in GF(q)).
    # Find a non-square c.
    def is_square(a, p):
        if a == 0:
            return True
        return pow(a, (p - 1) // 2, p) == 1
    c = 2
    while is_square(c % q, q):
        c += 1
    c = c % q
    # field element ops
    def fm(a, b):
        # multiply (a0,a1)*(b0,b1)
        a0, a1 = a
        b0, b1 = b
        # (a0 + a1 w)(b0 + b1 w) = a0 b0 + c a1 b1 + (a0 b1 + a1 b0) w
        r0 = (a0 * b0 + c * a1 * b1) % q
        r1 = (a0 * b1 + a1 * b0) % q
        return (r0, r1)
    # Singer construction: pick g a primitive element of GF(q^2)^* (order q^2-1)
    # Sidon set = { g^i : i = 0..q } (q+1 elements)
    # Find primitive element
    order = q * q - 1
    factors = []
    d = order
    p = 2
    while p * p <= d:
        if d % p == 0:
            factors.append(p)
            while d % p == 0:
                d //= p
        p += 1
    if d > 1:
        factors.append(d)
    def fpow(base, e):
        res = (1, 0)
        b = base
        while e > 0:
            if e & 1:
                res = fm(res, b)
            b = fm(b, b)
            e >>= 1
        return res
    def feq(x, y):
        return x[0] == y[0] and x[1] == y[1]
    # find primitive element
    prim = None
    for a0 in range(q):
        for a1 in range(q):
            if a0 == 0 and a1 == 0:
                continue
            e = (a0, a1)
            is_prim = True
            for f in factors:
                if feq(fpow(e, order // f), (1, 0)):
                    is_prim = False
                    break
            if is_prim:
                prim = e
                break
        if prim is not None:
            break
    if prim is None:
        return [0, 1, 3]
    # generate q+1 elements
    elems = []
    cur = (1, 0)
    for i in range(q + 1):
        elems.append(cur)
        cur = fm(cur, prim)
    # map to integers: value = a0 + a1 * q (unique since 0<=a0,a1<q)
    vals = [e[0] + e[1] * q for e in elems]
    vals = sorted(set(vals))
    if 0 not in vals:
        vals = [0] + vals
    return vals

def _singer_golomb(q):
    """Singer Golomb ruler (difference set) from GF(q) with q=3^k-1 prime power, q odd."""
    if q < 3 or (q - 1) % 2 != 0:
        return None
    # Singer difference set: parameters (v, k, lambda) = (q^2+q+1, q+1, 1)
    # Construct using GF(q) with q = 3^k - 1 (q must be prime power)
    # Standard: take primitive element g of GF(q), set D = { g^i : i=0..q } plus {infinity}
    # Represent GF(q) elements as integers mod q (q prime power, use polynomial if composite)
    # For simplicity, require q prime or small prime power; use integer field if prime
    # Check q prime
    def is_prime(n):
        if n < 2:
            return False
        for p in range(2, int(n ** 0.5) + 1):
            if n % p == 0:
                return False
        return True
    if not is_prime(q):
        return None
    # Singer construction in GF(q): D = { g^(2i) : i=0..(q-1)/2 } ∪ {∞}  (q+1 elements)
    # where g is primitive. Map ∞ to a large value.
    order = q - 1
    # find primitive element
    factors = []
    d = order
    p = 2
    while p * p <= d:
        if d % p == 0:
            factors.append(p)
            while d % p == 0:
                d //= p
        p += 1
    if d > 1:
        factors.append(d)
    def fpow(base, e):
        return pow(base, e, q)
    prim = None
    for a in range(1, q):
        is_prim = True
        for f in factors:
            if fpow(a, order // f) == 1:
                is_prim = False
                break
        if is_prim:
            prim = a
            break
    if prim is None:
        return None
    # D = { g^(2i) : i = 0..(q-1)/2 }
    half = (q - 1) // 2
    D = set()
    for i in range(half + 1):
        D.add(fpow(prim, 2 * i))
    # add infinity as q (so values 0..q, but 0 not in D since g^(2i) != 0)
    D.add(q)
    vals = sorted(D)
    # ensure 0 in set: shift so min is 0
    m = min(vals)
    vals = [v - m for v in vals]
    return vals

def _known_golomb(n):
    known = {
        1: [0], 2: [0, 1], 3: [0, 1, 3], 4: [0, 1, 4, 6], 5: [0, 1, 4, 10, 11],
        6: [0, 1, 4, 10, 12, 17], 7: [0, 1, 4, 10, 12, 17, 22],
        8: [0, 1, 4, 10, 12, 17, 22, 29], 9: [0, 1, 4, 10, 12, 17, 22, 29, 36],
        10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44], 11: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52],
        12: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60],
    }
    return known.get(n)

def _spectral_soft_loss(U, M, temp=2.0):
    """Smooth surrogate: soft-min of sumset support + peak sidelobe energy of autocorrelation."""
    max_u = max(U) if U else 0
    fft_len = _next_pow2(2 * (2 * max_u + 1) + 4)
    ind = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        ind[u] = 1.0
    f = fft(ind)
    conv = ifft(f * f).real  # linear autocorrelation (sumset multiplicities)
    # sumset support: positions 0..2*max_u
    ss_region = conv[:2 * max_u + 1]
    # soft "is present" via -log(1 - exp(-x/temp)) approx; use softplus of presence
    # presence indicator ~ sigmoid; soft support count = sum of soft indicator
    soft_presence = 1.0 / (1.0 + np.exp(-ss_region / temp))
    soft_sumset = float(np.sum(soft_presence))
    # difference set: autocorrelation of shifted U
    diff_vec = np.zeros(fft_len, dtype=np.float64)
    off = max_u
    for u in U:
        diff_vec[u + off] = 1.0
    fd = fft(diff_vec)
    diff_res = ifft(fd * np.conj(fd)).real
    # peak sidelobe: max |diff_res| excluding lag 0 (position off)
    diff_region = diff_res[:2 * max_u + 1]
    # exclude the central peak at index off
    if 0 <= off < len(diff_region):
        diff_region = np.concatenate([diff_region[:off], diff_region[off + 1:]])
    peak_sidelobe = float(np.max(np.abs(diff_region))) if len(diff_region) else 0.0
    # We want small soft_sumset (minimize) and large diffset (maximize distinct diffs ~ minimize peak)
    return soft_sumset + peak_sidelobe

def _local_search(U, max_iters=6000, deadline=None):
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 300
    no_improve = 0
    max_no_improve = 60

    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        if no_improve >= max_no_improve:
            break
        improved = False
        idxs = list(range(len(U)))
        random.shuffle(idxs)
        for i in idxs:
            if U[i] == 0:
                continue
            old_val = U[i]
            cands = set()
            for off in range(-40, 41):
                nv = old_val + off
                if nv > 0:
                    cands.add(nv)
            for _ in range(10):
                cands.add(random.randint(1, upper_bound))
            for nv in sorted(cands, key=lambda x: abs(x - old_val)):
                if nv == old_val:
                    continue
                new_U = U[:]
                new_U[i] = nv
                if len(set(new_U)) != len(new_U):
                    continue
                s = _score_set(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U[:]
                    U = new_U[:]
                    improved = True
                    no_improve = 0
                    break
            if improved:
                break
        if not improved:
            no_improve += 1
            # add
            if len(U) < 400:
                for _ in range(40):
                    nv = random.randint(1, upper_bound)
                    if nv in U:
                        continue
                    new_U = U + [nv]
                    s = _score_set(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        improved = True
                        no_improve = 0
                        break
            if not improved and len(U) > 2:
                for i in range(len(U)):
                    if U[i] == 0:
                        continue
                    new_U = U[:i] + U[i + 1:]
                    s = _score_set(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        improved = True
                        no_improve = 0
                        break
        if not improved and random.random() < 0.2:
            new_U = U[:]
            nc = random.randint(2, min(10, len(new_U) - 1))
            for _ in range(nc):
                idx = random.randint(1, len(new_U) - 1)
                new_U[idx] = random.randint(1, upper_bound)
            if len(set(new_U)) == len(new_U):
                s = _score_set(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U[:]
                    U = new_U[:]
                    no_improve = 0
    return sorted(set(best_U)), best_score

def _polish(U, deadline=None):
    U = sorted(set(U))
    n = len(U)
    if n > 60:
        return _local_search(U, max_iters=2000, deadline=deadline)
    best_score = _score_set_exact(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 120
    no_improve = 0
    for it in range(6000):
        if deadline and time.time() > deadline:
            break
        if no_improve >= 35:
            break
        improved = False
        idxs = list(range(1, len(U)))
        random.shuffle(idxs)
        for i in idxs:
            old_val = U[i]
            for off in range(-25, 26):
                nv = old_val + off
                if nv <= 0:
                    continue
                new_U = U[:]
                new_U[i] = nv
                if len(set(new_U)) != len(new_U):
                    continue
                s = _score_set_exact(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U[:]
                    U = new_U[:]
                    improved = True
                    no_improve = 0
                    break
            if improved:
                break
        if not improved:
            no_improve += 1
            if len(U) < 90:
                for _ in range(25):
                    nv = random.randint(1, upper_bound)
                    if nv in U:
                        continue
                    new_U = U + [nv]
                    s = _score_set_exact(new_U)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        improved = True
                        no_improve = 0
                        break
            if not improved and len(U) > 3:
                for i in range(1, len(U)):
                    new_U = U[:i] + U[i + 1:]
                    s = _score_set_exact(new_U)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        improved = True
                        no_improve = 0
                        break
    return sorted(set(best_U)), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s
    random.seed(12345)

    best_U = [0, 1, 3]
    best_score = _score_set(best_U)

    candidates = []

    # Bose Sidon sets for various odd prime powers q
    for q in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]:
        if time.time() > deadline - 50:
            break
        try:
            U = _bose_sidon(q)
            if U and 0 in U:
                candidates.append(U)
        except Exception:
            pass

    # Singer Golomb rulers for q = 3^k - 1 prime
    for k in [2, 3, 4]:
        q = 3 ** k - 1
        if time.time() > deadline - 50:
            break
        try:
            U = _singer_golomb(q)
            if U:
                candidates.append(U)
        except Exception:
            pass

    # Known Golomb rulers
    for n in range(3, 13):
        gr = _known_golomb(n)
        if gr:
            candidates.append(gr)

    # Random sets
    for trial in range(60):
        if time.time() > deadline - 50:
            break
        n = random.randint(3, 50)
        max_val = random.randint(40, 1200)
        U = [0] + random.sample(range(1, max_val + 1), n - 1)
        candidates.append(sorted(U))

    # Sparse random
    for trial in range(30):
        if time.time() > deadline - 50:
            break
        n = random.randint(5, 40)
        max_val = random.randint(150, 3000)
        U = [0] + random.sample(range(1, max_val + 1), n - 1)
        candidates.append(sorted(U))

    # Scaled versions of good small sets (scaling reduces maxU relative to n? No, scaling increases maxU)
    # Actually scaling multiplies maxU, which HURTS the log denom. Skip scaling up.
    # But we can try scaling DOWN is impossible. So skip.

    # Evaluate and keep top
    scored = []
    for U in candidates:
        if time.time() > deadline - 40:
            break
        U = sorted(set(U))
        if 0 not in U:
            continue
        s = _score_set(U)
        scored.append((s, U))
        if s > best_score:
            best_score = s
            best_U = U[:]

    scored.sort(key=lambda x: -x[0])

    # Local search on best
    if time.time() < deadline - 25:
        best_U, best_score = _local_search(best_U, max_iters=9000, deadline=deadline - 25)

    # Local search on top candidates
    for s, U in scored[:12]:
        if time.time() > deadline - 25:
            break
        if U == best_U:
            continue
        if s > best_score - 0.06:
            iU, is_ = _local_search(U, max_iters=3500, deadline=deadline - 25)
            if is_ > best_score:
                best_score = is_
                best_U = iU

    # Final polish
    if time.time() < deadline - 5:
        best_U, best_score = _polish(best_U, deadline=deadline - 2)

    return sorted(set(best_U))
# EVOLVE-BLOCK-END
