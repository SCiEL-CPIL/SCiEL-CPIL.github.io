import time
import random
import numpy as np

_START = time.time()
_DEADLINE = _START + 84.0  # safety margin under 90s kill

def _square_support_count(coeffs):
    nz = [(i, c) for i, c in enumerate(coeffs) if c != 0]
    sq = {}
    for ei, ci in nz:
        for ej, cj in nz:
            s = ei + ej
            sq[s] = sq.get(s, 0) + ci * cj
    return sum(1 for v in sq.values() if v != 0)

def _spectral_collision_score(coeffs):
    """Fast approximate score using FFT autocorrelation to estimate square support."""
    n = len(coeffs)
    # Pad to next power of 2 for FFT efficiency
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    f = np.zeros(nfft, dtype=np.float64)
    for i, c in enumerate(coeffs):
        f[i] = c
    F = np.fft.fft(f)
    power = F * np.conj(F)
    ac = np.fft.ifft(power).real[:2*n-1]
    # Count nonzero autocorrelation values (approximation of square support)
    return int(np.sum(np.abs(ac) > 0.5))

def _valid(coeffs):
    if len(coeffs) > 41:
        return False
    if not coeffs or coeffs[-1] == 0:
        return False
    if any(c < -64 or c > 64 for c in coeffs):
        return False
    if sum(1 for c in coeffs if c != 0) != 12:
        return False
    return True

def _trim(coeffs):
    arr = list(coeffs)
    while len(arr) > 1 and arr[-1] == 0:
        arr.pop()
    return arr

def construct():
    global _DEADLINE
    rng = random.Random(20240517)
    best_coeffs = None
    best_score = 10**9

    def random_poly():
        exponents = sorted(rng.sample(range(41), 12))
        coeffs = [0] * 41
        for e in exponents:
            # bias to small coefficients for easier cancellation
            r = rng.random()
            if r < 0.5:
                m = 1
            elif r < 0.8:
                m = rng.randint(1, 4)
            else:
                m = rng.randint(1, 64)
            coeffs[e] = rng.choice([-1, 1]) * m
        return coeffs

    def antisym_poly():
        # antisymmetric about center 20 to encourage cancellation
        coeffs = [0] * 41
        used = set()
        placed = 0
        attempts = 0
        while placed < 12 and attempts < 200:
            attempts += 1
            d = rng.randint(1, 20)
            e1 = 20 - d
            e2 = 20 + d
            if e1 in used or e2 in used or e1 < 0 or e2 >= 41:
                continue
            m = rng.randint(1, 8)
            coeffs[e1] = -m
            coeffs[e2] = m
            used.add(e1); used.add(e2)
            placed += 2
        # fill remaining if odd (shouldn't happen since 12 even)
        if placed < 12:
            nz = [i for i, c in enumerate(coeffs) if c != 0]
            free = [j for j in range(41) if j not in set(nz)]
            for j in free:
                if placed >= 12:
                    break
                coeffs[j] = rng.choice([-1, 1]) * rng.randint(1, 8)
                placed += 1
        if coeffs[-1] == 0:
            # ensure nonzero final
            nz = [i for i, c in enumerate(coeffs) if c != 0]
            if nz:
                # move a term to high exponent
                hi = 40
                if coeffs[hi] == 0:
                    src = nz[0]
                    coeffs[hi] = coeffs[src]
                    coeffs[src] = 0
        return coeffs if _valid(coeffs) else None

    def _find_cancellation_targets(coeffs):
        """Use spectral analysis to find exponent positions where moving terms
        might create additional cancellations in the square."""
        n = len(coeffs)
        nfft = 1
        while nfft < 2 * n:
            nfft *= 2
        f = np.zeros(nfft, dtype=np.float64)
        for i, c in enumerate(coeffs):
            f[i] = c
        F = np.fft.fft(f)
        power = F * np.conj(F)
        ac = np.fft.ifft(power).real[:2*n-1]
        
        # Find peaks in autocorrelation (positions with large magnitude)
        # These are where cross-terms concentrate
        abs_ac = np.abs(ac)
        # Identify top collision positions
        if len(abs_ac) == 0:
            return []
        top_k = min(10, len(abs_ac))
        peak_indices = np.argsort(abs_ac)[-top_k:][::-1]
        
        # For each peak, suggest nearby exponent moves that might cancel
        targets = []
        nz_positions = [i for i, c in enumerate(coeffs) if c != 0]
        for peak in peak_indices:
            peak = int(peak)
            if peak >= 2*n-1:
                continue
            # Suggest moving a term so that its cross-term with another lands near this peak
            for pos in nz_positions[:6]:  # limit to first few for speed
                # Try shifting pos by small amounts
                for delta in [-2, -1, 1, 2]:
                    new_pos = pos + delta
                    if 0 <= new_pos < n and new_pos != pos and coeffs[new_pos] == 0:
                        targets.append((pos, new_pos))
        return targets

    def improve(coeffs, max_iters=4000):
        cur = list(coeffs)
        cur_score = _square_support_count(cur)
        it = 0
        while it < max_iters and time.time() < _DEADLINE:
            it += 1
            nz_idx = [i for i, c in enumerate(cur) if c != 0]
            i = rng.choice(nz_idx)
            op = rng.random()
            new = list(cur)
            if op < 0.40:
                # change coefficient (small bias)
                r = rng.random()
                if r < 0.6:
                    m = 1
                elif r < 0.85:
                    m = rng.randint(1, 4)
                else:
                    m = rng.randint(1, 64)
                new[i] = rng.choice([-1, 1]) * m
            elif op < 0.70:
                # move exponent
                used = set(j for j, c in enumerate(new) if c != 0)
                free = [j for j in range(41) if j not in used]
                if not free:
                    continue
                new_exp = rng.choice(free)
                new[new_exp] = new[i]
                new[i] = 0
            elif op < 0.85:
                # spectral-guided move: use cancellation targets
                targets = _find_cancellation_targets(new)
                if targets:
                    src, dst = rng.choice(targets)
                    if new[src] != 0 and new[dst] == 0:
                        new[dst] = new[src]
                        new[src] = 0
                    else:
                        # fallback to random move
                        used = set(j for j, c in enumerate(new) if c != 0)
                        free = [j for j in range(41) if j not in used]
                        if free:
                            new_exp = rng.choice(free)
                            new[new_exp] = new[i]
                            new[i] = 0
                else:
                    used = set(j for j, c in enumerate(new) if c != 0)
                    free = [j for j in range(41) if j not in used]
                    if not free:
                        continue
                    new_exp = rng.choice(free)
                    new[new_exp] = new[i]
                    new[i] = 0
            else:
                # swap coefficients between two terms
                others = [j for j in nz_idx if j != i]
                if not others:
                    continue
                j = rng.choice(others)
                new[i], new[j] = new[j], new[i]
            if not _valid(new):
                continue
            # Use spectral score as a fast filter, then exact score
            spec_score = _spectral_collision_score(new)
            if spec_score > cur_score + 3:
                continue  # likely worse, skip exact computation
            s = _square_support_count(new)
            if s < cur_score:
                cur = new
                cur_score = s
                if cur_score <= 1:
                    break
        return cur, cur_score

    # Phase 1: random restarts
    for r in range(60):
        if time.time() > _DEADLINE:
            break
        coeffs = random_poly()
        c, sc = improve(coeffs, max_iters=2500)
        if sc < best_score:
            best_score = sc
            best_coeffs = c

    # Phase 2: structured antisymmetric restarts
    for r in range(40):
        if time.time() > _DEADLINE:
            break
        coeffs = antisym_poly()
        if coeffs is None:
            continue
        c, sc = improve(coeffs, max_iters=2500)
        if sc < best_score:
            best_score = sc
            best_coeffs = c

    # Phase 3: polish best with more iterations and spectral guidance
    if best_coeffs is not None and time.time() < _DEADLINE:
        c, sc = improve(best_coeffs, max_iters=20000)
        if sc < best_score:
            best_score = sc
            best_coeffs = c

    if best_coeffs is None:
        return [1] * 12
    arr = _trim(best_coeffs)
    if not _valid(arr):
        return [1] * 12
    return arr
