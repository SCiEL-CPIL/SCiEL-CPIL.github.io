"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
from numpy.fft import fft, ifft

# ---------------------------------------------------------------------------
# Scoring (FFT-based, exact for the sizes we use)
# ---------------------------------------------------------------------------
def _score_set_fft(U):
    """Compute the score for a finite set U using FFT for speed."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0

    total_len = 4 * max_u + 1
    fft_len = 1
    while fft_len < total_len:
        fft_len *= 2

    indicator = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        indicator[u] = 1.0

    freq = fft(indicator)

    # Sumset: autocorrelation of indicator (convolution with itself)
    sum_conv = ifft(freq * freq).real
    sumset_size = int(np.sum(sum_conv[:2 * max_u + 1] > 1e-9))

    # Difference set: count nonzero autocorrelation lags in [-max_u, max_u]
    autocorr = ifft(freq * np.conj(freq)).real
    diffset_size = 1  # difference 0
    for k in range(1, max_u + 1):
        if autocorr[k] > 1e-9:
            diffset_size += 1
        if autocorr[fft_len - k] > 1e-9:
            diffset_size += 1

    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom


def _score_set(U):
    """Compute the score for a finite set U containing 0."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0

    if n <= 25:
        sumset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
        diffset = set()
        for a in U:
            for b in U:
                diffset.add(a - b)
        ratio = len(diffset) / len(sumset)
        if ratio <= 0:
            return 0.0
        log_ratio = math.log(ratio)
        log_denom = math.log(2 * max_u + 1)
        if log_denom == 0:
            return 1.0
        return 1.0 + log_ratio / log_denom
    return _score_set_fft(U)


# ---------------------------------------------------------------------------
# Seed families
# ---------------------------------------------------------------------------
def _golomb_ruler(n):
    """Known (near-)optimal Golomb rulers: unique pairwise differences."""
    known = {
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 10, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 22],
        8: [0, 1, 4, 10, 12, 17, 22, 29],
        9: [0, 1, 4, 10, 12, 17, 22, 29, 36],
        10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
        11: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52],
    }
    if n in known:
        return known[n][:]
    if n <= 1:
        return [0]
    ruler = [0]
    for _ in range(1, n):
        pos = 1
        while True:
            cand = ruler + [pos]
            diffs = set()
            valid = True
            for j in range(len(cand)):
                for k in range(j + 1, len(cand)):
                    d = cand[k] - cand[j]
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if not valid:
                    break
            if valid:
                ruler.append(pos)
                break
            pos += 1
    return ruler


def _sidon_ruler(n):
    """Greedy Sidon (B2) set: few distinct pairwise sums, differences near-maximal."""
    ruler = [0]
    for _ in range(1, n):
        pos = 1
        while True:
            cand = ruler + [pos]
            sums = set()
            valid = True
            for i in range(len(cand)):
                for j in range(i, len(cand)):
                    s = cand[i] + cand[j]
                    if s in sums:
                        valid = False
                        break
                    sums.add(s)
                if not valid:
                    break
            if valid:
                ruler.append(pos)
                break
            pos += 1
    return ruler


# ---------------------------------------------------------------------------
# Spectral helpers: use FFT autocorrelation to score candidate positions
# ---------------------------------------------------------------------------
def _get_collision_map(U, upper_bound=None):
    """Return an array where collision_map[pos] = number of other elements
    that create a sum collision if pos were added. Lower is better."""
    U = sorted(set(U))
    if upper_bound is None:
        upper_bound = max(U) + 100
    n = len(U)
    total_len = 4 * upper_bound + 1
    fft_len = 1
    while fft_len < total_len:
        fft_len *= 2

    indicator = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        indicator[u] = 1.0

    freq = fft(indicator)
    # Convolution: sumset indicator
    sum_conv = ifft(freq * freq).real

    # collision_map[pos] = how many existing elements a+b=pos where a!=b
    # i.e., sum_conv[pos] - (1 if pos is already in U else 0) counts pairs
    # But we want: adding pos, new sums created = number of u in U such that pos+u is already in U+U
    # Simpler: collision_map[pos] = sum_conv[pos] (number of pairs (a,b) with a+b=pos)
    # If we add pos, new sums are pos+u for u in U. The number of NEW sums
    # is n minus the number of u where pos+u is already a sum.
    # Actually, let's just use sum_conv as a proxy: low sum_conv[pos] means
    # pos is "far" from the sumset structure.
    collision_map = np.zeros(upper_bound + 1, dtype=np.float64)
    for pos in range(upper_bound + 1):
        if pos < fft_len:
            collision_map[pos] = sum_conv[pos]
    return collision_map


def _spectral_local_search(U, max_iters=3000, deadline=None):
    """Local search guided by spectral (autocorrelation) collision landscape."""
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 150

    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False

        # Recompute collision map periodically
        if it % 10 == 0:
            collision_map = _get_collision_map(U, upper_bound)

        # 1) Move one element to a low-collision position
        for i in range(len(U)):
            if U[i] == 0:
                continue
            old_val = U[i]
            U_set = set(U)
            U_set.discard(old_val)

            # Generate candidates: nearby positions + low-collision positions
            candidates = set()
            for offset in range(-30, 31):
                nv = old_val + offset
                if 0 < nv <= upper_bound and nv not in U_set:
                    candidates.add(nv)
            # Add top low-collision positions
            if len(collision_map) > 0:
                # Find positions with lowest collision (excluding existing U)
                masked = collision_map.copy()
                for u in U:
                    if u <= upper_bound:
                        masked[u] = 1e18
                # Get 20 lowest collision positions
                flat_idx = np.argsort(masked)[:20]
                for idx in flat_idx:
                    if 0 < idx <= upper_bound and idx not in U_set:
                        candidates.add(int(idx))
            # Random candidates
            for _ in range(8):
                rv = random.randint(1, upper_bound)
                if rv not in U_set:
                    candidates.add(rv)

            best_cand_score = best_score
            best_cand_val = None
            for new_val in sorted(candidates):
                new_U = U[:]
                new_U[i] = new_val
                new_U.sort()
                if len(set(new_U)) != len(new_U):
                    continue
                s = _score_set(new_U)
                if s > best_cand_score + 1e-9:
                    best_cand_score = s
                    best_cand_val = new_val

            if best_cand_val is not None:
                best_score = best_cand_score
                best_U = U[:]
                best_U[i] = best_cand_val
                best_U.sort()
                U = best_U[:]
                max_u = max(U)
                upper_bound = max_u + 150
                improved = True
                break

        if not improved:
            # 2) Add an element at a low-collision position
            if len(U) < 50:
                U_set = set(U)
                # Pick from low-collision positions
                candidates = []
                if len(collision_map) > 0:
                    masked = collision_map.copy()
                    for u in U:
                        if u <= upper_bound:
                            masked[u] = 1e18
                    flat_idx = np.argsort(masked)[:15]
                    for idx in flat_idx:
                        if 0 < idx <= upper_bound and idx not in U_set:
                            candidates.append(int(idx))
                # Also try random
                for _ in range(10):
                    rv = random.randint(1, upper_bound)
                    if rv not in U_set:
                        candidates.append(rv)

                for new_val in candidates:
                    new_U = U + [new_val]
                    new_U.sort()
                    s = _score_set(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        max_u = max(U)
                        upper_bound = max_u + 150
                        improved = True
                        break

            # 3) Remove an element
            if not improved and len(U) > 3:
                for i in range(len(U)):
                    if U[i] == 0:
                        continue
                    new_U = U[:i] + U[i + 1:]
                    s = _score_set(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        max_u = max(U)
                        upper_bound = max_u + 150
                        improved = True
                        break

        if not improved:
            # 4) Perturbation
            if random.random() < 0.3:
                new_U = U[:]
                for _ in range(min(3, len(new_U) - 1)):
                    idx = random.randint(1, len(new_U) - 1)
                    new_U[idx] = random.randint(1, upper_bound)
                if len(set(new_U)) == len(new_U) and 0 in new_U:
                    s = _score_set(sorted(new_U))
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = sorted(new_U)
                        U = best_U[:]
                        max_u = max(U)
                        upper_bound = max_u + 150
                        improved = True
            if not improved:
                break

    return sorted(set(best_U)), best_score


def _local_search(U, max_iters=5000, deadline=None):
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 200

    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False

        # 1) swap / move one element
        for i in range(len(U)):
            if U[i] == 0:
                continue
            old_val = U[i]
            candidates = set()
            for offset in range(-40, 41):
                nv = old_val + offset
                if nv > 0:
                    candidates.add(nv)
            for _ in range(12):
                candidates.add(random.randint(1, upper_bound))
            for new_val in sorted(candidates):
                if new_val == old_val:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                if len(set(new_U)) != len(new_U):
                    continue
                s = _score_set(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U[:]
                    U = new_U[:]
                    max_u = max(U)
                    upper_bound = max_u + 200
                    improved = True
                    break
            if improved:
                break

        if not improved:
            # 2) add an element
            if len(U) < 60:
                for _ in range(30):
                    new_val = random.randint(1, upper_bound)
                    if new_val in U:
                        continue
                    new_U = U + [new_val]
                    s = _score_set(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        max_u = max(U)
                        upper_bound = max_u + 200
                        improved = True
                        break

            # 3) remove an element
            if not improved and len(U) > 3:
                for i in range(len(U)):
                    if U[i] == 0:
                        continue
                    new_U = U[:i] + U[i + 1:]
                    s = _score_set(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        max_u = max(U)
                        upper_bound = max_u + 200
                        improved = True
                        break

        if not improved:
            # 4) larger random perturbation to escape local optima
            if random.random() < 0.2:
                new_U = U[:]
                for _ in range(min(4, len(new_U))):
                    idx = random.randint(1, len(new_U) - 1)
                    new_U[idx] = random.randint(1, upper_bound)
                new_U_set = set(new_U)
                if len(new_U_set) == len(new_U):
                    s = _score_set(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        max_u = max(U)
                        upper_bound = max_u + 200
                        improved = True
            if not improved:
                break

    return sorted(set(best_U)), best_score


def _greedy_build(target_size, max_val, deadline=None):
    """Build a set greedily: start with {0}, add elements one by one
    choosing the element that maximizes incremental score."""
    U = [0]
    for step in range(1, target_size):
        if deadline and time.time() > deadline:
            break
        best_val = None
        best_s = _score_set(U)
        # Try a range of candidate values
        # Focus on values that are "spread out"
        candidates = set()
        # Near-multiples of existing gaps
        if len(U) > 1:
            gaps = [U[i+1] - U[i] for i in range(len(U)-1)]
            avg_gap = sum(gaps) / len(gaps)
            for g in gaps:
                for k in range(1, 5):
                    v = U[-1] + g * k
                    if 0 < v <= max_val:
                        candidates.add(v)
        # Random spread
        for _ in range(50):
            v = random.randint(1, max_val)
            candidates.add(v)
        # Systematic
        for v in range(1, min(max_val, 200) + 1, max(1, max_val // 80)):
            candidates.add(v)

        for v in sorted(candidates):
            if v in U:
                continue
            new_U = U + [v]
            new_U.sort()
            s = _score_set(new_U)
            if s > best_s + 1e-12:
                best_s = s
                best_val = v

        if best_val is not None:
            U.append(best_val)
            U.sort()
        else:
            break
    return U


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before the 150s limit

    random.seed(42)

    best_U = [0, 1, 3]
    best_score = _score_set(best_U)

    candidates = []

    # --- Greedy prime-style build: maximize incremental score ---
    for seed_n in [8, 12, 15, 20, 25, 30]:
        if time.time() > deadline - 80:
            break
        for max_val in [100, 300, 800, 2000]:
            if time.time() > deadline - 80:
                break
            gU = _greedy_build(seed_n, max_val, deadline=deadline - 80)
            if len(gU) >= 3:
                candidates.append(gU)

    # --- Sidon / B2 sets: directly minimize the sumset (the key refinement) ---
    for n in range(3, 20):
        if time.time() > deadline - 75:
            break
        candidates.append(_sidon_ruler(n))

    # --- Golomb rulers: maximize the difference set ---
    for n in range(3, 20):
        if time.time() > deadline - 75:
            break
        candidates.append(_golomb_ruler(n))

    # --- Structured: arithmetic progressions with various bases ---
    for base in [2, 3, 5, 7, 11, 13]:
        if time.time() > deadline - 75:
            break
        U = [0]
        for i in range(1, 25):
            U.append(i * base)
        candidates.append(U[:min(len(U), 25)])

    # --- Powers-based sets ---
    for p in [2, 3]:
        if time.time() > deadline - 75:
            break
        U = [0]
        for i in range(1, 15):
            U.append(p ** i)
            if U[-1] > 20000:
                break
        candidates.append(U)

    # --- Mixed structured: powers with small offsets ---
    for p in [2, 3]:
        if time.time() > deadline - 75:
            break
        U = [0]
        for i in range(1, 12):
            val = p ** i + random.randint(-5, 5)
            if val > 0 and val not in U:
                U.append(val)
        candidates.append(sorted(U))

    # --- Random sets ---
    for trial in range(20):
        if time.time() > deadline - 75:
            break
        n = random.randint(3, 40)
        max_val = random.randint(20, 600)
        U = [0] + random.sample(range(1, max_val + 1), n - 1)
        candidates.append(sorted(U))

    # Evaluate all candidates, keep the top few for polishing
    scored = []
    for U in candidates:
        if time.time() > deadline - 35:
            break
        U = sorted(set(U))
        if len(U) < 2:
            continue
        s = _score_set(U)
        scored.append((s, U))
        if s > best_score:
            best_score = s
            best_U = U[:]

    scored.sort(key=lambda x: -x[0])

    # Polish the best candidates using both spectral and standard local search
    seen = []
    for s, U in scored:
        if time.time() > deadline - 5:
            break
        key = tuple(U)
        if key in seen:
            continue
        seen.append(key)
        if s < best_score - 0.05:
            continue

        # Spectral local search
        if time.time() < deadline - 10:
            spec_U, spec_score = _spectral_local_search(U, max_iters=2000, deadline=deadline - 10)
            if spec_score > best_score:
                best_score = spec_score
                best_U = spec_U

        # Standard local search
        if time.time() < deadline - 5:
            std_U, std_score = _local_search(best_U if best_score > s else U, max_iters=3000, deadline=deadline - 5)
            if std_score > best_score:
                best_score = std_score
                best_U = std_U

        if len(seen) >= 5:
            break

    # Final polish: one more spectral pass on the best
    if time.time() < deadline - 3 and len(best_U) >= 3:
        final_U, final_score = _spectral_local_search(best_U, max_iters=1500, deadline=deadline - 3)
        if final_score > best_score:
            best_score = final_score
            best_U = final_U

    return sorted(set(best_U))
# EVOLVE-BLOCK-END
