"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START

def _score_set_fft(U):
    """Compute score using FFT-based convolution for sumset/diffset sizes."""
    U = sorted(set(U))
    n = len(U)
    if n < 2 or 0 not in U:
        return -1.0
    max_u = U[-1]
    if max_u <= 0:
        return -1.0
    
    # Build indicator vector
    M = 2 * max_u + 1  # max possible sum is 2*max_u
    vec = np.zeros(M, dtype=np.float64)
    for x in U:
        vec[x] = 1.0
    
    # Sumset: |U+U| via FFT convolution
    size = 1
    while size < 2 * M:
        size <<= 1
    fft_vec = np.fft.rfft(vec, size)
    conv = np.fft.irfft(fft_vec * fft_vec, size)
    n_sums = int(np.sum(conv[:M] > 0.5))
    
    # Diffset: |U-U| via FFT autocorrelation
    auto = np.fft.irfft(fft_vec * np.conj(fft_vec), size)
    # auto[k] for k=0..max_u gives positive lags; symmetric
    n_pos = int(np.sum(auto[1:max_u+1] > 0.5))
    n_diffs = 1 + 2 * n_pos
    
    if n_sums == 0:
        return -1.0
    ratio = n_diffs / n_sums
    if ratio <= 1:
        return -1.0
    log_denom = math.log(2 * max_u + 1)
    if log_denom <= 0:
        return -1.0
    return 1.0 + math.log(ratio) / log_denom


def _score_set(U):
    """Compute score = 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)."""
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return -1.0
    if 0 not in U:
        return -1.0
    max_u = U[-1]
    if max_u <= 0:
        return -1.0
    
    # For small sets, use Python sets (faster overhead)
    if n <= 20:
        sum_set = set()
        diff_set = set()
        for a in U:
            for b in U:
                sum_set.add(a + b)
                diff_set.add(a - b)
        n_sums = len(sum_set)
        n_diffs = len(diff_set)
    elif max_u <= 200000:
        # Use FFT for medium/large sets
        return _score_set_fft(U)
    else:
        # Fallback to numpy for very large max_u
        U_arr = np.array(U, dtype=np.int64)
        sums = U_arr[:, None] + U_arr[None, :]
        diffs = U_arr[:, None] - U_arr[None, :]
        n_sums = len(np.unique(sums))
        n_diffs = len(np.unique(diffs))
    
    if n_sums == 0:
        return -1.0
    ratio = n_diffs / n_sums
    if ratio <= 1:
        return -1.0
    log_denom = math.log(2 * max_u + 1)
    if log_denom <= 0:
        return -1.0
    return 1.0 + math.log(ratio) / log_denom


def _golomb_seed(n, max_val):
    """Greedy Golomb ruler: pick n elements with distinct pairwise differences."""
    U = [0]
    diffs = set()
    for _ in range(1, n):
        best_v = None
        for v in range(1, max_val + 1):
            if v in U:
                continue
            ok = True
            for a in U:
                d = abs(v - a)
                if d in diffs:
                    ok = False
                    break
            if ok:
                best_v = v
                break
        if best_v is None:
            break
        U.append(best_v)
        for a in U:
            diffs.add(abs(best_v - a))
    return U


def _local_search(initial, deadline, max_steps=2000):
    """Local search: add/remove/swap elements to improve score."""
    best_U = list(initial)
    best_score = _score_set(best_U)
    current_U = list(best_U)
    current_score = best_score

    for step in range(max_steps):
        if time.time() > deadline:
            break
        max_val = max(current_U) if current_U else 0
        improved = False

        # Try adding a new element
        candidates = set()
        # Elements just above max
        for delta in range(1, 30):
            candidates.add(max_val + delta)
        # Elements around 2*max
        for delta in range(-10, 30):
            v = max_val * 2 + delta
            if v > 0:
                candidates.add(v)
        # Some interior values (multiples of gaps)
        if len(current_U) >= 2:
            gaps = [current_U[i+1] - current_U[i] for i in range(len(current_U)-1)]
            for g in gaps[:5]:
                for k in range(1, 5):
                    v = k * g
                    if v > 0 and v not in current_U:
                        candidates.add(v)
                    v2 = max_val + k * g
                    candidates.add(v2)
        # Random interior
        for _ in range(20):
            v = random.randint(1, max_val * 3)
            if v not in current_U:
                candidates.add(v)
        
        cand_list = list(candidates)
        random.shuffle(cand_list)

        for new_val in cand_list[:20]:
            if new_val <= 0:
                continue
            new_U = list(current_U) + [new_val]
            new_U.sort()
            new_score = _score_set(new_U)
            if new_score > current_score + 1e-10:
                current_U = new_U
                current_score = new_score
                if current_score > best_score:
                    best_U = list(current_U)
                    best_score = current_score
                improved = True
                break

        if not improved:
            # Try removing an element
            if len(current_U) > 2:
                remove_candidates = list(range(1, len(current_U)))
                random.shuffle(remove_candidates)
                for idx in remove_candidates[:10]:
                    new_U = [x for i, x in enumerate(current_U) if i != idx]
                    new_score = _score_set(new_U)
                    if new_score > current_score + 1e-10:
                        current_U = new_U
                        current_score = new_score
                        if current_score > best_score:
                            best_U = list(current_U)
                            best_score = current_score
                        improved = True
                        break

        if not improved and step > 30:
            if random.random() < 0.08:
                # Perturb: replace a random element
                if len(current_U) > 2:
                    idx = random.randint(1, len(current_U) - 1)
                    new_val = random.randint(1, max(max_val * 2, 50))
                    if new_val not in current_U:
                        current_U[idx] = new_val
                        current_U.sort()
                        current_score = _score_set(current_U)
            elif step > 100:
                break

    return best_U, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    # Phase 1: Systematic search over small structured sets
    # Geometric progressions {0, 1, b, b^2, ..., b^(n-1)}
    for b in range(2, 20):
        if time.time() > deadline - 30:
            break
        for n in range(2, 10):
            U = [0]
            val = 1
            for i in range(1, n):
                val *= b
                U.append(val)
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
    
    # {0, 1, a, b}
    for a in range(2, 20):
        if time.time() > deadline - 25:
            break
        for b in range(a + 1, 100):
            U = [0, 1, a, b]
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
    
    # {0, 1, a, b, c}
    for a in range(2, 12):
        if time.time() > deadline - 20:
            break
        for b in range(a + 1, 60):
            if time.time() > deadline - 20:
                break
            for c in range(b + 1, 200):
                U = [0, 1, a, b, c]
                s = _score_set(U)
                if s > best_score:
                    best_score = s
                    best_U = U[:]
                if time.time() > deadline - 20:
                    break
    
    # Phase 2: Golomb ruler seeds + local search
    golomb_seeds = []
    for n in range(4, 12):
        if time.time() > deadline - 30:
            break
        for max_val in [n * n, n * n + n, n * n + 2 * n, n * n * 2]:
            if time.time() > deadline - 30:
                break
            U = _golomb_seed(n, max_val)
            if len(U) >= 4:
                golomb_seeds.append(U)
                s = _score_set(U)
                if s > best_score:
                    best_score = s
                    best_U = U[:]
    
    # Phase 3: Local search from multiple starting points
    start_points = [best_U]
    # Add top Golomb seeds
    golomb_seeds.sort(key=lambda u: _score_set(u), reverse=True)
    start_points.extend(golomb_seeds[:3])
    # Add a few geometric seeds
    for b in [2, 3, 5]:
        U = [0]
        val = 1
        for i in range(1, 6):
            val *= b
            U.append(val)
        start_points.append(U)
    
    for sp in start_points:
        if time.time() > deadline - 5:
            break
        U, score = _local_search(sp, deadline - 5, max_steps=1500)
        if score > best_score:
            best_score = score
            best_U = U
    
    # Phase 4: FFT-enabled broad random search over larger sets
    if time.time() < deadline - 10:
        for trial in range(200):
            if time.time() > deadline - 10:
                break
            n_elem = random.randint(5, 30)
            max_val = random.randint(100, 50000)
            U_set = {0}
            while len(U_set) < n_elem:
                v = random.randint(1, max_val)
                U_set.add(v)
            U = sorted(U_set)
            s = _score_set_fft(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
    
    # Phase 5: Local search from best found so far
    if time.time() < deadline - 3:
        U, score = _local_search(best_U, deadline - 1, max_steps=1000)
        if score > best_score:
            best_score = score
            best_U = U
    
    # Phase 6: Final refinement - try to extend best_U
    if time.time() < deadline - 1:
        max_val = max(best_U)
        for _ in range(50):
            if time.time() > deadline - 1:
                break
            v = random.randint(1, max_val * 2)
            if v not in best_U:
                new_U = sorted(best_U + [v])
                s = _score_set(new_U)
                if s > best_score:
                    best_score = s
                    best_U = new_U
                    max_val = max(best_U)
    
    return sorted(best_U)

# EVOLVE-BLOCK-END
