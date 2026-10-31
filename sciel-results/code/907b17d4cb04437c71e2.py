import math
import time
import random
import numpy as np

def construct():
    START = time.time()
    DEADLINE = START + 145  # 145s safety margin before 150s limit

    def elapsed():
        return time.time() - START

    def time_ok(limit_frac=1.0):
        return elapsed() < DEADLINE - 5 * limit_frac

    score_cache = {}

    def compute_score(u):
        """Compute score with caching. Uses numpy for larger sets."""
        u = sorted(set(u))
        if not u or 0 not in u:
            return -1.0
        key = tuple(u)
        if key in score_cache:
            return score_cache[key]
        
        n = len(u)
        if n == 0:
            score_cache[key] = -1.0
            return -1.0
        if 0 not in u:
            score_cache[key] = -1.0
            return -1.0
        max_u = int(u[-1])
        m = 2 * max_u + 1
        if m <= 1:
            score_cache[key] = 1.0
            return 1.0
        
        if n <= 6:
            # Small set: use pure Python sets
            sumset = set()
            diffset = set()
            for a in u:
                for b in u:
                    sumset.add(a + b)
                    diffset.add(a - b)
            sumset_size = len(sumset)
            diffset_size = len(diffset)
        else:
            # Larger set: use numpy
            u_arr = np.array(u, dtype=np.int64)
            sums = u_arr[:, None] + u_arr[None, :]
            sumset_size = len(np.unique(sums))
            diffs = u_arr[:, None] - u_arr[None, :]
            diffset_size = len(np.unique(diffs))
        
        r = diffset_size / sumset_size
        if r <= 0:
            score_cache[key] = -1.0
            return -1.0
        score = 1.0 + math.log(r) / math.log(m)
        score_cache[key] = score
        return score

    def is_golomb(u):
        u = sorted(u)
        diffs = set()
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                d = u[j] - u[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    # Known optimal/near-optimal Golomb rulers
    golomb_candidates = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25],       # n=7 optimal
        [0, 1, 4, 10, 18, 23, 29, 34],   # n=8 optimal
        [0, 1, 4, 10, 18, 23, 29, 37, 45], # n=9 (near-optimal)
        [0, 1, 4, 10, 18, 23, 29, 37, 45, 52], # n=10 (near-optimal)
    ]

    best_set = [0, 1, 3]
    best_score = compute_score(best_set)

    # Evaluate all known Golomb rulers
    for u in golomb_candidates:
        if not time_ok():
            break
        s = compute_score(u)
        if s > best_score:
            best_score = s
            best_set = list(u)

    # Simulated Annealing with FFT-based scoring for larger sets
    def fft_score(u):
        """Compute sumset and diffset sizes using FFT for large sets."""
        if not u:
            return -1.0
        max_u = max(u)
        m = 2 * max_u + 1
        if m <= 1:
            return 1.0
        
        # Create indicator array
        size = 2 * max_u + 1
        arr = np.zeros(size, dtype=np.float64)
        for x in u:
            arr[x] = 1.0
        
        # FFT for sumset (convolution)
        fft_arr = np.fft.fft(arr, n=2 * size)
        conv = np.fft.ifft(fft_arr * fft_arr).real
        sumset_size = int(np.sum(conv > 0.5))
        
        # For diffset, use cross-correlation
        # diffset = {a - b : a,b in u}
        # This is equivalent to the support of the autocorrelation
        # autocorr[k] = sum_x arr[x] * arr[x-k]
        # Use FFT: ifft(fft(arr) * conj(fft(arr)))
        autocorr = np.fft.ifft(fft_arr * np.conj(fft_arr)).real
        diffset_size = int(np.sum(autocorr > 0.5))
        
        r = diffset_size / sumset_size
        if r <= 0:
            return -1.0
        return 1.0 + math.log(r) / math.log(m)

    def simulated_annealing(start_u, time_budget=30):
        """Simulated annealing to explore the space of sets."""
        start_t = time.time()
        current_u = sorted(set(start_u))
        current_score = compute_score(current_u)
        best_u = list(current_u)
        best_s = current_score
        
        T = 0.5  # Initial temperature
        cooling = 0.995
        max_u_current = max(current_u)
        
        while time.time() - start_t < time_budget and time_ok():
            # Generate neighbor
            neighbor = list(current_u)
            move_type = random.random()
            
            if move_type < 0.4 and len(neighbor) < 15:
                # Add an element
                max_cur = max(neighbor)
                new_val = random.randint(max_cur + 1, max_cur + len(neighbor) * len(neighbor) + 5)
                neighbor.append(new_val)
                neighbor = sorted(set(neighbor))
            elif move_type < 0.7 and len(neighbor) > 2:
                # Remove an element (not 0)
                non_zero = [i for i in range(len(neighbor)) if neighbor[i] != 0]
                if non_zero:
                    idx = random.choice(non_zero)
                    neighbor.pop(idx)
                    neighbor = sorted(set(neighbor))
            else:
                # Shift an element
                non_zero = [i for i in range(len(neighbor)) if neighbor[i] != 0]
                if non_zero:
                    idx = random.choice(non_zero)
                    delta = random.randint(-3, 3)
                    new_val = neighbor[idx] + delta
                    if new_val >= 1:
                        neighbor[idx] = new_val
                        neighbor = sorted(set(neighbor))
            
            if len(neighbor) < 2 or 0 not in neighbor:
                continue
            
            new_score = compute_score(neighbor)
            delta_s = new_score - current_score
            
            # Acceptance criterion
            if delta_s > 0 or random.random() < math.exp(delta_s / T):
                current_u = neighbor
                current_score = new_score
                if new_score > best_s:
                    best_s = new_score
                    best_u = list(current_u)
            
            T *= cooling
        
        return best_u, best_s

    # Try local search from the best Golomb ruler found
    def local_search(start_u, time_budget=20):
        """Hill-climbing local search on the score function."""
        start_t = time.time()
        current_u = sorted(set(start_u))
        current_score = compute_score(current_u)
        best_u = list(current_u)
        best_s = current_score
        
        while time.time() - start_t < time_budget and time_ok():
            improved = False
            
            # Neighborhood 1: add a new element
            max_u = max(current_u)
            search_hi = max_u + len(current_u) * len(current_u) + 10
            for new_val in range(max_u + 1, search_hi + 1):
                candidate = sorted(current_u + [new_val])
                s = compute_score(candidate)
                if s > current_score + 1e-12:
                    current_u = candidate
                    current_score = s
                    if s > best_s:
                        best_s = s
                        best_u = list(candidate)
                    improved = True
                    break
            
            # Neighborhood 2: remove an element (if len > 2)
            if len(current_u) > 2:
                for i in range(len(current_u)):
                    if current_u[i] == 0:
                        continue
                    candidate = [x for j, x in enumerate(current_u) if j != i]
                    if len(candidate) >= 2:
                        s = compute_score(candidate)
                        if s > current_score + 1e-12:
                            current_u = candidate
                            current_score = s
                            if s > best_s:
                                best_s = s
                                best_u = list(candidate)
                            improved = True
                            break
            
            # Neighborhood 3: shift a non-zero element by small delta
            if not improved:
                for i in range(len(current_u)):
                    if current_u[i] == 0:
                        continue
                    for delta in [-3, -2, -1, 1, 2, 3]:
                        new_val = current_u[i] + delta
                        if new_val < 1:
                            continue
                        candidate = list(current_u)
                        candidate[i] = new_val
                        candidate = sorted(set(candidate))
                        if 0 not in candidate or len(candidate) != len(current_u):
                            continue
                        s = compute_score(candidate)
                        if s > current_score + 1e-12:
                            current_u = candidate
                            current_score = s
                            if s > best_s:
                                best_s = s
                                best_u = list(candidate)
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return best_u, best_s

    # Run local search from the best Golomb ruler
    if time_ok(2.0):
        ls_u, ls_s = local_search(best_set, time_budget=15)
        if ls_s > best_score:
            best_score = ls_s
            best_set = ls_u

    # Run simulated annealing from multiple starting points
    if time_ok(3.0):
        sa_u, sa_s = simulated_annealing(best_set, time_budget=20)
        if sa_s > best_score:
            best_score = sa_s
            best_set = sa_u

    # Try local search from other starting points
    if time_ok(3.0):
        for u in golomb_candidates[:5]:
            if not time_ok(3.0):
                break
            ls_u, ls_s = local_search(u, time_budget=8)
            if ls_s > best_score:
                best_score = ls_s
                best_set = ls_u

    # Run more simulated annealing with different starting points
    if time_ok(4.0):
        for start in golomb_candidates[3:6]:
            if not time_ok(4.0):
                break
            sa_u, sa_s = simulated_annealing(start, time_budget=10)
            if sa_s > best_score:
                best_score = sa_s
                best_set = sa_u

    # Final polish: try to improve with a tighter local search
    if time_ok(2.0):
        # Do a final local search with larger deltas
        current_u = sorted(set(best_set))
        current_score = compute_score(current_u)
        for _ in range(50):
            if not time_ok(2.0):
                break
            improved = False
            for i in range(len(current_u)):
                if current_u[i] == 0:
                    continue
                for delta in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
                    new_val = current_u[i] + delta
                    if new_val < 1:
                        continue
                    candidate = list(current_u)
                    candidate[i] = new_val
                    candidate = sorted(set(candidate))
                    if 0 not in candidate or len(candidate) != len(current_u):
                        continue
                    s = compute_score(candidate)
                    if s > current_score + 1e-12:
                        current_u = candidate
                        current_score = s
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        if current_score > best_score:
            best_score = current_score
            best_set = current_u

    # Final: ensure we return a valid set containing 0
    best_set = sorted(set(best_set))
    if 0 not in best_set:
        best_set = [0] + best_set
    
    return best_set
