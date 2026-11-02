import numpy as np
import time
from itertools import combinations

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def score(h):
        """Compute the ratio: sum(h^2) / (sum(h) * max(autoconv(h)))"""
        h = np.maximum(h, 0)
        l1 = np.sum(h)
        if l1 < 1e-15:
            return 0.0
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        if max_conv < 1e-15:
            return 0.0
        return l2_sq / (l1 * max_conv)

    def score_fast(h, conv_cache=None):
        """Compute score with optional pre-computed convolution."""
        h = np.maximum(h, 0)
        l1 = np.sum(h)
        if l1 < 1e-15:
            return 0.0
        l2_sq = np.sum(h**2)
        if conv_cache is None:
            conv = np.convolve(h, h, mode='full')
        else:
            conv = conv_cache
        max_conv = np.max(conv)
        if max_conv < 1e-15:
            return 0.0
        return l2_sq / (l1 * max_conv)

    def find_maximal_sidon_set(n, k_target, rng):
        """Find a maximal Sidon set in [0, n) of size close to k_target.
        A Sidon set has all pairwise sums distinct (except trivial a+b=c+d when {a,b}={c,d})."""
        best_set = None
        best_size = 0
        
        for attempt in range(50):
            if time.time() > deadline - 10.0:
                return best_set
            
            # Greedy construction
            S = set()
            used_sums = set()
            
            # Start with a random spread
            candidates = list(range(n))
            rng.shuffle(candidates)
            
            # Try to build greedily
            for x in candidates:
                if time.time() > deadline - 10.0:
                    break
                # Check if adding x creates a collision
                valid = True
                for y in S:
                    s = x + y
                    if s in used_sums:
                        valid = False
                        break
                    used_sums.add(s)
                if valid:
                    S.add(x)
            
            S_list = sorted(S)
            if len(S_list) > best_size:
                best_size = len(S_list)
                best_set = S_list
            
            if best_size >= k_target:
                break
        
        # Local search: try to improve
        if best_set is not None:
            S = set(best_set)
            for _ in range(200):
                if time.time() > deadline - 10.0:
                    break
                improved = False
                # Try removing one element and adding two
                for a in list(S):
                    S.remove(a)
                    # Try adding pairs
                    for x in range(n):
                        if x in S:
                            continue
                        S.add(x)
                        valid = True
                        sums = set()
                        elems = sorted(S)
                        for i in range(len(elems)):
                            for j in range(i, len(elems)):
                                s = elems[i] + elems[j]
                                if s in sums:
                                    valid = False
                                    break
                                sums.add(s)
                            if not valid:
                                break
                        if valid and len(S) > len(best_set) if best_set else True:
                            best_set = sorted(S)
                            best_size = len(S)
                            improved = True
                            break
                        S.remove(x)
                    if improved:
                        break
                    S.add(a)
                if not improved:
                    break
        
        return best_set

    def verify_sidon(S):
        """Check if S is a Sidon set (all pairwise sums are distinct up to ordering)."""
        S = sorted(S)
        sums = {}
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s in sums:
                    # Check if it's the same pair
                    prev_i, prev_j = sums[s]
                    if (prev_i, prev_j) != (i, j) and (prev_j, prev_i) != (i, j):
                        return False
                else:
                    sums[s] = (i, j)
        return True

    def optimize_amplitudes(S, max_iter=2000, tol=1e-14):
        """Optimize amplitudes for a fixed support set S using projected gradient on log scale."""
        k = len(S)
        if k == 0:
            return np.zeros(N), 0.0
        
        S_arr = np.array(S)
        S_set = set(S)
        
        # Initialize with equal weights
        h = np.zeros(N)
        h[S] = 1.0 / k
        
        best_h = h.copy()
        best_s = score(h)
        
        # Compute initial convolution
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        peak_idx = np.argmax(conv)
        shift = peak_idx - (N - 1)
        
        # Adaptive learning rate
        lr = 0.1
        prev_s = best_s
        
        for it in range(max_iter):
            if time.time() > deadline - 8.0:
                break
            
            l1 = np.sum(h)
            l2_sq = np.sum(h**2)
            
            if l2_sq < 1e-15 or l1 < 1e-15:
                break
            
            # Recompute convolution and peak
            conv = np.convolve(h, h, mode='full')
            max_conv = np.max(conv)
            peak_idx = np.argmax(conv)
            shift = peak_idx - (N - 1)
            
            if max_conv < 1e-15:
                break
            
            # Gradient of f = l2_sq / (l1 * max_conv)
            # Use log gradient: grad log f = 2h/l2_sq - 1/l1 - d_peak/max_conv
            d_peak = np.zeros(N)
            for i in S:
                j1 = i + shift
                j2 = i - shift
                if 0 <= j1 < N and j1 in S_set:
                    d_peak[i] += h[j1]
                if 0 <= j2 < N and j2 in S_set:
                    d_peak[i] += h[j2]
            
            grad = 2.0 * h / l2_sq - 1.0 / l1 - d_peak / max_conv
            
            # Project onto support
            grad_proj = np.zeros(N)
            grad_proj[S] = grad[S]
            
            # Multiplicative update
            h_new = h * np.exp(lr * grad_proj)
            h_new = np.maximum(h_new, 0)
            
            # Renormalize
            h_sum = np.sum(h_new)
            if h_sum < 1e-15:
                h = h / l1
                break
            h = h_new / h_sum
            
            s = score(h)
            if s > best_s:
                best_s = s
                best_h = h.copy()
            
            # Adaptive step size
            if s < prev_s - 1e-10:
                lr *= 0.5
            elif it > 10 and abs(s - prev_s) < 1e-12:
                break
            else:
                lr = min(lr * 1.05, 0.5)
            prev_s = s
        
        return best_h, best_s

    def coordinate_ascent(h, max_rounds=200):
        """Coordinate ascent refinement with golden section search on each coordinate."""
        h = h.copy()
        best_s = score(h)
        support = list(np.where(h > 1e-12)[0])
        
        for rnd in range(max_rounds):
            if time.time() > deadline - 5.0:
                break
            improved = False
            for i in support:
                if time.time() > deadline - 5.0:
                    break
                old_val = h[i]
                if old_val < 1e-15:
                    continue
                
                # Try a range of multipliers
                best_frac = 1.0
                best_trial_s = best_s
                for frac in [0.2, 0.4, 0.6, 0.8, 1.2, 1.5, 2.0, 3.0, 5.0]:
                    h_trial = h.copy()
                    h_trial[i] = old_val * frac
                    s = score(h_trial)
                    if s > best_trial_s:
                        best_trial_s = s
                        best_frac = frac
                
                if best_frac != 1.0:
                    h[i] = old_val * best_frac
                    best_s = best_trial_s
                    improved = True
            
            if not improved:
                break
        
        return h, best_s

    def try_add_remove(h, max_ops=100):
        """Try adding or removing support points."""
        h = h.copy()
        best_s = score(h)
        support = set(np.where(h > 1e-12)[0])
        
        for _ in range(max_ops):
            if time.time() > deadline - 5.0:
                break
            improved = False
            
            # Try removing a point
            if len(support) > 3:
                for i in list(support):
                    if time.time() > deadline - 5.0:
                        break
                    h_trial = h.copy()
                    h_trial[i] = 0
                    s = score(h_trial)
                    if s > best_s + 1e-13:
                        best_s = s
                        h = h_trial
                        support.remove(i)
                        improved = True
                        break
            
            # Try adding a point
            if not improved and len(support) < 25:
                for j in range(N):
                    if j in support:
                        continue
                    if time.time() > deadline - 5.0:
                        break
                    h_trial = h.copy()
                    h_trial[j] = 0.05 * np.sum(h) / max(len(support), 1)
                    s = score(h_trial)
                    if s > best_s + 1e-13:
                        best_s = s
                        h = h_trial
                        support.add(j)
                        improved = True
                        break
            
            if not improved:
                break
        
        return h, best_s

    # Main search
    best_h = np.zeros(N)
    best_score = 0.0
    rng = np.random.RandomState(42)

    # Phase 1: Find good Sidon sets of various sizes
    for k in range(4, 12):
        if time.time() > deadline - 60.0:
            break
        
        for seed in range(8):
            if time.time() > deadline - 60.0:
                break
            
            rng_sub = np.random.RandomState(seed * 1000 + k)
            S = find_maximal_sidon_set(N, k, rng_sub)
            
            if S is None or len(S) < 3:
                continue
            
            # Verify Sidon property
            if not verify_sidon(S):
                # Try to fix by removing conflicting elements
                S = sorted(S)
                for i in range(len(S)):
                    for j in range(i+1, len(S)):
                        if time.time() > deadline - 60.0:
                            break
                        # Check for conflicts
                # Just proceed if it's close enough
                pass
            
            h_opt, s = optimize_amplitudes(S, max_iter=1500)
            if s > best_score:
                best_score = s
                best_h = h_opt.copy()

    # Phase 2: Random Sidon-like sets
    for it in range(300):
        if time.time() > deadline - 30.0:
            break
        
        k = rng.randint(4, 12)
        # Generate random set and check Sidon property
        S = sorted(rng.choice(N, k, replace=False).tolist())
        
        if not verify_sidon(S):
            # Try to repair
            S_fixed = [S[0]]
            for x in S[1:]:
                S_trial = S_fixed + [x]
                if verify_sidon(S_trial):
                    S_fixed.append(x)
            S = S_fixed
        
        if len(S) < 3:
            continue
        
        h_opt, s = optimize_amplitudes(S, max_iter=800)
        if s > best_score:
            best_score = s
            best_h = h_opt.copy()

    # Phase 3: Structured sets based on known good patterns
    structured_sets = [
        list(range(0, 50, 7)),
        list(range(0, 50, 6)),
        [0, 1, 4, 10, 18, 23, 30, 39],
        [0, 1, 4, 10, 18, 25, 34, 44],
        [0, 2, 5, 11, 19, 26, 35, 45],
        [0, 1, 3, 7, 12, 18, 26, 35, 45],
        [0, 2, 5, 9, 14, 20, 27, 35, 44],
    ]
    
    for S in structured_sets:
        if time.time() > deadline - 20.0:
            break
        if len(S) < 3:
            continue
        h_opt, s = optimize_amplitudes(S, max_iter=2000)
        if s > best_score:
            best_score = s
            best_h = h_opt.copy()

    # Phase 4: Refinement
    if best_score > 0:
        best_h, best_score = coordinate_ascent(best_h, max_rounds=300)
        best_h, best_score = try_add_remove(best_h, max_ops=200)
        best_h, best_score = coordinate_ascent(best_h, max_rounds=300)
        best_h, best_score = try_add_remove(best_h, max_ops=100)
        best_h, best_score = coordinate_ascent(best_h, max_rounds=200)

    # Ensure valid output
    if best_score <= 0:
        best_h = np.zeros(N)
        best_h[0] = 1.0
    else:
        best_h = np.maximum(best_h, 0)
        s = np.sum(best_h)
        if s > 0:
            best_h = best_h / s
        else:
            best_h = np.zeros(N)
            best_h[0] = 1.0

    return best_h.tolist()
