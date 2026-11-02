"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
from itertools import combinations, product
from scipy.optimize import differential_evolution

def _is_prime(n):
    """Check if n is prime."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _primes_up_to(limit):
    """Generate all primes up to limit."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

def _score_set(U):
    """Compute the score for a finite set U containing 0."""
    U_set = set(U)
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 1.0
    
    # Use numpy FFT-based approach for larger sets
    if n > 20:
        return _score_set_fft(U_set)
    
    # Use set operations for efficiency
    U_list = sorted(U_set)
    U_arr = list(U_list)
    
    sumset = set()
    diffset = set()
    
    for a in U_arr:
        for b in U_arr:
            sumset.add(a + b)
            diffset.add(a - b)
            
    if not sumset:
        return 0.0
        
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    if ratio == 1.0:
        return 1.0
        
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 1.0
        
    return 1.0 + log_ratio / log_denom

def _score_set_fft(U_set):
    """Compute score using FFT for larger sets."""
    U_list = sorted(U_set)
    n = len(U_list)
    max_u = max(U_list)
    
    if max_u == 0:
        return 1.0
    
    # Sumset via convolution: indicator * indicator, support in [0, 2*max_u]
    sum_size = 2 * max_u + 1
    sum_fft = np.zeros(sum_size, dtype=np.float64)
    for u in U_list:
        sum_fft[u] = 1.0
    sum_conv = np.fft.fft(sum_fft)
    sum_conv *= sum_conv
    sum_result = np.real(np.fft.ifft(sum_conv))
    sumset_count = int(np.sum(sum_result > 0.5))
    
    # Difference set: U - U has support in [-max_u, max_u], size 2*max_u+1
    # Use autocorrelation: correlate U with reversed U
    # diff[x] = sum_u 1[u in U] * 1[u+x in U] for x in [-max_u, max_u]
    # Equivalently: autocorrelation of indicator
    diff_size = 2 * max_u + 1
    diff_fft = np.zeros(diff_size, dtype=np.float64)
    for u in U_list:
        diff_fft[u] = 1.0
    diff_conv = np.fft.fft(diff_fft)
    diff_conv *= np.conj(diff_conv)
    diff_result = np.real(np.fft.ifft(diff_conv))
    # diff_result[k] corresponds to difference (k - max_u) for k in [0, 2*max_u]
    # Actually autocorrelation: r[k] = sum_j f[j] * f[j+k], so support is [-max_u, max_u]
    # The ifft gives circular correlation; for our size it should be fine
    diffset_count = int(np.sum(diff_result > 0.5))
    
    if sumset_count == 0:
        return 0.0
        
    ratio = diffset_count / sumset_count
    if ratio <= 0:
        return 0.0
    if ratio == 1.0:
        return 1.0
        
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 1.0
        
    return 1.0 + log_ratio / log_denom

def _search_small_sets(max_n=6, max_val=35, deadline=None):
    """Exhaustive search for small sets U."""
    best_score = 0.0
    best_U = [0, 1, 3]
    
    start_time = time.time()
    
    for n in range(3, max_n + 1):
        # Generate combinations of n-1 numbers from 1 to max_val
        # For n=6, max_val=35: C(35,5) = 324,632 - manageable
        # For n=7, we'd need a tighter max_val
        
        for combo in combinations(range(1, max_val + 1), n - 1):
            if deadline and time.time() > deadline:
                return best_U, best_score
                
            U = [0] + list(combo)
            U.sort()
            
            # Calculate score
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
                
    return best_U, best_score

def _prime_step_construction(deadline=None):
    """Construct sets using sparse arithmetic progressions with prime-step gaps.
    
    Uses U = {0, p1, p1+p2, p1+p2+p3, ...} where pi are distinct primes,
    exploiting additive independence to minimize sumset collisions while
    maximizing difference set diversity.
    """
    best_score = 0.0
    best_U = [0, 1, 3]
    
    # Generate primes for use
    primes = _primes_up_to(200)
    primes = [p for p in primes if p >= 2]
    
    # Strategy 1: Greedy prime-step construction
    # Start with {0}, greedily add next element to maximize score
    random.seed(123)
    
    for trial in range(30):
        if deadline and time.time() > deadline:
            break
        
        # Choose a random subset of primes to use
        num_primes = random.randint(3, min(8, len(primes)))
        selected_primes = random.sample(primes[:50], num_primes)
        random.shuffle(selected_primes)
        
        # Build cumulative sum set
        U = [0]
        current = 0
        for p in selected_primes:
            current += p
            U.append(current)
        
        U.sort()
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
        
        # Also try with some primes doubled or with small adjustments
        U2 = [0]
        current = 0
        for i, p in enumerate(selected_primes):
            if i % 3 == 0:
                current += p * 2
            else:
                current += p
            U2.append(current)
        U2.sort()
        if len(set(U2)) == len(U2):
            s2 = _score_set(U2)
            if s2 > best_score:
                best_score = s2
                best_U = U2[:]
    
    # Strategy 2: Mian-Chowla-like sequence with prime gaps
    # The Mian-Chowla sequence has all pairwise sums and differences distinct
    # We modify it by using prime-step increments
    mc_base = [0, 1, 3]  # Start with known good base
    mc_primes = _primes_up_to(100)
    mc_primes = [p for p in mc_primes if p >= 5]
    
    for start_idx in range(min(5, len(mc_primes))):
        if deadline and time.time() > deadline:
            break
        
        U = [0]
        current = 0
        for i in range(start_idx, min(start_idx + 7, len(mc_primes))):
            current += mc_primes[i]
            U.append(current)
        
        U.sort()
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Strategy 3: Mixed construction - combine AP segments with prime jumps
    for trial in range(20):
        if deadline and time.time() > deadline:
            break
        
        U = [0]
        current = 0
        num_segments = random.randint(2, 5)
        
        for seg in range(num_segments):
            # Add a small AP segment
            seg_len = random.randint(1, 3)
            step = random.randint(1, 3)
            for j in range(seg_len):
                current += step
                U.append(current)
            
            # Then a prime jump
            if seg < num_segments - 1:
                p = random.choice(primes[:30])
                current += p
                U.append(current)
        
        U.sort()
        if len(set(U)) == len(U) and 0 in U:
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
    
    return best_U, best_score

def _construct_heuristic_sets(deadline=None):
    """Construct sets based on known good structures."""
    candidates = []
    
    # Structure 1: Almost AP
    # {0, 1, ..., k-2, M}
    for k in range(3, 10):
        for M in range(k, 50):
            if deadline and time.time() > deadline:
                break
            U = list(range(k - 1)) + [M]
            if len(set(U)) == len(U):
                candidates.append(U)
                
    # Structure 2: Golomb rulers (might have R < 1, but good to check)
    # Known optimal Golomb rulers
    known_gr = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 22],
        [0, 1, 4, 10, 12, 17, 22, 28],
    ]
    candidates.extend(known_gr)
    
    # Structure 3: Randomized local search on AP
    random.seed(42)
    for _ in range(50):
        if deadline and time.time() > deadline:
            break
        n = random.randint(3, 8)
        base_ap = list(range(n))
        # Perturb one element
        i = random.randint(1, n - 1)
        new_val = random.randint(1, 60)
        U = base_ap[:]
        U[i] = new_val
        if len(set(U)) == len(U) and 0 in U:
            candidates.append(sorted(set(U)))
    
    # Structure 4: Prime-step sparse constructions
    primes = _primes_up_to(150)
    primes = [p for p in primes if p >= 2]
    
    for trial in range(30):
        if deadline and time.time() > deadline:
            break
        # Random prime-step construction
        num_steps = random.randint(3, 7)
        selected = random.sample(primes[:40], num_steps)
        U = [0]
        current = 0
        for p in selected:
            current += p
            U.append(current)
        U.sort()
        if len(set(U)) == len(U):
            candidates.append(U)
    
    # Structure 5: Sparse AP with large gaps
    for gap in [5, 7, 11, 13, 17]:
        for start in range(0, 10):
            U = [0]
            current = 0
            for i in range(start, start + 5):
                current += gap * (i + 1)
                U.append(current)
            if len(set(U)) == len(U):
                candidates.append(U)
    
    return candidates

def _local_search(U, max_iters=300, deadline=None):
    """Improve set U via local search with add/remove/move operations."""
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 40
    
    # Precompute primes for guided moves
    primes = _primes_up_to(upper_bound + 20)
    primes = [p for p in primes if p >= 2]
    
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False
        
        # Try moving one element - prioritize prime-step moves
        for i in range(len(best_U)):
            if improved:
                break
            old_val = best_U[i]
            if old_val == 0:
                continue
            
            # Generate candidate new values: nearby values + prime offsets
            candidates = set()
            # Nearby values
            for delta in range(-5, 6):
                nv = old_val + delta
                if 1 <= nv <= upper_bound:
                    candidates.add(nv)
            # Prime offsets from current position
            for p in primes[:20]:
                nv = old_val + p
                if 1 <= nv <= upper_bound:
                    candidates.add(nv)
                nv = old_val - p
                if 1 <= nv <= upper_bound:
                    candidates.add(nv)
            # Random exploration
            for _ in range(5):
                nv = random.randint(1, upper_bound)
                candidates.add(nv)
            
            for new_val in candidates:
                if new_val == old_val:
                    continue
                new_U = best_U[:]
                new_U[i] = new_val
                if len(set(new_U)) != len(new_U):
                    continue
                s = _score_set(new_U)
                if s > best_score + 1e-10:
                    best_score = s
                    best_U = new_U[:]
                    improved = True
                    break
        
        if improved:
            continue
            
        # Try adding a new element - guided by prime steps
        existing = set(best_U)
        add_candidates = set()
        # Prime offsets from existing elements
        for u in best_U:
            for p in primes[:15]:
                nv = u + p
                if 1 <= nv <= upper_bound and nv not in existing:
                    add_candidates.add(nv)
        # Nearby gaps
        for i in range(len(best_U) - 1):
            mid = (best_U[i] + best_U[i+1]) // 2
            if mid > best_U[i] and mid < best_U[i+1] and mid not in existing:
                add_candidates.add(mid)
        # Random
        for _ in range(10):
            nv = random.randint(1, upper_bound)
            if nv not in existing:
                add_candidates.add(nv)
        
        for new_val in add_candidates:
            if new_val in existing:
                continue
            new_U = best_U[:] + [new_val]
            new_U.sort()
            s = _score_set(new_U)
            if s > best_score + 1e-10:
                best_score = s
                best_U = new_U[:]
                improved = True
                break
        
        if improved:
            continue
            
        # Try removing an element (if size > 3)
        if len(best_U) > 3:
            for i in range(1, len(best_U)):
                new_U = best_U[:i] + best_U[i+1:]
                s = _score_set(new_U)
                if s > best_score + 1e-10:
                    best_score = s
                    best_U = new_U[:]
                    improved = True
                    break
        
        if not improved:
            break
            
    return sorted(set(best_U)), best_score

def _de_surrogate(x, grid_max, target_size):
    """Smooth surrogate for DE optimization.
    
    x is a vector of length grid_max+1, values in [0,1].
    We round to get a binary indicator, then compute score.
    To make it smooth-ish, we use a soft threshold.
    """
    # Round to binary
    indicator = (x > 0.5).astype(float)
    # Force position 0 to be included
    indicator[0] = 1.0
    
    U_list = [i for i, v in enumerate(indicator) if v > 0.5]
    if len(U_list) < 2:
        return -1.0
    if len(U_list) > target_size + 5:
        # Penalize too many elements
        return -1.0 - (len(U_list) - target_size) * 0.1
    
    U_set = set(U_list)
    score = _score_set(U_set)
    
    # Add small penalty for deviation from target size to guide search
    size_penalty = abs(len(U_list) - target_size) * 0.01
    return score - size_penalty

def _de_continuous_search(deadline, target_size=8, grid_max=60):
    """Use differential evolution on a continuous relaxation of the indicator vector."""
    if time.time() > deadline - 20:
        return None, 0.0
    
    # Grid: positions 0 to grid_max
    bounds = [(0.0, 1.0)] * (grid_max + 1)
    
    # Objective: maximize score, so we minimize negative score
    def objective(x):
        if time.time() > deadline - 15:
            return 100.0  # Large penalty to stop
        return -_de_surrogate(x, grid_max, target_size)
    
    try:
        result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=50,
            popsize=15,
            tol=0.01,
            seed=42,
            polish=False,
            mutation=(0.3, 1.0),
            recombination=0.7
        )
        
        if result.success or result.fun < 50:
            x_opt = result.x
            indicator = (x_opt > 0.5).astype(int)
            indicator[0] = 1
            U_list = [i for i, v in enumerate(indicator) if v > 0]
            if len(U_list) >= 2:
                U_set = set(U_list)
                score = _score_set(U_set)
                return sorted(U_set), score
    except Exception:
        pass
    
    return None, 0.0

def _multi_scale_de_search(deadline):
    """Run DE on multiple grid scales to explore different set sizes and ranges."""
    best_U = None
    best_score = 0.0
    
    configs = [
        (6, 40),    # small set, small range
        (8, 50),    # medium set
        (10, 70),   # larger set
        (12, 90),   # even larger
    ]
    
    for target_size, grid_max in configs:
        if time.time() > deadline - 15:
            break
        U, s = _de_continuous_search(deadline, target_size=target_size, grid_max=grid_max)
        if U is not None and s > best_score:
            best_score = s
            best_U = U
    
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    # 1. Exhaustive search for small sets
    try:
        U_sol, s_sol = _search_small_sets(max_n=6, max_val=35, deadline=deadline)
        if s_sol > best_score:
            best_score = s_sol
            best_U = U_sol
    except Exception:
        pass
        
    # 2. Prime-step construction
    if time.time() < deadline - 30:
        try:
            U_prime, s_prime = _prime_step_construction(deadline=deadline)
            if s_prime > best_score:
                best_score = s_prime
                best_U = U_prime
        except Exception:
            pass
    
    # 3. Heuristic candidates
    candidates = _construct_heuristic_sets(deadline=deadline)
    for U in candidates:
        if deadline and time.time() > deadline:
            break
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
            
    # 4. DE continuous relaxation search (new)
    if time.time() < deadline - 25:
        try:
            U_de, s_de = _multi_scale_de_search(deadline)
            if U_de is not None and s_de > best_score:
                best_score = s_de
                best_U = U_de
        except Exception:
            pass
    
    # 5. Local search on best found
    if time.time() < deadline - 10:
        best_U, best_score = _local_search(best_U, max_iters=300, deadline=deadline)
        
    # 6. Final polish with multiple restarts
    if time.time() < deadline - 5:
        restarts = 0
        while time.time() < deadline - 5 and restarts < 5:
            restarts += 1
            perturbed = best_U[:]
            if len(perturbed) > 3:
                num_changes = random.randint(1, min(3, len(perturbed) - 1))
                indices = random.sample(range(1, len(perturbed)), num_changes)
                for idx in indices:
                    old_val = perturbed[idx]
                    new_val = old_val + random.randint(-10, 10)
                    if new_val > 0 and new_val not in perturbed:
                        perturbed[idx] = new_val
                perturbed.sort()
                perturbed = list(set(perturbed))
                if 0 in perturbed and len(perturbed) >= 3:
                    s = _score_set(perturbed)
                    if s > best_score:
                        best_score = s
                        best_U = perturbed[:]
            
            if time.time() < deadline - 5:
                best_U, best_score = _local_search(best_U, max_iters=150, deadline=deadline)
        
    return sorted(set(best_U))
# EVOLVE-BLOCK-END
