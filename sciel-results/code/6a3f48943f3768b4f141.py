import math
import time
import random
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 150.0
_SAFETY_MARGIN = 15.0

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _TIME_LIMIT - _SAFETY_MARGIN - _elapsed()

def _score_fft(U):
    """
    Fast score calculation using FFT.
    U: list of non-negative integers.
    """
    U = sorted(set(int(x) for x in U if x >= 0))
    if len(U) < 2:
        return 0.0
    M = max(U)
    if M == 0:
        return 0.0
    
    # Determine FFT size
    L = 1
    while L < 2 * M + 1:
        L *= 2
        
    # Create indicator vector
    A = np.zeros(L, dtype=np.float64)
    for u in U:
        A[u] = 1.0
        
    # Sumset: A * A
    F_A = np.fft.fft(A)
    F_S = F_A * F_A
    S = np.fft.ifft(F_S).real
    
    # Count non-zero elements in S[0 : 2*M+1]
    sum_count = np.sum(S[:2*M+1] > 0.1)
    
    # Diffset: Use autocorrelation
    # Autocorrelation R[k] = sum A[i] A[i+k]
    # For differences, we look at the support of R.
    # R is centered at L//2 in 'full' mode, but here we use circular autocorr via FFT.
    # F_R = F_A * conj(F_A)
    F_R = F_A * np.conj(F_A)
    R = np.fft.ifft(F_R).real
    
    # The differences a-b range from -M to M.
    # In the circular convolution/autocorrelation of size L, the difference d corresponds to index d % L.
    # Since L > 2M, indices 0..M correspond to differences 0..M, and L-M..L-1 correspond to -M..-1.
    # We need the count of non-zero entries in R for these indices.
    diff_indices = list(range(0, M+1)) + list(range(L-M, L))
    diff_count = np.sum(R[diff_indices] > 0.1)
    
    denom = math.log(2 * M + 1, 2)
    if denom == 0:
        return 0.0
    return 1.0 + math.log(diff_count / sum_count, 2) / denom

def _score_small(U):
    """Exact O(n^2) score for small sets."""
    U = sorted(set(int(x) for x in U if x >= 0))
    if len(U) < 2:
        return 0.0
    M = max(U)
    if M == 0:
        return 0.0
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    denom = math.log(2 * M + 1, 2)
    if denom == 0:
        return 0.0
    return 1.0 + math.log(len(diffset) / len(sumset), 2) / denom

def _get_score(U):
    if max(U) < 1000:
        return _score_small(U)
    return _score_fft(U)

def _greedy_construct(n, max_val, rng):
    """
    Construct a set of size n by greedily adding elements that maximize the score increment.
    Start with {0}.
    """
    U = [0]
    current_score = 0.0
    
    # Precompute potential candidates? No, just sample from range.
    # To be efficient, we sample candidates from a range proportional to current max.
    
    for _ in range(n - 1):
        best_candidate = None
        best_score = -1.0
        
        # Determine search range for new element
        # Ideally, we want to spread elements out.
        # Let's sample from [1, max_val] but weighted towards larger values if max is small.
        # Actually, for Sidon/Golomb properties, larger gaps are better.
        # Let's sample from a range that grows.
        current_max = max(U)
        search_max = min(max_val, max(10, current_max * 2))
        
        # Sample K candidates
        K = 50
        candidates = rng.sample(range(1, search_max + 1), min(K, search_max))
        
        for c in candidates:
            if c in U:
                continue
            new_U = U + [c]
            new_U.sort()
            s = _get_score(new_U)
            if s > best_score:
                best_score = s
                best_candidate = c
        
        if best_candidate is not None:
            U.append(best_candidate)
            U.sort()
        else:
            # Fallback: random
            c = rng.randint(1, max_val)
            while c in U:
                c = rng.randint(1, max_val)
            U.append(c)
            U.sort()
            
    return U

def _simulated_annealing(initial_U, time_budget, rng):
    """
    Simulated Annealing with Swap moves.
    """
    U = list(initial_U)
    U.sort()
    n = len(U)
    current_score = _get_score(U)
    best_U = list(U)
    best_score = current_score
    
    # Temperature schedule
    # Start high, cool down
    T_start = 0.1
    T_end = 0.001
    steps = 2000 # Fixed number of steps to estimate budget usage
    
    # Estimate time per step
    # We need to be careful with the time budget.
    # Let's run a small batch to estimate.
    
    start_time = time.time()
    
    # Dynamic step count based on time
    # We'll run in loops until time runs out.
    
    while time.time() - start_time < time_budget and _remaining() > 1.0:
        # Pick two indices to swap
        if n < 2:
            break
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
        
        # Swap values
        old_i, old_j = U[i], U[j]
        U[i], U[j] = U[j], U[i]
        
        # Ensure 0 is still in the set (swapping doesn't remove 0, just moves it)
        # But we must ensure all elements are non-negative.
        # Since we only swap existing elements, this is guaranteed.
        
        new_score = _get_score(U)
        delta = new_score - current_score
        
        # Metropolis acceptance
        if delta > 0:
            # Accept
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_U = list(U)
        else:
            # Accept with probability exp(delta / T)
            # T needs to be scaled. Score differences are small.
            # Let's use a fixed T for now, or adapt.
            T = T_start * (T_end / T_start) ** ((time.time() - start_time) / time_budget)
            if T < 1e-6:
                T = 1e-6
            p = math.exp(delta / T)
            if rng.random() < p:
                current_score = new_score
            else:
                # Reject: swap back
                U[i], U[j] = U[j], U[i]
                
    return best_U, best_score

def construct():
    global _START_TIME
    _START_TIME = time.time()
    rng = random.Random(42)
    
    # 1. Generate diverse initial candidates using Greedy Construction
    # with different parameters (size, max_val)
    candidates = []
    
    # Try different sizes
    for n in [10, 15, 20, 30, 50]:
        # Max value should be large enough to allow spreading
        # For n=50, max val 1000 might be too small? 
        # Golomb ruler length is ~n^2. So for n=50, length ~2500.
        # Let's scale max_val with n^2.
        max_val = n * n * 2 
        try:
            U = _greedy_construct(n, max_val, rng)
            if 0 not in U:
                U.append(0)
            U.sort()
            candidates.append(U)
        except Exception:
            pass
            
    # 2. Add some random sets for diversity
    for n in [10, 20, 30]:
        max_val = n * n * 2
        U = [0]
        while len(U) < n:
            x = rng.randint(1, max_val)
            if x not in U:
                U.append(x)
        U.sort()
        candidates.append(U)
        
    # 3. Score all candidates
    scored = []
    for U in candidates:
        if _remaining() < 5.0:
            break
        s = _get_score(U)
        scored.append((s, U))
        
    if not scored:
        return [0, 1]
        
    scored.sort(key=lambda x: -x[0])
    best_s, best_U = scored[0]
    
    # 4. Run Simulated Annealing on top candidates
    # Allocate remaining time to SA
    # We have ~100s left. 
    # Run SA on the best 3 candidates.
    
    time_per_sa = min(20.0, _remaining() / 3.0)
    
    for s, U in scored[:3]:
        if _remaining() < time_per_sa + 2.0:
            break
            
        # Run SA
        res_U, res_s = _simulated_annealing(U, time_per_sa, rng)
        
        if res_s > best_s:
            best_s = res_s
            best_U = res_U
            
    # Final polish: Local search on the best result
    if _remaining() > 5.0:
        # Simple local search: try moving each element by small delta
        U = list(best_U)
        U.sort()
        n = len(U)
        current_s = best_s
        
        # Budget for local search
        ls_budget = min(10.0, _remaining() - 2.0)
        start_ls = time.time()
        
        improved = True
        while improved and time.time() - start_ls < ls_budget and _remaining() > 2.0:
            improved = False
            indices = list(range(n))
            rng.shuffle(indices)
            
            for i in indices:
                if _remaining() < 2.0:
                    break
                orig = U[i]
                # Try small moves
                # Delta range based on density
                max_delta = 5
                deltas = list(range(-max_delta, max_delta + 1))
                rng.shuffle(deltas)
                
                for d in deltas:
                    if d == 0:
                        continue
                    new_val = orig + d
                    if new_val < 0:
                        continue
                    if new_val in U and U.index(new_val) != i:
                        continue
                    
                    old_val = U[i]
                    U[i] = new_val
                    U.sort()
                    
                    new_s = _get_score(U)
                    
                    if new_s > current_s + 1e-10:
                        current_s = new_s
                        best_U = list(U)
                        best_s = current_s
                        improved = True
                        break
                    else:
                        U[i] = old_val
                        U.sort()
                        
    # Ensure validity
    best_U = sorted(set(int(x) for x in best_U if x >= 0))
    if 0 not in best_U:
        best_U.insert(0, 0)
        
    return best_U
