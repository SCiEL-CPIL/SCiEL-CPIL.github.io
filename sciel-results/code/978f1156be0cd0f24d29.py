"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from numpy.fft import fft, ifft

def _score_set_fft(U):
    """Compute the score for a finite set U using FFT for speed."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Compute sumset and difference set using FFT
    total_len = 4 * max_u + 1
    fft_len = 1
    while fft_len < total_len:
        fft_len *= 2
    
    indicator = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        indicator[u] = 1.0
    
    freq = fft(indicator)
    
    # Sumset
    sum_conv = ifft(freq * freq).real
    sumset_size = np.sum(sum_conv[:2*max_u+1] > 1e-9)
    
    # Difference set via autocorrelation
    autocorr = ifft(freq * np.conj(freq)).real
    diffset_size = 1
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
    else:
        return _score_set_fft(U)

def _spectral_local_search(U, max_iters=2000, deadline=None):
    """
    Spectral local search: optimize the set by working in Fourier domain.
    
    Key insight: The autocorrelation R[k] = sum_x f[x]f[x-k] determines |U-U|.
    We want to maximize the number of nonzero R[k] for k != 0 while keeping
    the convolution C[k] = sum_x f[x]f[k-x] sparse (few nonzero entries).
    
    Strategy: Try small perturbations in the spatial domain that correspond to
    smooth changes in the Fourier domain, guided by the gradient of the score.
    """
    U = sorted(set(U))
    n = len(U)
    max_u = max(U)
    best_score = _score_set(U)
    best_U = U[:]
    
    # Precompute FFT length
    total_len = 4 * max_u + 1
    fft_len = 1
    while fft_len < total_len:
        fft_len *= 2
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        improved = False
        
        # Compute current autocorrelation and convolution
        indicator = np.zeros(fft_len, dtype=np.float64)
        for u in U:
            indicator[u] = 1.0
        
        freq = fft(indicator)
        autocorr = ifft(freq * np.conj(freq)).real
        sum_conv = ifft(freq * freq).real
        
        # Try swapping elements with emphasis on reducing sumset collisions
        # while maintaining difference set diversity
        for i in range(len(U)):
            if U[i] == 0:
                continue
            
            old_val = U[i]
            candidates = set()
            
            # Focus on positions that reduce autocorrelation peaks
            # Try values that are "far" from existing elements in a spectral sense
            for offset in range(-50, 51):
                nv = old_val + offset
                if nv > 0:
                    candidates.add(nv)
            
            # Add candidates based on spectral analysis
            # Find positions where autocorrelation is already high (bad)
            # and try to move away from them
            high_autocorr_positions = set()
            for k in range(1, min(max_u + 1, 200)):
                if autocorr[k] > 2 * n * 0.1:  # relatively high autocorrelation
                    for u in U:
                        if u + k <= max_u + 100:
                            high_autocorr_positions.add(u + k)
                        if u - k > 0:
                            high_autocorr_positions.add(u - k)
            
            # Add some candidates avoiding high-autocorrelation positions
            for pos in high_autocorr_positions:
                for delta in [-2, -1, 1, 2]:
                    cand = pos + delta
                    if cand > 0:
                        candidates.add(cand)
            
            # Random candidates
            for _ in range(20):
                candidates.add(random.randint(1, max_u + 300))
            
            for new_val in sorted(candidates):
                if new_val == old_val:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                new_U_set = set(new_U)
                if len(new_U_set) != len(new_U):
                    continue
                
                # Quick spectral check before full scoring
                new_indicator = np.zeros(fft_len, dtype=np.float64)
                for u in new_U:
                    if u < fft_len:
                        new_indicator[u] = 1.0
                new_freq = fft(new_indicator)
                new_autocorr = ifft(new_freq * np.conj(new_freq)).real
                new_sum_conv = ifft(new_freq * new_freq).real
                
                new_sumset_size = np.sum(new_sum_conv[:2*max_u+1] > 1e-9)
                new_diffset_size = 1
                for k in range(1, max_u + 1):
                    if new_autocorr[k] > 1e-9:
                        new_diffset_size += 1
                    if new_autocorr[fft_len - k] > 1e-9:
                        new_diffset_size += 1
                
                if new_sumset_size > 0:
                    new_ratio = new_diffset_size / new_sumset_size
                    if new_ratio > 0:
                        new_log_ratio = math.log(new_ratio)
                        new_log_denom = math.log(2 * max_u + 1)
                        if new_log_denom > 0:
                            new_score = 1.0 + new_log_ratio / new_log_denom
                            if new_score > best_score + 1e-9:
                                best_score = new_score
                                best_U = new_U[:]
                                U = new_U[:]
                                improved = True
                                break
            
            if improved:
                break
        
        if not improved:
            # Try adding an element
            if len(U) < 400:
                for _ in range(40):
                    new_val = random.randint(1, max_u + 300)
                    if new_val in U:
                        continue
                    new_U = U + [new_val]
                    s = _score_set(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        improved = True
                        break
            
            if not improved:
                # Try removing an element
                if len(U) > 3:
                    for i in range(len(U)):
                        if U[i] == 0:
                            continue
                        new_U = U[:i] + U[i+1:]
                        s = _score_set(new_U)
                        if s > best_score + 1e-9:
                            best_score = s
                            best_U = new_U[:]
                            U = new_U[:]
                            improved = True
                            break
        
        if not improved:
            # Random restart-like perturbation
            if random.random() < 0.1:
                new_U = U[:]
                for _ in range(min(3, len(new_U))):
                    idx = random.randint(1, len(new_U) - 1)
                    new_U[idx] = random.randint(1, max_u + 300)
                new_U_set = set(new_U)
                if len(new_U_set) == len(new_U):
                    s = _score_set(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        improved = True
            
            if not improved:
                break
    
    return sorted(set(best_U)), best_score

def _polish_set(U, deadline=None):
    """Final high-precision polish pass with tighter tolerance."""
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    max_u = max(U)
    
    # Very fine-grained local search
    for pass_num in range(3):
        for i in range(len(U)):
            if U[i] == 0:
                continue
            old_val = U[i]
            # Try all nearby values with fine granularity
            for offset in range(-10, 11):
                nv = old_val + offset
                if nv <= 0 or nv == old_val:
                    continue
                new_U = U[:]
                new_U[i] = nv
                new_U_set = set(new_U)
                if len(new_U_set) != len(new_U):
                    continue
                s = _score_set(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U[:]
                    U = new_U[:]
        
        if deadline and time.time() > deadline:
            break
    
    return sorted(set(best_U)), best_score

def _golomb_ruler(n):
    """Generate a Golomb ruler of order n using greedy construction."""
    if n <= 1:
        return [0]
    known = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 10, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 22],
        8: [0, 1, 4, 10, 12, 17, 22, 29],
        9: [0, 1, 4, 10, 12, 17, 22, 29, 36],
        10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
    }
    if n in known:
        return known[n]
    ruler = [0]
    for i in range(1, n):
        pos = 1
        while True:
            candidate = ruler + [pos]
            diffs = set()
            valid = True
            for j in range(len(candidate)):
                for k in range(j + 1, len(candidate)):
                    d = candidate[k] - candidate[j]
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

def _local_search(U, max_iters=5000, deadline=None):
    """Improve set U via local search (swap elements)."""
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 200
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False
        for i in range(len(U)):
            if U[i] == 0:
                continue
            old_val = U[i]
            candidates = set()
            for offset in range(-30, 31):
                nv = old_val + offset
                if nv > 0:
                    candidates.add(nv)
            for _ in range(15):
                candidates.add(random.randint(1, upper_bound))
            
            for new_val in sorted(candidates):
                if new_val == old_val:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                new_U_set = set(new_U)
                if len(new_U_set) != len(new_U):
                    continue
                s = _score_set(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U[:]
                    U = new_U[:]
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            if len(U) < 300:
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
                        improved = True
                        break
            
            if not improved:
                if len(U) > 2:
                    for i in range(len(U)):
                        if U[i] == 0:
                            continue
                        new_U = U[:i] + U[i+1:]
                        s = _score_set(new_U)
                        if s > best_score + 1e-9:
                            best_score = s
                            best_U = new_U[:]
                            U = new_U[:]
                            improved = True
                            break
        
        if not improved:
            if random.random() < 0.15:
                new_U = U[:]
                for _ in range(min(5, len(new_U))):
                    idx = random.randint(1, len(new_U) - 1)
                    new_U[idx] = random.randint(1, upper_bound)
                new_U_set = set(new_U)
                if len(new_U_set) == len(new_U):
                    s = _score_set(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        improved = True
            
            if not improved:
                break
    
    return sorted(set(best_U)), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    candidates = []
    
    # Golomb rulers
    for n in range(3, 30):
        if time.time() > deadline - 80:
            break
        gr = _golomb_ruler(n)
        candidates.append(gr)
    
    # Random sets
    random.seed(42)
    for trial in range(50):
        if time.time() > deadline - 80:
            break
        n = random.randint(3, 80)
        max_val = random.randint(20, 1200)
        U = [0] + random.sample(range(1, max_val + 1), n - 1)
        U = sorted(U)
        candidates.append(U)
    
    # Structured sets: arithmetic progressions
    for base in [2, 3, 5, 7, 11, 13, 17]:
        if time.time() > deadline - 80:
            break
        U = [0]
        for i in range(1, 30):
            U.append(i * base)
        candidates.append(U[:min(len(U), 30)])
    
    # Powers-based sets
    for p in [2, 3]:
        if time.time() > deadline - 80:
            break
        U = [0]
        for i in range(1, 18):
            U.append(p ** i)
            if U[-1] > 50000:
                break
        candidates.append(U)
    
    # Mixed structured sets
    for p in [2, 3]:
        if time.time() > deadline - 80:
            break
        U = [0]
        for i in range(1, 15):
            val = p ** i + random.randint(-10, 10)
            if val > 0 and val not in U:
                U.append(val)
        candidates.append(sorted(U))
    
    # Evaluate candidates
    for U in candidates:
        if time.time() > deadline - 40:
            break
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Spectral local search on best candidate
    if time.time() < deadline - 20:
        best_U, best_score = _spectral_local_search(best_U, max_iters=3000, deadline=deadline - 20)
    
    # Try spectral local search on other promising candidates
    for U in candidates[:15]:
        if time.time() > deadline - 20:
            break
        if U == best_U:
            continue
        s = _score_set(U)
        if s > best_score - 0.05:
            improved_U, improved_score = _spectral_local_search(U, max_iters=2000, deadline=deadline - 20)
            if improved_score > best_score:
                best_score = improved_score
                best_U = improved_U
    
    # Final polish pass
    if time.time() < deadline - 5:
        best_U, best_score = _polish_set(best_U, deadline=deadline - 5)
    
    return sorted(set(best_U))
# EVOLVE-BLOCK-END
