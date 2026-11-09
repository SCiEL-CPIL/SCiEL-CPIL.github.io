import math
import time
import random
import numpy as np

def _score(U):
    U = sorted(list(set(U)))
    if 0 not in U:
        return 0.0
    if len(U) < 2:
        return 0.0
    mx = max(U)
    if mx == 0:
        return 0.0
    
    # Use FFT for fast cardinality computation
    n_plus, n_minus = _sumset_diffset_card(U)
    
    if n_plus == 0:
        return 0.0
        
    ratio = n_minus / n_plus
    if ratio <= 1:
        return 1.0
        
    try:
        val = 1.0 + math.log(ratio) / math.log(2 * mx + 1)
    except (ValueError, ZeroDivisionError):
        return 0.0
    return val

def _sumset_diffset_card(U):
    """Compute |U+U| and |U-U| using FFT convolution."""
    mx = max(U)
    # For sumset: values range 0 to 2*mx
    # For diffset: values range -mx to mx
    # We use indicator vectors and FFT
    
    # Sumset: A = indicator of U on [0, mx]
    # |U+U| = number of k in [0, 2*mx] where sum_{a in U} I(a=k-b) > 0 for some b in U
    # This is the support of A * A (convolution)
    
    # Use power of 2 for FFT efficiency
    n = 2 * mx + 1
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    
    A = np.zeros(fft_size)
    for u in U:
        A[u] = 1.0
    
    # Convolution for sumset
    A_fft = np.fft.rfft(A)
    conv = np.fft.irfft(A_fft * A_fft, n=fft_size)
    # Sumset support: positions 0 to 2*mx where conv > 0.5
    n_plus = int(np.sum(conv[:2*mx+1] > 0.5))
    
    # For diffset: |U-U| = number of k in [-mx, mx] where exists a,b in U with a-b=k
    # This is the support of A * A_flipped where A_flipped[i] = A[mx-i]
    # Actually, diffset: for each pair (a,b), we get a-b. 
    # Let B[i] = 1 if i in U. Then (B * B_rev)[k] = sum_j B[j] * B_rev[k-j] = sum_j B[j] * B[k-j+mx]
    # Hmm, let me think differently.
    # |U-U| = |{a-b : a,b in U}|
    # Let's use: define C[i] = 1 if i in U, for i in [0,mx]
    # Then for difference d = a-b, we need a = b+d, so sum over b of C[b]*C[b+d]
    # This is like a cross-correlation.
    # Alternatively: |U-U| = number of nonzero entries in the "difference matrix"
    # Simpler: just compute directly for small sets, use FFT for large
    
    # For efficiency, compute diffset via: 
    # Let D[k] = number of pairs (a,b) with a-b=k, for k in [-mx, mx]
    # D[k] = sum_{b} I(b in U) * I(b+k in U)
    # This is the autocorrelation of the indicator.
    # Using FFT: autocorr = ifft(|fft(C)|^2) gives circular autocorrelation
    # We need linear autocorrelation.
    
    # For small mx, direct computation is fine
    if mx <= 200:
        minus_set = set()
        for a in U:
            for b in U:
                minus_set.add(a - b)
        n_minus = len(minus_set)
    else:
        # Use FFT-based approach
        # Pad to avoid circular convolution issues
        fft_size2 = 1
        while fft_size2 < 2 * mx + 1:
            fft_size2 *= 2
        
        C = np.zeros(fft_size2)
        for u in U:
            C[u] = 1.0
        
        C_fft = np.fft.rfft(C)
        # Autocorrelation: ifft(C_fft * conj(C_fft))
        autocorr = np.fft.irfft(C_fft * np.conj(C_fft), n=fft_size2)
        # autocorr[k] for k=0,...,fft_size2-1
        # The linear autocorrelation at lag d is at index d (for d>=0) and index fft_size2-d (for d<0)
        # n_minus = number of d in [-mx, mx] where autocorr[d mod fft_size2] > 0.5
        count = 0
        for d in range(-mx, mx+1):
            idx = d % fft_size2
            if autocorr[idx] > 0.5:
                count += 1
        n_minus = count
    
    return n_plus, n_minus

def _random_set(n, max_val):
    U = {0}
    while len(U) < n:
        U.add(random.randint(1, max_val))
    return list(U)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_score = 0.0
    best_U = [0, 1, 3]
    
    # Check some known good candidates
    candidates = [
        [0, 1, 3],
        [0, 1, 3, 9],
        [0, 1, 3, 9, 27],
        [0, 1, 4, 6],
        [0, 1, 4, 10],
        [0, 2, 3, 7],
        [0, 1, 2, 4, 8],
        [0, 1, 3, 8],
        [0, 1, 5, 6],
        [0, 1, 3, 12],
        [0, 1, 4, 15],
        [0, 2, 5, 17],
    ]
    
    for U in candidates:
        s = _score(U)
        if s > best_score:
            best_score = s
            best_U = U
    
    # Simulated annealing
    U = list(best_U)
    current_score = best_score
    T = 0.1  # Initial temperature
    cooling_rate = 0.999
    min_T = 0.001
    
    step = 0
    no_improve = 0
    
    while time.time() < deadline:
        step += 1
        
        # Generate perturbation
        new_U = list(U)
        op = random.random()
        
        if op < 0.4:
            # Add element
            new_U.append(random.randint(1, 200))
        elif op < 0.7 and len(U) > 2:
            # Remove element (keep 0)
            idx = random.randint(1, len(U)-1)
            new_U.pop(idx)
        elif op < 0.85 and len(U) > 1:
            # Shift an element
            idx = random.randint(1, len(U)-1)
            shift = random.randint(-10, 10)
            new_val = new_U[idx] + shift
            if new_val >= 1:
                new_U[idx] = new_val
            else:
                new_U[idx] = random.randint(1, 200)
        else:
            # Swap two elements
            if len(U) >= 3:
                i, j = random.sample(range(1, len(U)), 2)
                new_U[i], new_U[j] = new_U[j], new_U[i]
            else:
                idx = random.randint(1, len(U)-1)
                new_U[idx] = random.randint(1, 200)
        
        # Ensure 0 is present and values are nonnegative
        if 0 not in new_U:
            new_U = [0] + new_U
        new_U = [x for x in new_U if x >= 0]
        if len(new_U) < 2:
            new_U = [0, 1]
        
        # Remove duplicates
        new_U = list(set(new_U))
        if 0 not in new_U:
            new_U.append(0)
        
        new_score = _score(new_U)
        
        # Acceptance criterion
        if new_score > current_score:
            U = new_U
            current_score = new_score
            no_improve = 0
            if new_score > best_score:
                best_score = new_score
                best_U = list(U)
        else:
            # Metropolis criterion
            delta = new_score - current_score
            if delta < 0:
                accept_prob = math.exp(delta / T) if T > 0 else 0
                if random.random() < accept_prob:
                    U = new_U
                    current_score = new_score
                else:
                    no_improve += 1
            
            # Cool down
            T = max(min_T, T * cooling_rate)
            
            # If stuck for too long, restart from best with higher temperature
            if no_improve > 500:
                U = list(best_U)
                current_score = best_score
                T = 0.1
                no_improve = 0
    
    return best_U
