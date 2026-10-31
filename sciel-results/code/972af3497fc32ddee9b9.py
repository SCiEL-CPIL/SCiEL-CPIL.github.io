import numpy as np

def construct():
    # Initial best set
    best_set = [0, 1, 2, 6]
    best_score = 0.0
    
    def calc_score(U):
        U = sorted(set(U))
        if not U or 0 not in U:
            return -np.inf
        max_u = max(U)
        if max_u == 0:
            return 1.0 # log(1/1)/log(1) is 0/0, limit is 1?
        
        # Use FFT to compute sum and diff set sizes
        # Pad to power of 2
        size = 2 * max_u + 1
        n_fft = 1
        while n_fft < size:
            n_fft *= 2
            
        # Indicator vector for U
        # U is non-negative
        vec = np.zeros(n_fft, dtype=np.float64)
        for x in U:
            vec[x] = 1.0
            
        # FFT
        fft_vec = np.fft.fft(vec)
        
        # Sumset U+U
        # Coefficients of polynomial squared
        sum_poly = np.fft.ifft(fft_vec * fft_vec).real
        # Round to nearest integer to avoid floating point errors
        sum_counts = np.round(sum_poly).astype(int)
        # Size of U+U is number of non-zero counts
        # Note: U+U is within [0, 2*max_u]
        size_sum = np.sum(sum_counts[:2*max_u+1] > 0)
        
        # Diffset U-U
        # U-U corresponds to convolution of U and reversed U?
        # Or just compute directly for small sets?
        # For efficiency in search, direct set ops might be faster for small N.
        # But directive says use FFT.
        # Diffset: x - y = d.
        # This is cross-correlation.
        # fft_vec * conj(fft_vec) gives autocorrelation.
        # Autocorrelation at lag k is sum_{i} U[i] U[i+k].
        # This gives counts of differences.
        corr = np.fft.ifft(fft_vec * np.conj(fft_vec)).real
        diff_counts = np.round(corr).astype(int)
        # Difference range is [-max_u, max_u]
        # In FFT output, lags are 0..n_fft-1.
        # Lag k corresponds to diff k.
        # Lag n_fft-k corresponds to diff -k.
        size_diff = 0
        for d in range(-max_u, max_u+1):
            if d >= 0:
                idx = d
            else:
                idx = n_fft + d
            if diff_counts[idx] > 0:
                size_diff += 1
                
        if size_sum == 0 or size_diff == 0:
            return -np.inf
            
        ratio = size_diff / size_sum
        if ratio <= 0:
            return -np.inf
            
        score = 1 + np.log(ratio) / np.log(2 * max_u + 1)
        return score

    current_score = calc_score(best_set)
    best_score = current_score
    
    # Simple local search
    # Try adding or removing elements, or swapping
    import random
    random.seed(42)
    
    max_iterations = 1000
    for _ in range(max_iterations):
        U = best_set.copy()
        # Random mutation
        op = random.choice(['add', 'remove', 'swap'])
        if op == 'add' and len(U) < 20:
            # Add random element
            max_u = max(U)
            # Try small range first
            new_val = random.randint(0, max_u + 10)
            if new_val not in U:
                U.append(new_val)
        elif op == 'remove' and len(U) > 1:
            # Remove random element (not 0)
            idx = random.randint(1, len(U)-1)
            U.pop(idx)
        elif op == 'swap':
            if len(U) > 1:
                idx = random.randint(1, len(U)-1)
                max_u = max(U)
                U[idx] = random.randint(0, max_u + 10)
        
        if 0 not in U:
            continue
            
        new_score = calc_score(U)
        if new_score > best_score:
            best_score = new_score
            best_set = U
            
    return sorted(best_set)
