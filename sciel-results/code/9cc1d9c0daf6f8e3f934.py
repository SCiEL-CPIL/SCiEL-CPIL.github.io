import numpy as np
import random
import time

def construct():
    # Configuration
    MAX_SET_SIZE = 500  # Allow larger sets
    MAX_RANGE = 2000    # Upper bound for elements to keep FFT size manageable
    NUM_ITERATIONS = 500
    RANDOM_SEED = 42
    
    random.seed(RANDOM_SEED)
    
    def calc_score(U):
        U = sorted(set(U))
        if not U or 0 not in U:
            return -np.inf
        max_u = max(U)
        if max_u == 0:
            return 1.0
        
        # FFT size must be at least 2 * max_u + 1
        size = 2 * max_u + 1
        n_fft = 1
        while n_fft < size:
            n_fft *= 2
            
        # Indicator vector
        vec = np.zeros(n_fft, dtype=np.float64)
        for x in U:
            vec[x] = 1.0
            
        # FFT
        fft_vec = np.fft.fft(vec)
        
        # Sumset U+U
        sum_poly = np.fft.ifft(fft_vec * fft_vec).real
        sum_counts = np.round(sum_poly).astype(int)
        # U+U is in [0, 2*max_u]
        size_sum = np.sum(sum_counts[:2*max_u+1] > 0)
        
        # Diffset U-U
        # Autocorrelation
        corr = np.fft.ifft(fft_vec * np.conj(fft_vec)).real
        diff_counts = np.round(corr).astype(int)
        
        # Difference range is [-max_u, max_u]
        # In FFT output (linear convolution of vec and reversed vec? No, autocorr)
        # Autocorrelation of a signal x is defined as sum x[i] conj(x[i+k])
        # Here we are doing polynomial multiplication effectively?
        # Wait, fft_vec * conj(fft_vec) corresponds to autocorrelation in time domain.
        # The index in the IFFT result for lag k is at index k (for k>=0) and n_fft-k (for k<0).
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

    # Initialize with a random set
    # Start with a moderate size
    initial_size = 50
    current_U = set()
    current_U.add(0)
    # Add random elements
    for _ in range(initial_size - 1):
        val = random.randint(1, MAX_RANGE // 2)
        current_U.add(val)
    
    best_U = list(current_U)
    best_score = calc_score(best_U)
    
    # Local Search
    for i in range(NUM_ITERATIONS):
        # Check time
        if time.time() > 110: # Leave 10s buffer
            break
            
        U = set(best_U)
        max_u = max(U) if U else 0
        
        # Mutation strategies
        # 1. Swap: pick an element, move it to a new random position
        # 2. Add: add a new random element
        # 3. Remove: remove a random element (not 0)
        
        op = random.choice(['swap', 'add', 'remove'])
        
        if op == 'swap' and len(U) > 1:
            elements = list(U)
            # Don't move 0 usually, or allow it? 0 is required.
            non_zero_elements = [e for e in elements if e != 0]
            if non_zero_elements:
                elem_to_move = random.choice(non_zero_elements)
                U.remove(elem_to_move)
                # New position: random within current range or slightly expanded
                new_val = random.randint(0, max_u + 50)
                U.add(new_val)
        elif op == 'add' and len(U) < MAX_SET_SIZE:
            # Add element, potentially expanding range
            new_val = random.randint(0, max_u + 100)
            U.add(new_val)
        elif op == 'remove' and len(U) > 10:
            non_zero_elements = [e for e in U if e != 0]
            if non_zero_elements:
                U.remove(random.choice(non_zero_elements))
        
        if 0 not in U:
            continue
            
        new_score = calc_score(list(U))
        
        if new_score > best_score:
            best_score = new_score
            best_U = list(U)
            
            # Optional: if we found a good one, we might want to try to grow it
            # But simple hill climbing is safer for now.
            
    return sorted(best_U)
