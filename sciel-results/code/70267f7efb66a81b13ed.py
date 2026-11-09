import numpy as np
import time

N = 100

def compute_score_fast(h):
    """Compute max overlap using FFT-based circular cross-correlation."""
    hft = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(hft * np.conj(hft)))
    min_ac = np.min(autocorr[1:])
    return N / 2.0 - min_ac

def construct():
    deadline = time.time() + 145
    best_h = None
    best_score = np.inf
    
    np.random.seed(42)
    
    # Generate diverse initial binary solutions
    candidates = []
    
    # Alternating pattern
    h_alt = np.zeros(N)
    h_alt[::2] = 1.0
    candidates.append(h_alt.copy())
    
    # Random binary with N/2 ones
    for restart in range(80):
        h_rand = np.zeros(N)
        ones = np.random.choice(N, N // 2, replace=False)
        h_rand[ones] = 1.0
        candidates.append(h_rand)
    
    # Block patterns
    for block_size in range(2, 20):
        h_block = np.zeros(N)
        for i in range(0, N, block_size * 2):
            end = min(i + block_size, N)
            h_block[i:end] = 1.0
        ones_count = np.sum(h_block == 1.0)
        if ones_count > N // 2:
            ones_pos = np.where(h_block == 1.0)[0]
            to_zero = ones_pos[-(ones_count - N // 2):]
            h_block[to_zero] = 0.0
        elif ones_count < N // 2:
            zeros_pos = np.where(h_block == 0.0)[0]
            to_one = zeros_pos[:N // 2 - ones_count]
            h_block[to_one] = 1.0
        candidates.append(h_block)
    
    # Evaluate all candidates
    for h_cand in candidates:
        score = compute_score_fast(h_cand)
        if score < best_score:
            best_score = score
            best_h = h_cand.copy()
    
    # Simulated annealing on binary sequences
    h_curr = best_h.copy()
    current_score = best_score
    ones = np.where(h_curr == 1)[0]
    zeros = np.where(h_curr == 0)[0]
    
    T = 1.0
    cooling = 0.995
    min_T = 0.005
    
    while time.time() < deadline - 10 and T > min_T:
        if len(ones) == 0 or len(zeros) == 0:
            break
        i = np.random.choice(ones)
        j = np.random.choice(zeros)
        
        h_curr[i] = 0
        h_curr[j] = 1
        new_score = compute_score_fast(h_curr)
        delta = new_score - current_score
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_score = new_score
            ones = np.where(h_curr == 1)[0]
            zeros = np.where(h_curr == 0)[0]
            
            if current_score < best_score:
                best_score = current_score
                best_h = h_curr.copy()
        else:
            h_curr[i] = 1
            h_curr[j] = 0
        
        T *= cooling
    
    # Hill climbing polish
    h_curr = best_h.copy()
    current_score = best_score
    ones = np.where(h_curr == 1)[0]
    zeros = np.where(h_curr == 0)[0]
    
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for _ in range(2000):
            if time.time() > deadline - 3:
                break
            if len(ones) == 0 or len(zeros) == 0:
                break
            i = np.random.choice(ones)
            j = np.random.choice(zeros)
            
            h_curr[i] = 0
            h_curr[j] = 1
            new_score = compute_score_fast(h_curr)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
                ones = np.where(h_curr == 1)[0]
                zeros = np.where(h_curr == 0)[0]
                
                if current_score < best_score:
                    best_score = current_score
                    best_h = h_curr.copy()
            else:
                h_curr[i] = 1
                h_curr[j] = 0
    
    # Final validation
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    s = np.sum(best_h)
    if abs(s - N / 2.0) > 1e-10:
        if s > N / 2.0:
            ones_pos = np.where(best_h == 1.0)[0]
            for p in ones_pos:
                if np.sum(best_h) <= N / 2.0:
                    break
                best_h[p] = 0.0
        else:
            zeros_pos = np.where(best_h == 0.0)[0]
            for p in zeros_pos:
                if np.sum(best_h) >= N / 2.0:
                    break
                best_h[p] = 1.0
    
    return best_h
