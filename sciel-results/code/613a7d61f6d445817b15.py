import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 40s margin before 150s kill

    k = np.arange(1, N)

    def compute_acf(h):
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.fft(h, n=n_fft)
        power = np.abs(F) ** 2
        acf = np.fft.ifft(power).real
        return acf[1:N]

    def get_metric(acf):
        return np.max(np.abs(acf))

    def local_search_greedy(h, max_passes=500):
        acf = compute_acf(h)
        current_val = get_metric(acf)
        
        improved = True
        passes = 0
        while improved and passes < max_passes:
            if time.time() > deadline - 20.0:
                break
            improved = False
            passes += 1
            for i in range(N):
                if time.time() > deadline - 20.0:
                    return h, current_val
                
                idx1 = i + k
                valid1 = idx1 < N
                term1 = np.zeros(N-1)
                if np.any(valid1):
                    term1[valid1] = h[idx1[valid1]]
                
                idx2 = i - k
                valid2 = idx2 >= 0
                term2 = np.zeros(N-1)
                if np.any(valid2):
                    term2[valid2] = h[idx2[valid2]]
                
                delta = -2.0 * h[i] * (term1 + term2)
                
                new_acf = acf + delta
                new_val = get_metric(new_acf)
                
                if new_val < current_val:
                    h[i] *= -1.0
                    acf = new_acf
                    current_val = new_val
                    improved = True
        
        return h, current_val

    def simulated_annealing(h, max_iters=20000):
        acf = compute_acf(h)
        current_val = get_metric(acf)
        
        T = 2.0
        T_min = 0.1
        cooling = 0.9999
        
        for _ in range(max_iters):
            if time.time() > deadline - 10.0:
                break
            
            i = np.random.randint(N)
            
            idx1 = i + k
            valid1 = idx1 < N
            term1 = np.zeros(N-1)
            if np.any(valid1):
                term1[valid1] = h[idx1[valid1]]
            
            idx2 = i - k
            valid2 = idx2 >= 0
            term2 = np.zeros(N-1)
            if np.any(valid2):
                term2[valid2] = h[idx2[valid2]]
            
            delta = -2.0 * h[i] * (term1 + term2)
            
            new_acf = acf + delta
            new_val = get_metric(new_acf)
            
            delta_val = new_val - current_val
            
            if delta_val < 0 or np.random.rand() < np.exp(-delta_val / T):
                h[i] *= -1.0
                acf = new_acf
                current_val = new_val
            
            T *= cooling
            if T < T_min:
                T = T_min
                
        return h, current_val

    # --- Structural Change: Ensemble of Structured Initializations ---
    # Instead of a single M-sequence, we use a diverse set of initializations
    # including M-sequences, Thue-Morse, and random, then apply the same
    # optimization pipeline to each, keeping the best result.
    
    best_h = None
    best_val = np.inf

    def gen_m_seq(m, poly_taps):
        state = 1
        seq = []
        for _ in range(2**m - 1):
            seq.append(state & 1)
            fb = 0
            for tap in poly_taps:
                if tap == 0:
                    fb ^= (state & 1)
                else:
                    fb ^= (state >> tap) & 1
            state = ((state >> 1) | (fb << (m - 1))) & ((1 << m) - 1)
        return np.array(seq, dtype=np.float64) * 2 - 1

    initializations = []
    
    # 1. M-sequence (as before)
    m_seq_511 = gen_m_seq(9, [4, 0])
    initializations.append(m_seq_511[:N].copy())
    
    # 2. Thue-Morse sequence
    tm = np.zeros(N)
    for i in range(N):
        c = 0
        m_val = i
        while m_val:
            c += m_val & 1
            m_val >>= 1
        tm[i] = 1.0 if c % 2 == 0 else -1.0
    initializations.append(tm)
    
    # 3. Random initialization with fixed seed for reproducibility
    rng = np.random.RandomState(42)
    initializations.append(rng.choice([-1, 1], size=N))
    
    # 4. Another M-sequence with different taps if available
    # x^9 + x^5 + x^4 + x^3 + 1 is also primitive? Let's try a known one.
    # Actually, let's just use a different seed for random to get diversity.
    initializations.append(rng.choice([-1, 1], size=N))
    
    # Process each initialization
    for h_init in initializations:
        if time.time() > deadline - 30.0:
            break
            
        h = h_init.copy()
        
        # Greedy Local Search
        h, val = local_search_greedy(h, max_passes=100)
        
        # Simulated Annealing
        h, val = simulated_annealing(h, max_iters=10000)
        
        # Final Polish
        h, val = local_search_greedy(h, max_passes=50)
        
        if val < best_val:
            best_val = val
            best_h = h.copy()

    # If no initialization was processed (unlikely), fallback to all ones
    if best_h is None:
        best_h = np.ones(N)

    return best_h
