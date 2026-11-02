import numpy as np
import time

N = 400
DEADLINE = 140.0

def _calc_acf(h):
    # h is +/- 1
    # Pad to 2*N to get linear autocorrelation
    h_padded = np.zeros(2*N)
    h_padded[:N] = h
    H = np.fft.fft(h_padded)
    acf = np.fft.ifft(H * np.conj(H)).real
    # The linear autocorrelation for lags 0 to N-1 is in acf[:N]
    # Note: acf[N:] contains the negative lags (symmetric)
    return acf[:N]

def _score(h):
    acf = _calc_acf(h)
    # Peak sidelobe: max absolute value for k != 0
    if len(acf) > 1:
        return np.max(np.abs(acf[1:]))
    return 0.0

def construct():
    start_time = time.time()
    
    # Initialize with random
    np.random.seed(42)
    h = np.random.choice([-1, 1], N)
    best_h = h.copy()
    best_score = _score(h)
    
    # Simulated Annealing
    T = 1.0 # Start temp. Scores are ~20-50. Delta is integer.
    T_min = 0.01
    cooling = 0.999
    
    current_h = h.copy()
    current_score = best_score
    
    def elapsed_ok():
        return (time.time() - start_time) < DEADLINE

    while elapsed_ok():
        # Move: flip a random bit
        idx = np.random.randint(0, N)
        new_h = current_h.copy()
        new_h[idx] *= -1
        
        new_score = _score(new_h)
        
        if new_score < current_score:
            current_h = new_h
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = new_h.copy()
        else:
            delta = new_score - current_score
            # delta is usually integer or 0
            if delta == 0:
                current_h = new_h
                current_score = new_score
            elif np.random.rand() < np.exp(-delta / T):
                current_h = new_h
                current_score = new_score
        
        T *= cooling
        if T < T_min:
            # Reheat or restart
            if elapsed_ok():
                # Try a few random restarts
                for _ in range(20):
                    if not elapsed_ok():
                        break
                    cand_h = np.random.choice([-1, 1], N)
                    cand_score = _score(cand_h)
                    if cand_score < best_score:
                        best_score = cand_score
                        best_h = cand_h.copy()
                        current_h = best_h.copy()
                        current_score = best_score
                        T = 1.0
                break

    # Greedy Polish
    # Try flipping each bit to see if it improves
    improved = True
    while improved and elapsed_ok():
        improved = False
        for i in range(N):
            if not elapsed_ok():
                break
            # Flip i
            best_h[i] *= -1
            new_score = _score(best_h)
            if new_score < best_score:
                best_score = new_score
                improved = True
            else:
                # Revert
                best_h[i] *= -1

    return best_h.astype(np.float64)
