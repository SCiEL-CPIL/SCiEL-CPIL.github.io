import numpy as np
import time

N = 400

def _thue_morse(n):
    tm = np.array([1.0])
    while len(tm) < n:
        tm = np.concatenate([tm, -tm])
    return tm[:n]

def _calc_ac(h):
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    ac = np.fft.irfft(F * np.conj(F), n=n_fft)[:N]
    ac[0] = float(N)
    return ac

def construct():
    try:
        t0 = time.time()
        deadline = t0 + 115.0
        
        # Initializations
        inits = [
            _thue_morse(N),
            np.random.choice([-1.0, 1.0], size=N),
            np.random.choice([-1.0, 1.0], size=N),
            np.cos(2 * np.pi * np.arange(N)**2 / N) * 1.0 # Chirp-like, signs
        ]
        # Ensure binary
        for i in range(len(inits)):
            inits[i] = np.sign(inits[i])
            inits[i][inits[i] == 0] = 1.0
            
        best_h = None
        best_score = 1e9
        
        for h0 in inits:
            ac = _calc_ac(h0)
            score = np.max(np.abs(ac)) / N
            if score < best_score:
                best_score = score
                best_h = h0.copy()
                
        h = best_h.copy()
        ac = _calc_ac(h)
        cur_score = best_score
        cur_max = np.max(np.abs(ac))
        
        # Main Loop: Greedy Hill Climbing with SA-like restarts
        T = 0.05
        decay = 0.9999
        stuck = 0
        
        while time.time() < deadline - 10.0:
            j = np.random.randint(0, N)
            
            k_max = max(N - j - 1, j)
            if k_max < 0:
                continue
                
            delta = np.zeros(k_max + 1)
            
            if N - j - 1 >= 0:
                k1_max = N - j - 1
                delta[0:k1_max+1] -= 2.0 * h[j] * h[j:j+k1_max+1]
                
            if j >= 0:
                k2_max = j
                delta[0:k2_max+1] -= 2.0 * h[j] * h[0:j+1][::-1]
                
            delta[0] = 0.0
            
            old_ac_seg = ac[0:k_max+1].copy()
            ac[0:k_max+1] += delta
            
            new_max = np.max(np.abs(ac))
            new_score = new_max / N
            
            if new_score < cur_score:
                h[j] *= -1.0
                cur_max = new_max
                cur_score = new_score
                stuck = 0
                if new_score < best_score:
                    best_score = new_score
                    best_h = h.copy()
            else:
                ac[0:k_max+1] = old_ac_seg
                stuck += 1
                # If stuck, do a random restart or kick
                if stuck > 500:
                    # Kick: flip 5 random bits
                    for _ in range(5):
                        k = np.random.randint(0, N)
                        h[k] *= -1.0
                    ac = _calc_ac(h)
                    cur_max = np.max(np.abs(ac))
                    cur_score = cur_max / N
                    stuck = 0
                    if cur_score < best_score:
                        best_score = cur_score
                        best_h = h.copy()
                        
            T = max(1e-6, T * decay)
            
        # Final Polish: Strict Greedy
        h = best_h.copy()
        ac = _calc_ac(h)
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for j in range(N):
                k_max = max(N - j - 1, j)
                if k_max < 0: continue
                delta = np.zeros(k_max + 1)
                if N - j - 1 >= 0:
                    k1_max = N - j - 1
                    delta[0:k1_max+1] -= 2.0 * h[j] * h[j:j+k1_max+1]
                if j >= 0:
                    k2_max = j
                    delta[0:k2_max+1] -= 2.0 * h[j] * h[0:j+1][::-1]
                delta[0] = 0.0
                
                old_ac_seg = ac[0:k_max+1].copy()
                ac[0:k_max+1] += delta
                new_max = np.max(np.abs(ac))
                new_score = new_max / N
                
                if new_score < best_score - 1e-12:
                    h[j] *= -1.0
                    best_score = new_score
                    best_h = h.copy()
                    improved = True
                else:
                    ac[0:k_max+1] = old_ac_seg
                    
        return best_h

    except Exception:
        return np.ones(N)
