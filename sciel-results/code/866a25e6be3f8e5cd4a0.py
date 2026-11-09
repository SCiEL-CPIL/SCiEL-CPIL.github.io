import numpy as np
import time

N = 400
SIZE = 1
while SIZE < 2 * N - 1:
    SIZE <<= 1

def _autoconv(f):
    F = np.fft.rfft(f, SIZE)
    power = F * np.conj(F)
    auto = np.fft.irfft(power, SIZE)[:2 * N - 1]
    return auto

def _score(auto, s):
    if abs(s) < 1e-12:
        return float('inf')
    return float(np.max(np.abs(auto)) / (s * s))

def _update_autoconv_incremental(auto, f, i, delta):
    n = len(f)
    new_auto = auto.copy()
    shift = np.zeros(2 * n - 1)
    shift[i:i+n] = f
    new_auto += 2 * delta * shift
    if 0 <= 2 * i < 2 * n - 1:
        new_auto[2 * i] += delta * delta
    return new_auto

def _greedy_flip(f, auto, s, current_score, deadline, margin=25):
    improved = True
    while improved and time.time() < deadline - margin:
        improved = False
        order = np.random.default_rng(int(time.time()*1000) % 100000).permutation(N)
        for i in order:
            if time.time() > deadline - margin:
                break
            old_val = f[i]
            delta = -2.0 * old_val
            new_auto = _update_autoconv_incremental(auto, f, i, delta)
            new_s = s + delta
            new_score = _score(new_auto, new_s)
            if new_score < current_score - 1e-14:
                f[i] = old_val + delta
                auto = new_auto
                s = new_s
                current_score = new_score
                improved = True
    return f, auto, s, current_score

def _spectral_init(rng, target_pct=40.0, iters=30):
    f = rng.choice([-1.0, 1.0], size=N)
    for _ in range(iters):
        F = np.fft.rfft(f, SIZE)
        mag = np.abs(F)
        phase = np.angle(F)
        target_mag = np.percentile(mag[1:], target_pct)
        new_mag = mag.copy()
        new_mag[1:] = target_mag
        new_F = new_mag * np.exp(1j * phase)
        new_f = np.fft.irfft(new_F, SIZE)[:N]
        f = np.sign(new_f)
        f[f == 0] = 1.0
    return f

def _walsh_hadamard_init(rng):
    # Generate a random Hadamard-like sequence using Walsh functions
    # Use a simple approach: random signs with some structure
    f = rng.choice([-1.0, 1.0], size=N)
    # Apply a few rounds of spectral flattening
    for _ in range(20):
        F = np.fft.rfft(f, SIZE)
        mag = np.abs(F)
        phase = np.angle(F)
        target_mag = np.percentile(mag[1:], 50.0)
        new_mag = mag.copy()
        new_mag[1:] = target_mag
        new_F = new_mag * np.exp(1j * phase)
        new_f = np.fft.irfft(new_F, SIZE)[:N]
        f = np.sign(new_f)
        f[f == 0] = 1.0
    return f

def _simulated_annealing(f, auto, s, current_score, deadline, margin=5):
    rng = np.random.default_rng(789)
    start_temp = 0.05
    end_temp = 0.001
    max_iters = 150000
    it = 0
    while it < max_iters and time.time() < deadline - margin:
        it += 1
        progress = it / max_iters
        temp = start_temp * (end_temp / start_temp) ** progress
        
        i = int(rng.integers(N))
        old_val = f[i]
        delta = -2.0 * old_val
        new_auto = _update_autoconv_incremental(auto, f, i, delta)
        new_s = s + delta
        new_score = _score(new_auto, new_s)
        
        if new_score < current_score:
            f[i] = old_val + delta
            auto = new_auto
            s = new_s
            current_score = new_score
        else:
            delta_score = new_score - current_score
            if delta_score > 0 and temp > 1e-8:
                if rng.random() < np.exp(-delta_score / temp):
                    f[i] = old_val + delta
                    auto = new_auto
                    s = new_s
                    current_score = new_score
    return f, auto, s, current_score

def construct():
    deadline = time.time() + 140
    
    # Try multiple diverse initializations
    rng = np.random.default_rng(42)
    
    best_f = np.ones(N, dtype=np.float64)
    best_auto = _autoconv(best_f)
    best_s = float(np.sum(best_f))
    best_score = _score(best_auto, best_s)
    
    # Generate several diverse starting points
    inits = []
    # All ones
    inits.append(best_f.copy())
    # Spectral inits with different parameters
    for pct in [25, 35, 45, 55]:
        inits.append(_spectral_init(rng, target_pct=pct, iters=25))
    # Walsh-Hadamard init
    inits.append(_walsh_hadamard_init(rng))
    # Random
    inits.append(rng.choice([-1.0, 1.0], size=N))
    
    # Evaluate each init and pick the best few for local search
    scored_inits = []
    for f_init in inits:
        auto = _autoconv(f_init)
        s = float(np.sum(f_init))
        sc = _score(auto, s)
        scored_inits.append((sc, f_init.copy(), auto.copy(), s))
    
    scored_inits.sort(key=lambda x: x[0])
    
    # Take top 3 initializations for local search
    top_inits = scored_inits[:3]
    
    for sc, f_init, auto_init, s_init in top_inits:
        if sc < best_score:
            best_score = sc
            best_f = f_init.copy()
            best_auto = auto_init.copy()
            best_s = s_init
        
        if time.time() > deadline - 30:
            break
        
        # Greedy local search from this init
        f = f_init.copy()
        auto = auto_init.copy()
        s = s_init
        current_score = sc
        
        f, auto, s, current_score = _greedy_flip(f, auto, s, current_score, deadline, margin=30)
        
        if current_score < best_score:
            best_score = current_score
            best_f = f.copy()
            best_auto = auto.copy()
            best_s = s
    
    # Simulated annealing from best
    if time.time() < deadline - 10:
        f = best_f.copy()
        auto = best_auto.copy()
        s = best_s
        current_score = best_score
        
        f, auto, s, current_score = _simulated_annealing(f, auto, s, current_score, deadline, margin=10)
        
        if current_score < best_score:
            best_score = current_score
            best_f = f.copy()
            best_auto = auto.copy()
            best_s = s
    
    # Final greedy polish
    if time.time() < deadline - 2:
        f = best_f.copy()
        auto = best_auto.copy()
        s = best_s
        current_score = best_score
        
        f, auto, s, current_score = _greedy_flip(f, auto, s, current_score, deadline, margin=2)
        
        if current_score < best_score:
            best_score = current_score
            best_f = f.copy()
            best_auto = auto.copy()
            best_s = s
    
    return best_f
