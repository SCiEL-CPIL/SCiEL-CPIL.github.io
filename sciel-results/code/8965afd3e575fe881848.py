"""
Combinatorial Sign-Flip Strategy for Signed Autocorrelation Minimization.

Strategy:
1. Generate a high-quality continuous base signal using a simple spectral method 
   (random phase with flat magnitude) to ensure the signal has low initial peaks 
   and a non-zero sum.
2. Treat the problem as a combinatorial optimization over the signs of the signal 
   components. The metric is scale-invariant, so we can normalize the sum to 1 
   (or close to it) and minimize max |autoconv|.
   Actually, sign flips change the sum. We need to be careful.
   Let x be the base signal. We consider x_s = s * x where s_i in {+1, -1}.
   The metric is M(x_s) = max_k |c_k| / (sum x_s)^2.
   Note that c_k = (x_s * x_s)_k.
   This is a discrete search space of size 2^N.
3. Use Simulated Annealing to navigate this space.
   - Start with the base signal signs (all +1).
   - At each step, flip the sign of one or a few random indices.
   - Calculate the change in the metric.
   - Accept the change if it improves the metric, or with probability exp(-delta/T) 
     if it worsens it, where T is the temperature.
4. The key advantage is that the objective function is exact (no soft-max 
   approximation error) and the search is robust to the non-smoothness of 
   max |.|.
5. Since the search space is huge, we rely on the fact that the metric is 
   scale-invariant and the "shape" of the autoconvolution is what matters. 
   Flipping signs can significantly reduce peak autoconvolution by creating 
   destructive interference in the time domain peaks.
"""

import numpy as np
import time

N = 400
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_max_abs(x):
    """Compute max |autoconv(x)|."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))

def _integral(x):
    return float(np.sum(x))

def _metric(x):
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _generate_base_signal():
    """
    Generate a base signal with good spectral properties.
    We use a random phase spectrum with flat magnitude.
    """
    rng = np.random.default_rng(42)
    n_freq = _FFT_SIZE // 2 + 1
    mag = np.ones(n_freq)
    # Taper the edges slightly to reduce ringing
    window = np.hanning(n_freq)
    window[0] = 1.0
    window[-1] = 1.0
    mag *= window
    
    phase = rng.uniform(-np.pi, np.pi, size=n_freq)
    fx = mag * np.exp(1j * phase)
    x = np.fft.irfft(fx, _FFT_SIZE)[:N]
    
    # Ensure the sum is not zero. If it is, flip all signs or add a small DC.
    # A flat spectrum with random phase usually has a non-zero sum, but let's be safe.
    if abs(np.sum(x)) < 1e-6:
        x += 0.01 * np.ones(N)
        
    return x

def construct():
    deadline = time.time() + 145.0
    rng = np.random.default_rng(2024)
    
    # 1. Generate base signal
    x_base = _generate_base_signal()
    
    # Initial state: all signs are +1
    # We work with the signs array and the base values
    # x_eff = signs * x_base
    signs = np.ones(N, dtype=np.int8)
    
    # Precompute base autoconv? No, because x_eff changes with signs.
    # However, we can compute the metric quickly.
    
    def get_metric_from_signs(s):
        x_eff = s * x_base
        return _metric(x_eff)
    
    best_signs = signs.copy()
    best_val = get_metric_from_signs(best_signs)
    
    # Simulated Annealing
    # Temperature schedule: geometric cooling
    T_start = 1.0
    T_end = 0.001
    n_steps = 100000  # We will break based on time
    
    # We need to estimate the scale of the metric changes to set T appropriately.
    # Let's run a few random flips to estimate typical delta.
    deltas = []
    for _ in range(10):
        idx = rng.integers(0, N)
        signs[idx] *= -1
        v_new = get_metric_from_signs(signs)
        deltas.append(abs(v_new - best_val))
        signs[idx] *= -1 # revert
        
    if not deltas:
        return x_base
        
    typical_delta = np.median(deltas)
    if typical_delta < 1e-9:
        typical_delta = 1e-6
        
    T = T_start * typical_delta
    
    # Optimization loop
    step = 0
    last_improve_time = time.time()
    no_improve_count = 0
    
    while time.time() < deadline - 5:
        step += 1
        
        # Cooling
        if step % 100 == 0:
            T = max(T_end, T * 0.999)
            
        # Propose a move: flip 1 or 2 signs
        # Flipping 1 sign is local, 2 signs allows more exploration
        if rng.random() < 0.8:
            k = 1
        else:
            k = 2
            
        indices = rng.choice(N, size=k, replace=False)
        
        # Apply flip
        for idx in indices:
            signs[idx] *= -1
            
        v_new = get_metric_from_signs(signs)
        
        # Acceptance criterion
        delta = v_new - best_val
        
        if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
            best_val = v_new
            best_signs = signs.copy()
            no_improve_count = 0
            last_improve_time = time.time()
        else:
            # Revert
            for idx in indices:
                signs[idx] *= -1
            no_improve_count += 1
            
        # If we haven't improved in a long time and T is low, maybe restart with a new base?
        # Or just keep going. The problem is non-convex, so SA can get stuck.
        # But with N=400, the landscape is complex.
        
        # Additional strategy: If we stagnate, try a "kick" by flipping a random block
        if no_improve_count > 1000 and rng.random() < 0.01:
            # Flip a random contiguous block of length 5-10
            start = rng.integers(0, N - 10)
            length = rng.integers(5, 10)
            for idx in range(start, start + length):
                signs[idx] *= -1
            v_kick = get_metric_from_signs(signs)
            if v_kick < best_val:
                best_val = v_kick
                best_signs = signs.copy()
                no_improve_count = 0
            else:
                # Revert kick
                for idx in range(start, start + length):
                    signs[idx] *= -1
                # Reheat
                T = max(T, T_start * typical_delta * 0.5)

    # Final result
    x_final = best_signs * x_base
    return x_final

if __name__ == "__main__":
    # Simple test
    x = construct()
    print(f"Metric: {_metric(x)}")
