"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import linprog

N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog

def _autoconvolve(x):
    """Compute the full autoconvolution of x."""
    n = len(x)
    # Full convolution length is 2*n - 1
    # Using FFT for efficiency
    size = 2 * n
    X = np.fft.rfft(x, n=size)
    conv = np.fft.irfft(X * X, n=size)[:2*n-1]
    return conv

def _score(x):
    """Compute the score: max(|autoconv|) / (sum(x))^2"""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    conv = _autoconvolve(x)
    max_abs = np.max(np.abs(conv))
    return max_abs / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Start with all ones as baseline
    best_x = np.ones(N)
    best_score = _score(best_x)
    
    # Strategy: Use a binary template (sign pattern) and solve for optimal amplitudes
    # We'll try several templates and pick the best one
    
    # Generate some candidate binary templates (sign patterns)
    # Template 1: All positive
    # Template 2: Alternating signs
    # Template 3: Random signs with fixed seed
    # Template 4: Block patterns
    
    templates = []
    
    # All positive
    templates.append(np.ones(N, dtype=int))
    
    # Alternating
    templates.append(np.array([1 if i % 2 == 0 else -1 for i in range(N)]))
    
    # Block of positives then negatives
    templates.append(np.concatenate([np.ones(N//2), -np.ones(N - N//2)]))
    
    # Random templates
    rng = np.random.RandomState(42)
    for _ in range(5):
        templates.append(rng.choice([1, -1], size=N))
    
    # For each template, solve a QP-like problem to find optimal amplitudes
    # We want to minimize max(|autoconv|) subject to sign(x_i) = template[i]
    
    # Simplified approach: For a given sign template s, let x = s * a where a >= 0
    # autoconv(x) = s * (a * a) * s (element-wise in some sense)
    # Actually, autoconv(s*a) is more complex
    
    # Let's use a different approach: 
    # For a fixed sign pattern, scale the amplitudes to minimize the peak-to-integral ratio
    
    # Actually, let's try a simpler but effective approach:
    # Use the fact that for step functions, we can try specific structured patterns
    
    # Try: x_i = 1 for i in some set, -1 for others, with optimized boundary positions
    
    # Pattern: Two blocks
    # x = [1]*k + [-1]*(N-k)
    for k in range(1, N):
        x = np.concatenate([np.ones(k), -np.ones(N-k)])
        s = np.sum(x)
        if abs(s) < 1e-12:
            continue
        conv = _autoconvolve(x)
        score = np.max(np.abs(conv)) / (s*s)
        if score < best_score:
            best_score = score
            best_x = x.copy()
        
        if time.time() > deadline:
            break
    
    # Try: Three blocks
    # x = [1]*k1 + [-1]*k2 + [1]*k3
    if time.time() < deadline:
        for k1 in range(1, N//2 + 1):
            for k2 in range(1, N - k1 + 1):
                k3 = N - k1 - k2
                if k3 < 0:
                    continue
                x = np.concatenate([np.ones(k1), -np.ones(k2), np.ones(k3)])
                s = np.sum(x)
                if abs(s) < 1e-12:
                    continue
                conv = _autoconvolve(x)
                score = np.max(np.abs(conv)) / (s*s)
                if score < best_score:
                    best_score = score
                    best_x = x.copy()
            
            if time.time() > deadline:
                break
    
    # Try: Four blocks with alternating signs
    if time.time() < deadline:
        for k1 in range(1, N//3 + 1):
            for k2 in range(1, (N - k1)//2 + 1):
                for k3 in range(1, N - k1 - k2 + 1):
                    k4 = N - k1 - k2 - k3
                    if k4 < 0:
                        continue
                    x = np.concatenate([np.ones(k1), -np.ones(k2), np.ones(k3), -np.ones(k4)])
                    s = np.sum(x)
                    if abs(s) < 1e-12:
                        continue
                    conv = _autoconvolve(x)
                    score = np.max(np.abs(conv)) / (s*s)
                    if score < best_score:
                        best_score = score
                        best_x = x.copy()
                
                if time.time() > deadline:
                    break
            
            if time.time() > deadline:
                break
    
    # Try: Sinusoidal-like sign patterns
    if time.time() < deadline:
        for freq in range(1, 20):
            phase = 0
            t = np.linspace(0, 2*np.pi*freq, N)
            x = np.sign(np.sin(t + phase))
            x[x == 0] = 1
            s = np.sum(x)
            if abs(s) < 1e-12:
                continue
            conv = _autoconvolve(x)
            score = np.max(np.abs(conv)) / (s*s)
            if score < best_score:
                best_score = score
                best_x = x.copy()
    
    # Try: Optimized amplitude scaling for the best sign pattern found
    # For the best sign pattern, try scaling individual segments
    
    # Final: return the best sequence found
    return best_x
# EVOLVE-BLOCK-END
