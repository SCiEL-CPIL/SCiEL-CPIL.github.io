import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 600

def compute_score(h):
    h = np.maximum(h, 0)
    area = np.sum(h)
    max_val = np.max(h)
    if area == 0 or max_val == 0:
        return float('inf')
    conv = np.convolve(h, h, mode='full')
    return np.max(conv) / (area * max_val)

def compute_score_fft(h):
    """FFT-based convolution for speed."""
    h = np.maximum(h, 0)
    area = np.sum(h)
    max_val = np.max(h)
    if area == 0 or max_val == 0:
        return float('inf')
    n = len(h)
    # Use FFT for convolution
    size = 2 * n
    f = np.fft.rfft(h, size)
    conv = np.fft.irfft(f * f, size)[:2*n-1]
    return np.max(conv) / (area * max_val)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    i = np.arange(N)
    
    # Initial best: uniform
    h_uniform = np.ones(N) / N
    best_score = compute_score(h_uniform)
    best_h = h_uniform.copy()
    
    # Also try single spike
    M = 1.0 / np.sqrt(N)
    h1 = np.full(N, (1.0 - M) / (N - 1))
    h1[0] = M
    s1 = compute_score(h1)
    if s1 < best_score:
        best_score = s1
        best_h = h1.copy()
    
    # Structural change: coarse block optimization
    # Divide into K blocks of size B
    K = 60  # number of blocks
    B = N // K  # block size (10)
    
    def blocks_to_h(blocks):
        """Convert block heights to full step function."""
        blocks = np.maximum(blocks, 0)
        if np.sum(blocks) > 0:
            blocks = blocks / np.sum(blocks) * N  # scale so total sum = N (each unit height contributes B)
        h = np.repeat(blocks, B)
        # Adjust for remainder
        if len(h) < N:
            h = np.concatenate([h, np.zeros(N - len(h))])
        return h
    
    def h_to_blocks(h):
        """Convert full step function to block heights."""
        blocks = np.zeros(K)
        for k in range(K):
            start = k * B
            end = start + B
            blocks[k] = np.mean(h[start:end])
        return blocks
    
    def objective_blocks(blocks):
        h = blocks_to_h(blocks)
        return compute_score_fft(h)
    
    # Try multiple random initializations for Nelder-Mead
    rng = np.random.RandomState(42)
    
    for trial in range(20):
        if time.time() > deadline - 30:
            break
        
        # Different initial block patterns
        if trial == 0:
            blocks0 = np.ones(K) / K
        elif trial == 1:
            blocks0 = np.zeros(K)
            blocks0[K//2] = 1.0
        elif trial == 2:
            # Two peaks
            blocks0 = np.zeros(K)
            blocks0[K//4] = 0.5
            blocks0[3*K//4] = 0.5
        elif trial == 3:
            # Ramp
            blocks0 = np.linspace(0.1, 1.0, K)
        elif trial == 4:
            # Sine
            blocks0 = np.abs(np.sin(np.pi * np.arange(K) / K)) + 0.1
        elif trial == 5:
            # Exponential decay
            blocks0 = np.exp(-np.arange(K) / 10.0)
        else:
            # Random
            blocks0 = rng.random(K) + 0.01
        
        blocks0 = blocks0 / np.sum(blocks0)
        
        try:
            result = minimize(objective_blocks, blocks0, method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10,
                                    'adaptive': True})
            if result.fun < best_score:
                best_score = result.fun
                best_h = blocks_to_h(result.x)
        except:
            pass
    
    # Local refinement: hill-climbing on individual steps
    if time.time() < deadline - 20:
        h = best_h.copy()
        improved = True
        max_rounds = 5
        
        for round_num in range(max_rounds):
            if time.time() > deadline - 20:
                break
            improved = False
            # Try moving mass from one step to another
            for step in range(0, N, 3):  # Check every 3rd step for speed
                if time.time() > deadline - 20:
                    break
                for target in range(step+1, min(step+30, N)):
                    # Try moving a small amount from target to step
                    for frac in [0.1, 0.2, 0.3, 0.5]:
                        h_test = h.copy()
                        delta = h_test[target] * frac
                        if delta > 0:
                            h_test[target] -= delta
                            h_test[step] += delta
                            h_test = np.maximum(h_test, 0)
                            s_test = compute_score_fft(h_test)
                            if s_test < best_score - 1e-10:
                                best_score = s_test
                                best_h = h_test.copy()
                                h = h_test.copy()
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
    
    return list(best_h)
