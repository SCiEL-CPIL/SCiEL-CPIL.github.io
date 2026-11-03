"""Rewritten program for signed autocorrelation maximum absolute-value variant."""
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

# Fixed parameters
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

def _autoconv(x):
    """Compute full linear autoconvolution of x with itself."""
    if len(x) == 0:
        return np.array([])
    # Use FFT-based convolution for speed
    n = len(x)
    f_len = 2 * n - 1
    # Pad to next power of 2 for efficiency
    size = 1
    while size < f_len:
        size *= 2
    fx = np.fft.rfft(x, size)
    conv = np.fft.irfft(fx * fx, size)[:f_len]
    return conv

def _objective(x):
    """
    Objective: minimize max(|autoconv(x)|) / (sum(x)^2)
    x is a vector of length N.
    """
    conv = _autoconv(x)
    max_abs = np.max(np.abs(conv))
    integral_sq = np.sum(x) ** 2
    if integral_sq < 1e-12:
        return 1e10
    return max_abs / integral_sq

def _gradient(x):
    """
    Numerical gradient for the objective function.
    """
    eps = 1e-7
    grad = np.zeros_like(x)
    f0 = _objective(x)
    for i in range(len(x)):
        x_old = x[i]
        x[i] += eps
        f1 = _objective(x)
        x[i] = x_old - eps
        f2 = _objective(x)
        x[i] = x_old
        grad[i] = (f1 - f2) / (2 * eps)
    return grad

def _project_to_signed(x):
    """
    Project x to signed values: each entry is either +1 or -1.
    Use sign function with a small threshold for stability.
    """
    return np.sign(x)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 145s to leave safety margin before 150s kill
    
    best_x = None
    best_score = float('inf')
    
    # Strategy: Use a block/pulse design approach.
    # Represent x as a sum of a few rectangular pulses with signs.
    # Then refine via local search.
    
    # Initial guess: alternating blocks
    # Try several initial structures
    
    candidates = []
    
    # 1. Simple alternating signs
    x_alt = np.ones(N)
    x_alt[1::2] = -1
    candidates.append(x_alt)
    
    # 2. Block pattern: +1 for first half, -1 for second half
    x_block = np.ones(N)
    x_block[N//2:] = -1
    candidates.append(x_block)
    
    # 3. Random signed vectors
    np.random.seed(42)
    for _ in range(5):
        x_rand = np.random.choice([-1, 1], size=N)
        candidates.append(x_rand)
    
    # 4. Pulse-based design: a few wide pulses
    # Try designs with k pulses of equal width
    for k in [2, 3, 4, 5]:
        width = N // k
        for shift in range(min(3, width)):
            x_pulse = np.zeros(N)
            for i in range(k):
                start = i * width + shift
                end = min(start + width, N)
                if start < N:
                    x_pulse[start:end] = 1 if i % 2 == 0 else -1
            # Fill remaining with sign of last pulse
            if end < N:
                x_pulse[end:] = 1 if (k-1) % 2 == 0 else -1
            candidates.append(x_pulse)
    
    # Evaluate all candidates
    for x in candidates:
        score = _objective(x)
        if score < best_score:
            best_score = score
            best_x = x.copy()
        if time.time() > deadline - 30:
            break
    
    # Local search refinement: flip individual bits
    if best_x is not None and time.time() < deadline - 10:
        x = best_x.copy()
        current_score = best_score
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # Try flipping each position
            for i in range(N):
                x_old = x[i]
                x[i] = -x_old
                new_score = _objective(x)
                if new_score < current_score - 1e-10:
                    current_score = new_score
                    improved = True
                    if new_score < best_score:
                        best_score = new_score
                        best_x = x.copy()
                else:
                    x[i] = x_old
                if time.time() > deadline - 15:
                    break
    
    # Simulated annealing for further refinement
    if best_x is not None and time.time() < deadline - 5:
        x = best_x.copy()
        current_score = best_score
        T = 0.01
        cooling = 0.95
        max_iters = 5000
        iters = 0
        while T > 1e-6 and iters < max_iters and time.time() < deadline - 5:
            iters += 1
            i = np.random.randint(N)
            x_old = x[i]
            x[i] = -x_old
            new_score = _objective(x)
            delta = new_score - current_score
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_x = x.copy()
            else:
                x[i] = x_old
            T *= cooling
    
    if best_x is None:
        best_x = np.ones(N)
    
    return best_x
# EVOLVE-BLOCK-END
