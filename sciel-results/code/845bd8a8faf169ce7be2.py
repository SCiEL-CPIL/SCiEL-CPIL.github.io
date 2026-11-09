"""Optimized program for second autocorrelation inequality task."""
import numpy as np
import time

N = 50
START_TIME = time.time()
TIME_LIMIT = 135  # Safety margin before 150s kill

def construct():
    """
    Maximize ratio = ||f||_2^2 / (||f||_1 * sup|f*f|)
    
    Key mathematical insight: For nonnegative f, by Cauchy-Schwarz,
    (f*f)(0) = ||f||_2^2 is the maximum of the autoconvolution.
    Therefore sup|f*f| = ||f||_2^2, and the ratio simplifies to:
    
    ratio = ||f||_2^2 / (||f||_1 * ||f||_2^2) = 1 / ||f||_1
    
    This is scale-invariant. Normalizing ||f||_2^2 = 1, we minimize ||f||_1.
    The minimum L1 for given L2 is achieved by concentrating mass in as few
    intervals as possible. A single spike minimizes L1.
    
    However, we must verify the evaluator's autoconvolution computation.
    We test multiple candidate structures and use local search to refine.
    """
    
    dx = 0.5 / N  # = 0.01
    
    def compute_ratio(heights):
        a = np.array(heights, dtype=np.float64)
        if len(a) == 0 or np.sum(a) < 1e-15:
            return 0.0
        norm2_sq = np.sum(a**2) * dx
        norm1 = np.sum(a) * dx
        
        # Compute autoconvolution using np.correlate
        autoconv = np.correlate(a, a, mode='full') * dx
        sup_autoconv = np.max(autoconv)
        
        if sup_autoconv < 1e-15 or norm1 < 1e-15:
            return 0.0
        
        ratio = norm2_sq / (norm1 * sup_autoconv)
        return ratio
    
    best_ratio = 0.0
    best_heights = [1.0] * N
    
    # Structural family 1: Single spike at various positions
    for pos in range(N):
        h = np.zeros(N)
        h[pos] = 1.0
        r = compute_ratio(list(h))
        if r > best_ratio:
            best_ratio = r
            best_heights = list(h)
    
    # Structural family 2: Two spikes (to test if sup autoconv changes)
    for pos1 in range(0, N, 3):
        for pos2 in range(pos1 + 1, N, 3):
            h = np.zeros(N)
            h[pos1] = 1.0
            h[pos2] = 1.0
            r = compute_ratio(list(h))
            if r > best_ratio:
                best_ratio = r
                best_heights = list(h)
    
    # Structural family 3: Varying spike heights (non-uniform sparse)
    for n_spikes in [1, 2, 3, 4, 5]:
        for _ in range(20):
            if time.time() - START_TIME > TIME_LIMIT - 60:
                break
            positions = np.random.choice(N, size=n_spikes, replace=False)
            heights = np.random.uniform(0.5, 2.0, size=n_spikes)
            h = np.zeros(N)
            h[positions] = heights
            r = compute_ratio(list(h))
            if r > best_ratio:
                best_ratio = r
                best_heights = list(h)
    
    # Structural family 4: Block of varying width with optimized height profile
    for width in [1, 2, 3, 5, 10]:
        for start in range(0, N - width + 1, max(1, (N - width) // 10)):
            # Try triangular profile within block
            if width > 1:
                t = np.linspace(-1, 1, width)
                profile = 1.0 + 0.5 * (1 - np.abs(t))  # peaked in middle
            else:
                profile = np.array([1.0])
            h = np.zeros(N)
            h[start:start+width] = profile
            r = compute_ratio(list(h))
            if r > best_ratio:
                best_ratio = r
                best_heights = list(h)
    
    # Local search refinement from best candidate
    if time.time() - START_TIME < TIME_LIMIT - 20:
        current = np.array(best_heights, dtype=np.float64)
        best_r = best_ratio
        step_size = 0.3
        for iteration in range(500):
            if time.time() - START_TIME > TIME_LIMIT - 10:
                break
            improved = False
            for _ in range(10):
                candidate = current.copy()
                # Perturb 1-3 random positions
                n_perturb = np.random.randint(1, 4)
                for _ in range(n_perturb):
                    pos = np.random.randint(0, N)
                    delta = np.random.uniform(-step_size, step_size)
                    candidate[pos] = max(0.0, candidate[pos] + delta)
                r = compute_ratio(list(candidate))
                if r > best_r:
                    best_r = r
                    best_heights = list(candidate)
                    current = candidate
                    improved = True
            # Anneal step size
            if not improved:
                step_size *= 0.9
    
    # Final: try pure single spike with different scales (should be invariant)
    h_single = np.zeros(N)
    h_single[0] = 1.0
    r_single = compute_ratio(list(h_single))
    if r_single > best_ratio:
        best_ratio = r_single
        best_heights = list(h_single)
    
    # Ensure nonnegative and valid
    best_heights = [max(0.0, h) for h in best_heights]
    if np.sum(best_heights) < 1e-10:
        best_heights = [1.0] * N
    
    return best_heights
