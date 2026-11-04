# EVOLVE-BLOCK-START
import numpy as np
import time

N = 400

def _compute_score(h):
    """Compute the sum of squares of non-zero autocorrelation coefficients."""
    # h is length N, values +/- 1
    # c[k] = sum_{j=0}^{N-1-k} h[j] h[j+k]
    # We want sum_{k=1}^{N-1} c[k]^2
    c = np.zeros(N)
    for k in range(1, N):
        c[k] = np.dot(h[:N-k], h[k:])
    return np.sum(c[1:]**2), c

def construct():
    start_time = time.time()
    deadline = 110.0  # 10s margin
    
    # Initialize with random sequence
    rng = np.random.RandomState(42)
    best_h = rng.choice([-1.0, 1.0], size=N)
    best_score, best_c = _compute_score(best_h)
    
    current_h = best_h.copy()
    current_score, current_c = best_score, best_c
    
    while time.time() < deadline:
        # Compute gradient for each position i
        # G[i] = sum_{k=1}^{N-1} c[k] * (h[i+k] + h[i-k])
        # Flipping h[i] changes c[k] by -2*h[i]*(h[i+k]+h[i-k])
        # Change in score J = sum c[k]^2 is approx 2 * sum c[k] * dc[k]
        # dc[k] = -2*h[i]*(h[i+k]+h[i-k])
        # dJ = -4 * h[i] * sum c[k]*(h[i+k]+h[i-k])
        # We want to minimize J, so we want dJ < 0.
        # Since h[i] is +/- 1, we look for sign(h[i]) * G[i] > 0 to flip?
        # If h[i]=1, dJ = -4 G[i]. We want G[i] > 0.
        # If h[i]=-1, dJ = 4 G[i]. We want G[i] < 0.
        # So we want to flip i if h[i] * G[i] > 0.
        # The magnitude of improvement is 4 * |h[i] * G[i]| = 4 * |G[i]|.
        
        G = np.zeros(N)
        for k in range(1, N):
            ck = current_c[k]
            if ck == 0:
                continue
            # Term h[i+k] contributes to G[i] for i in 0..N-1-k
            G[:N-k] += ck * current_h[k:]
            # Term h[i-k] contributes to G[i] for i in k..N-1
            G[k:] += ck * current_h[:N-k]
        
        # Calculate improvement for each flip
        # dJ_i = -4 * current_h[i] * G[i]
        improvements = -4.0 * current_h * G
        
        # Find best flip
        max_improvement = np.max(improvements)
        if max_improvement > 1e-9:
            idx = np.argmax(improvements)
            current_h[idx] *= -1
            # Recalculate score
            current_score, current_c = _compute_score(current_h)
            if current_score < best_score:
                best_score = current_score
                best_h = current_h.copy()
        else:
            # Stuck in local minimum, try random restart or perturbation
            # Perturb a few random bits
            for _ in range(10):
                idx = rng.randint(N)
                current_h[idx] *= -1
            current_score, current_c = _compute_score(current_h)
            if current_score < best_score:
                best_score = current_score
                best_h = current_h.copy()
            else:
                # Revert perturbation? Or keep going?
                # Let's revert to best known and restart search from there
                current_h = best_h.copy()
                current_score, current_c = _compute_score(current_h)
                # Maybe break and try a different random seed?
                # For now, just continue, but it might loop.
                # Let's add a counter to force restart if stuck too long
                pass

    return best_h
# EVOLVE-BLOCK-END
