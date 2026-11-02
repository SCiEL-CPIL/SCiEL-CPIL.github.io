"""Rewritten program: use periodic m-sequence (balanced binary) with local refinement."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 150.0
_START = time.time()

def _time_remaining():
    return _DEADLINE - 5.0 - (time.time() - _START)  # 5s safety margin


def _max_overlap(h):
    """Compute max overlap of h with shifted (1-h) over all circular shifts.
    
    Overlap at shift k = sum_i h[i] * (1 - h[(i+k) % N])
    = sum_i h[i] - sum_i h[i]*h[(i+k)%N]
    = N/2 - autocorr[k]  where autocorr[k] = sum_i h[i]*h[(i+k)%N]
    
    So max overlap = N/2 - min_k autocorr[k]
    """
    # autocorr via FFT
    n = len(h)
    fft_h = np.fft.fft(h)
    power = np.abs(fft_h)**2
    autocorr = np.fft.ifft(power).real
    # autocorr[k] for k=0..n-1 (circular)
    min_ac = np.min(autocorr)
    return n / 2.0 - min_ac


def _score(h):
    """Return the overlap value (lower is better)."""
    return _max_overlap(h)


def _project_to_constraint(h, target_sum=None):
    """Project h to [0,1]^N with sum = N/2 using bisection on Lagrange multiplier."""
    if target_sum is None:
        target_sum = N / 2.0
    
    def clip_sum(tau):
        # h_i = clip(h_i - tau, 0, 1), sum them
        clipped = np.clip(h - tau, 0.0, 1.0)
        return np.sum(clipped)
    
    # Find tau such that sum(clip(h - tau, 0, 1)) = target_sum
    lo, hi = -1.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        s = clip_sum(mid)
        if s > target_sum:
            lo = mid
        else:
            hi = mid
    tau = (lo + hi) / 2.0
    return np.clip(h - tau, 0.0, 1.0)


def _generate_msequence_period(p):
    """Generate a balanced binary sequence with good autocorrelation properties.
    
    For a prime p, use a quadratic residue sequence:
    s[i] = 1 if i^2 mod p is a quadratic residue (and i != 0), else 0.
    This gives a sequence of length p with low autocorrelation.
    """
    p = p
    # Quadratic residues mod p
    residues = set()
    for i in range(1, p):
        residues.add((i * i) % p)
    
    seq = np.zeros(p)
    for i in range(p):
        if i == 0:
            seq[i] = 0.0
        elif i in residues:
            seq[i] = 1.0
        else:
            seq[i] = 0.0
    
    # The sum should be (p-1)/2
    # We need sum = N/2, so if p != N, we tile/truncate
    return seq


def _tile_to_length(seq, n):
    """Tile sequence to length n, then project to have sum n/2."""
    if len(seq) >= n:
        h = seq[:n].copy()
    else:
        repeats = (n + len(seq) - 1) // len(seq)
        h = np.tile(seq, repeats)[:n]
    return _project_to_constraint(h)


def _local_refinement(h, max_iter=500):
    """Local search: try flipping individual bits or small perturbations."""
    h = h.copy()
    current_score = _score(h)
    
    for it in range(max_iter):
        if _time_remaining() < 10.0:
            break
        
        # Try random single-coordinate perturbation
        i = np.random.randint(N)
        old_val = h[i]
        
        # Try moving toward 0 or 1
        if old_val < 0.5:
            new_val = 1.0
        else:
            new_val = 0.0
        
        h[i] = new_val
        h = _project_to_constraint(h)
        new_score = _score(h)
        
        if new_score < current_score:
            current_score = new_score
        else:
            # Revert
            h = _project_to_constraint(h)  # This won't revert exactly, let's be more careful
    
    return h, current_score


def _refine_with_gradient(h, steps=200):
    """Subgradient-like refinement using the shift that achieves max overlap."""
    h = h.copy()
    current_score = _score(h)
    
    for step in range(steps):
        if _time_remaining() < 10.0:
            break
        
        # Find which shift gives the max overlap
        n = len(h)
        fft_h = np.fft.fft(h)
        power = np.abs(fft_h)**2
        autocorr = np.fft.ifft(power).real
        min_ac_idx = np.argmin(autocorr)
        # The problematic shift is min_ac_idx
        # Overlap at shift k: sum_i h[i]*(1-h[(i+k)%n])
        # To reduce this, we want to reduce h[i] where h[(i+k)%n] is also high,
        # or increase h[i] where h[(i+k)%n] is low
        
        k = min_ac_idx % n
        # Gradient of overlap w.r.t. h[i] at this shift:
        # d/dh[i] [sum_j h[j]*(1-h[(j+k)%n])] = (1 - h[(i+k)%n]) - h[(i-k)%n]
        grad = (1.0 - h[(np.arange(n) + k) % n]) - h[(np.arange(n) - k) % n]
        
        # Subgradient step
        lr = 0.05
        h_new = h - lr * grad
        h_new = _project_to_constraint(h_new)
        
        new_score = _score(h_new)
        if new_score < current_score:
            h = h_new
            current_score = new_score
        else:
            # Try smaller step
            lr = 0.01
            h_new = h - lr * grad
            h_new = _project_to_constraint(h_new)
            new_score = _score(h_new)
            if new_score < current_score:
                h = h_new
                current_score = new_score
    
    return h, current_score


def construct():
    global _START
    _START = time.time()
    
    best_h = None
    best_score = float('inf')
    
    # Strategy 1: Try m-sequence based periodic patterns with various primes
    primes_to_try = [97, 89, 83, 79, 73, 71, 67, 61, 59, 53, 47, 43, 41, 37, 31, 29, 23, 19, 17, 13, 11, 7, 5, 3]
    
    for p in primes_to_try:
        if _time_remaining() < 20.0:
            break
        
        seq = _generate_msequence_period(p)
        h = _tile_to_length(seq, N)
        s = _score(h)
        
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # Also try the simple alternating pattern
    h_alt = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(N)])
    h_alt = _project_to_constraint(h_alt)
    s = _score(h_alt)
    if s < best_score:
        best_score = s
        best_h = h_alt.copy()
    
    # Strategy 2: Random restarts with binary-ish initialization
    np.random.seed(42)
    for restart in range(20):
        if _time_remaining() < 15.0:
            break
        
        # Random binary-ish initialization
        h_rand = np.random.randint(0, 2, N).astype(float)
        h_rand = _project_to_constraint(h_rand)
        s = _score(h_rand)
        
        if s < best_score:
            best_score = s
            best_h = h_rand.copy()
    
    # Strategy 3: Refine the best candidate
    if best_h is not None and _time_remaining() > 10.0:
        best_h, best_score = _refine_with_gradient(best_h, steps=100)
    
    # Strategy 4: More random restarts + refinement
    for restart in range(10):
        if _time_remaining() < 10.0:
            break
        
        h_rand = np.random.randint(0, 2, N).astype(float)
        # Add some noise to break symmetry
        h_rand += np.random.randn(N) * 0.1
        h_rand = _project_to_constraint(h_rand)
        s = _score(h_rand)
        
        if s < best_score:
            best_score = s
            best_h = h_rand.copy()
        
        # Quick refinement
        if _time_remaining() > 15.0:
            h_ref, s_ref = _refine_with_gradient(h_rand, steps=50)
            if s_ref < best_score:
                best_score = s_ref
                best_h = h_ref.copy()
    
    # Strategy 5: Final refinement of best
    if best_h is not None and _time_remaining() > 10.0:
        best_h, best_score = _refine_with_gradient(best_h, steps=100)
    
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    # Ensure exact constraint satisfaction
    best_h = _project_to_constraint(best_h)
    
    return best_h
# EVOLVE-BLOCK-END
