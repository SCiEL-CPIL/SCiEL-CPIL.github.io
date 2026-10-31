"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.signal import fftconvolve


def _score(signs):
    """Compute the C3-prime value for a binary sign sequence."""
    s = signs.astype(np.int32)
    integral = np.sum(s)
    if integral == 0:
        return np.inf
    denom = integral ** 2
    ac = fftconvolve(s, s, mode='full')
    max_ac = np.max(np.abs(ac))
    return float(max_ac) / denom


def construct():
    """
    Construct a signed step function (binary signs in {-1, 1}) that minimizes
    the maximum absolute autoconvolution divided by squared integral.
    
    For unit-magnitude steps, the integral is sum(s), and we want to minimize
    max|ac| / (sum(s))^2. We use discrete gradient descent on sign flips.
    """
    # Start with all ones, then apply structured perturbations
    signs = np.ones(N, dtype=np.int32)
    
    # Try several different initial patterns and keep the best
    best_signs = signs.copy()
    best_score = _score(signs)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. All ones
    candidates.append(np.ones(N, dtype=np.int32))
    
    # 2. Alternating
    alt = np.ones(N, dtype=np.int32)
    alt[1::2] = -1
    candidates.append(alt)
    
    # 3. Blocks of 2
    blocks2 = np.ones(N, dtype=np.int32)
    blocks2[2::4] = -1
    blocks2[3::4] = -1
    candidates.append(blocks2)
    
    # 4. Blocks of 4
    blocks4 = np.ones(N, dtype=np.int32)
    blocks4[4::8] = -1
    blocks4[5::8] = -1
    blocks4[6::8] = -1
    blocks4[7::8] = -1
    candidates.append(blocks4)
    
    # 5. Random patterns
    rng = np.random.RandomState(42)
    for _ in range(5):
        rand_signs = rng.choice([-1, 1], size=N).astype(np.int32)
        candidates.append(rand_signs)
    
    # 6. Thue-Morse-like pattern
    tm = np.ones(N, dtype=np.int32)
    for i in range(N):
        # Count trailing ones in binary representation
        count = 0
        n = i
        while n & 1:
            count += 1
            n >>= 1
        if count % 2 == 1:
            tm[i] = -1
    candidates.append(tm)
    
    # Evaluate all candidates
    for cand in candidates:
        score = _score(cand)
        if score < best_score:
            best_score = score
            best_signs = cand.copy()
    
    signs = best_signs.copy()
    current_score = best_score
    
    # Local search: try flipping individual signs
    max_iterations = 50
    no_improve_count = 0
    
    for iteration in range(max_iterations):
        improved = False
        # Shuffle the order of positions to avoid bias
        perm = np.random.RandomState(iteration).permutation(N)
        
        for i in perm:
            original_sign = signs[i]
            signs[i] = -original_sign
            
            new_score = _score(signs)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                signs[i] = original_sign
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= 3:
                break
        else:
            no_improve_count = 0
    
    return signs
# EVOLVE-BLOCK-END
