"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _max_overlap(h):
    """
    Compute the maximum overlap for all shifts k=0..N-1.
    For shift k: overlap_k = sum_{i=0}^{N-k-1} h[i] * (1 - h[i+k]) * dw
    """
    dw = 2.0 / N
    max_overlap = 0.0
    for k in range(N):
        valid = N - k
        if valid <= 0:
            continue
        overlap = np.sum(h[:valid] * (1.0 - h[k:])) * dw
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap

def _max_overlap_fast(h_binary):
    """
    Fast max overlap for binary h (values 0 or 1).
    For binary h, h[i]*(1-h[i+k]) = 1 only when h[i]=1 and h[i+k]=0.
    So overlap_k = count of (h[i]=1, h[i+k]=0) for i in [0, N-k-1] * dw
    """
    dw = 2.0 / N
    max_overlap = 0.0
    for k in range(1, N):
        valid = N - k
        # Count positions where h[i]=1 and h[i+k]=0
        overlap = np.sum(h_binary[:valid] * (1.0 - h_binary[k:])) * dw
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    half = N // 2  # 50
    
    # Strategy: Discrete local search over binary patterns with anti-periodic constraint
    # h[i] + h[i+half] = 1 for i in [0, half)
    # This means if h[i]=1, then h[i+half]=0, and vice versa.
    # The sum constraint: sum(h) = N/2 = 50 is automatically satisfied 
    # if exactly half of the first-half positions are 1 (and corresponding second-half are 0).
    # Wait: sum(h) = sum(first_half) + sum(1-first_half) = sum(first_half) + half - sum(first_half) = half = 50.
    # So ANY binary first_half gives sum = 50. 
    
    # We need to find binary first_half of length 50 that minimizes max overlap.
    # Use simulated annealing / local search.
    
    rng = np.random.RandomState(42)
    
    # Start with a random binary pattern
    best_bits = rng.randint(0, 2, size=half)
    best_bits[half // 2] = 1  # ensure at least one 1
    
    def bits_to_h(bits):
        h = np.empty(N)
        h[:half] = bits.astype(float)
        h[half:] = 1.0 - bits.astype(float)
        return h
    
    best_h = bits_to_h(best_bits)
    best_score = _max_overlap_fast(best_h)
    
    # Local search: flip random bits, accept if improves or with probability (annealing)
    temperature = 0.05
    cooling = 0.999
    no_improve = 0
    max_no_improve = 2000
    
    for iteration in range(100000):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            break
        
        # Pick a random position to flip
        idx = rng.randint(0, half)
        new_bits = best_bits.copy()
        new_bits[idx] = 1 - new_bits[idx]
        
        new_h = bits_to_h(new_bits)
        new_score = _max_overlap_fast(new_h)
        
        if new_score < best_score:
            best_bits = new_bits
            best_h = new_h
            best_score = new_score
            no_improve = 0
        else:
            # Accept with probability exp(-(new_score - best_score)/temperature)
            delta = new_score - best_score
            if delta < 0 or rng.random() < np.exp(-delta / temperature):
                best_bits = new_bits
                best_h = new_h
                best_score = new_score
            no_improve += 1
        
        temperature *= cooling
        if no_improve > max_no_improve:
            # Restart with new random pattern
            no_improve = 0
            temperature = 0.05
            best_bits = rng.randint(0, 2, size=half)
            best_h = bits_to_h(best_bits)
            best_score = _max_overlap_fast(best_h)
    
    # After binary search, try to refine with small continuous perturbations
    # around the best binary solution
    h_refined = best_h.copy()
    
    # Try small perturbations: for each position, try h[i] +/- epsilon
    eps = 0.01
    for _ in range(500):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            break
        
        idx = rng.randint(0, N)
        old_val = h_refined[idx]
        
        # Try perturbing
        for delta in [eps, -eps, 2*eps, -2*eps]:
            new_val = old_val + delta
            if new_val < 0 or new_val > 1:
                continue
            h_test = h_refined.copy()
            h_test[idx] = new_val
            # Must maintain anti-periodic constraint: h[idx] + h[idx+half] = 1
            # So if we change h[idx], we must change h[idx+half] = 1 - h[idx]
            # But if idx >= half, the paired index is idx - half
            if idx < half:
                h_test[idx + half] = 1.0 - new_val
            else:
                h_test[idx - half] = 1.0 - new_val
            
            new_score = _max_overlap(h_test)
            if new_score < best_score:
                h_refined = h_test
                best_score = new_score
                break
        else:
            continue
    
    # Final: ensure sum is exactly N/2
    current_sum = np.sum(h_refined)
    if abs(current_sum - N/2.0) > 1e-12:
        diff = N/2.0 - current_sum
        # Distribute correction evenly, then clip and re-correct
        h_refined = h_refined + diff / N
        h_refined = np.clip(h_refined, 0.0, 1.0)
        current_sum = np.sum(h_refined)
        if abs(current_sum - N/2.0) > 1e-12:
            diff = N/2.0 - current_sum
            # Add to the element closest to 0.5
            idx = np.argmin(np.abs(h_refined - 0.5))
            h_refined[idx] += diff
            h_refined[idx] = np.clip(h_refined[idx], 0.0, 1.0)
    
    return h_refined
# EVOLVE-BLOCK-END
