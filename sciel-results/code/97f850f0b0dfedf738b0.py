"""
Erdos minimum overlap: Block-Interleaved Multi-Level Strategy.

Strategy:
1. Represent the sequence h as a piecewise constant function with K distinct levels.
2. Arrange these levels in a block-interleaved pattern to ensure good spreading properties.
3. Use a discrete search over the block structure (periods and shifts) and continuous 
   optimization over the level values to minimize the maximum overlap.
4. The key insight is that random or binary SA gets stuck in local minima of the 
   autocorrelation landscape. Structured blocks (like those in Golay sequences or 
   Zadoff-Chu derived patterns) provide a better starting point for the level optimization.
"""

import numpy as np
import time

N = 100

def _compute_overlap(h):
    """
    Compute the maximum overlap of h with shifted (1-h).
    Overlap at shift k: sum_i h_i * (1 - h_{(i+k)%N})
    = sum_i h_i - sum_i h_i * h_{(i+k)%N}
    = N/2 - C_k
    where C_k is the circular autocorrelation.
    Minimizing max overlap is equivalent to maximizing min C_k.
    """
    # Ensure sum is exactly N/2 for valid score calculation context, 
    # though the formula works generally.
    H = np.fft.fft(h)
    C = np.fft.ifft(H * np.conj(H)).real
    min_c = np.min(C)
    return N / 2.0 - min_c

def _optimize_levels(mask, values):
    """
    Given a binary mask (0/1) indicating which of two values are used,
    and two values v0, v1, adjust v0, v1 to satisfy sum=N/2 and bounds [0,1]
    to minimize overlap.
    Actually, we can just define h based on a multi-level mask.
    
    For this strategy, we will construct h directly from a pattern.
    """
    pass

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    best_h = np.full(N, 0.5)
    best_score = _compute_overlap(best_h)
    
    # Strategy: Try various structured patterns.
    # Pattern type: Repeated blocks of values.
    # Example: [a, a, b, b, a, a, b, b, ...] or [a, b, b, a, a, b, b, a...]
    # We search over the "skeleton" of the sequence (which positions get which level)
    # and then solve for the optimal level values.
    
    # To be more general, let's use a 3-level structure.
    # h_i = x_{i mod p} where x is a period-p sequence.
    # But N=100 is not prime. 100 = 4 * 25.
    
    # Let's try a "Block Cyclic" approach.
    # Divide N into B blocks of length L.
    # Within each block, the values follow a specific pattern.
    # We optimize the values in the "prototype" block.
    
    # Since we need sum(h) = N/2, average value is 0.5.
    
    # Candidate 1: Binary sequence optimized via local search on the structure
    # but starting from good structures (e.g., Thue-Morse, Golay).
    
    # Let's generate a set of candidate binary masks.
    masks = []
    
    # 1. Random masks
    for _ in range(10):
        masks.append(np.random.randint(0, 2, N))
        
    # 2. Thue-Morse based
    def thue_morse(n):
        t = np.zeros(n, dtype=int)
        for i in range(1, n):
            t[i] = t[i//2] ^ (i % 2)
        return t
    tm = thue_morse(N)
    masks.append(tm)
    masks.append(1 - tm)
    
    # 3. Modular patterns: h_i = 1 if (i * k) % N < N/2
    for k in range(1, N//2 + 1):
        m = np.zeros(N, dtype=int)
        m[(np.arange(N) * k) % N < N//2] = 1
        masks.append(m)
        
    # 4. Block patterns: Alternating blocks of 1s and 0s
    for block_len in range(1, N//2 + 1):
        m = np.zeros(N, dtype=int)
        for i in range(0, N, block_len * 2):
            m[i:i+block_len] = 1
        masks.append(m)
        
    # For each mask, we have a binary structure.
    # We can refine this by allowing the "1"s to be a and "0"s to be b.
    # Constraint: k*a + (N-k)*b = N/2.
    # This implies b = (N/2 - k*a) / (N-k).
    # We want to minimize overlap.
    
    # However, the current seed score is 1.0 (overlap 25).
    # To beat 1.0, we need overlap < 25.
    # Overlap = 50 - min_autocorr.
    # So we need min_autocorr > 25.
    # For random binary, autocorr is roughly 0.
    # For constant 0.5, autocorr is 25.
    # We need a sequence with "high" minimum autocorrelation.
    # This is equivalent to a "Low Autocorrelation" sequence? No.
    # C_k = sum h_i h_{i+k}.
    # If h is constant 0.5, C_k = 25 for all k. Min is 25. Overlap 25.
    # If we can make C_k > 25 for all k, we win.
    # But sum h_i^2 is fixed? No.
    # If h is binary {0,1}, sum h_i = 50.
    # C_0 = sum h_i^2 = 50.
    # Average C_k = (1/N) sum C_k = (1/N) (sum h_i)^2 = 25.
    # So the average autocorrelation is 25.
    # To have min C_k > 25, ALL C_k must be > 25 (since avg is 25).
    # This is impossible for binary sequences because C_0 = 50 > 25, but others fluctuate around 25.
    # In fact, for binary, min C_k is typically < 25.
    # Thus, binary sequences usually have overlap > 25 (score < 1.0).
    
    # Therefore, we MUST use non-binary values (a != b) to "smooth" the autocorrelation.
    # If we use a and b close to 0.5, we approach the constant case.
    # The gain comes from the structure reducing the variance of C_k.
    
    # Let's optimize the levels a, b for each mask.
    
    def eval_mask_optimized(mask):
        k = np.sum(mask)
        if k == 0 or k == N:
            return 25.0, np.full(N, 0.5)
        
        # b = (50 - k*a) / (N-k)
        # We need 0 <= a <= 1 and 0 <= b <= 1.
        
        # Let's scan a small range of 'a' values.
        # Symmetry suggests a and b around 0.5.
        # Let a = 0.5 + delta, b = 0.5 - delta * k / (N-k).
        
        best_overlap = np.inf
        best_h = None
        
        # Scan delta
        # Max delta such that a,b in [0,1]
        # a <= 1 => delta <= 0.5
        # b <= 1 => 0.5 - delta * r <= 1 => -delta * r <= 0.5 => delta >= -0.5/r
        # b >= 0 => 0.5 - delta * r >= 0 => delta * r <= 0.5 => delta <= 0.5/r
        # a >= 0 => delta >= -0.5
        
        r = k / (N - k)
        d_min = max(-0.5, -0.5/r)
        d_max = min(0.5, 0.5/r)
        
        if d_min >= d_max:
             return 25.0, np.full(N, 0.5)
            
        deltas = np.linspace(d_min, d_max, 50)
        
        for d in deltas:
            a = 0.5 + d
            b = 0.5 - d * r
            h = np.where(mask == 1, a, b)
            # Check bounds strictly
            if np.any(h < -1e-9) or np.any(h > 1 + 1e-9):
                continue
            h = np.clip(h, 0, 1)
            ov = _compute_overlap(h)
            if ov < best_overlap:
                best_overlap = ov
                best_h = h.copy()
                
        return best_overlap, best_h

    # Iterate through candidates
    for m in masks:
        if time.time() > deadline:
            break
        ov, h = eval_mask_optimized(m)
        if ov < best_score:
            best_score = ov
            best_h = h
            
    # Local search on the best found h?
    # The space of h is continuous. 
    # We can try small perturbations on best_h.
    # But h is constrained to sum=50 and [0,1].
    
    # Let's try a few random restarts of the "level optimization" with different masks
    # specifically focusing on masks with low variance in autocorrelation.
    
    # Alternative: Directly optimize h using gradient descent?
    # Objective: min max_k (N/2 - C_k)
    # This is non-smooth.
    
    # Let's stick to the discrete structure search but expand the search space.
    # Try "Gray Code" like sequences?
    
    # Add a few more structured masks
    # 1. Alternating single elements
    m_alt = np.array([1,0]*50)
    masks.append(m_alt)
    
    # 2. Blocks of 1, 2, 3, 4...
    for pattern_len in [2, 4, 5, 10, 20, 25, 50]:
         if pattern_len > N: continue
         m = np.zeros(N, dtype=int)
         # Simple repeating pattern of 1s and 0s
         # e.g. 1,0,1,1,0,0...
         # Let's just use modular arithmetic with different moduli
         for mod in [3, 4, 5, 7, 10, 13, 20, 25]:
             if N % mod == 0:
                 # Create a base pattern of length mod
                 # Try all subsets of size mod/2? Too many.
                 # Just try a few random subsets of the base pattern
                 for _ in range(5):
                     base = np.random.randint(0, 2, mod)
                     # Ensure balance in base? Not strictly necessary if we optimize a,b
                     m = np.tile(base, N//mod)
                     masks.append(m)
                     
    for m in masks:
        if time.time() > deadline:
            break
        ov, h = eval_mask_optimized(m)
        if ov < best_score:
            best_score = ov
            best_h = h
            
    return best_h

if __name__ == "__main__":
    # Quick test
    h = construct()
    print(f"Score: {1.0 / _compute_overlap(h)}")
    print(f"Overlap: {_compute_overlap(h)}")
    print(f"Sum: {np.sum(h)}")
