"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize

N = 400

# EVOLVE-BLOCK-START

def _autoconv_full(x):
    """Full linear autoconvolution of x (length 2N-1)."""
    n = len(x)
    size = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(x, n=size)
    conv = np.fft.irfft(F * F, n=size)
    return conv[:2 * n - 1]

def _score_offdiag(x):
    """
    Compute the metric: max(|autoconv|) for k != 0 / (sum(x))^2
    We want to minimize this.
    """
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return 1e12
    conv = _autoconv_full(x)
    if len(conv) <= 1:
        return 0.0
    max_abs_offdiag = np.max(np.abs(conv[1:]))
    return max_abs_offdiag / (integral * integral)

def _generate_eca_sequence(rule, initial_state, length):
    """
    Generate a sequence of length 'length' by evolving an Elementary Cellular Automaton.
    rule: 0-255
    initial_state: binary array of length 'length'
    Returns: float array of +1/-1 of length 'length'
    """
    # Convert to 0/1 for ECA logic
    # We want the output to be +/- 1.
    # Let's map 0 -> -1, 1 -> +1.
    
    # ECA update rule
    # new[i] = rule bit corresponding to (left, center, right)
    
    # Precompute rule bits for speed
    rule_bits = np.array([(rule >> i) & 1 for i in range(7)], dtype=np.uint8)
    # Indices: 111 (7), 110 (6), 101 (5), 100 (4), 011 (3), 010 (2), 001 (1), 000 (0)
    # But numpy indexing for rule bits is usually 0-7 corresponding to binary 000-111.
    # Let's map pattern (l, c, r) -> index 4*l + 2*c + r
    pattern_to_bit = np.zeros(8, dtype=np.uint8)
    for pat in range(8):
        # pat is binary lcr
        # bit index in rule is pat
        pattern_to_bit[pat] = (rule >> pat) & 1
    
    # State as 0/1 integers
    state = initial_state.copy()
    
    # Evolve for a few steps to "mix" the state
    # 1 step is often enough for random init, but let's do 1-2 steps
    # Actually, just taking the initial state is also a valid "trajectory" (length 1).
    # But the prompt suggests "trajectory". Let's evolve 1 step to get non-trivial structure.
    
    # Wrap-around boundary conditions
    left = np.roll(state, 1)
    right = np.roll(state, -1)
    
    # Calculate pattern index
    # pattern = 4*left + 2*center + 1*right
    pattern_idx = 4 * left + 2 * state + right
    
    new_state = pattern_to_bit[pattern_idx]
    
    # Convert 0/1 to -1/+1
    # 0 -> -1, 1 -> +1
    return 2.0 * new_state - 1.0

def _chunk_flip_search(current, current_score, best_score, best_h, rng, deadline, n, use_offdiag=True):
    """
    Fast bit-flip local search.
    """
    size = 1 << (2 * n - 1).bit_length()
    
    # Track peak location for targeting (off-diagonal)
    ac = _autoconv_full(current)
    if use_offdiag and len(ac) > 1:
        peak_idx = np.argmax(np.abs(ac[1:])) + 1
    else:
        peak_idx = np.argmax(np.abs(ac))
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            break
        
        # Peak-targeted single bit flip
        # 60% chance to target near peak, 40% random
        if rng.random() < 0.6:
            target = peak_idx % n
            spread = max(5, n // 8)
            start = int(rng.integers(max(0, target - spread), min(n - 1, target + spread) + 1))
        else:
            start = int(rng.integers(0, n))
            
        # Flip bit at start
        current[start] = -current[start]
        
        # Compute new score
        s = np.sum(current)
        if abs(s) < 1e-12:
            new_score = np.inf
        else:
            H = np.fft.rfft(current, size)
            ac_new = np.fft.irfft(H * np.conj(H), size)[:2 * n - 1]
            if use_offdiag and len(ac_new) > 1:
                max_abs = np.max(np.abs(ac_new[1:]))
            else:
                max_abs = np.max(np.abs(ac_new))
            new_score = max_abs / (s * s)
        
        # Accept or reject
        if new_score <= current_score:
            current_score = new_score
            no_improve_count = 0
            if new_score < best_score:
                best_score = new_score
                best_h = current.copy()
                if use_offdiag and len(ac_new) > 1:
                    peak_idx = np.argmax(np.abs(ac_new[1:])) + 1
                else:
                    peak_idx = np.argmax(np.abs(ac_new))
        else:
            current[start] = -current[start]
            no_improve_count += 1
    
    return current, current_score, best_score, best_h

def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    
    rng = np.random.default_rng(42)
    
    best_score = np.inf
    best_h = np.ones(N)
    
    # Phase 1: ECA Search
    # Sample a diverse set of ECA rules and initial states
    # Total budget for ECA generation + scoring: ~60s
    eca_deadline = t0 + 60.0
    
    # We will test a few thousand combinations
    # Rules: 0-255
    # Initial states: random binary vectors
    
    # To speed up, we can vectorize the initial state generation?
    # No, ECA is iterative. But 1 step is vectorizable.
    
    # Let's try all 256 rules with ~100 random initial states each?
    # 256 * 100 = 25,600 candidates.
    # Each candidate: 1 FFT (size 512) -> very fast.
    # 25,600 * ~10us = 0.25s. We can do much more.
    
    # Let's do 500 random initial states per rule?
    # Or just random pairs of (rule, init_state)
    
    num_candidates = 20000
    
    # Generate candidates in batches to monitor time
    batch_size = 1000
    batches = num_candidates // batch_size
    
    for b in range(batches):
        if time.time() > eca_deadline:
            break
            
        # Random rules
        rules = rng.integers(0, 256, size=batch_size)
        
        # Random initial states (binary)
        # Shape: (batch_size, N)
        inits = rng.integers(0, 2, size=(batch_size, N), dtype=np.uint8)
        
        # Vectorized ECA step 1
        # left = roll(inits, 1, axis=1)
        # right = roll(inits, -1, axis=1)
        # pattern = 4*left + 2*inits + right
        
        left = np.roll(inits, 1, axis=1)
        right = np.roll(inits, -1, axis=1)
        pattern_idx = 4 * left + 2 * inits + right # shape (batch_size, N)
        
        # We need to apply the rule bits.
        # rules shape (batch_size,)
        # We need to extract the bit from `rules` corresponding to `pattern_idx`
        # This is tricky to vectorize directly without lookup tables per rule.
        
        # Alternative: Precompute a lookup table for all 256 rules?
        # LUT[rule, pattern] -> bit
        # 256 * 8 = 2048 entries. Small.
        
        # Let's build the LUT once outside the loop
        # Actually, let's build it inside construct but outside the batch loop
        # Wait, I'm inside the loop. Let's restructure.
        
        # For now, let's just do it in a python loop for the batch if vectorization is complex.
        # Or better: build the LUT before the batch loop.
        
        # Since I can't easily refactor the loop structure without breaking the code block flow,
        # I will assume the LUT is available or build it once.
        # Let's build the LUT here if not exists? No, just build it once at start of construct.
        
        # I will rewrite the loop to use a precomputed LUT.
        
        pass # This block is replaced by the optimized version below

    # Optimized ECA Search with LUT
    # Precompute LUT: lut[rule, pattern] -> bit (0 or 1)
    lut = np.zeros((256, 8), dtype=np.uint8)
    for r in range(256):
        for p in range(8):
            lut[r, p] = (r >> p) & 1
            
    # Re-run ECA search with LUT
    eca_deadline = t0 + 60.0
    batch_size = 2000
    # Estimate time: 2000 candidates * (1 FFT + overhead) ~ 20ms?
    # 60s / 0.02s = 3000 batches.
    # We can do many more.
    
    while time.time() < eca_deadline:
        # Random rules
        rules = rng.integers(0, 256, size=batch_size)
        
        # Random initial states
        inits = rng.integers(0, 2, size=(batch_size, N), dtype=np.uint8)
        
        # ECA Step 1
        left = np.roll(inits, 1, axis=1)
        right = np.roll(inits, -1, axis=1)
        pattern_idx = 4 * left + 2 * inits + right
        
        # Apply LUT
        # We need lut[rules[i], pattern_idx[i, j]]
        # Advanced indexing:
        # indices = pattern_idx (shape B, N)
        # rule_indices = rules[:, np.newaxis] (shape B, 1) -> broadcast to (B, N)
        # new_states = lut[rule_indices, pattern_idx]
        
        rule_indices = rules[:, np.newaxis]
        new_states = lut[rule_indices, pattern_idx]
        
        # Convert to +/- 1
        signals = 2.0 * new_states - 1.0
        
        # Score each signal
        # Vectorized FFT?
        # np.fft.rfft supports batched input along last axis?
        # Yes, if shape is (..., N).
        
        size = 1 << (2 * N - 1).bit_length()
        
        # Compute FFT for batch
        F = np.fft.rfft(signals, n=size, axis=1)
        conv = np.fft.irfft(F * F, n=size, axis=1)
        conv = conv[:, :2 * N - 1]
        
        # Compute scores
        integrals = np.sum(signals, axis=1)
        
        # Avoid division by zero
        valid = np.abs(integrals) > 1e-12
        
        scores = np.full(batch_size, 1e12)
        
        if np.any(valid):
            # max_abs_offdiag
            # conv[:, 1:]
            abs_conv = np.abs(conv[:, 1:])
            max_abs = np.max(abs_conv, axis=1)
            
            # Calculate score
            scores[valid] = max_abs[valid] / (integrals[valid] * integrals[valid])
            
            # Find best in this batch
            best_in_batch_idx = np.argmin(scores)
            best_score_batch = scores[best_in_batch_idx]
            
            if best_score_batch < best_score:
                best_score = best_score_batch
                best_h = signals[best_in_batch_idx].copy()

    # Phase 2: Local Search Refinement
    # Use the best ECA output as seed
    if time.time() < deadline - 10:
        phase2_deadline = deadline - 5
        best_h, best_score, best_score, best_h = _chunk_flip_search(
            best_h.copy(), best_score, best_score, best_h.copy(), rng, phase2_deadline, N, use_offdiag=True
        )
    
    return best_h

# EVOLVE-BLOCK-END
