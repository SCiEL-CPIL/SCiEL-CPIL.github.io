"""Additive base with few representations: NumPy-accelerated local search with targeted moves.

Uses convolution for fast multiplicity computation and prioritizes moves that
reduce the peak multiplicity by targeting high-count sums.
"""
import time
import random
import numpy as np

# Fixed deadline guard (evaluator kills at 90 s wall-clock; keep a large margin).
_DEADLINE = time.time() + 80.0


def _compute_multiplicities_fast(A):
    """Compute unordered-pair representation counts for all s in [128,256] using FFT convolution.
    
    Returns array of length 257 where mult[s] = number of unordered pairs (a,b) in A
    with a <= b and a+b = s.
    """
    if not A:
        return np.zeros(257, dtype=np.int32)
    
    A_arr = np.array(sorted(A), dtype=np.int32)
    # Use FFT-based convolution for speed
    ind = np.zeros(129, dtype=np.float64)
    ind[A_arr] = 1.0
    # Use next power of 2 for FFT efficiency
    n_fft = 256
    ind_fft = np.fft.rfft(ind, n=n_fft)
    conv_fft = np.fft.irfft(ind_fft * ind_fft, n=n_fft)
    conv = np.round(conv_fft[:257]).astype(np.int32)
    
    A_set = set(A)
    mult = np.zeros(257, dtype=np.int32)
    for s in range(128, 257):
        if s % 2 == 0 and s // 2 in A_set:
            mult[s] = (conv[s] + 1) // 2
        else:
            mult[s] = conv[s] // 2
    
    return mult


def _peak(A):
    """Largest unordered-pair representation multiplicity over [128,256]."""
    mult = _compute_multiplicities_fast(A)
    return int(np.max(mult[128:257]))


def _is_valid(A):
    """True iff every s in [128,256] is a sum of two elements of A."""
    mult = _compute_multiplicities_fast(A)
    return bool(np.all(mult[128:257] > 0))


def _get_high_mult_elements(A, mult, threshold):
    """Find elements that appear in sums with multiplicity >= threshold."""
    if not A:
        return set()
    
    A_set = set(A)
    high_elements = set()
    for s in range(128, 257):
        if mult[s] >= threshold:
            for a in A:
                b = s - a
                if a <= b and b in A_set:
                    high_elements.add(a)
                    high_elements.add(b)
    return high_elements


def _generate_random_valid_set(min_size=30, max_size=80):
    """Generate a random set that is likely to cover [128,256]."""
    # Start with a guaranteed valid base
    base = set(range(64, 129))
    
    # Randomly remove some elements while maintaining validity
    A = set(base)
    while len(A) > min_size and time.time() < _DEADLINE - 5:
        candidates = list(A)
        random.shuffle(candidates)
        removed = False
        for elem in candidates:
            new_A = A - {elem}
            if _is_valid(new_A):
                A = new_A
                removed = True
                break
        if not removed:
            break
    
    # Optionally add some random elements to explore different structures
    if len(A) < max_size:
        for v in range(0, 129):
            if v not in A and random.random() < 0.3:
                A.add(v)
                if len(A) >= max_size:
                    break
    
    return A


def _local_search(A, max_iterations=10000, time_budget=None, polish=False):
    """Targeted local search: prioritize moves affecting high-multiplicity sums.
    
    If polish=True, use stricter acceptance and more iterations for final refinement.
    """
    if time_budget is None:
        time_budget = _DEADLINE - time.time() - 5
    
    A = list(set(A))
    mult = _compute_multiplicities_fast(A)
    peak = int(np.max(mult[128:257]))
    best_A = A[:]
    best_peak = peak
    
    # For polish mode, use stricter acceptance
    accept_equal_prob = 0.05 if polish else 0.2
    
    for iteration in range(max_iterations):
        if time.time() > _DEADLINE - 5:
            break
        
        # Identify high-multiplicity elements for targeted removal
        if peak > 1:
            high_elems = _get_high_mult_elements(A, mult, peak)
            if high_elems:
                # Try removing a high-multiplicity element
                candidates = [e for e in A if e in high_elems]
                random.shuffle(candidates)
                for elem in candidates[:3]:  # Try up to 3 candidates
                    cand = [x for x in A if x != elem]
                    if _is_valid(cand):
                        p = _peak(cand)
                        if p < peak:
                            A = cand
                            mult = _compute_multiplicities_fast(A)
                            peak = p
                            best_A = A[:]
                            best_peak = peak
                            break
                        elif p == peak and random.random() < accept_equal_prob:
                            A = cand
                            mult = _compute_multiplicities_fast(A)
                            break
        
        # Try removing a random element (for diversity)
        if len(A) > 10:
            idx = random.randint(0, len(A) - 1)
            cand = A[:idx] + A[idx + 1:]
            if _is_valid(cand):
                p = _peak(cand)
                if p < peak:
                    A = cand
                    mult = _compute_multiplicities_fast(A)
                    peak = p
                    best_A = A[:]
                    best_peak = peak
                elif p == peak and random.random() < accept_equal_prob * 0.5:
                    A = cand
                    mult = _compute_multiplicities_fast(A)
        
        # Try adding a random element
        if len(A) < 128:
            v = random.randint(0, 128)
            if v not in A:
                cand = A + [v]
                p = _peak(cand)
                if p < peak:
                    A = cand
                    mult = _compute_multiplicities_fast(A)
                    peak = p
                    best_A = A[:]
                    best_peak = peak
                elif p == peak and random.random() < accept_equal_prob * 0.25:
                    A = cand
                    mult = _compute_multiplicities_fast(A)
        
        # Try swapping: remove one, add another
        if len(A) > 5 and len(A) < 127:
            idx = random.randint(0, len(A) - 1)
            v = random.randint(0, 128)
            if v not in A:
                cand = A[:idx] + A[idx + 1:] + [v]
                if _is_valid(cand):
                    p = _peak(cand)
                    if p < peak:
                        A = cand
                        mult = _compute_multiplicities_fast(A)
                        peak = p
                        best_A = A[:]
                        best_peak = peak
                    elif p == peak and random.random() < accept_equal_prob * 0.5:
                        A = cand
                        mult = _compute_multiplicities_fast(A)
    
    return best_A, best_peak


def _polish(A):
    """Final high-precision polish pass with more iterations and stricter criteria."""
    A = list(A)
    # Run a focused local search with polish=True for additional refinement
    A_opt, peak = _local_search(A, max_iterations=5000, polish=True)
    return A_opt, peak


def construct():
    t0 = time.time()
    random.seed(42)  # Reproducibility
    
    best_A = None
    best_peak = float('inf')
    
    # Multiple random restarts
    num_restarts = 15
    for restart in range(num_restarts):
        if time.time() > _DEADLINE - 15:
            break
        
        # Generate a random valid starting set
        A = _generate_random_valid_set(min_size=25, max_size=70)
        
        if not _is_valid(A):
            continue
        
        # Run local search
        A_opt, peak = _local_search(A, max_iterations=8000)
        
        if peak < best_peak:
            best_peak = peak
            best_A = A_opt
        
        # Early exit if we found optimal
        if best_peak <= 1:
            break
    
    # If no valid set found, use fallback
    if best_A is None:
        best_A = list(range(64, 129))
        best_peak = _peak(best_A)
    
    # Final polish pass if we have time
    if time.time() < _DEADLINE - 10 and best_peak > 1:
        A_polished, peak_polished = _polish(best_A)
        if peak_polished < best_peak:
            best_A = A_polished
            best_peak = peak_polished
    
    return sorted(best_A)
