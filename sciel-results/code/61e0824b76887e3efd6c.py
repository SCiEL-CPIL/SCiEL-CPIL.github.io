"""Additive bases with few representations – structural redesign.

Strategy:
  1. Start from a valid construction (contiguous block) that guarantees coverage.
  2. Use a greedy local-search / simulated-annealing style heuristic to reduce
     the peak multiplicity of pair sums in [128, 256].
  3. Represent A as a boolean numpy array of length 129.
  4. Compute pair-sum multiplicities via convolution (FFT) for speed.
  5. Move elements in/out of A to balance the distribution of sums.
"""
import numpy as np
import time
import random

def construct():
    """Return a nonempty subset A of [0,128] such that every integer in
    [128,256] is a sum of two elements of A, minimizing the largest
    representation multiplicity."""
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    n = 128
    # A is a boolean array of length n+1 = 129
    # Initial guess: contiguous block that is known to work.
    # Using [0, 128] gives all sums 0..256 but with high multiplicity.
    # A better start: a random valid set or a structured one.
    
    # Let's start with a random set of size ~80 and repair it.
    # But to guarantee coverage, we can start with a known valid set
    # and then optimize.
    
    # Known valid: A = [0, 128] (all elements). Sums cover [0, 256].
    # But multiplicity is very high.
    
    # Better: A = [0, 63] ∪ [64, 128]? That's just [0,128].
    
    # Let's try a structured approach:
    # A = [0, 128] is valid. Peak multiplicity for sum s in [128,256]:
    # number of pairs (a,b) with a+b=s, a,b in [0,128].
    # For s in [128, 192]: multiplicity = s - 0 ... wait, a,b >= 0, a,b <= 128.
    # For s <= 128: pairs are (0,s), (1,s-1), ..., (s,0) but a,b <= 128.
    #   If s <= 128: all pairs (a, s-a) for a in [0,s] are valid if s-a <= 128, which is true.
    #   So multiplicity = s+1 for s <= 128.
    # For s in [128, 256]: 
    #   a ranges from max(0, s-128) to min(128, s).
    #   Number of values = min(128, s) - max(0, s-128) + 1.
    #   For s in [128, 256]: max(0, s-128) = s-128, min(128, s) = 128 if s>=128.
    #   So count = 128 - (s-128) + 1 = 257 - s.
    #   At s=128: 257-128 = 129. At s=256: 257-256 = 1.
    # So peak is 129 at s=128.
    
    # We want to reduce this peak.
    
    # Let's use a random valid set and optimize.
    # A set A is valid if for every s in [128, 256], there exists a in A such that s-a in A.
    
    # Start with a random set of size ~70-90 and check validity.
    # If invalid, add elements to repair.
    
    best_A = None
    best_peak = float('inf')
    
    # Initial candidate: the full set [0, 128]
    A_full = np.ones(n+1, dtype=bool)
    best_A = A_full.copy()
    best_peak = _peak_multiplicity(A_full)
    
    # Try random restarts with local search
    rng = np.random.default_rng(42)
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Random initialization: pick ~75-95 random elements
        size = rng.integers(70, 100)
        A = np.zeros(n+1, dtype=bool)
        A[rng.choice(n+1, size=size, replace=False)] = True
        
        # Repair: ensure coverage
        A = _repair(A, n)
        
        # Local search
        A, peak = _local_search(A, n, rng, deadline)
        
        if peak < best_peak:
            best_peak = peak
            best_A = A.copy()
        
        if best_peak <= 2:  # Can't do better than 1 or 2 realistically
            break
    
    # Convert boolean array to list of integers
    result = [i for i in range(n+1) if best_A[i]]
    return result


def _peak_multiplicity(A):
    """Compute the peak multiplicity of pair sums in [128, 256]."""
    # Use convolution to compute all pair sums
    # A is boolean array of length 129
    # Pair sum s has multiplicity = number of (a,b) with a+b=s, a,b in A
    # This is the convolution of A with A
    conv = np.convolve(A.astype(float), A.astype(float), mode='full')
    # conv[s] = sum of A[a]*A[s-a] for all a
    # We want sums in [128, 256]
    peak = np.max(conv[128:257])
    return int(peak)


def _repair(A, n):
    """Repair A to ensure coverage of [128, 256]."""
    A = A.copy()
    # Check which sums are missing
    conv = np.convolve(A.astype(float), A.astype(float), mode='full')
    
    for s in range(128, 257):
        if conv[s] == 0:
            # Need to add elements to cover s
            # Try adding a random element that helps
            # For sum s, we need a in A and s-a in A
            # Add a random a, then check if s-a is in A or add it too
            attempts = 0
            while conv[s] == 0 and attempts < 200:
                a = np.random.randint(0, n+1)
                b = s - a
                if 0 <= b <= n:
                    A[a] = True
                    A[b] = True
                    # Recompute conv (expensive but okay for repair)
                    conv = np.convolve(A.astype(float), A.astype(float), mode='full')
                attempts += 1
            # If still not covered, add all elements in [0, n] as last resort
            if conv[s] == 0:
                A[:] = True
                conv = np.convolve(A.astype(float), A.astype(float), mode='full')
                break
    
    return A


def _local_search(A, n, rng, deadline):
    """Local search to reduce peak multiplicity."""
    A = A.copy()
    peak = _peak_multiplicity(A)
    conv = np.convolve(A.astype(float), A.astype(float), mode='full')
    
    while time.time() < deadline:
        # Pick a random move: add or remove an element
        i = rng.integers(0, n+1)
        
        if A[i]:
            # Try removing i
            A[i] = False
            conv = np.convolve(A.astype(float), A.astype(float), mode='full')
            # Check coverage
            if np.min(conv[128:257]) >= 1:
                new_peak = int(np.max(conv[128:257]))
                if new_peak <= peak:
                    peak = new_peak
                    # Accept
                else:
                    A[i] = True  # Reject
            else:
                A[i] = True  # Reject, coverage broken
        else:
            # Try adding i
            A[i] = True
            conv = np.convolve(A.astype(float), A.astype(float), mode='full')
            new_peak = int(np.max(conv[128:257]))
            if new_peak < peak:
                peak = new_peak
                # Accept
            else:
                # Accept with some probability (simulated annealing-like)
                # For simplicity, reject if not better
                A[i] = False
    
    return A, peak
