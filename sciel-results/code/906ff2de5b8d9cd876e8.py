import numpy as np
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 128
    target_low = 128
    target_high = 256
    num_targets = target_high - target_low + 1  # 129 targets
    
    # Represent A as boolean array of length 129 (indices 0..128)
    # A[i] = True means i is in A
    
    def compute_multiplicities(A_bool):
        """Compute multiplicity for each target in [128,256].
        Returns array of length num_targets where result[k] = number of unordered pairs
        (a,b) with a<=b, a,b in A, a+b = target_low+k."""
        # Use convolution: convolve A with A, but we need unordered pairs with a<=b
        # For unordered pairs allowing equal: count = (conv[s] + (1 if s even and A[s//2] else 0)) // 2
        # Actually: number of ordered pairs (a,b) with a,b in A, a+b=s is conv[s]
        # Number of unordered pairs with a<=b: if s is even and s//2 in A, then (conv[s]+1)//2, else conv[s]//2
        A_arr = A_bool.astype(np.float64)
        conv = np.convolve(A_arr, A_arr, mode='full')
        # conv[s] = number of ordered pairs (a,b) with a+b=s
        mults = np.zeros(num_targets, dtype=np.float64)
        for k in range(num_targets):
            s = target_low + k
            ordered = conv[s] if s < len(conv) else 0.0
            if s % 2 == 0:
                half = s // 2
                if 0 <= half <= n and A_bool[half]:
                    mults[k] = (ordered + 1) / 2.0
                else:
                    mults[k] = ordered / 2.0
            else:
                mults[k] = ordered / 2.0
        return mults
    
    def is_valid(A_bool):
        """Check if every target in [128,256] has at least one representation."""
        mults = compute_multiplicities(A_bool)
        return np.all(mults >= 1.0)
    
    def peak_multiplicity(A_bool):
        """Return the maximum multiplicity, or -1 if invalid."""
        mults = compute_multiplicities(A_bool)
        if np.any(mults < 1.0):
            return -1  # invalid
        return int(np.max(mults))
    
    # Greedy initialization: build A to cover all targets
    # Start with a reasonable initial set
    A_bool = np.zeros(n + 1, dtype=bool)
    
    # Initial strategy: use elements around the middle to cover the target range
    # For target t in [128,256], we need a,b in [0,128] with a+b=t
    # If we include elements from [64,128], then a+b ranges from [128,256]
    # But we want to minimize multiplicity, so let's be smarter
    
    # Start with a sparse initial set and greedily add elements
    # Priority: targets that are currently uncovered
    
    # Initialize with some random elements to break symmetry
    random.seed(42)
    
    # Start with a minimal valid set: {64, 65, ..., 128} works but has high multiplicity
    # Let's try a different approach: start empty and greedily add
    
    # Actually, let's start with the known valid set and try to improve
    A_bool[64:129] = True  # [64,128]
    
    current_peak = peak_multiplicity(A_bool)
    if current_peak < 0:
        # Shouldn't happen with [64,128]
        A_bool = np.zeros(n + 1, dtype=bool)
        A_bool[64:129] = True
        current_peak = peak_multiplicity(A_bool)
    
    best_A = A_bool.copy()
    best_peak = current_peak
    
    # Local search: try removing elements that don't break validity
    # and try adding elements to reduce peak
    
    def try_improve(A_bool, current_peak, max_iters=5000):
        """Local search to reduce peak multiplicity."""
        A = A_bool.copy()
        peak = current_peak
        improved = False
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Try removing an element
            if peak > 1:
                # Find elements whose removal might reduce peak
                # Try random removals
                candidates = [i for i in range(n + 1) if A[i]]
                if candidates:
                    i = random.choice(candidates)
                    A[i] = False
                    new_peak = peak_multiplicity(A)
                    if new_peak >= 0 and new_peak < peak:
                        peak = new_peak
                        improved = True
                    else:
                        A[i] = True  # revert
            
            # Try adding an element to reduce peak
            # If peak is high, adding an element might not help directly,
            # but it could allow removal of other elements later
            # For now, focus on removals
            
            if not improved and peak <= 2:
                break
        
        return A, peak, improved
    
    # Phase 1: Greedy removal
    # Try to remove elements one by one, keeping validity
    for i in range(n + 1):
        if time.time() > deadline - 10:
            break
        if A_bool[i]:
            A_bool[i] = False
            if is_valid(A_bool):
                new_peak = peak_multiplicity(A_bool)
                if new_peak >= 0 and new_peak <= current_peak:
                    current_peak = new_peak
                    best_A = A_bool.copy()
                    best_peak = current_peak
                else:
                    A_bool[i] = True
            else:
                A_bool[i] = True
    
    # Phase 2: Local search with simulated annealing
    temperature = 1.0
    cooling = 0.999
    
    for iteration in range(100000):
        if time.time() > deadline - 5:
            break
        
        # Random move: toggle a random element
        i = random.randint(0, n)
        A_bool[i] = not A_bool[i]
        
        new_peak = peak_multiplicity(A_bool)
        
        if new_peak >= 0:
            # Valid
            delta = new_peak - current_peak
            if delta < 0 or random.random() < np.exp(-delta / max(temperature, 0.01)):
                current_peak = new_peak
                if new_peak < best_peak:
                    best_peak = new_peak
                    best_A = A_bool.copy()
            else:
                A_bool[i] = not A_bool[i]  # revert
        else:
            # Invalid, revert unless we want to explore (but we need validity)
            A_bool[i] = not A_bool[i]
        
        temperature *= cooling
        if temperature < 0.01:
            temperature = 0.01
    
    # Phase 3: Final greedy cleanup
    for i in range(n + 1):
        if time.time() > deadline - 2:
            break
        if best_A[i]:
            best_A[i] = False
            if is_valid(best_A):
                new_peak = peak_multiplicity(best_A)
                if new_peak >= 0 and new_peak <= best_peak:
                    best_peak = new_peak
                else:
                    best_A[i] = True
            else:
                best_A[i] = True
    
    # Return the best found
    result = [i for i in range(n + 1) if best_A[i]]
    if not result:
        # Fallback
        result = list(range(64, 129))
    
    return result
