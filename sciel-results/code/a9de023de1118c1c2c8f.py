"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    N = 1000
    # FFT size for convolution: need to handle sums up to 2000
    fft_size = 4096
    # Precompute FFT of indicator vector
    # We'll use a greedy approach with FFT to check validity quickly
    
    # Start with a random initial set
    best_set = None
    best_size = 0
    
    # First, try the simple greedy to get a baseline
    chosen, counts = [], {}
    for x in range(1, 1001):
        new_sums = [x + y for y in chosen] + [2 * x]
        trial = dict(counts)
        valid = True
        for s in new_sums:
            trial[s] = trial.get(s, 0) + 1
            if trial[s] > 2:
                valid = False
                break
        if valid:
            chosen.append(x)
            counts = trial
    best_set = list(chosen)
    best_size = len(chosen)
    
    # Now try FFT-based local search / greedy with different ordering
    # Use FFT to compute representation counts for a candidate set
    
    def compute_counts_fft(indicator):
        """Compute representation counts using FFT convolution."""
        # indicator is a boolean array of length N+1 (index 0 unused)
        # We want c[s] = number of pairs (a,b) with a<=b, a+b=s, both in set
        # Using convolution: if f(x) = 1 for x in set, then (f*f)[s] = sum_a f[a]*f[s-a]
        # This counts ordered pairs. For a<=b: 
        #   if s is even, the diagonal term f[s/2]^2 is counted once in ordered
        #   For a<b, both (a,b) and (b,a) appear, so ordered count = 2*(# with a<b) + 1*(# with a=b)
        #   We want: # with a<b + # with a=b = (ordered - 1)/2 + 1 if s even and s/2 in set, else (ordered)/2
        # Simpler: just compute ordered and check if ordered/2 <= 2 (approximately)
        # Actually let's be precise:
        #   Let O[s] = sum_a f[a] * f[s-a] (ordered pairs, a and b can be equal)
        #   Number of unordered pairs with a<=b: 
        #     if s even: (O[s] - f[s//2]) / 2 + f[s//2]  = (O[s] + f[s//2]) / 2
        #     if s odd:  O[s] / 2
        #   We need this <= 2 for all s
        
        f = np.zeros(fft_size)
        f[1:N+1] = indicator[1:N+1].astype(float)
        F = np.fft.fft(f)
        conv = np.fft.ifft(F * F).real
        # conv[s] = O[s] for s=0..fft_size-1
        # Check constraints
        max_rep = 0
        for s in range(2, 2*N+1):
            if s % 2 == 0:
                mid = s // 2
                if 1 <= mid <= N:
                    rep = (conv[s] + indicator[mid]) / 2.0
                else:
                    rep = conv[s] / 2.0
            else:
                rep = conv[s] / 2.0
            if rep > max_rep:
                max_rep = rep
        return max_rep
    
    def is_valid_fft(indicator):
        """Check if indicator vector represents a valid B2[2] set."""
        max_rep = 0
        f = np.zeros(fft_size)
        f[1:N+1] = indicator[1:N+1].astype(float)
        F = np.fft.fft(f)
        conv = np.fft.ifft(F * F).real
        for s in range(2, 2*N+1):
            if s % 2 == 0:
                mid = s // 2
                if 1 <= mid <= N:
                    rep = (conv[s] + indicator[mid]) / 2.0
                else:
                    rep = conv[s] / 2.0
            else:
                rep = conv[s] / 2.0
            if rep > 2.0 + 1e-9:
                return False
        return True
    
    # Greedy with FFT: try adding elements in different orders
    # Strategy: try multiple random permutations, use FFT to quickly check
    
    np.random.seed(42)
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Create a random permutation of candidates
        candidates = list(range(1, N+1))
        np.random.shuffle(candidates)
        
        # Greedy: add element if it doesn't violate constraint
        indicator = np.zeros(N+1, dtype=np.uint8)
        chosen = []
        
        for x in candidates:
            if time.time() > deadline:
                break
            indicator[x] = 1
            if not is_valid_fft(indicator):
                indicator[x] = 0
            else:
                chosen.append(x)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = list(chosen)
    
    # Local search: try to swap elements to increase size
    if time.time() < deadline - 5:
        # Try removing an element and adding two others
        for _ in range(100):
            if time.time() > deadline:
                break
            
            indicator = np.zeros(N+1, dtype=np.uint8)
            for x in best_set:
                indicator[x] = 1
            
            # Try random swaps
            for _ in range(20):
                if time.time() > deadline:
                    break
                if len(best_set) < 2:
                    break
                # Remove a random element
                idx = np.random.randint(len(best_set))
                removed = best_set[idx]
                indicator[removed] = 0
                new_set = [x for i, x in enumerate(best_set) if i != idx]
                
                # Try to add two random elements
                added = []
                for _ in range(2):
                    x = np.random.randint(1, N+1)
                    if indicator[x] == 0:
                        indicator[x] = 1
                        if is_valid_fft(indicator):
                            added.append(x)
                        else:
                            indicator[x] = 0
                
                if len(added) == 2:
                    new_set.extend(added)
                    if len(new_set) > len(best_set):
                        best_set = new_set
                        best_size = len(best_set)
                    else:
                        # Revert
                        for x in added:
                            indicator[x] = 0
                        indicator[removed] = 1
                else:
                    # Revert
                    for x in added:
                        indicator[x] = 0
                    indicator[removed] = 1
    
    return best_set
# EVOLVE-BLOCK-END
