import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    N = 128
    TARGETS = list(range(128, 257))
    NUM_TARGETS = len(TARGETS)
    
    # Precompute target indices for quick lookup
    # We only care about t in [128, 256]
    
    def compute_multiplicities(A_set):
        """
        Returns an array of multiplicities for targets 0..256.
        Uses convolution for speed.
        """
        # Create boolean array for A
        A_bool = np.zeros(2 * N + 1, dtype=int)
        for a in A_set:
            A_bool[a] = 1
        
        # Convolution gives count of ordered pairs (a, b) such that a+b=t
        # Note: np.convolve does linear convolution
        conv = np.convolve(A_bool, A_bool)
        
        # conv[t] is the number of ordered pairs (a,b) in A x A such that a+b=t.
        # We want unordered pairs with replacement.
        # m(t) = (conv[t] + delta) / 2
        # where delta = 1 if t is even and t/2 is in A, else 0.
        
        mults = np.zeros(2 * N + 1, dtype=int)
        for t in range(128, 257):
            delta = 0
            if t % 2 == 0:
                half = t // 2
                if half in A_set:
                    delta = 1
            # conv[t] is integer
            # If t is even and half in A, conv[t] is odd (contains (half, half) once, others in pairs)
            # If t is odd or half not in A, conv[t] is even (all in pairs)
            # So (conv[t] + delta) is always even.
            mults[t] = (conv[t] + delta) // 2
            
        return mults

    def is_complete(mults):
        """Check if all targets in [128, 256] have multiplicity > 0."""
        return np.all(mults[128:257] > 0)

    def get_max_mult(mults):
        return np.max(mults[128:257])

    best_A = None
    best_mult = float('inf')
    
    # Strategy: Try different sizes and use Simulated Annealing / Local Search
    # Optimal size is likely around 30-40.
    
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
        
        # Random initial size
        k = random.randint(30, 45)
        
        # Initial random set
        A_set = set(random.sample(range(0, N + 1), k))
        
        # Phase 1: Ensure Coverage
        # If not complete, greedily add elements
        max_adds = 50
        for _ in range(max_adds):
            if time.time() > deadline - 5:
                break
            mults = compute_multiplicities(A_set)
            if is_complete(mults):
                break
            
            # Find uncovered targets
            uncovered = [t for t in TARGETS if mults[t] == 0]
            if not uncovered:
                break
            
            # Try to find an element that covers the most uncovered targets
            # An element x covers target t if t-x is in A_set
            best_gain = 0
            best_elem = None
            
            # Sample candidates to speed up if A is large, but N=128 is small
            candidates = [x for x in range(N + 1) if x not in A_set]
            # Shuffle for randomness
            random.shuffle(candidates)
            
            for x in candidates:
                gain = 0
                for t in uncovered:
                    y = t - x
                    if 0 <= y <= N and y in A_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_elem = x
                    
            if best_gain > 0 and best_elem is not None:
                A_set.add(best_elem)
            else:
                # If no single element helps, just add a random one? 
                # Or break and rely on next attempt?
                # Let's add a random element to keep going
                if candidates:
                    A_set.add(candidates[0])
                else:
                    break
        
        if time.time() > deadline - 5:
            break
        
        # Check completeness
        mults = compute_multiplicities(A_set)
        if not is_complete(mults):
            # Force complete if still missing (rare if greedy worked)
            # Just add necessary pairs
            for t in TARGETS:
                if mults[t] == 0:
                    # Add t//2 and t - t//2
                    a = t // 2
                    b = t - a
                    A_set.add(a)
                    A_set.add(b)
            mults = compute_multiplicities(A_set)
        
        current_mult = get_max_mult(mults)
        
        if current_mult < best_mult:
            best_mult = current_mult
            best_A = set(A_set)
            
        # Phase 2: Local Search (Swap) to reduce peak multiplicity
        # We try to swap elements in A with elements outside A
        # to reduce the max multiplicity.
        
        # Simulated Annealing parameters
        temp = 10.0
        cooling = 0.95
        steps = 200
        
        A_set = set(best_A) if best_A else set(A_set)
        mults = compute_multiplicities(A_set)
        current_max = get_max_mult(mults)
        
        for step in range(steps):
            if time.time() > deadline - 2:
                break
            
            # Pick a random target with high multiplicity
            # To be efficient, pick from top 10%
            target_slice = mults[128:257]
            # Find indices where mult == current_max
            max_indices = np.where(target_slice == current_max)[0]
            if len(max_indices) == 0:
                break
                
            t_idx = max_indices[random.randint(0, len(max_indices) - 1)]
            t = 128 + t_idx
            
            # Find a pair (a, b) in A such that a + b = t
            # Iterate through A
            pair_found = False
            for a in A_set:
                b = t - a
                if a <= b and b in A_set:
                    # Found pair (a, b)
                    # Try swapping a with a random c not in A
                    # Or swapping b with c
                    # Let's try swapping a
                    candidates = [x for x in range(N + 1) if x not in A_set]
                    if not candidates:
                        break
                    c = random.choice(candidates)
                    
                    # Test swap: remove a, add c
                    new_A_set = set(A_set)
                    new_A_set.remove(a)
                    new_A_set.add(c)
                    
                    # Check if coverage is maintained
                    # We only need to check targets that might be affected
                    # But computing full mults is fast enough (N=128)
                    new_mults = compute_multiplicities(new_A_set)
                    if not is_complete(new_mults):
                        continue
                        
                    new_max = get_max_mult(new_mults)
                    
                    # Acceptance criterion
                    if new_max < current_max:
                        # Improvement, always accept
                        A_set = new_A_set
                        mults = new_mults
                        current_max = new_max
                        if current_max < best_mult:
                            best_mult = current_max
                            best_A = set(A_set)
                    elif new_max == current_max:
                        # Same max, check if variance reduced (optional, or just accept with prob)
                        # Let's accept with probability 0.5 to explore
                        if random.random() < 0.5:
                            A_set = new_A_set
                            mults = new_mults
                    else:
                        # Worse, accept with probability exp(-(new_max - current_max) / temp)
                        delta = new_max - current_max
                        prob = np.exp(-delta / temp)
                        if random.random() < prob:
                            A_set = new_A_set
                            mults = new_mults
                            current_max = new_max
                            
                    pair_found = True
                    break # Restart loop with new state
                    
            if not pair_found:
                # No pair found for this t? Should not happen if covered.
                # Maybe t is covered by (a,a) where a was swapped?
                # Just break inner loop
                pass
                
            temp *= cooling
            
            # Early stop if we hit 1 (perfect)
            if current_max == 1:
                break

    if best_A is None:
        # Fallback
        return list(range(64, 129))
        
    return list(best_A)
