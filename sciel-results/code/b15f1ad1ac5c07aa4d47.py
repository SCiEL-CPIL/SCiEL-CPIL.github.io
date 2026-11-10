import numpy as np
import time
import random

N = 5000

def _primes(limit):
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0:2] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p*p:limit+1:p] = False
    return [int(p) for p in np.where(sieve)[0]]

def _beam_search_greedy(primes, beam_width=50, branching=5, time_limit=280):
    """
    Performs a beam search to find a large admissible tuple.
    """
    start_time = time.time()
    # Initial state: all ones
    # We represent the beam as a list of boolean arrays
    beam = [np.ones(N + 1, dtype=bool)]
    
    # We only need to consider primes up to the current max size
    # Max size is N+1
    # We iterate through primes in increasing order
    
    # Pre-compute primes up to N+1
    # We will stop when p > max_size_in_beam
    
    for p in primes:
        if time.time() > start_time + time_limit:
            break
            
        new_beam = []
        max_size_in_beam = max(np.sum(s) for s in beam)
        
        if p > max_size_in_beam:
            # No further constraints for this prime or any larger prime
            # We can stop processing primes, but we should keep the current beam
            # Actually, if p > max_size, then for all future primes p' > p, p' > max_size too.
            # So we can break.
            break
            
        for state in beam:
            size = np.sum(state)
            if p > size:
                # This specific state doesn't need to satisfy p
                # But wait, if p > size, we don't need to check p.
                # However, other states in the beam might be larger.
                # The constraint is on the final set.
                # If a state has size < p, it doesn't need to satisfy p.
                # So it passes through unchanged.
                new_beam.append(state)
                continue
                
            indices = np.where(state)[0]
            if len(indices) == 0:
                new_beam.append(state)
                continue
                
            residues = indices % p
            counts = np.bincount(residues, minlength=p)
            
            if not np.all(counts > 0):
                # Admissible for this prime
                new_beam.append(state)
            else:
                # Not admissible, need to remove a residue class
                min_count = np.min(counts)
                min_classes = np.where(counts == min_count)[0]
                
                # Select up to 'branching' classes to remove
                # To add diversity, we can pick random ones from the min classes
                # Or just the first 'branching' ones
                # Let's pick 'branching' random ones if available, else all
                if len(min_classes) > branching:
                    # Random sample
                    selected_classes = np.random.choice(min_classes, branching, replace=False)
                else:
                    selected_classes = min_classes
                    
                for r in selected_classes:
                    new_state = state.copy()
                    mask = (residues == r)
                    new_state[indices[mask]] = False
                    new_beam.append(new_state)
                    
        # Select top 'beam_width' states from new_beam based on size
        if len(new_beam) > beam_width:
            # Sort by size descending
            # To make it faster, we can just find the top K
            # But sorting 50*5=250 items is fast
            new_beam.sort(key=lambda x: np.sum(x), reverse=True)
            beam = new_beam[:beam_width]
        else:
            beam = new_beam
            
        # If beam is empty, break (should not happen)
        if not beam:
            break
            
    # Return the best state from the final beam
    if not beam:
        return np.zeros(N + 1, dtype=bool)
        
    best_state = max(beam, key=lambda x: np.sum(x))
    return best_state

def _local_search_swap(S, primes, time_limit):
    """
    Attempts to improve S by swapping elements.
    """
    S = S.copy()
    start_time = time.time()
    current_size = np.sum(S)
    indices = np.where(S)[0]
    
    # Precompute primes up to N+1
    # We only care about primes <= current_size + 1 (since size can increase by 1 in a swap? No, swap keeps size same)
    # Swap: remove x, add y. Size remains same.
    # So we only need to check primes <= current_size.
    
    # To speed up admissibility check, we can maintain residue counts?
    # But indices change.
    
    # Simple local search:
    # Try swapping a random element in S with a random element not in S.
    # If the new set is admissible, accept.
    
    # This is slow if we check from scratch.
    # Let's try a few hundred swaps.
    
    not_in_S = np.where(~S)[0]
    if len(not_in_S) == 0:
        return S
        
    rng = random.Random(123)
    
    for _ in range(1000): # Fixed number of attempts, or time based
        if time.time() > start_time + time_limit:
            break
            
        if len(indices) == 0:
            break
            
        # Pick random x in S, y not in S
        x_idx = rng.randint(0, len(indices) - 1)
        x = indices[x_idx]
        y_idx = rng.randint(0, len(not_in_S) - 1)
        y = not_in_S[y_idx]
        
        # Create new set
        S_new = S.copy()
        S_new[x] = False
        S_new[y] = True
        
        # Check admissibility
        # Size is same
        is_adm = True
        new_indices = np.where(S_new)[0]
        
        for p in primes:
            if p > current_size:
                break
            residues = new_indices % p
            counts = np.bincount(residues, minlength=p)
            if np.all(counts > 0):
                is_adm = False
                break
                
        if is_adm:
            S = S_new
            # Update indices and not_in_S?
            # This is O(N) to update.
            # Since we are doing this in a loop, it might be slow.
            # Let's just recompute indices if we accept.
            indices = np.where(S)[0]
            not_in_S = np.where(~S)[0]
            if len(not_in_S) == 0:
                break
                
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    primes = _primes(N + 1)
    
    # Run Beam Search
    # We can run multiple beam searches with different parameters
    best_S = np.zeros(N + 1, dtype=bool)
    best_size = -1
    
    # Parameters for beam search
    # Try a few different beam widths and branching factors
    configs = [
        (50, 5),
        (100, 3),
        (200, 2),
    ]
    
    for bw, br in configs:
        if time.time() > deadline - 30:
            break
            
        # Random seed for numpy
        np.random.seed(int(time.time() * 1000) % 10000)
        
        S = _beam_search_greedy(primes, beam_width=bw, branching=br, time_limit=60)
        size = np.sum(S)
        
        if size > best_size:
            best_size = size
            best_S = S.copy()
            
    # Local search on the best result
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 5
        S = _local_search_swap(best_S, primes, remaining)
        size = np.sum(S)
        if size > best_size:
            best_size = size
            best_S = S
            
    return [int(x) for x in np.where(best_S)[0]]
