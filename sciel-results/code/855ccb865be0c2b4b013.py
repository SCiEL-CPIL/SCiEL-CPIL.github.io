import time
import numpy as np
import random

def _score_set(U):
    """Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2 max U + 1)."""
    if not U or len(U) < 2:
        return 0.0
    arr = np.array(U, dtype=np.int64)
    if 0 not in arr:
        arr = np.concatenate(([0], arr))
    arr = np.unique(arr)
    n = len(arr)
    mx = int(arr.max())
    if mx == 0:
        return 0.0
    
    N = 2 * mx + 1
    
    # Efficient counting via FFT
    sig = np.zeros(N, dtype=np.float64)
    sig[arr] = 1.0
    
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
        
    F = np.fft.rfft(sig, n=fft_size)
    
    # Sumset
    S_sum = np.fft.irfft(F * F, n=fft_size)
    sumset_size = np.sum(S_sum[:N] > 1e-6) # Sums range 0..2*mx which is 0..N-1
    
    # Diffset
    flip_sig = np.zeros(N, dtype=np.float64)
    flip_sig[:] = sig[::-1]
    
    F_flip = np.fft.rfft(flip_sig, n=fft_size)
    S_diff = np.fft.irfft(F * F_flip, n=fft_size)
    
    diffset_size = np.sum(S_diff > 1e-6)
    
    if sumset_size == 0:
        return 0.0
        
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
        
    denom = np.log(2 * mx + 1)
    if denom == 0:
        return 0.0
        
    return 1.0 + np.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_score = 0.0
    best_set = [0, 1]
    
    # Strategy: Greedy construction of U.
    # We build U element by element.
    # To make this efficient, we maintain bitsets for the sumset and diffset.
    # However, as max(U) grows, the bitset size grows.
    # We will limit the size of U and the max element to keep it tractable.
    
    # We try multiple random restarts with different "greediness" or randomization.
    
    # We'll use a simple set-based approach for small U, and switch to numpy if needed.
    # Given the time limit, we can afford a decent number of iterations.
    
    # Let's use a heuristic search.
    # Start with U = {0}.
    # At each step, choose x to add to U such that the score improves.
    # This is O(N * |U| * SearchSpace).
    # To speed up, we can approximate the gain.
    
    # A faster approach:
    # Generate candidate sets using a "Mian-Chowla" like sequence but with randomization.
    # The Mian-Chowla sequence is defined by a_1=1, a_{n+1} = smallest integer such that all pairwise sums are distinct.
    # This maximizes the sumset size for a given max element? No, it minimizes collisions.
    # We want to maximize |U-U|/|U+U|.
    
    # Let's try a local search on the elements of U.
    # Representation: U is a sorted list.
    # Move: Replace an element, or add an element, or remove an element.
    
    # To make this fast, we need a fast scoring function.
    # _score_set uses FFT. For small U (say < 1000), it's fast.
    
    # Let's run a few independent local search processes.
    
    def local_search(initial_U, time_limit):
        current_U = sorted(initial_U)
        current_score = _score_set(current_U)
        best_local_score = current_score
        best_local_U = current_U[:]
        
        # Precompute sumset and diffset as sets for fast updates?
        # Updating sumset/diffset incrementally is complex when removing elements.
        # So we recompute from scratch.
        
        # We will use a simple hill climbing.
        # To avoid getting stuck, we allow some random restarts or perturbations.
        
        while time.time() < time_limit:
            # Try a few random moves
            improved = False
            for _ in range(10):
                new_U = current_U[:]
                move_type = random.choice(['add', 'remove', 'replace'])
                
                if move_type == 'add':
                    # Add a random element
                    # Range: up to 2 * max(current_U) or so
                    max_val = max(current_U) if current_U else 0
                    # Try adding a value that might extend the set
                    # Heuristic: try values near the current max or in gaps
                    candidates = []
                    if max_val > 0:
                        # Try around max
                        for delta in range(1, 10):
                            candidates.append(max_val + delta)
                            candidates.append(max_val - delta)
                        # Try random in range
                        candidates.append(random.randint(1, max_val + 10))
                        # Try small values if not present
                        candidates.append(random.randint(1, 5))
                    else:
                        candidates = [1, 2, 3, 4, 5]
                    
                    # Pick a random candidate
                    x = random.choice(candidates)
                    if x not in new_U:
                        new_U.append(x)
                        
                elif move_type == 'remove':
                    if len(new_U) > 2:
                        # Remove a random element (keep 0)
                        if len(new_U) > 2:
                            idx = random.randint(1, len(new_U)-1)
                            new_U.pop(idx)
                        else:
                            continue
                            
                elif move_type == 'replace':
                    if len(new_U) > 2:
                        idx = random.randint(1, len(new_U)-1)
                        old_val = new_U[idx]
                        # Replace with a nearby value
                        new_val = old_val + random.randint(-5, 5)
                        if new_val >= 0 and new_val not in new_U:
                            new_U[idx] = new_val
                        else:
                            continue
                    else:
                        continue
                        
                new_U = sorted(set(new_U))
                
                # Ensure 0 is in U
                if 0 not in new_U:
                    new_U.append(0)
                    new_U = sorted(new_U)
                    
                if len(new_U) < 2:
                    continue
                    
                # Check size constraint for FFT
                if max(new_U) > 100000:
                    continue
                    
                new_score = _score_set(new_U)
                
                if new_score > current_score:
                    current_score = new_score
                    current_U = new_U
                    improved = True
                    if current_score > best_local_score:
                        best_local_score = current_score
                        best_local_U = current_U[:]
                    
            if not improved and random.random() < 0.1:
                # Perturbation: jump to a random neighborhood
                # Or restart with a slightly different seed
                pass
                
        return best_local_score, best_local_U

    # Run multiple local searches
    # We divide the time among several restarts
    num_restarts = 5
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Initialize with different strategies
        if i == 0:
            # Mian-Chowla like
            U = [0, 1]
            # Generate a few more terms greedily
            for _ in range(10):
                # Find next term
                # This is slow, so we just use a simple heuristic
                # Let's just start with a small set and let local search do the work
                pass
        elif i == 1:
            U = [0, 1, 3, 7]
        elif i == 2:
            U = [0, 2, 4, 8, 16]
        elif i == 3:
            U = [0, 1, 2, 4, 8]
        else:
            U = [0] + [random.randint(1, 50) for _ in range(10)]
            U = sorted(set(U))
            
        t_limit = time.time() + time_per_restart
        s, U_found = local_search(U, t_limit)
        
        if s > best_score:
            best_score = s
            best_set = U_found
            
    return best_set

if __name__ == "__main__":
    s = construct()
    print(f"Result: {s[:10]}... Size: {len(s)}, Score: {_score_set(s)}")
