import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
_START = time.time()
_DEADLINE = _START + 118.0  # Safety margin vs 150s kill

def _score_from_vector(vec):
    """
    Compute score from a binary vector representing set U.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    """
    # Find max index with 1
    indices = np.nonzero(vec)[0]
    if len(indices) < 2:
        return 0.0
    
    max_val = indices[-1]
    if max_val < 1:
        return 0.0
    
    L = max_val + 1
    # FFT length: next power of 2 >= 2*L - 1
    fft_len = 1
    while fft_len < 2 * L - 1:
        fft_len *= 2
        
    # Pad vector
    f_vec = np.zeros(fft_len, dtype=np.float64)
    f_vec[:L] = vec
    
    f = np.fft.fft(f_vec)
    
    # Sumset: |U+U|
    # Convolution is IFFT(f * f). Non-zero entries indicate sum exists.
    conv = np.fft.ifft(f * f).real
    # We only care about sums in [0, 2*max_val]
    # The first 2*max_val + 1 entries of the convolution are valid (no wrap-around if padded correctly)
    sums_count = np.sum(np.abs(conv[:2 * max_val + 1]) > 1e-9)
    
    # Difference set: |U-U|
    # Correlation is IFFT(conj(f) * f). 
    # The output is circular. 
    # Index 0 is diff 0.
    # Indices 1..max_val are positive diffs.
    # Indices fft_len - max_val .. fft_len - 1 are negative diffs.
    corr = np.fft.ifft(np.conj(f) * f).real
    # Count non-zeros in the valid range
    # Valid diffs are in [-max_val, max_val]
    # In the circular array of length fft_len:
    # 0 is at 0.
    # 1..max_val are at 1..max_val.
    # -1..-max_val are at fft_len-1..fft_len-max_val.
    diffs_count = 0
    # 0 diff
    if abs(corr[0]) > 1e-9:
        diffs_count += 1
    # positive diffs
    if max_val > 0:
        diffs_count += np.sum(np.abs(corr[1:max_val+1]) > 1e-9)
        # negative diffs
        diffs_count += np.sum(np.abs(corr[fft_len-max_val:fft_len]) > 1e-9)
        
    if sums_count == 0:
        return 0.0
        
    ratio = diffs_count / float(sums_count)
    if ratio <= 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / math.log(2 * max_val + 1)

def _vec_to_list(vec):
    """Convert binary vector to sorted list of integers."""
    return list(np.nonzero(vec)[0])

def _list_to_vec(lst, size):
    """Convert list of integers to binary vector of given size."""
    vec = np.zeros(size, dtype=np.float64)
    for x in lst:
        if 0 <= x < size:
            vec[x] = 1.0
    return vec

def _construct_initial_set(target_size=10, max_range=200):
    """
    Construct an initial set using a perturbed Golomb ruler or random Sidon-like structure.
    We want a diverse starting point.
    """
    # Simple greedy Sidon (Golomb)
    U = [0, 1]
    diffs = {1}
    current_max = 1
    
    for _ in range(target_size - 2):
        candidate = current_max + 1
        while True:
            valid = True
            for u in U:
                d = candidate - u
                if d in diffs:
                    valid = False
                    break
            if valid:
                U.append(candidate)
                for u in U[:-1]:
                    diffs.add(candidate - u)
                current_max = candidate
                break
            else:
                candidate += 1
                
    # Perturb to allow non-Sidon structures
    # We will use this as a seed for the vector representation
    # Determine range
    max_val = max(U)
    # Add some slack to range to allow moves
    range_size = max(2 * max_val + 10, 100)
    
    vec = _list_to_vec(U, range_size)
    return vec

def _simulated_annealing(initial_vec, time_limit):
    """
    Simulated Annealing on the binary vector representation.
    Moves:
    1. Flip a bit (add/remove element)
    2. Shift an element (remove at i, add at j)
    """
    current_vec = initial_vec.copy()
    current_score = _score_from_vector(current_vec)
    
    best_vec = current_vec.copy()
    best_score = current_score
    
    # Temperature schedule
    start_time = time.time()
    total_time = time_limit
    T_start = 1.0
    T_end = 0.001
    
    random.seed(42)
    
    while time.time() - start_time < total_time:
        # Calculate current temperature
        elapsed = time.time() - start_time
        progress = min(1.0, elapsed / total_time)
        # Exponential cooling
        T = T_start * math.exp(-5 * progress)
        
        # Generate a neighbor
        # Determine range: we can expand range if we want, but for simplicity fix range 
        # or dynamically adjust? 
        # For now, fix range based on initial, but allow moves within.
        # Actually, if max element is removed, range effectively shrinks, which might improve score.
        # If we add a large element, range grows.
        # Let's allow moves within [0, len(current_vec)-1].
        
        n = len(current_vec)
        if n < 2:
            break
            
        move_type = random.random()
        
        new_vec = current_vec.copy()
        
        if move_type < 0.5:
            # Flip a bit
            idx = random.randint(0, n - 1)
            new_vec[idx] = 1.0 - new_vec[idx]
        else:
            # Shift: pick an existing 1, remove it, add a new 1 elsewhere
            ones = np.nonzero(current_vec)[0]
            if len(ones) == 0:
                continue
            remove_idx = ones[random.randint(0, len(ones) - 1)]
            add_idx = random.randint(0, n - 1)
            if add_idx == remove_idx:
                continue
            new_vec[remove_idx] = 0.0
            new_vec[add_idx] = 1.0
            
        # Ensure 0 is always in the set? 
        # The task says "containing 0". 
        # If we flip 0 off, we must turn it back on or reject.
        if new_vec[0] == 0.0:
            new_vec[0] = 1.0
            # If we just set 0 to 1, did we violate size? No, we are just enforcing constraint.
            # But if we removed 0 in a shift, we should probably reject or repair.
            # Let's just enforce: if 0 is not in set, add it.
            # This effectively makes the move "shift other element" if 0 was removed.
            # To keep it simple, if 0 is removed, we revert that part of the move.
            # Actually, let's just force 0=1 in the new_vec.
            
        # Calculate new score
        new_score = _score_from_vector(new_vec)
        
        # Acceptance criterion
        if new_score > current_score:
            current_vec = new_vec
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_vec = new_vec.copy()
        else:
            delta = new_score - current_score
            # Avoid log(0) or negative args
            if delta < 0 and T > 0:
                accept_prob = math.exp(delta / T)
                if random.random() < accept_prob:
                    current_vec = new_vec
                    current_score = new_score
    
    return best_vec, best_score

def construct():
    global _DEADLINE
    
    best_score = 0.0
    best_set = [0, 1]
    
    # We will try a few different initial configurations and ranges
    # Range 1: Small set, small range
    # Range 2: Larger set, larger range
    
    configs = [
        {"size": 8, "range": 150},
        {"size": 12, "range": 400},
        {"size": 20, "range": 1000},
        {"size": 5, "range": 50},
    ]
    
    # Allocate time to each config
    total_time = _DEADLINE - time.time()
    time_per_config = total_time / len(configs)
    
    for config in configs:
        if time.time() > _DEADLINE - 2.0:
            break
            
        size = config["size"]
        range_size = config["range"]
        
        # Create initial vector
        # We can start with a random Sidon-like set or just a random set of that size
        # Let's start with a greedy Golomb ruler of that size, then pad/truncate to range_size
        U = [0, 1]
        diffs = {1}
        current_max = 1
        
        for _ in range(size - 2):
            candidate = current_max + 1
            while True:
                valid = True
                for u in U:
                    d = candidate - u
                    if d in diffs:
                        valid = False
                        break
                if valid:
                    U.append(candidate)
                    for u in U[:-1]:
                        diffs.add(candidate - u)
                    current_max = candidate
                    break
                else:
                    candidate += 1
                    
        # If current_max > range_size, we have a problem. 
        # For small sizes, Golomb rulers fit in small ranges.
        # For size 20, max is ~ 400? 
        # Let's check: n=20, max ~ 20^2 = 400. 
        # If range_size is too small, we can't fit it. 
        # In that case, just use random elements within range.
        if max(U) < range_size:
            init_vec = _list_to_vec(U, range_size)
        else:
            # Fallback: random set
            init_vec = np.zeros(range_size, dtype=np.float64)
            init_vec[0] = 1.0
            indices = random.sample(range(1, range_size), size - 1)
            for i in indices:
                init_vec[i] = 1.0
                
        # Run SA
        vec, score = _simulated_annealing(init_vec, time_per_config)
        
        if score > best_score:
            best_score = score
            best_set = _vec_to_list(vec)
            
    # Final sanity check
    if not best_set or 0 not in best_set:
        best_set = [0, 1]
        
    best_set = sorted(set(int(x) for x in best_set))
    
    # One last quick check: if we have time, try a few random restarts on the best structure found?
    # Or just return.
    
    return best_set
# EVOLVE-BLOCK-END
