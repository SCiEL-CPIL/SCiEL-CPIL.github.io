import time
import random
import numpy as np

def _score_bitset_np(mask):
    """
    Count distinct d > 0 such that exists a with a, a+d, a+2d all in the set.
    mask: int where bit i is set iff value (i+1) is in the set.
    Uses vectorized operations for speed.
    """
    # Convert mask to array of 0/1 values
    bits = np.array([(mask >> i) & 1 for i in range(512)], dtype=np.uint8)
    
    diffs = 0
    for d in range(1, 256):
        # Check if there exists a such that bits[a], bits[a+d], bits[a+2d] are all 1
        # This is equivalent to checking if (bits & bits[d:] & bits[2*d:]) has any 1
        if np.any(bits[:512-2*d] & bits[d:512-d] & bits[2*d:]):
            diffs += 1
    return diffs

def _score_fast(S):
    """Score a set of 32 integers using bitset for fast membership."""
    mask = 0
    for v in S:
        mask |= (1 << (v - 1))
    return _score_bitset_np(mask)

def _local_search(current, deadline, rng):
    """Perform local search with simulated annealing using fast scoring."""
    N = 32
    MAXV = 512
    current_score = _score_fast(current)
    best = set(current)
    best_score = current_score
    
    # Precompute the mask for fast updates
    current_mask = 0
    for v in current:
        current_mask |= (1 << (v - 1))
    
    T = 20.0
    T_min = 0.5
    cooling = 0.999
    
    # Precompute list of current elements for fast choice
    curr_list = list(current)
    
    while time.time() < deadline:
        # Generate a move: swap one element
        idx = rng.randrange(N)
        to_remove = curr_list[idx]
        
        # Choose to_add from values not in current
        to_add = rng.randint(1, MAXV)
        # Simple retry loop to avoid collision
        while to_add in current:
            to_add = rng.randint(1, MAXV)
        
        new_mask = (current_mask & ~(1 << (to_remove - 1))) | (1 << (to_add - 1))
        new_score = _score_bitset_np(new_mask)
        
        if new_score > current_score:
            current.discard(to_remove)
            current.add(to_add)
            curr_list[idx] = to_add
            current_mask = new_mask
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best = set(current)
        else:
            delta = new_score - current_score
            if delta < 0 and rng.random() < np.exp(delta / T):
                current.discard(to_remove)
                current.add(to_add)
                curr_list[idx] = to_add
                current_mask = new_mask
                current_score = new_score
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best, best_score

def _greedy_improve(S, deadline, rng):
    """Try greedy swaps that improve the score."""
    current = set(S)
    current_score = _score_fast(current)
    best = set(current)
    best_score = current_score
    
    current_mask = 0
    for v in current:
        current_mask |= (1 << (v - 1))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for to_remove in list(current):
            if time.time() >= deadline:
                break
            # Try multiple random additions
            for _ in range(15):
                to_add = rng.randint(1, 512)
                if to_add in current:
                    continue
                new_mask = current_mask & ~(1 << (to_remove - 1)) | (1 << (to_add - 1))
                new_score = _score_bitset_np(new_mask)
                if new_score > current_score:
                    current.discard(to_remove)
                    current.add(to_add)
                    current_mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best = set(current)
                    improved = True
                    break
    return best, best_score

def construct():
    start_time = time.time()
    # Set a conservative deadline to ensure we finish well within 150s
    deadline = start_time + 140.0
    
    N = 32
    MAXV = 512
    rng = random.Random(42)
    
    # Start with a structured initial set that has many progressions
    best_overall = set(range(1, N + 1))
    best_overall_score = _score_fast(best_overall)
    
    # Try various structured starts with different patterns
    structured_starts = [
        set(range(1, N + 1)),           # consecutive 1-32
        set(range(1, 64, 2)),           # evens 2,4,...,62
        set(range(2, 64, 2)),           # evens 2,4,...,62
        set(range(1, 63, 2)),           # odds 1,3,...,61
        set(range(1, 33)),              # 1-32
        set(range(33, 65)),             # 33-64
        set(range(1, 129, 4)),          # multiples of 4: 1,5,9,...,125
        set(range(1, 129, 3)),          # 1,4,7,...,124
    ]
    
    for start_set in structured_starts:
        if len(start_set) == N:
            s = _score_fast(start_set)
            if s > best_overall_score:
                best_overall_score = s
                best_overall = set(start_set)
    
    # Multiple restarts with local search
    # Use shorter slices to allow more diverse restarts
    slice_duration = 3.0
    
    while time.time() < deadline:
        # Generate diverse initial sets
        start_type = rng.random()
        
        if start_type < 0.3:
            # Random set
            current = set(rng.sample(range(1, MAXV + 1), N))
        elif start_type < 0.5:
            # Structured: consecutive block
            start = rng.randint(1, MAXV - N + 1)
            current = set(range(start, start + N))
        elif start_type < 0.7:
            # Structured: arithmetic progression
            d = rng.randint(1, 16)
            start = rng.randint(1, MAXV - (N-1)*d)
            current = set(start + i*d for i in range(N))
        elif start_type < 0.85:
            # Structured: two blocks
            block1_start = rng.randint(1, MAXV // 2 - 16)
            block2_start = rng.randint(MAXV // 2 + 1, MAXV - 16)
            current = set(range(block1_start, block1_start + 16)) | set(range(block2_start, block2_start + 16))
        else:
            # Structured: sparse with gaps
            current = set()
            for _ in range(N):
                v = rng.randint(1, MAXV)
                current.add(v)
            # Ensure exactly 32 distinct
            while len(current) < N:
                current.add(rng.randint(1, MAXV))
            current = list(current)[:N]
            current = set(current)
        
        current_score = _score_fast(current)
        
        if current_score > best_overall_score:
            best_overall = set(current)
            best_overall_score = current_score
        
        # Run local search with a time slice
        slice_deadline = min(deadline, time.time() + slice_duration)
        best, best_score = _local_search(current, slice_deadline, rng)
        
        if best_score > best_overall_score:
            best_overall = best
            best_overall_score = best_score
        
        # If we have plenty of time left, do a greedy improvement pass on the best
        if time.time() < deadline - 15.0:
            g_best, g_score = _greedy_improve(best_overall, deadline - 5.0, rng)
            if g_score > best_overall_score:
                best_overall = g_best
                best_overall_score = g_score
    
    return sorted(list(best_overall))
