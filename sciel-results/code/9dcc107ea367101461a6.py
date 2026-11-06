"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

N = 256

def _count_mono4_fast(c):
    """Count monochromatic 4-term APs in a 0/1 numpy array of length N using vectorization."""
    count = 0
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        c0 = c[0:max_a]
        c1 = c[d:d + max_a]
        c2 = c[2*d:2*d + max_a]
        c3 = c[3*d:3*d + max_a]
        # All four equal: c0==c1 and c0==c2 and c0==c3
        mask = (c0 == c1) & (c0 == c2) & (c0 == c3)
        count += np.count_nonzero(mask)
    return count

def _delta_count(c, idx):
    """
    Compute the change in mono4 count if we flip bit at position idx (0-indexed).
    Returns the delta (new_count - old_count).
    """
    old_bit = c[idx]
    new_bit = 1 - old_bit
    delta = 0
    
    # We need to check all 4-term APs that include position idx.
    # A 4-term AP is (a, a+d, a+2d, a+3d). Position idx can be at any of the 4 slots.
    # Slot 0: idx = a -> a = idx, need a+d, a+2d, a+3d in range
    # Slot 1: idx = a+d -> a = idx-d, need a, a+2d, a+3d in range
    # Slot 2: idx = a+2d -> a = idx-2d, need a, a+d, a+3d in range
    # Slot 3: idx = a+3d -> a = idx-3d, need a, a+d, a+2d in range
    
    for d in range(1, N // 3 + 1):
        # Slot 0: a = idx
        a = idx
        if a + 3*d < N:
            # AP: a, a+d, a+2d, a+3d
            # Currently: c[a], c[a+d], c[a+2d], c[a+3d]
            # After flip: new_bit, c[a+d], c[a+2d], c[a+3d]
            # Old mono: c[a]==c[a+d]==c[a+2d]==c[a+3d]
            # New mono: new_bit==c[a+d]==c[a+2d]==c[a+3d]
            others = [c[a+d], c[a+2*d], c[a+3*d]]
            old_mono = 1 if (old_bit == others[0] == others[1] == others[2]) else 0
            new_mono = 1 if (new_bit == others[0] == others[1] == others[2]) else 0
            delta += new_mono - old_mono
        
        # Slot 1: a = idx - d
        a = idx - d
        if a >= 0 and a + 3*d < N:
            # AP: a, a+d, a+2d, a+3d where a+d = idx
            others = [c[a], c[a+2*d], c[a+3*d]]
            old_mono = 1 if (old_bit == others[0] == others[1] == others[2]) else 0
            new_mono = 1 if (new_bit == others[0] == others[1] == others[2]) else 0
            delta += new_mono - old_mono
        
        # Slot 2: a = idx - 2d
        a = idx - 2*d
        if a >= 0 and a + 3*d < N:
            # AP: a, a+d, a+2d, a+3d where a+2d = idx
            others = [c[a], c[a+d], c[a+3*d]]
            old_mono = 1 if (old_bit == others[0] == others[1] == others[2]) else 0
            new_mono = 1 if (new_bit == others[0] == others[1] == others[2]) else 0
            delta += new_mono - old_mono
        
        # Slot 3: a = idx - 3d
        a = idx - 3*d
        if a >= 0 and a + 3*d < N:
            # AP: a, a+d, a+2d, a+3d where a+3d = idx
            others = [c[a], c[a+d], c[a+2*d]]
            old_mono = 1 if (old_bit == others[0] == others[1] == others[2]) else 0
            new_mono = 1 if (new_bit == others[0] == others[1] == others[2]) else 0
            delta += new_mono - old_mono
    
    return delta

def _local_search(c, max_iter=100000):
    """Local search: repeatedly flip the bit that gives the best improvement."""
    c = c.copy()
    current_count = _count_mono4_fast(c)
    
    for it in range(max_iter):
        best_delta = 0
        best_idx = -1
        # Try all positions
        for idx in range(N):
            d = _delta_count(c, idx)
            if d < best_delta:
                best_delta = d
                best_idx = idx
        
        if best_delta >= 0:
            break  # No improvement possible
        
        c[best_idx] = 1 - c[best_idx]
        current_count += best_delta
    
    return c, current_count

def _simulated_annealing(c, start_temp=1.0, end_temp=0.01, max_iter=50000):
    """Simulated annealing to escape local minima."""
    c = c.copy()
    current_count = _count_mono4_fast(c)
    best_c = c.copy()
    best_count = current_count
    
    for it in range(max_iter):
        temp = start_temp * (end_temp / start_temp) ** (it / max_iter)
        idx = random.randint(0, N - 1)
        delta = _delta_count(c, idx)
        
        if delta < 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
            c[idx] = 1 - c[idx]
            current_count += delta
            
            if current_count < best_count:
                best_count = current_count
                best_c = c.copy()
    
    return best_c, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    best_c = None
    best_count = float('inf')
    
    # Strategy 1: Start with Thue-Morse sequence (known to be good for avoiding APs)
    # Thue-Morse: t(n) = parity of number of 1s in binary representation of n
    def thue_morse(n):
        return bin(n).count('1') % 2
    
    tm_c = np.array([thue_morse(i) for i in range(N)], dtype=np.int8)
    tm_count = _count_mono4_fast(tm_c)
    
    if tm_count < best_count:
        best_count = tm_count
        best_c = tm_c.copy()
    
    # Local search from Thue-Morse
    if time.time() < deadline:
        ls_c, ls_count = _local_search(tm_c, max_iter=200000)
        if ls_count < best_count:
            best_count = ls_count
            best_c = ls_c.copy()
    
    # Strategy 2: Random restarts with local search
    random.seed(42)
    num_restarts = 0
    while time.time() < deadline:
        # Random initial coloring
        rand_c = np.random.randint(0, 2, N, dtype=np.int8)
        rand_count = _count_mono4_fast(rand_c)
        
        # Local search
        ls_c, ls_count = _local_search(rand_c, max_iter=50000)
        
        if ls_count < best_count:
            best_count = ls_count
            best_c = ls_c.copy()
        
        num_restarts += 1
        
        # If we've found a very good solution, do simulated annealing to refine
        if time.time() < deadline - 10 and num_restarts % 5 == 0:
            sa_c, sa_count = _simulated_annealing(best_c, start_temp=0.5, end_temp=0.01, max_iter=20000)
            if sa_count < best_count:
                best_count = sa_count
                best_c = sa_c.copy()
    
    # Return as list of ints
    return [int(x) for x in best_c]
# EVOLVE-BLOCK-END
