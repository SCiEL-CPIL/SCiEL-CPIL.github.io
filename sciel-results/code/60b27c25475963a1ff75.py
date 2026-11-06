"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy.fft import fft, ifft

# EVOLVE-BLOCK-START

def _score_set(s):
    """
    Compute the number of distinct d > 0 such that there exist a, a+d, a+2d in s.
    Uses FFT-based convolution for speed.
    """
    s_set = set(s)
    max_val = max(s_set) if s_set else 0
    # We need to check for each d > 0, if there exists a such that a, a+d, a+2d in s.
    # This is equivalent to checking if there exists a pair (x, y) in s with y - x = 2d,
    # and (x + d) in s.
    # Alternatively, for each d, we check if (s & (s - d) & (s - 2*d)) is non-empty.
    
    # Using FFT:
    # Let f be an indicator array where f[i] = 1 if i in s.
    # The number of 3-APs with difference d is sum_a f[a] * f[a+d] * f[a+2d].
    # We want to know for which d this sum > 0.
    
    # A faster way:
    # For a given d, the condition is: exists a such that a in s, a+d in s, a+2d in s.
    # This is the same as: (s & (s shifted by -d) & (s shifted by -2d)) is non-empty.
    
    # Let's use the difference set approach with FFT.
    # Actually, a simpler approach for checking distinct d:
    # For each pair (i, j) in s with i < j, let diff = j - i.
    # Then a 3-AP with difference d exists if there is a pair (a, a+2d) in s such that a+d is also in s.
    # So for each possible d, we need to check if there exists a such that a in s, a+2d in s, and a+d in s.
    
    # Let's precompute the set as a boolean array for fast lookups.
    arr = np.zeros(max_val + 1, dtype=bool)
    for x in s_set:
        arr[x] = True
    
    diffs_found = set()
    # We can iterate over possible d values. d can range from 1 to max_val//2.
    # But a smarter way: for each pair (i, j) in s, d = (j-i)//2 must be integer, and i+d must be in s.
    # Actually, let's just iterate over all pairs and check.
    
    lst = sorted(s_set)
    n = len(lst)
    
    # For each pair (i, j) where i < j, if (j-i) is even, then d = (j-i)//2.
    # Check if i + d = (i+j)//2 is in s.
    # But this only finds 3-APs where the middle element is (i+j)//2.
    # Wait, a 3-AP is a, a+d, a+2d. So the middle is a+d.
    # If we have two elements x, y in s with y > x, then for them to be the first and third terms of a 3-AP,
    # we need y - x = 2d, so d = (y-x)/2 must be integer, and the middle element (x+y)/2 must be in s.
    
    # So: for each pair (x, y) in s with x < y and (y-x) even, if (x+y)//2 in s, then d = (y-x)//2 is a valid difference.
    
    for i in range(n):
        for j in range(i+1, n):
            diff = lst[j] - lst[i]
            if diff % 2 == 0:
                mid = (lst[i] + lst[j]) // 2
                if arr[mid]:
                    diffs_found.add(diff // 2)
    
    return len(diffs_found)


def _score_set_fast(s):
    """
    Faster scoring using numpy operations.
    """
    s_set = set(s)
    max_val = max(s_set) if s_set else 0
    arr = np.zeros(max_val + 1, dtype=np.int8)
    for x in s_set:
        arr[x] = 1
    
    diffs_found = set()
    lst = sorted(s_set)
    n = len(lst)
    
    # For each pair (i, j), check if (j-i) is even and mid is in set.
    for i in range(n):
        for j in range(i+1, n):
            diff = lst[j] - lst[i]
            if diff % 2 == 0:
                mid = (lst[i] + lst[j]) // 2
                if arr[mid]:
                    diffs_found.add(diff // 2)
    
    return len(diffs_found)


def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin
    
    random.seed(12345)
    
    # Start with a good initial set
    # Use a geometric progression or a set with many small differences
    # Let's start with {1, 2, 4, 8, 16, 32, 64, 128, 256} plus some others
    # Actually, let's start with a random set and improve
    
    # Initial set: try a set that's dense in a small range
    current_set = list(range(1, 33))
    current_score = _score_set(current_set)
    
    best_set = list(current_set)
    best_score = current_score
    
    # Simulated annealing / local search
    temperature = 50.0
    cooling = 0.995
    
    while time.time() < deadline:
        # Pick a random element to remove
        idx = random.randint(0, 31)
        removed = current_set[idx]
        
        # Pick a random element to add
        new_elem = random.randint(1, 512)
        
        if new_elem in current_set:
            continue
        
        # Create new set
        new_set = current_set[:idx] + current_set[idx+1:] + [new_elem]
        new_set = list(set(new_set))  # Ensure uniqueness (should be 32 elements)
        
        if len(new_set) != 32:
            continue
        
        new_score = _score_set(new_set)
        
        # Accept if better, or with probability exp((new_score - current_score) / temperature)
        if new_score > current_score:
            current_set = new_set
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_set = list(new_set)
        else:
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / temperature)
                if random.random() < prob:
                    current_set = new_set
                    current_score = new_score
        
        temperature *= cooling
    
    return best_set

# EVOLVE-BLOCK-END
