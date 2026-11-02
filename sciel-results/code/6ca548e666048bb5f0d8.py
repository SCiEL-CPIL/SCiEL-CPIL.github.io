"""Optimal-control / flat-autocorrelation design for the Erdos minimum-overlap task.

We minimize the worst-case overlap of h with shifted (1-h).
With sum(h) fixed = N/2, overlap(k) = N/2 - autocorr(k), so minimizing
max_k overlap(k) is equivalent to maximizing min_k autocorr(k).
We therefore build a balanced 0/1 sequence with as flat (high) an
autocorrelation as possible via a Tabu Search algorithm that directly drives up the minimum lag.
"""
import numpy as np

N = 100


def _autocorr_min_fast(x):
    # x is 0/1 array of length N, sum = N/2
    # Compute circular autocorrelation efficiently.
    # ac[k] = sum_i x[i] * x[(i+k) % N]
    # Using FFT is O(N log N), but for N=100, direct computation might be faster due to lower overhead.
    # Let's use a vectorized approach: 
    # ac[k] = dot(x, roll(x, -k))
    # We can compute all lags at once using matrix multiplication or just a loop with rolls.
    # For N=100, a loop of 100 dot products is very fast.
    n = len(x)
    ac = np.zeros(n)
    for k in range(n):
        ac[k] = np.dot(x, np.roll(x, -k))
    return ac.min()

def _max_overlap(x):
    # overlap(k) = sum_i x[i]*(1 - x[(i+k)%N]); max over k (k>=1)
    n = len(x)
    # We want max overlap, which is n/2 - min autocorr (for k != 0)
    ac = np.zeros(n)
    for k in range(1, n):
        ac[k] = np.dot(x, np.roll(x, -k))
    return n / 2.0 - ac.min()

def _tabu_search(x, iters=50000, tabu_size=100):
    n = len(x)
    target = n // 2
    # Ensure sum is correct
    s = int(x.sum())
    if s != target:
        # Adjust
        if s > target:
            idx = np.where(x == 1)[0]
            for k in range(s - target):
                x[idx[k]] = 0
        else:
            idx = np.where(x == 0)[0]
            for k in range(target - s):
                x[idx[k]] = 1
        
    best_x = x.copy()
    best_min = _autocorr_min_fast(best_x)
    cur_min = best_min
    
    rng = np.random.default_rng(42)
    tabu_list = [] # List of (i, j) moves, storing indices
    tabu_until = {} # Dict mapping (i,j) to iteration it is tabu until
    
    ones = np.where(x == 1)[0]
    zeros = np.where(x == 0)[0]
    
    for it in range(iters):
        # Select a random move
        if len(ones) == 0 or len(zeros) == 0:
            break
            
        i = ones[rng.integers(len(ones))]
        j = zeros[rng.integers(len(zeros))]
        
        if i == j:
            continue
            
        move = (i, j)
        
        # Check tabu
        if move in tabu_until and tabu_until[move] > it:
            # Try another move or accept if improvement is huge (aspiration)
            # For simplicity, skip if tabu
            continue
            
        # Perform swap
        x[i], x[j] = x[j], x[i]
        
        # Update sets
        # x[i] became 0, x[j] became 1
        ones = ones[ones != i]
        zeros = zeros[zeros != j]
        ones = np.append(ones, j)
        zeros = np.append(zeros, i)
        
        new_min = _autocorr_min_fast(x)
        
        # Aspiration criterion: accept if new_min > best_min
        if new_min > best_min:
            best_min = new_min
            best_x = x.copy()
            cur_min = new_min
        elif new_min > cur_min:
            cur_min = new_min
        else:
            # Reject move, revert
            x[i], x[j] = x[j], x[i]
            ones = ones[ones != j]
            zeros = zeros[zeros != i]
            ones = np.append(ones, i)
            zeros = np.append(zeros, j)
            continue
            
        # Update tabu list
        tabu_until[move] = it + tabu_size
        # Also tabu the reverse move
        tabu_until[(j, i)] = it + tabu_size
        
        # Clean up old tabu entries occasionally
        if it % 1000 == 0:
            for k in list(tabu_until.keys()):
                if tabu_until[k] <= it:
                    del tabu_until[k]

    return best_x, best_min

def construct():
    n = N
    target = n // 2
    
    best_x_global = None
    best_min_global = -np.inf
    
    # Multiple restarts with different initializations
    # 1. Thue-Morse
    def thue_morse(n):
        t = np.zeros(n, dtype=np.int8)
        for i in range(n):
            t[i] = bin(i).count('1') % 2
        return t.astype(np.float64)
        
    # 2. Random Balanced
    def random_balanced(n, seed):
        rng = np.random.default_rng(seed)
        x = np.zeros(n)
        indices = rng.permutation(n)
        x[indices[:target]] = 1
        return x

    # 3. Checkerboard (if n even)
    def checkerboard(n):
        x = np.zeros(n)
        x[::2] = 1
        return x

    initializations = [
        thue_morse(n),
        random_balanced(n, 1),
        random_balanced(n, 2),
        random_balanced(n, 3),
        checkerboard(n)
    ]
    
    for x0 in initializations:
        # Ensure exact sum
        s = int(x0.sum())
        if s != target:
            if s > target:
                idx = np.where(x0 == 1)[0]
                for k in range(s - target):
                    x0[idx[k]] = 0
            else:
                idx = np.where(x0 == 0)[0]
                for k in range(target - s):
                    x0[idx[k]] = 1
                    
        x_opt, min_val = _tabu_search(x0.copy(), iters=20000, tabu_size=50)
        
        if min_val > best_min_global:
            best_min_global = min_val
            best_x_global = x_opt
            
    # Final polish: Steepest Descent on the best found
    x = best_x_global.copy()
    cur_min = _autocorr_min_fast(x)
    improved = True
    max_polish_iters = 100 # Limit to avoid long runs
    polish_iter = 0
    
    while improved and polish_iter < max_polish_iters:
        improved = False
        polish_iter += 1
        ones = np.where(x == 1)[0]
        zeros = np.where(x == 0)[0]
        
        best_swap = None
        best_swap_min = cur_min
        
        for i in ones:
            for j in zeros:
                x[i], x[j] = x[j], x[i]
                nm = _autocorr_min_fast(x)
                if nm > best_swap_min + 1e-12:
                    best_swap_min = nm
                    best_swap = (i, j)
                x[i], x[j] = x[j], x[i]
                
        if best_swap:
            i, j = best_swap
            x[i], x[j] = x[j], x[i]
            cur_min = best_swap_min
            improved = True

    h = x.astype(np.float64)
    if abs(h.sum() - n / 2.0) > 1e-12:
        diff = n / 2.0 - h.sum()
        h[0] += diff
    return h
