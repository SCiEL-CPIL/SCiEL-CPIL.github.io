"""
Strategy: Discrete Binary Sequence Design via Simulated Annealing.

The task asks to minimize the maximum overlap of $h$ with shifted $1-h$.
Let $g(x) = 1 - h(x)$.
Overlap at shift $k$: $O_k = \sum_{i} h_i g_{i+k} = \sum_{i} h_i (1 - h_{i+k}) = \sum h_i - \sum h_i h_{i+k}$.
Given $\sum h_i = N/2$, $O_k = N/2 - A_k$, where $A_k = \sum h_i h_{i+k}$ is the circular autocorrelation.
Minimizing $\max_k O_k$ is equivalent to maximizing $\min_k A_k$.

The current continuous approach yields a score of 1.0 (which corresponds to a specific overlap value, likely $N/4$ for constant $h=0.5$, or similar).
By restricting $h$ to binary values $\{0, 1\}$ with exactly $N/2$ ones (balanced binary sequence), we transform this into a combinatorial design problem.
Balanced binary sequences (like Barker sequences or Legendre sequences) have well-studied autocorrelation properties.
For a binary sequence with $N/2$ ones, the autocorrelation $A_k$ takes integer values.
We use Simulated Annealing to search the space of balanced binary permutations to maximize the minimum autocorrelation.
This is a structurally different approach: discrete combinatorial search vs continuous gradient-free optimization.
"""

import numpy as np
import time
import random

N = 100

def _compute_overlap(h):
    """
    Compute the maximum overlap of h with shifted 1-h.
    h is a 1D array of step heights.
    O_k = sum h[i] (1 - h[(i+k)%N])
    """
    # Autocorrelation A_k = sum h[i] h[(i+k)%N]
    H = np.fft.fft(h)
    A = np.real(np.fft.ifft(H * np.conj(H)))
    # Max overlap = N/2 - min(A)
    return N / 2.0 - np.min(A)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # 1. Initialize a balanced binary sequence
    # Start with a random permutation of N/2 ones and N/2 zeros
    h = np.zeros(N)
    ones_indices = np.random.choice(N, N // 2, replace=False)
    h[ones_indices] = 1.0
    
    current_overlap = _compute_overlap(h)
    best_h = h.copy()
    best_overlap = current_overlap
    
    # 2. Simulated Annealing
    # We perform moves that preserve the sum constraint (swap a 1 and a 0)
    # Temperature schedule: Geometric cooling
    
    T = 1.0
    T_min = 0.001
    cooling_rate = 0.999
    
    # Precompute indices for faster access
    ones = np.where(h == 1.0)[0]
    zeros = np.where(h == 0.0)[0]
    
    # We will use a list for ones and zeros to allow fast swapping
    # But numpy arrays are easier for the FFT objective.
    # Since N=100, FFT is very fast.
    
    max_iterations = 500000
    iterations = 0
    
    while time.time() < deadline and iterations < max_iterations:
        if T < T_min:
            break
            
        # Choose a random 1 and a random 0 to swap
        i = np.random.choice(ones)
        j = np.random.choice(zeros)
        
        # Apply swap
        h[i], h[j] = h[j], h[i]
        
        # Update sets
        # This is O(N) if done naively, but we can just recompute or maintain lists.
        # For N=100, recomputing sets is cheap, but let's just update the arrays in place
        # and rely on the fact that we pick from the current state.
        # Wait, if we pick from stale 'ones' and 'zeros' lists, we might pick invalid indices.
        # Better to maintain the lists or recompute.
        # Recomputing ones/zeros every step is O(N log N). 500k steps * 100 log 100 ~ 50M ops. Feasible.
        # Or, simpler: just pick random i, j. If h[i]==1 and h[j]==0, swap.
        # Probability of valid swap is 0.5.
        
        new_overlap = _compute_overlap(h)
        delta = new_overlap - current_overlap
        
        # Metropolis criterion
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            # Accept
            current_overlap = new_overlap
            # Update best
            if current_overlap < best_overlap:
                best_overlap = current_overlap
                best_h = h.copy()
                # Update sets for valid moves
                # We don't strictly need to update sets if we use the "random i, j" method below
        else:
            # Reject: swap back
            h[i], h[j] = h[j], h[i]
            
        T *= cooling_rate
        iterations += 1
        
        # Periodic check for time
        if iterations % 1000 == 0:
            if time.time() > deadline:
                break

    return best_h

# EVOLVE-BLOCK-START
# The code above is the strategy.
# We need to wrap it in the construct function properly.
# The prompt asks to keep function names.
# I will re-structure slightly to ensure the 'construct' is the main entry point
# and uses the simulated annealing logic derived above.
# EVOLVE-BLOCK-END

# Final check on the logic:
# The "random i, j" approach:
# Pick i, j from [0, N).
# If h[i] == 1 and h[j] == 0:
#   Swap.
#   Calculate new cost.
#   Accept/Reject.
# This is simpler and avoids maintaining sets.
# Let's refine the construct function with this simpler logic.

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initialize
    h = np.zeros(N)
    ones_indices = np.random.choice(N, N // 2, replace=False)
    h[ones_indices] = 1.0
    
    current_overlap = _compute_overlap(h)
    best_h = h.copy()
    best_overlap = current_overlap
    
    T = 1.0
    T_min = 0.001
    cooling_rate = 0.9995
    
    max_iterations = 2000000
    iterations = 0
    
    while time.time() < deadline and iterations < max_iterations:
        if T < T_min:
            break
        
        # Random swap attempt
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        
        if i == j:
            continue
            
        # Check if it's a valid swap (1 and 0)
        if h[i] == 1.0 and h[j] == 0.0:
            # Perform swap
            h[i] = 0.0
            h[j] = 1.0
            
            new_overlap = _compute_overlap(h)
            delta = new_overlap - current_overlap
            
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                # Accept
                current_overlap = new_overlap
                if current_overlap < best_overlap:
                    best_overlap = current_overlap
                    best_h = h.copy()
            else:
                # Reject: swap back
                h[i] = 1.0
                h[j] = 0.0
                
        T *= cooling_rate
        iterations += 1
        
    return best_h

# EVOLVE-BLOCK-START
# The code above is the implementation.
# EVOLVE-BLOCK-END

# Wait, I need to output the COMPLETE program in a single code block.
# I will combine the helper functions and the construct function.

import numpy as np
import time

N = 100

def _compute_overlap(h):
    """
    Compute the maximum overlap of h with shifted 1-h.
    O_k = sum h[i] (1 - h[(i+k)%N])
    """
    H = np.fft.fft(h)
    A = np.real(np.fft.ifft(H * np.conj(H)))
    return N / 2.0 - np.min(A)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # 1. Initialize a balanced binary sequence
    h = np.zeros(N)
    ones_indices = np.random.choice(N, N // 2, replace=False)
    h[ones_indices] = 1.0
    
    current_overlap = _compute_overlap(h)
    best_h = h.copy()
    best_overlap = current_overlap
    
    # 2. Simulated Annealing
    T = 1.0
    T_min = 0.001
    cooling_rate = 0.9995
    
    max_iterations = 2000000
    iterations = 0
    
    while time.time() < deadline and iterations < max_iterations:
        if T < T_min:
            break
        
        # Random swap attempt
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        
        if i == j:
            continue
            
        # Check if it's a valid swap (1 and 0)
        if h[i] == 1.0 and h[j] == 0.0:
            # Perform swap
            h[i] = 0.0
            h[j] = 1.0
            
            new_overlap = _compute_overlap(h)
            delta = new_overlap - current_overlap
            
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                # Accept
                current_overlap = new_overlap
                if current_overlap < best_overlap:
                    best_overlap = current_overlap
                    best_h = h.copy()
            else:
                # Reject: swap back
                h[i] = 1.0
                h[j] = 0.0
                
        T *= cooling_rate
        iterations += 1
        
    return best_h
