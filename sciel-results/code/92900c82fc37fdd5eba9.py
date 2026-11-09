"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 58.0  # seconds, safety margin before 60s kill
    
    def elapsed():
        return time.time() - start_time

    N = 256
    numbers = list(range(2, N + 1))
    n = len(numbers)
    
    # Precompute conflict matrix as a NumPy array of uint8 (0/1)
    # conflict[i, j] = 1 if lcm(numbers[i], numbers[j]) <= 256
    conflict = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        a = numbers[i]
        for j in range(i + 1, n):
            b = numbers[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= N:
                conflict[i, j] = 1
                conflict[j, i] = 1
                
    weights = np.array([1.0 / x for x in numbers], dtype=np.float64)
    
    # Helper to check if a node can be added to a set represented by a boolean mask
    # mask[i] == True means i is in the set
    # Returns True if node i is not in mask and has no conflicts with any node in mask
    def can_add(i, mask):
        if mask[i]:
            return False
        # Check if there are any 1s in conflict[i] where mask is also 1
        # We can use bitwise operations or matrix multiplication
        # sum(conflict[i] * mask) == 0
        return np.dot(conflict[i], mask) == 0

    # Helper to calculate the sum of weights for a mask
    def calc_sum(mask):
        return np.dot(weights, mask)
        
    # 1. Generate a good initial solution using Greedy
    # Sort indices by weight descending (i.e., number ascending)
    order = np.argsort(-weights)
    
    initial_mask = np.zeros(n, dtype=bool)
    for i in order:
        if can_add(i, initial_mask):
            initial_mask[i] = True
            
    best_mask = initial_mask.copy()
    best_sum = calc_sum(best_mask)
    current_mask = initial_mask.copy()
    current_sum = best_sum
    
    # 2. Simulated Annealing
    # Start with a relatively high temperature, cool down geometrically
    # The "energy" is negative sum of weights, so we want to minimize energy (maximize sum)
    # Delta E = E_new - E_old = -sum_new - (-sum_old) = sum_old - sum_new
    # If Delta E < 0, always accept.
    # If Delta E > 0, accept with probability exp(-Delta E / T)
    
    T = 0.05  # Initial temperature. Weights are around 1/2 to 1/256. Sum changes are small.
    cooling = 0.99
    min_T = 1e-4
    
    # Randomize for SA
    rng = np.random.default_rng(42)
    
    while T > min_T and elapsed() < DEADLINE:
        # Pick a random node to flip
        i = rng.integers(0, n)
        
        if current_mask[i]:
            # Try removing i
            new_mask = current_mask.copy()
            new_mask[i] = False
            new_sum = current_sum - weights[i]
            delta_e = current_sum - new_sum # Wait, E = -S. Delta E = E_new - E_old = -S_new - (-S_old) = S_old - S_new
            # If we remove, S_new < S_old, so Delta E > 0.
            # Actually, let's just work with Sum directly.
            # We want to maximize Sum.
            # If new_sum > current_sum, accept.
            # Else accept with prob exp((current_sum - new_sum) / T)
            
            gain = new_sum - current_sum
            if gain > 0:
                current_mask = new_mask
                current_sum = new_sum
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_mask = current_mask.copy()
            else:
                # gain <= 0
                # Probability of accepting a worse solution
                # p = exp(gain / T)  (since gain is negative, this is < 1)
                if rng.random() < np.exp(gain / T):
                    current_mask = new_mask
                    current_sum = new_sum
        else:
            # Try adding i
            if can_add(i, current_mask):
                new_mask = current_mask.copy()
                new_mask[i] = True
                new_sum = current_sum + weights[i]
                gain = new_sum - current_sum # Always positive here
                current_mask = new_mask
                current_sum = new_sum
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_mask = current_mask.copy()
            # If can't add, do nothing (or count as a step)
            
        T *= cooling
        
        # Check time every few iterations to avoid overhead of time.time() every loop
        # time.time() is fast, but let's be safe.
        if rng.integers(0, 10) == 0 and elapsed() > DEADLINE:
            break

    # 3. Final local search (Greedy Add) to clean up
    # Since SA might leave "holes" that can be filled
    improved = True
    while improved and elapsed() < DEADLINE:
        improved = False
        for i in range(n):
            if not best_mask[i]:
                if can_add(i, best_mask):
                    best_mask[i] = True
                    best_sum += weights[i]
                    improved = True
                    # Restart to check for more
                    break
                    
    # Convert back to list of numbers
    result = [numbers[i] for i in range(n) if best_mask[i]]
    return result
# EVOLVE-BLOCK-END
