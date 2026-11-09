import time
import numpy as np
import random

N = 18
SUBSET_SIZE = 2000
FULL_SIZE = 1 << N

# Pre-generate subset inputs
np.random.seed(42)
subset_inputs = np.random.randint(0, 2, size=(SUBSET_SIZE, N))

# Pre-generate full inputs (memory ~40MB, acceptable)
# Using a faster method to generate binary representations
def generate_full_inputs():
    # Create an array of 0s and 1s
    # Use np.frombuffer?
    # Simpler:
    inputs = np.zeros((FULL_SIZE, N), dtype=np.uint8)
    for i in range(N):
        # Column i corresponds to bit (N-1-i)
        # Pattern repeats every 2^(N-i)
        # We can use a trick or just loop
        # Looping 18 times over 262k elements is fast
        inputs[:, i] = ((np.arange(FULL_SIZE) >> (N - 1 - i)) & 1).astype(np.uint8)
    return inputs

full_inputs = generate_full_inputs()

def simulate(network, inputs):
    X = inputs.copy()
    for i, j in network:
        min_val = np.minimum(X[:, i], X[:, j])
        max_val = np.maximum(X[:, i], X[:, j])
        X[:, i] = min_val
        X[:, j] = max_val
    return X

def is_sorted(X):
    return np.all(np.diff(X, axis=1) >= 0)

def check_subset(network):
    X = simulate(network, subset_inputs)
    return is_sorted(X)

def check_full(network):
    X = simulate(network, full_inputs)
    return is_sorted(X)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85  # 5s margin
    
    # Seed network: Odd-Even Transposition
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    best_network = network[:]
    best_size = len(network)
    
    # Heuristic Search: Greedy Removal + Swaps
    # We keep the current network and try to improve it.
    current_network = network[:]
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try Removals
        # Try removing each comparator in random order
        indices = list(range(len(current_network)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline: break
            
            new_network = current_network[:idx] + current_network[idx+1:]
            
            # Quick check
            if check_subset(new_network):
                # Full check
                if check_full(new_network):
                    current_network = new_network
                    best_network = current_network[:]
                    best_size = len(current_network)
                    improved = True
                    break # Restart search from new state
        
        if time.time() > deadline: break
        
        # 2. If no removal worked, try Swaps
        if not improved:
            # Try swapping a comparator with a random new one
            # Pick a random existing comparator to remove
            idx_remove = random.randint(0, len(current_network) - 1)
            
            # Pick a random new comparator to add
            # Avoid duplicates
            possible_pairs = [(i, j) for i in range(N) for j in range(i+1, N) if (i, j) not in current_network]
            if not possible_pairs:
                break # No more pairs to add
            
            new_comp = random.choice(possible_pairs)
            
            new_network = current_network[:idx_remove] + current_network[idx_remove+1:] + [new_comp]
            
            if check_subset(new_network):
                if check_full(new_network):
                    current_network = new_network
                    best_network = current_network[:]
                    best_size = len(current_network)
                    improved = True
        
        if not improved:
            # If stuck, maybe restart from a slightly perturbed seed?
            # Or just break.
            # Given time constraints, breaking is safer.
            break
            
    return best_network
# EVOLVE-BLOCK-END
