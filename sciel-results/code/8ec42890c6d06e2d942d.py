"""Size-efficient sorting networks on 16 wires.

Design:
- Start from a known-good 120-comparator odd-even transposition network.
- Use a fast vectorized zero-one verification (NumPy) to test candidate networks.
- Perform iterative local search: randomly remove a comparator, then try to
  re-insert a single comparator (from a pool of candidate pairs) to restore
  validity. If removal alone preserves validity, keep it.
- Use simulated annealing / hill-climbing with restarts to escape local minima.
- Track elapsed time and return the best valid network found before the deadline.

The score is the number of comparators (lower is better). The reference is 60,
so any reduction below 120 improves the score.
"""

import time
import random
import numpy as np

N = 16
NUM_ZEROS = 1 << N  # 65536

# Precompute all zero-one inputs as a (65536, 16) uint8 array
def _build_inputs():
    """Build the 65536 x 16 matrix of all Boolean inputs."""
    bits = np.arange(NUM_ZEROS, dtype=np.uint32)
    # Each row is the bit-representation of the input
    inputs = np.zeros((NUM_ZEROS, N), dtype=np.uint8)
    for k in range(N):
        inputs[:, k] = (bits >> k) & 1
    return inputs

# Precompute the expected sorted output for each input (all 0s then all 1s)
def _build_expected():
    """Expected output: for each input with c ones, output has first (N-c) zeros and last c ones."""
    expected = np.zeros((NUM_ZEROS, N), dtype=np.uint8)
    for idx in range(NUM_ZEROS):
        c = idx & ((1 << N) - 1)  # number of ones is just the index? No.
        # Actually, the number of ones in input idx is popcount(idx)
        # But we need to map: input idx has popcount(idx) ones.
        # After sorting, the last popcount(idx) positions are 1.
    # More efficient: compute popcount for all
    popcounts = np.zeros(NUM_ZEROS, dtype=np.uint8)
    for idx in range(NUM_ZEROS):
        popcounts[idx] = bin(idx).count('1')
    for idx in range(NUM_ZEROS):
        c = popcounts[idx]
        expected[idx, N - c:] = 1
    return expected

def _apply_comparator(inputs, i, j):
    """Apply a single compare-exchange (i, j) to the inputs matrix.
    min goes to wire i, max goes to wire j.
    Returns a new array."""
    out = inputs.copy()
    a = out[:, i]
    b = out[:, j]
    # min(a,b) -> i, max(a,b) -> j
    out[:, i] = np.minimum(a, b)
    out[:, j] = np.maximum(a, b)
    return out

def _verify_network(comparators):
    """Verify that the network sorts all 65536 inputs. Returns True if valid."""
    if not comparators:
        # Empty network only valid if N <= 1
        return N <= 1
    inputs = _build_inputs()
    for (i, j) in comparators:
        inputs = _apply_comparator(inputs, i, j)
    expected = _build_expected()
    return np.array_equal(inputs, expected)

def _verify_network_fast(comparators, inputs_cache, expected_cache):
    """Fast verification using precomputed caches. Modifies inputs_cache in place."""
    inputs = inputs_cache.copy()
    for (i, j) in comparators:
        a = inputs[:, i]
        b = inputs[:, j]
        inputs[:, i] = np.minimum(a, b)
        inputs[:, j] = np.maximum(a, b)
    return np.array_equal(inputs, expected_cache)

# Precompute at module level for speed
_INPUTS = _build_inputs()
_EXPECTED = _build_expected()

def _is_valid(comparators):
    """Check if the network is a valid sorting network."""
    return _verify_network_fast(comparators, _INPUTS, _EXPECTED)

def _initial_network():
    """Odd-even transposition network: 120 comparators."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _all_comparator_pairs():
    """All possible (i, j) pairs with i < j."""
    return [(i, j) for i in range(N) for j in range(i + 1, N)]

def construct():
    """Construct a sorting network with as few comparators as possible.
    
    Returns an ordered list of (i, j) pairs.
    """
    deadline = time.time() + 80.0  # 80s safety margin out of 90s
    
    best = _initial_network()
    best_len = len(best)
    
    # Precompute all possible comparator pairs for mutation
    all_pairs = _all_comparator_pairs()
    
    # Local search: try to remove comparators one at a time
    # For each comparator, try removing it and see if the network is still valid.
    # If not, try adding a different comparator to fix it.
    
    current = list(best)
    current_len = len(current)
    
    # Hill-climbing with random restarts
    for restart in range(50):
        if time.time() > deadline:
            break
        
        if restart > 0:
            # Restart from best found so far with some randomization
            current = list(best)
            # Randomly shuffle a few comparators
            for _ in range(random.randint(1, 5)):
                if len(current) > 1:
                    idx = random.randint(0, len(current) - 1)
                    # Replace with a random pair
                    current[idx] = random.choice(all_pairs)
            if not _is_valid(current):
                current = list(best)
        
        current_len = len(current)
        
        # Try to remove each comparator
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
            
            # Try removing each comparator in random order
            indices = list(range(len(current)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() > deadline:
                    break
                
                # Try removing comparator at idx
                candidate = current[:idx] + current[idx+1:]
                
                if _is_valid(candidate):
                    # Valid without this comparator
                    current = candidate
                    current_len -= 1
                    improved = True
                    if current_len < best_len:
                        best = list(current)
                        best_len = current_len
                    break  # Restart the inner loop with the new shorter network
        
        # If no removals worked, try some random mutations to escape local minima
        if time.time() > deadline:
            break
        
        # Random mutation: replace one comparator with another
        for _ in range(20):
            if time.time() > deadline:
                break
            
            candidate = list(current)
            idx = random.randint(0, len(candidate) - 1)
            candidate[idx] = random.choice(all_pairs)
            
            if _is_valid(candidate):
                current = candidate
                if current_len < best_len:
                    best = list(current)
                    best_len = current_len
    
    # Final check: make sure best is valid
    if not _is_valid(best):
        best = _initial_network()
    
    return best
