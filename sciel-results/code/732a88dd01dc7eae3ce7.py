"""Optimized sorting network minimizer using NumPy vectorized verification."""
import time
import random
import numpy as np

N = 17
_DEADLINE = 85  # seconds, safety margin before 90s kill
_NUM_INPUTS = 1 << N  # 131072

# Precompute expected sorted output for each of the 2^N inputs
def _build_expected_sorted():
    """Precompute expected sorted output as a NumPy array."""
    expected = np.zeros(_NUM_INPUTS, dtype=np.int32)
    for mask in range(_NUM_INPUTS):
        cnt = bin(mask).count('1')
        expected[mask] = ((1 << cnt) - 1) << (N - cnt)
    return expected

_EXPECTED = _build_expected_sorted()

def _verify_network(network):
    """Verify sorting network using NumPy vectorized simulation.
    Returns True if network sorts all 2^N binary inputs."""
    # State: for each input mask, the current wire assignment
    # Initialize: wire k of input m holds bit k of m
    state = np.arange(_NUM_INPUTS, dtype=np.int32)
    
    for i, j in network:
        vi = (state >> i) & 1
        vj = (state >> j) & 1
        # Where vi > vj: set i to 1, j to 0
        swap_down = vi > vj
        # Where vi < vj: set i to 0, j to 1
        swap_up = vi < vj
        # Apply swaps
        state[swap_down] = (state[swap_down] | (1 << i)) & ~(1 << j)
        state[swap_up] = (state[swap_up] & ~(1 << i)) | (1 << j)
    
    # Check if all states are sorted
    return np.array_equal(state, _EXPECTED)

def _seed_network():
    """Odd-even transposition sorting network for N wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _try_remove_comparator(network, idx):
    """Try removing comparator at index idx and verify."""
    candidate = network[:idx] + network[idx+1:]
    return candidate if _verify_network(candidate) else None

def _greedy_remove(network, start_time, deadline_margin=2):
    """Perform greedy removal passes on the network."""
    best = list(network)
    best_len = len(best)
    
    # Multiple passes with different directions
    for direction in ['forward', 'reverse', 'forward', 'reverse']:
        if time.time() - start_time > _DEADLINE - deadline_margin:
            break
        
        if direction == 'forward':
            i = 0
            while i < len(network):
                if time.time() - start_time > _DEADLINE - deadline_margin:
                    break
                candidate = _try_remove_comparator(network, i)
                if candidate is not None:
                    network = candidate
                    if len(network) < best_len:
                        best = list(network)
                        best_len = len(network)
                else:
                    i += 1
        else:
            i = len(network) - 1
            while i >= 0:
                if time.time() - start_time > _DEADLINE - deadline_margin:
                    break
                candidate = _try_remove_comparator(network, i)
                if candidate is not None:
                    network = candidate
                    if len(network) < best_len:
                        best = list(network)
                        best_len = len(network)
                    i = len(network) - 1
                else:
                    i -= 1
    
    return best

def _swap_comparators(network, i, j):
    """Swap comparators at positions i and j."""
    candidate = list(network)
    candidate[i], candidate[j] = candidate[j], candidate[i]
    return candidate

def _random_reorder(network):
    """Randomly reorder the network."""
    candidate = list(network)
    random.shuffle(candidate)
    return candidate

def _simulated_annealing(network, start_time, deadline_margin=2):
    """Simulated annealing search to improve the network."""
    best = list(network)
    best_len = len(best)
    current = list(network)
    current_len = len(current)
    
    # Temperature schedule
    initial_temp = 1.0
    cooling_rate = 0.999
    min_temp = 0.01
    
    temp = initial_temp
    rng = random.Random(42)
    
    while time.time() - start_time < _DEADLINE - deadline_margin and temp > min_temp:
        # Random move: try to remove a random comparator
        if len(current) > 1:
            idx = rng.randint(0, len(current) - 1)
            candidate = current[:idx] + current[idx+1:]
            if _verify_network(candidate):
                current = candidate
                current_len = len(current)
                if current_len < best_len:
                    best = list(current)
                    best_len = current_len
            else:
                # With some probability, accept the invalid move (to escape local minima)
                if rng.random() < temp:
                    current = candidate
                    current_len = len(current)
        
        # Random move: swap two comparators
        if len(current) > 1:
            i, j = rng.sample(range(len(current)), 2)
            candidate = _swap_comparators(current, i, j)
            if _verify_network(candidate):
                current = candidate
                current_len = len(current)
                if current_len < best_len:
                    best = list(current)
                    best_len = current_len
        
        # Random move: insert a random comparator
        if len(current) < 200:  # Keep size reasonable
            i = rng.randint(0, N - 2)
            j = rng.randint(i + 1, N - 1)
            pos = rng.randint(0, len(current))
            candidate = current[:pos] + [(i, j)] + current[pos:]
            if _verify_network(candidate):
                current = candidate
                current_len = len(current)
                if current_len < best_len:
                    best = list(current)
                    best_len = current_len
        
        # Cool down
        temp *= cooling_rate
    
    return best

def construct():
    start_time = time.time()
    
    # Start with seed network
    network = _seed_network()
    
    # Perform greedy removal first
    improved = _greedy_remove(network, start_time)
    
    # If time permits, try simulated annealing
    if time.time() - start_time < _DEADLINE - 5:
        improved = _simulated_annealing(improved, start_time)
    
    # Final verification
    if not _verify_network(improved):
        return _seed_network()
    
    return improved
