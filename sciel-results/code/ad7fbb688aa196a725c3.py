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

def _batcher_network():
    """Batcher's odd-even merge sort network."""
    def _odd_even_merge(n):
        """Generate odd-even merge sort network for n wires (must be power of 2)."""
        if n == 1:
            return []
        half = n // 2
        # Recursively sort halves
        left = _odd_even_merge(half)
        right = [(i + half, j + half) for i, j in left]
        # Odd-even merge
        merge = []
        for k in range(half - 1, -1, -1):
            # Odd phase
            for i in range(k, n - 1, 2):
                merge.append((i, i + 1))
            # Even phase
            for i in range(k + 1, n - 1, 2):
                merge.append((i, i + 1))
        return left + right + merge
    
    # For 17 wires, use a hybrid approach
    # First, sort 16 wires with Batcher, then merge the 17th
    base16 = _odd_even_merge(16)
    # Insert wire 16 by comparing with all others in a pattern
    # Simple approach: add comparisons to insert wire 16
    extra = []
    for i in range(16):
        extra.append((i, 16))
    # Then re-sort with a few passes
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            extra.append((i, i + 1))
    return base16 + extra

def _bitonic_network():
    """Bitonic sort network for 17 wires (padded to 32)."""
    def _bitonic_stage(n, width, direction):
        """Generate comparators for one stage of bitonic sort."""
        comparators = []
        if width == 1:
            return comparators
        half = width // 2
        # Recursively process halves
        for offset in range(0, n, width):
            comparators.extend(_bitonic_stage(n, half, direction))
            # Cross comparisons
            for i in range(half):
                a = offset + i
                b = offset + i + half
                if direction == 1:
                    comparators.append((a, b))
                else:
                    comparators.append((b, a))
        return comparators
    
    # Bitonic sort for 32 wires, then extract first 17
    n = 32
    comparators = []
    for width in [2, 4, 8, 16, 32]:
        for start in range(0, n, width * 2):
            # Create bitonic sequence
            comparators.extend(_bitonic_stage(n, width, 1))
            comparators.extend(_bitonic_stage(n, width, 0))
    
    # Filter to only comparators involving wires 0-16
    filtered = [(i, j) for i, j in comparators if i < N and j < N]
    return filtered

def _random_network(num_comparators):
    """Generate a random network with the given number of comparators."""
    comparators = []
    for _ in range(num_comparators):
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        comparators.append((i, j))
    return comparators

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
                        best_len = len(best)
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
                        best_len = len(best)
                    i = len(network) - 1
                else:
                    i -= 1
    
    return best

def construct():
    start_time = time.time()
    
    # Try multiple initial network constructions
    initial_networks = [
        _seed_network(),
    ]
    
    # Add Batcher if time permits
    if time.time() - start_time < _DEADLINE - 5:
        try:
            batcher = _batcher_network()
            if _verify_network(batcher):
                initial_networks.append(batcher)
        except Exception:
            pass
    
    # Add bitonic if time permits
    if time.time() - start_time < _DEADLINE - 5:
        try:
            bitonic = _bitonic_network()
            if _verify_network(bitonic):
                initial_networks.append(bitonic)
        except Exception:
            pass
    
    # Try random networks
    if time.time() - start_time < _DEADLINE - 3:
        for _ in range(5):
            if time.time() - start_time > _DEADLINE - 3:
                break
            rand_net = _random_network(random.randint(50, 100))
            if _verify_network(rand_net):
                initial_networks.append(rand_net)
    
    # For each initial network, perform greedy removal
    best_overall = None
    best_len_overall = float('inf')
    
    for init_net in initial_networks:
        if time.time() - start_time > _DEADLINE - 2:
            break
        
        if not _verify_network(init_net):
            continue
        
        improved = _greedy_remove(init_net, start_time)
        
        if len(improved) < best_len_overall:
            best_len_overall = len(improved)
            best_overall = improved
    
    # If no valid network found, return seed
    if best_overall is None:
        return _seed_network()
    
    # Final verification
    if not _verify_network(best_overall):
        return _seed_network()
    
    return best_overall
