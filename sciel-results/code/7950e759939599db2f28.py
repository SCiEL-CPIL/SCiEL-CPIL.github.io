"""
Direct flat-list local search for sorting network optimization.
Uses bit-level simulation for fast validation and aggressive comparator removal.
"""
import numpy as np
import time
import random

N = 17
DEADLINE = 85  # safety margin before 90s kill

def _simulate_network(network, inputs):
    """Simulate network on given inputs (list of uint32 bitmasks).
    Returns True if all outputs are sorted (bit 0 is smallest)."""
    if not inputs:
        return True
    n = len(inputs)
    # wires[i] is a numpy array of bits for wire i across all inputs
    wires = [np.empty(n, dtype=np.uint8) for _ in range(N)]
    for i in range(N):
        wires[i] = (np.array(inputs, dtype=np.uint32) >> i) & 1
    
    for (a, b) in network:
        wa = wires[a]
        wb = wires[b]
        # min goes to a, max goes to b
        wires[a] = wa & wb
        wires[b] = wa | wb
    
    # Check sorted: wire 0 should be <= wire 1 <= ... <= wire N-1
    # i.e., for all i, if wire[i] has a 1 where wire[i+1] has a 0, it's unsorted
    for i in range(N - 1):
        if np.any((wires[i] == 1) & (wires[i+1] == 0)):
            return False
    return True

def _is_sorted_network_full(network):
    """Check if network sorts all 131072 boolean inputs."""
    all_inputs = np.arange(2**N, dtype=np.uint32)
    return _simulate_network(network, all_inputs)

def _is_sorted_network_sample(network, sample_inputs):
    """Check if network sorts given sample boolean inputs."""
    return _simulate_network(network, sample_inputs)

def _odd_even_transposition_network():
    """Create the seed odd-even transposition sort network."""
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
    return network

def _remove_redundant_comparators(network, sample_inputs, max_iterations=10000):
    """Try to remove comparators one by one while maintaining correctness on samples."""
    current = list(network)
    improved = True
    iterations = 0
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Try removing each comparator
        for idx in range(len(current) - 1, -1, -1):
            # Skip if too slow
            if iterations % 10 == 0 and time.time() > DEADLINE - 10:
                return current
            
            candidate = current[:idx] + current[idx+1:]
            if _is_sorted_network_sample(candidate, sample_inputs):
                current = candidate
                improved = True
                break  # Restart from beginning after a removal
    
    return current

def _local_search_add(network, sample_inputs, time_limit):
    """Try to add comparators to fix any unsorted inputs found in samples."""
    current = list(network)
    start_time = time.time()
    
    # Find unsorted inputs
    unsorted_inputs = []
    for inp in sample_inputs:
        # Simulate this single input
        wires = [(inp >> i) & 1 for i in range(N)]
        for (a, b) in current:
            wa, wb = wires[a], wires[b]
            wires[a] = wa & wb
            wires[b] = wa | wb
        # Check sorted
        sorted_ok = True
        for i in range(N - 1):
            if wires[i] > wires[i+1]:
                sorted_ok = False
                break
        if not sorted_ok:
            unsorted_inputs.append(inp)
    
    if not unsorted_inputs:
        return current
    
    # Try adding comparators at various positions
    max_additions = 20
    added = 0
    while unsorted_inputs and added < max_additions and time.time() - start_time < time_limit:
        # Pick a random unsorted input
        inp = random.choice(unsorted_inputs)
        wires = [(inp >> i) & 1 for i in range(N)]
        
        # Find where the inversion is
        # Try adding a comparator to fix it
        best_pos = None
        best_comp = None
        best_score = float('inf')
        
        for pos in range(len(current) + 1):
            for a in range(N):
                for b in range(a + 1, N):
                    # Simulate with this comparator added at pos
                    test_wires = wires[:]
                    for i, (ca, cb) in enumerate(current):
                        if i == pos:
                            # Insert new comparator here
                            wa, wb = test_wires[a], test_wires[b]
                            test_wires[a] = wa & wb
                            test_wires[b] = wa | wb
                        wa, wb = test_wires[ca], test_wires[cb]
                        test_wires[ca] = wa & wb
                        test_wires[cb] = wa | wb
                    
                    # Check if this fixes the inversion
                    inversions = 0
                    for i in range(N - 1):
                        if test_wires[i] > test_wires[i+1]:
                            inversions += 1
                    
                    if inversions < best_score:
                        best_score = inversions
                        best_pos = pos
                        best_comp = (a, b)
                    
                    if best_score == 0:
                        break
                if best_score == 0:
                    break
            if best_score == 0:
                break
        
        if best_comp is not None and best_score < float('inf'):
            current.insert(best_pos, best_comp)
            added += 1
            # Re-check which inputs are still unsorted
            new_unsorted = []
            for inp in unsorted_inputs:
                wires = [(inp >> i) & 1 for i in range(N)]
                for (a, b) in current:
                    wa, wb = wires[a], wires[b]
                    wires[a] = wa & wb
                    wires[b] = wa | wb
                sorted_ok = True
                for i in range(N - 1):
                    if wires[i] > wires[i+1]:
                        sorted_ok = False
                        break
                if not sorted_ok:
                    new_unsorted.append(inp)
            unsorted_inputs = new_unsorted
    
    return current

def construct():
    """Construct a sorting network using local search with comparator removal."""
    start_time = time.time()
    remaining = DEADLINE - (time.time() - start_time)
    
    if remaining < 15:
        return _odd_even_transposition_network()
    
    # Start with seed
    network = _odd_even_transposition_network()
    
    # Use a smaller sample for fast iteration
    np.random.seed(42)
    sample_size = 4096
    sample_inputs = np.random.randint(0, 2**N, size=sample_size, dtype=np.uint32)
    
    # Phase 1: Remove redundant comparators
    removal_time_budget = remaining * 0.4
    network = _remove_redundant_comparators(network, sample_inputs)
    
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    
    if remaining < 10:
        # Verify and return
        if _is_sorted_network_full(network):
            return network
        return _odd_even_transposition_network()
    
    # Phase 2: Local search to fix any issues and potentially add useful comparators
    fix_time_budget = remaining * 0.3
    network = _local_search_add(network, sample_inputs, fix_time_budget)
    
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    
    if remaining < 5:
        if _is_sorted_network_full(network):
            return network
        return _odd_even_transposition_network()
    
    # Final verification
    if _is_sorted_network_full(network):
        return network
    
    # Fallback to seed
    return _odd_even_transposition_network()
