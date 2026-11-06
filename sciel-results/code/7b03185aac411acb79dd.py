"""
Greedy removal + targeted insertion for sorting network optimization.
Starts from a valid network, greedily removes comparators while maintaining
validity, then attempts targeted insertions to find better configurations.
"""
import numpy as np
import time
import random

N = 17
DEADLINE = 85  # safety margin before 90s kill

def _simulate_network_fast(network, inputs):
    """Vectorized simulation of network on multiple boolean inputs."""
    wires = [np.empty(len(inputs), dtype=np.uint8) for _ in range(N)]
    for i in range(N):
        wires[i] = (inputs >> i) & 1
    
    for (a, b) in network:
        wa = wires[a]
        wb = wires[b]
        wires[a] = wa & wb
        wires[b] = wa | wb
    
    # Check sorted: for each adjacent pair, no 1 followed by 0
    for i in range(N - 1):
        if np.any((wires[i] == 1) & (wires[i+1] == 0)):
            return False
    return True

def _is_sorted_network_full(network):
    """Check if network sorts all 131072 boolean inputs (exact)."""
    all_inputs = np.arange(2**N, dtype=np.uint32)
    return _simulate_network_fast(network, all_inputs)

def _count_inversions(network, inputs):
    """Count total inversions in output for given inputs. Lower is better."""
    wires = [np.empty(len(inputs), dtype=np.uint8) for _ in range(N)]
    for i in range(N):
        wires[i] = (inputs >> i) & 1
    
    for (a, b) in network:
        wa = wires[a]
        wb = wires[b]
        wires[a] = wa & wb
        wires[b] = wa | wb
    
    total_inv = 0
    for i in range(N - 1):
        total_inv += int(np.sum((wires[i] == 1) & (wires[i+1] == 0)))
    return total_inv

def _odd_even_transposition_network():
    """Generate odd-even transposition sort network (seed)."""
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
    return network

def _bitonic_sort_network(n):
    """Generate bitonic sort network for n wires (power of 2)."""
    # For 17 wires, we'll use a hybrid: bitonic on 16 + merge with 1
    # This is complex, so we'll use odd-even transposition as seed
    pass

def _greedy_remove(network, time_budget):
    """
    Greedily try to remove each comparator if the network remains valid.
    Returns the minimized network.
    """
    start_time = time.time()
    current = list(network)
    
    # Try removing from the end first (later comparators are often more redundant)
    # Then try all positions
    for attempt in range(3):  # Multiple passes
        if time.time() - start_time > time_budget:
            break
            
        i = len(current) - 1
        while i >= 0:
            if time.time() - start_time > time_budget:
                break
            
            # Try removing comparator at position i
            candidate = current[:i] + current[i+1:]
            
            if _is_sorted_network_full(candidate):
                current = candidate
                # Don't decrement i, check same position again (new element moved here)
            else:
                i -= 1
    
    return current

def _try_insert_comparator(network, time_budget):
    """
    Try to insert a comparator at various positions to see if it allows
    further removals elsewhere. Returns best network found.
    """
    start_time = time.time()
    best = list(network)
    best_len = len(best)
    
    # Generate candidate positions and wires
    max_pos = len(best) + 5
    candidate_wires = []
    for i in range(N):
        for j in range(i+1, N):
            candidate_wires.append((i, j))
    
    random.shuffle(candidate_wires)
    
    # Try a limited number of insertions
    tried = 0
    max_tries = 50
    for (a, b) in candidate_wires:
        if time.time() - start_time > time_budget or tried >= max_tries:
            break
        
        tried += 1
        
        # Try inserting at a few random positions
        for _ in range(3):
            pos = random.randint(0, len(best))
            candidate = best[:pos] + [(a, b)] + best[pos:]
            
            # Check if this insertion allows removing more comparators
            # Quick check: is the network still valid?
            if not _is_sorted_network_full(candidate):
                continue
            
            # Try greedy removal on this candidate
            reduced = _greedy_remove(candidate, time_budget - (time.time() - start_time))
            
            if len(reduced) < best_len:
                best = reduced
                best_len = len(best)
    
    return best

def _local_search_remove(network, time_budget):
    """
    More aggressive local search: try removing pairs of comparators.
    """
    start_time = time.time()
    current = list(network)
    
    # Try removing pairs
    for _ in range(10):  # Limited iterations
        if time.time() - start_time > time_budget:
            break
        
        improved = False
        n = len(current)
        
        # Try random pairs
        for _ in range(20):
            if time.time() - start_time > time_budget:
                break
            
            if n < 2:
                break
                
            i, j = random.sample(range(n), 2)
            i, j = min(i, j), max(i, j)
            
            # Remove both
            candidate = current[:i] + current[i+1:j] + current[j+1:]
            
            if _is_sorted_network_full(candidate):
                current = candidate
                n = len(current)
                improved = True
                break
        
        if not improved:
            break
    
    return current

def construct():
    """Construct a sorting network using greedy removal + targeted insertion."""
    start_time = time.time()
    time_budget = DEADLINE - (time.time() - start_time) - 5
    
    if time_budget < 10:
        # Not enough time, return seed
        return _odd_even_transposition_network()
    
    # Start with odd-even transposition (136 comparators)
    network = _odd_even_transposition_network()
    
    # Phase 1: Greedy removal (most important)
    phase1_time = time_budget * 0.5
    network = _greedy_remove(network, phase1_time)
    
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed - 5
    
    if remaining < 5:
        return network
    
    # Phase 2: Local search with pair removals
    phase2_time = remaining * 0.4
    network = _local_search_remove(network, phase2_time)
    
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed - 5
    
    if remaining < 5:
        return network
    
    # Phase 3: Targeted insertion to enable further removals
    phase3_time = remaining * 0.5
    network = _try_insert_comparator(network, phase3_time)
    
    # Final verification
    if not _is_sorted_network_full(network):
        # Fallback to seed
        return _odd_even_transposition_network()
    
    return network
