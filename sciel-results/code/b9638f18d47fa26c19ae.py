"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18

import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    SAT-inspired sorting network synthesis on 18 wires.
    
    Approach:
    1. Formulate network synthesis as constraint satisfaction using the 0-1 principle.
    2. Use a layered structure with variable comparator placement per layer.
    3. Employ iterative constraint propagation and local search to minimize comparators.
    4. For each boolean input, track which comparators are "critical" for sorting.
    """
    import random
    import time
    from collections import defaultdict
    
    start_time = time.time()
    timeout = 55  # seconds, leave buffer
    
    def simulate_network(comparators, mask):
        """Simulate the network on a single boolean input, return final state."""
        state = [(mask >> i) & 1 for i in range(N)]
        for (i, j) in comparators:
            if state[i] > state[j]:
                state[i], state[j] = state[j], state[i]
        return state
    
    def is_sorting_network(comparators):
        """Check if network sorts all 2^N boolean inputs (0-1 principle)."""
        if not comparators:
            return N <= 1
        for mask in range(1 << N):
            state = simulate_network(comparators, mask)
            # Check if sorted (non-decreasing)
            for k in range(N - 1):
                if state[k] > state[k + 1]:
                    return False
        return True
    
    def find_failing_inputs(comparators, sample_size=None):
        """Find boolean inputs that fail to sort. Returns list of failing masks."""
        failing = []
        if sample_size is None:
            sample_size = 1 << N
        else:
            # Random sample for speed
            all_masks = list(range(1 << N))
            random.shuffle(all_masks)
            all_masks = all_masks[:sample_size]
        
        for mask in all_masks:
            state = simulate_network(comparators, mask)
            for k in range(N - 1):
                if state[k] > state[k + 1]:
                    failing.append(mask)
                    break
        return failing
    
    def is_sorting_network_fast(comparators, precomputed_fails=None):
        """Fast check: if we have precomputed failing inputs, only check those."""
        if not comparators:
            return N <= 1
        
        if precomputed_fails is not None:
            # Only check previously failing inputs
            for mask in precomputed_fails:
                state = simulate_network(comparators, mask)
                for k in range(N - 1):
                    if state[k] > state[k + 1]:
                        return False
            # Also spot-check some random inputs
            for _ in range(100):
                mask = random.randint(0, (1 << N) - 1)
                state = simulate_network(comparators, mask)
                for k in range(N - 1):
                    if state[k] > state[k + 1]:
                        return False
            return True
        else:
            return is_sorting_network(comparators)
    
    # Generate initial dense layered network
    def generate_layered_network(num_layers, density=0.7):
        """Generate a layered sorting network with random comparator placement."""
        comps = []
        for layer in range(num_layers):
            # In each layer, place non-adjacent comparators
            available = list(range(N))
            random.shuffle(available)
            i = 0
            while i < N:
                if i + 1 < N and random.random() < density:
                    a, b = available[i], available[i + 1]
                    if a < b:
                        comps.append((a, b))
                    else:
                        comps.append((b, a))
                    i += 2
                else:
                    i += 1
        return comps
    
    # Odd-even transposition as baseline
    def odd_even_transposition():
        comps = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                comps.append((i, i + 1))
        return comps
    
    # Start with odd-even transposition
    current = odd_even_transposition()
    best = list(current)
    best_len = len(current)
    
    # Precompute all boolean inputs for fast checking
    all_masks = list(range(1 << N))
    random.seed(42)
    
    # Phase 1: Aggressive removal with constraint tracking
    # Track which comparators are "used" by each failing input
    def constraint_based_removal(comparators, max_iterations=100):
        """Remove comparators using constraint analysis."""
        comps = list(comparators)
        for _ in range(max_iterations):
            if time.time() - start_time > timeout:
                break
            if not comps:
                break
            
            # Find failing inputs
            failing = find_failing_inputs(comps, sample_size=5000)
            if not failing:
                break
            
            # For each comparator, check if it's critical for any failing input
            # A comparator is critical if removing it causes a new failure
            # Use a simpler heuristic: try removing each comparator and check
            indices_to_try = list(range(len(comps)))
            random.shuffle(indices_to_try)
            
            removed = False
            for idx in indices_to_try:
                if time.time() - start_time > timeout:
                    break
                new_comps = comps[:idx] + comps[idx+1:]
                if is_sorting_network_fast(new_comps, failing):
                    comps = new_comps
                    removed = True
                    break
            
            if not removed:
                break
        
        return comps
    
    # Phase 2: Simulated annealing with local search
    def simulated_annealing(comparators, iterations=500):
        """Use simulated annealing to find better networks."""
        comps = list(comparators)
        current_len = len(comps)
        best_local = list(comps)
        best_local_len = current_len
        
        temperature = 50.0
        cooling = 0.98
        
        for it in range(iterations):
            if time.time() - start_time > timeout:
                break
            
            # Generate neighbor: remove, add, or swap a comparator
            op = random.choice(['remove', 'add', 'swap'])
            
            if op == 'remove' and comps:
                idx = random.randint(0, len(comps) - 1)
                neighbor = comps[:idx] + comps[idx+1:]
            elif op == 'add':
                i, j = random.sample(range(N), 2)
                if i > j:
                    i, j = j, i
                pos = random.randint(0, len(comps))
                neighbor = comps[:pos] + [(i, j)] + comps[pos:]
            else:  # swap
                if len(comps) < 2:
                    continue
                i, j = random.sample(range(len(comps)), 2)
                neighbor = list(comps)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            
            neighbor_len = len(neighbor)
            
            # Check if neighbor is valid
            if is_sorting_network_fast(neighbor):
                if neighbor_len < best_local_len:
                    best_local = list(neighbor)
                    best_local_len = neighbor_len
                # Accept if better or with probability based on temperature
                delta = neighbor_len - current_len
                if delta <= 0:
                    comps = neighbor
                    current_len = neighbor_len
                else:
                    prob = min(1.0, temperature / max(1, delta))
                    if random.random() < prob:
                        comps = neighbor
                        current_len = neighbor_len
            else:
                # Reject invalid neighbors
                pass
            
            temperature *= cooling
        
        return best_local
    
    # Main optimization loop
    random.seed(42)
    
    # First, do aggressive constraint-based removal
    current = constraint_based_removal(current, max_iterations=200)
    if len(current) < best_len:
        best = list(current)
        best_len = len(current)
    
    # Then use simulated annealing for refinement
    for sa_iter in range(5):
        if time.time() - start_time > timeout:
            break
        
        # Start from best and run SA
        result = simulated_annealing(best, iterations=300)
        if len(result) < best_len:
            best = list(result)
            best_len = len(result)
        
        # Try from current as well
        result2 = simulated_annealing(current, iterations=200)
        if len(result2) < best_len:
            best = list(result2)
            best_len = len(result2)
    
    # Final greedy removal pass
    if time.time() - start_time < timeout:
        current = constraint_based_removal(best, max_iterations=100)
        if len(current) < best_len:
            best = list(current)
            best_len = len(current)
    
    # Verify final result
    if not is_sorting_network(best):
        # Fallback to odd-even transposition
        return odd_even_transposition()
    
    return best
# EVOLVE-BLOCK-END
