"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from collections import defaultdict

N = 17
SEED_COMPARATORS = 136

# Precompute all Boolean inputs
def _make_all_inputs():
    n_inputs = 1 << N
    indices = np.arange(n_inputs, dtype=np.uint32)[:, None]
    bit_positions = np.arange(N, dtype=np.uint32)[None, :]
    inputs = ((indices >> bit_positions) & 1).astype(np.uint8)
    return inputs

_ALL_INPUTS = None
_SAMPLE_INPUTS = None

def _get_all_inputs():
    global _ALL_INPUTS
    if _ALL_INPUTS is None:
        _ALL_INPUTS = _make_all_inputs()
    return _ALL_INPUTS

def _get_sample_inputs(n_samples=8192):
    global _SAMPLE_INPUTS
    if _SAMPLE_INPUTS is None:
        all_inputs = _get_all_inputs()
        rng = np.random.default_rng(12345)
        idx = rng.choice(len(all_inputs), size=n_samples, replace=False)
        _SAMPLE_INPUTS = all_inputs[idx]
    return _SAMPLE_INPUTS

def simulate_network(comparators, inputs):
    """Simulate compare-exchange network. Returns output array."""
    state = inputs.copy()
    for (i, j) in comparators:
        mi = state[:, i] & state[:, j]
        mj = state[:, i] | state[:, j]
        state[:, i] = mi
        state[:, j] = mj
    return state

def _check_sorted(output):
    """Check if output rows are sorted (non-decreasing)."""
    for k in range(N - 1):
        if np.any(output[:, k] > output[:, k + 1]):
            return False
    return True

def is_sorted_fast(comparators, inputs=None):
    """Quick check on sample inputs."""
    if inputs is None:
        inputs = _get_sample_inputs()
    output = simulate_network(comparators, inputs)
    return _check_sorted(output)

def is_sorted_network(comparators):
    """Full check on all 2^N Boolean inputs."""
    inputs = _get_all_inputs()
    output = simulate_network(comparators, inputs)
    return _check_sorted(output)

def odd_even_merge_sort_network(n):
    """
    Generate an odd-even merge sort network for n wires.
    Uses the recursive odd-even merge construction.
    Returns list of (i, j) comparator pairs.
    """
    if n == 1:
        return []
    
    def _build(n):
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        half = n // 2
        left = _build(half)
        right = [(i + half, j + half) for (i, j) in _build(n - half)]
        merged = _odd_even_merge(n)
        return left + right + merged
    
    def _odd_even_merge(n):
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        comparators = []
        
        if n % 2 == 0:
            for i in range(0, n, 2):
                comparators.append((i, i + 1))
            even_merged = _odd_even_merge(n // 2)
            for (a, b) in even_merged:
                comparators.append((2 * a, 2 * b))
            odd_merged = _odd_even_merge(n // 2)
            for (a, b) in odd_merged:
                comparators.append((2 * a + 1, 2 * b + 1))
        else:
            for i in range(1, n - 1, 2):
                comparators.append((i, i + 1))
            if n - 2 > 0:
                mid_merged = _odd_even_merge(n - 2)
                for (a, b) in mid_merged:
                    comparators.append((a + 1, b + 1))
        
        return comparators
    
    return _build(n)

def compute_redundancy_scores(comparators):
    """
    For each comparator, estimate how 'redundant' it is by checking
    how often it actually changes the output on random inputs.
    Lower score = more redundant = better candidate for removal.
    Uses vectorized simulation for speed.
    """
    sample = _get_sample_inputs(4096)
    n_comps = len(comparators)
    scores = np.zeros(n_comps, dtype=np.float64)
    
    # Vectorized simulation tracking swaps
    state = sample.copy()
    for ci, (i, j) in enumerate(comparators):
        a = state[:, i]
        b = state[:, j]
        swap_mask = a > b
        scores[ci] = np.sum(swap_mask)
        # Apply swap where needed
        state[:, i] = np.where(swap_mask, b, a)
        state[:, j] = np.where(swap_mask, a, b)
    
    return scores

def improve_by_removal_smart(comparators, deadline):
    """
    Remove comparators in order of redundancy score (most redundant first).
    Multiple passes with re-scoring.
    """
    best = list(comparators)
    
    while time.time() < deadline - 5:
        if len(best) == 0:
            break
        
        # Score all comparators
        scores = compute_redundancy_scores(best)
        
        # Sort by score ascending (most redundant first)
        order = sorted(range(len(best)), key=lambda x: scores[x])
        
        removed_any = False
        for idx in order:
            if time.time() > deadline - 5:
                break
            if idx >= len(best):
                break
            
            candidate = best[:idx] + best[idx+1:]
            if is_sorted_fast(candidate):
                if is_sorted_network(candidate):
                    best = candidate
                    removed_any = True
                    # Re-score since indices changed
                    break
        
        if not removed_any:
            break
    
    return best

def improve_by_removal_sequential(comparators, deadline):
    """
    Sequential pass: try removing each comparator in order.
    Multiple passes. Uses a quick pre-filter with fewer samples.
    """
    best = list(comparators)
    
    # Use a smaller sample for fast pre-filtering
    pre_sample = _get_sample_inputs(2048)
    
    def quick_check(candidate):
        """Very fast check on a small sample."""
        state = pre_sample.copy()
        for (i, j) in candidate:
            mi = state[:, i] & state[:, j]
            mj = state[:, i] | state[:, j]
            state[:, i] = mi
            state[:, j] = mj
        for k in range(N - 1):
            if np.any(state[:, k] > state[:, k + 1]):
                return False
        return True
    
    while time.time() < deadline - 5:
        removed_any = False
        i = 0
        while i < len(best):
            if time.time() > deadline - 5:
                break
            candidate = best[:i] + best[i+1:]
            if quick_check(candidate):
                if is_sorted_network(candidate):
                    best = candidate
                    removed_any = True
                    # Don't increment i
                else:
                    i += 1
            else:
                i += 1
        
        if not removed_any:
            break
    
    return best

def try_insert_and_remove(best, deadline):
    """
    Try inserting a random comparator at a random position, then removing redundancies.
    """
    if time.time() > deadline - 5:
        return best
    
    best_len = len(best)
    attempts = 0
    max_attempts = 200
    
    # Pre-compute a small sample for quick checks
    pre_sample = _get_sample_inputs(1024)
    
    def quick_check(candidate):
        state = pre_sample.copy()
        for (i, j) in candidate:
            mi = state[:, i] & state[:, j]
            mj = state[:, i] | state[:, j]
            state[:, i] = mi
            state[:, j] = mj
        for k in range(N - 1):
            if np.any(state[:, k] > state[:, k + 1]):
                return False
        return True
    
    while time.time() < deadline - 5 and attempts < max_attempts:
        attempts += 1
        
        pos = random.randint(0, len(best))
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        candidate = best[:pos] + [(i, j)] + best[pos:]
        
        # Try to remove comparators from the candidate
        k = 0
        improved = True
        while k < len(candidate) and improved:
            improved = False
            if time.time() > deadline - 3:
                break
            test = candidate[:k] + candidate[k+1:]
            if quick_check(test):
                if is_sorted_network(test):
                    candidate = test
                    improved = True
                else:
                    k += 1
            else:
                k += 1
        
        if len(candidate) < best_len:
            best = candidate
            best_len = len(best)
    
    return best

def build_layers(comparators):
    """
    Build depth-parallel layers from a sequential comparator list.
    Each layer contains disjoint comparators (no shared wires).
    Returns list of layers, where each layer is a list of (i,j) pairs.
    """
    layers = []
    current_layer = []
    used_wires = set()
    
    for (i, j) in comparators:
        if i in used_wires or j in used_wires:
            if current_layer:
                layers.append(current_layer)
            current_layer = []
            used_wires = set()
        current_layer.append((i, j))
        used_wires.add(i)
        used_wires.add(j)
    
    if current_layer:
        layers.append(current_layer)
    
    return layers

def flatten_layers(layers):
    """Flatten layers back to a sequential list."""
    result = []
    for layer in layers:
        result.extend(layer)
    return result

def simulate_layers(layers, inputs):
    """Simulate a network given as layers of disjoint comparators."""
    state = inputs.copy()
    for layer in layers:
        if not layer:
            continue
        # Extract all wires involved in this layer
        wire_indices = []
        for (i, j) in layer:
            wire_indices.append(i)
            wire_indices.append(j)
        wire_indices = list(set(wire_indices))
        wire_indices.sort()
        
        # Get current values on these wires
        vals = state[:, wire_indices]
        n_wires = len(wire_indices)
        
        # Apply all comparators in this layer
        for (i, j) in layer:
            # Find positions of i and j in wire_indices
            pos_i = wire_indices.index(i)
            pos_j = wire_indices.index(j)
            a = vals[:, pos_i]
            b = vals[:, pos_j]
            mi = a & b
            mj = a | b
            vals[:, pos_i] = mi
            vals[:, pos_j] = mj
        
        # Write back
        state[:, wire_indices] = vals
    
    return state

def is_sorted_network_layers(layers):
    """Check if a layer-based network sorts all Boolean inputs."""
    inputs = _get_all_inputs()
    output = simulate_layers(layers, inputs)
    return _check_sorted(output)

def is_sorted_network_layers_sample(layers, n_samples=4096):
    """Quick check on sample inputs for layer-based network."""
    all_inputs = _get_all_inputs()
    rng = np.random.default_rng(12345)
    idx = rng.choice(len(all_inputs), size=min(n_samples, len(all_inputs)), replace=False)
    sample = all_inputs[idx]
    output = simulate_layers(layers, sample)
    return _check_sorted(output)

def optimize_layer_structure(layers, deadline):
    """
    Optimize a layer-based network by removing redundant comparators.
    Works directly with the layer representation for efficiency.
    """
    best_layers = [list(layer) for layer in layers]
    
    pre_sample = _get_sample_inputs(2048)
    
    def quick_check(candidate_layers):
        state = pre_sample.copy()
        for layer in candidate_layers:
            if not layer:
                continue
            for (i, j) in layer:
                mi = state[:, i] & state[:, j]
                mj = state[:, i] | state[:, j]
                state[:, i] = mi
                state[:, j] = mj
        for k in range(N - 1):
            if np.any(state[:, k] > state[:, k + 1]):
                return False
        return True
    
    def full_check(candidate_layers):
        return is_sorted_network_layers(candidate_layers)
    
    # Try removing comparators from each layer, starting from the end
    # (comparators later in the network are more likely to be redundant)
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Process layers from last to first
        for layer_idx in range(len(best_layers) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            if not best_layers[layer_idx]:
                continue
            
            # Try removing each comparator in this layer
            for comp_idx in range(len(best_layers[layer_idx]) - 1, -1, -1):
                if time.time() > deadline - 5:
                    break
                
                # Create candidate with this comparator removed
                candidate_layers = [list(l) for l in best_layers]
                candidate_layers[layer_idx] = candidate_layers[layer_idx][:comp_idx] + \
                                                 candidate_layers[layer_idx][comp_idx+1:]
                
                # Quick check first
                if quick_check(candidate_layers):
                    # Full check
                    if full_check(candidate_layers):
                        best_layers = candidate_layers
                        improved = True
                        # Continue trying to remove more from this layer
                        continue
                    else:
                        comp_idx -= 1  # Move to previous comparator
                else:
                    comp_idx -= 1
        
        # Also try merging adjacent layers that have no conflicts
        # This can sometimes allow further removals
        if time.time() < deadline - 10:
            merged = False
            for i in range(len(best_layers) - 1):
                if time.time() > deadline - 5:
                    break
                if not best_layers[i] or not best_layers[i+1]:
                    continue
                
                # Check if we can merge these two layers
                wires_i = set()
                for (a, b) in best_layers[i]:
                    wires_i.add(a)
                    wires_i.add(b)
                wires_j = set()
                for (a, b) in best_layers[i+1]:
                    wires_j.add(a)
                    wires_j.add(b)
                
                if not wires_i & wires_j:
                    # Can merge
                    candidate_layers = [list(l) for l in best_layers]
                    candidate_layers[i] = best_layers[i] + best_layers[i+1]
                    candidate_layers.pop(i+1)
                    
                    if quick_check(candidate_layers):
                        if full_check(candidate_layers):
                            best_layers = candidate_layers
                            merged = True
                            break
            
            if merged:
                improved = True
    
    return best_layers

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Use odd-even merge sort as the base network
    base = odd_even_merge_sort_network(N)
    
    # Verify base is correct
    if not is_sorted_network(base):
        base = [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    best = list(base)
    best_len = len(best)
    
    # Phase 1: Sequential removal (most effective, do it first)
    if time.time() < deadline - 25:
        best = improve_by_removal_sequential(best, deadline)
        best_len = len(best)
    
    # Phase 2: Convert to layer representation and optimize
    if time.time() < deadline - 15:
        layers = build_layers(best)
        optimized_layers = optimize_layer_structure(layers, deadline)
        best = flatten_layers(optimized_layers)
        best_len = len(best)
    
    # Phase 3: Smart removal based on redundancy scores
    if time.time() < deadline - 10:
        best = improve_by_removal_smart(best, deadline)
        best_len = len(best)
    
    # Phase 4: Another sequential pass to catch remaining redundancies
    if time.time() < deadline - 8:
        best = improve_by_removal_sequential(best, deadline)
        best_len = len(best)
    
    # Phase 5: Convert back to layers and optimize again
    if time.time() < deadline - 5:
        layers = build_layers(best)
        optimized_layers = optimize_layer_structure(layers, deadline)
        best = flatten_layers(optimized_layers)
        best_len = len(best)
    
    # Phase 6: Final sequential removal
    if time.time() < deadline - 3:
        best = improve_by_removal_sequential(best, deadline)
        best_len = len(best)
    
    # Verify final result
    if not is_sorted_network(best):
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    return best
# EVOLVE-BLOCK-END
