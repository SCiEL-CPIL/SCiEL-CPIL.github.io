"""
Size-efficient sorting network search on 17 wires.
Uses batch-vectorized validation and layer-aware local search.
"""
import time
import random
import numpy as np

N = 17
SEED_COMPARATORS = 136
N_INPUTS = 1 << N  # 131072

# Precompute all Boolean inputs once
def _make_all_inputs():
    indices = np.arange(N_INPUTS, dtype=np.uint32)[:, None]
    bit_positions = np.arange(N, dtype=np.uint32)[None, :]
    return ((indices >> bit_positions) & 1).astype(np.uint8)

_ALL_INPUTS = _make_all_inputs()

def simulate_network(comparators, inputs=None):
    """Simulate compare-exchange network on all inputs. Returns output array."""
    if inputs is None:
        inputs = _ALL_INPUTS
    state = inputs.copy()
    for (i, j) in comparators:
        mi = state[:, i] & state[:, j]
        mj = state[:, i] | state[:, j]
        state[:, i] = mi
        state[:, j] = mj
    return state

def is_sorted_network(comparators):
    """Full check: does the network sort all 2^17 Boolean inputs?"""
    output = simulate_network(comparators)
    for k in range(N - 1):
        if np.any(output[:, k] > output[:, k + 1]):
            return False
    return True

def is_sorted_sample(comparators, n_samples=4096, seed=42):
    """Fast check on a random subset of inputs."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(N_INPUTS, size=n_samples, replace=False)
    return is_sorted_network(comparators)  # fallback: just do full check for safety

def odd_even_merge_sort_network(n):
    """Generate odd-even merge sort network for n wires."""
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

def build_layers(comparators):
    """Group comparators into depth-parallel layers (disjoint wires per layer)."""
    layers = []
    current_layer = []
    used = set()
    for (i, j) in comparators:
        if i in used or j in used:
            layers.append(current_layer)
            current_layer = []
            used = set()
        current_layer.append((i, j))
        used.add(i)
        used.add(j)
    if current_layer:
        layers.append(current_layer)
    return layers

def flatten_layers(layers):
    """Flatten layers back to sequential list."""
    result = []
    for layer in layers:
        result.extend(layer)
    return result

def try_remove_at(comparators, idx):
    """Try removing comparator at index idx. Return (success, new_list)."""
    candidate = comparators[:idx] + comparators[idx+1:]
    if is_sorted_network(candidate):
        return True, candidate
    return False, None

def greedy_layer_removal(comparators, deadline):
    """
    Remove comparators in a structured way:
    - Work layer by layer, trying to remove each comparator.
    - After a removal, rebuild layers.
    - Multiple passes until no more removals.
    """
    best = list(comparators)
    
    while time.time() < deadline - 3:
        removed_any = False
        layers = build_layers(best)
        
        for layer_idx in range(len(layers)):
            if time.time() > deadline - 3:
                break
            # Try removing each comparator in this layer (in reverse to preserve indices)
            for comp_idx in range(len(layers[layer_idx]) - 1, -1, -1):
                if time.time() > deadline - 3:
                    break
                global_idx = sum(len(layers[l]) for l in range(layer_idx)) + comp_idx
                if global_idx >= len(best):
                    continue
                ok, new_best = try_remove_at(best, global_idx)
                if ok:
                    best = new_best
                    removed_any = True
                    break  # Rebuild layers after any removal
            if removed_any:
                break
        
        if not removed_any:
            break
    
    return best

def random_local_search(comparators, deadline):
    """
    Randomly try removing and re-adding comparators to escape local minima.
    Also try swapping adjacent comparators.
    """
    best = list(comparators)
    rng = random.Random(12345)
    
    # Try random removals
    while time.time() < deadline - 5:
        n = len(best)
        if n == 0:
            break
        idx = rng.randint(0, n - 1)
        ok, new_best = try_remove_at(best, idx)
        if ok:
            best = new_best
            # After a removal, do a quick greedy pass
            if time.time() < deadline - 5:
                best = greedy_layer_removal(best, deadline)
    
    # Try inserting a random comparator then removing
    while time.time() < deadline - 3:
        pos = rng.randint(0, len(best))
        i = rng.randint(0, N - 2)
        j = rng.randint(i + 1, N - 1)
        candidate = best[:pos] + [(i, j)] + best[pos:]
        
        # Try to remove some comparators from the candidate
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            m = len(candidate)
            if m == 0:
                break
            idx = rng.randint(0, m - 1)
            ok, new_cand = try_remove_at(candidate, idx)
            if ok:
                candidate = new_cand
                improved = True
        
        if len(candidate) < len(best):
            best = candidate
    
    return best

def pairwise_swap_search(comparators, deadline):
    """Try swapping adjacent comparators to find better orderings that enable removals."""
    best = list(comparators)
    n = len(best)
    
    while time.time() < deadline - 5:
        improved = False
        for i in range(n - 1):
            if time.time() > deadline - 5:
                break
            candidate = best[:]
            candidate[i], candidate[i+1] = best[i+1], best[i]
            # Try removing a comparator from the swapped version
            m = len(candidate)
            for k in range(m):
                if time.time() > deadline - 5:
                    break
                ok, new_cand = try_remove_at(candidate, k)
                if ok:
                    best = new_cand
                    improved = True
                    n = len(best)
                    break
            if improved:
                break
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 82  # 8s safety margin before 90s kill
    
    # Base network: odd-even merge sort
    base = odd_even_merge_sort_network(N)
    
    if not is_sorted_network(base):
        # Fallback: odd-even transposition
        base = [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    best = list(base)
    
    # Phase 1: Greedy layer-based removal
    if time.time() < deadline - 10:
        best = greedy_layer_removal(best, deadline)
    
    # Phase 2: Random local search (remove + insert-then-remove)
    if time.time() < deadline - 8:
        best = random_local_search(best, deadline)
    
    # Phase 3: Another greedy pass
    if time.time() < deadline - 6:
        best = greedy_layer_removal(best, deadline)
    
    # Phase 4: Pairwise swap search
    if time.time() < deadline - 4:
        best = pairwise_swap_search(best, deadline)
    
    # Phase 5: Final greedy pass
    if time.time() < deadline - 2:
        best = greedy_layer_removal(best, deadline)
    
    # Final verification
    if not is_sorted_network(best):
        best = [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    return best
