"""
Strategy:
- Start from Batcher's odd-even merge sort network for 16 wires (known valid, ~63 comparators).
- Use a fast, vectorized zero-one principle check over all 2^16 Boolean inputs.
- Perform an aggressive local search:
    1. Greedy removal: repeatedly try to remove each comparator; accept the first valid removal.
    2. If no single removal works, perform random replacement moves (remove one, add another) to escape local minima.
    3. Occasionally accept temporary insertions to navigate the search space.
- Maintain a deadline guard to ensure we return within the time limit.
- The goal is to minimize comparator count; start from a good seed and iteratively reduce.
"""

import time
import numpy as np

N = 16
MAX_TIME = 58.0


def _generate_batcher_network():
    """
    Generates Batcher's Odd-Even Merge Sort network for N=16.
    Returns a flat list of comparators.
    """
    def merge(n, offset):
        if n == 1:
            return [(offset, offset + 1)]
        
        h = n // 2
        step1 = [(offset + i, offset + i + n) for i in range(n)]
        
        def strided_merge(n, offset, stride):
            if n == 1:
                return [(offset, offset + stride)]
            
            h = n // 2
            step1 = [(offset + i * stride, offset + (n + i) * stride) for i in range(n)]
            even = strided_merge(h, offset, 2 * stride)
            odd = strided_merge(h, offset + stride, 2 * stride)
            return step1 + even + odd
        
        even_merge = strided_merge(h, offset, 2)
        odd_merge = strided_merge(h, offset + 1, 2)
        
        return step1 + even_merge + odd_merge

    def sort(n, offset):
        if n <= 1:
            return []
        h = n // 2
        left = sort(h, offset)
        right = sort(h, offset + h)
        m = merge(h, offset)
        return left + right + m

    return sort(N, 0)


def _apply_network_vectorized(comps, inputs):
    """
    Apply the network to a batch of inputs.
    """
    vec = inputs.copy()
    for i, j in comps:
        min_val = np.minimum(vec[:, i], vec[:, j])
        max_val = np.maximum(vec[:, i], vec[:, j])
        vec[:, i] = min_val
        vec[:, j] = max_val
    return vec


def _is_sorted_network_vectorized(comps, bits):
    """
    Check if the network sorts all 2^16 boolean inputs.
    """
    out = _apply_network_vectorized(comps, bits)
    diffs = np.diff(out, axis=1)
    return np.all(diffs >= 0)


def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    # Pre-compute all boolean inputs
    n_inputs = 1 << N
    indices = np.arange(n_inputs)
    bits = (indices[:, None] >> np.arange(N)[::-1]) & 1
    
    # 1. Generate initial network (Batcher's, ~63 comparators)
    initial_comps = _generate_batcher_network()
    
    # Verify initial network
    if not _is_sorted_network_vectorized(initial_comps, bits):
        # Fallback: Odd-Even Transposition
        initial_comps = []
        for i in range(N):
            if i % 2 == 0:
                for j in range(0, N - 1, 2):
                    initial_comps.append((j, j + 1))
            else:
                for j in range(1, N - 1, 2):
                    initial_comps.append((j, j + 1))
    
    best_comps = initial_comps[:]
    best_len = len(best_comps)
    
    rng = np.random.default_rng(12345)
    
    # Pre-compute all possible comparators
    all_comps = [(i, j) for i in range(N) for j in range(i + 1, N)]
    all_comps_set = set(all_comps)
    
    # Local Search Loop
    while time.time() < deadline:
        if time.time() > deadline - 2.0:
            break
            
        current_comps = best_comps[:]
        improved = False
        
        # Phase 1: Try to remove each comparator in random order
        order = rng.permutation(len(current_comps))
        for idx in order:
            if time.time() > deadline - 2.0:
                break
            if len(current_comps) <= 1:
                break
            removed = current_comps.pop(idx)
            if _is_sorted_network_vectorized(current_comps, bits):
                best_comps = current_comps
                best_len = len(best_comps)
                improved = True
                break
            else:
                current_comps.insert(idx, removed)
        
        # Phase 2: If no removal worked, try random replacement
        if not improved and len(current_comps) > 1:
            # Try multiple random replacements
            for _ in range(50):
                if time.time() > deadline - 2.0:
                    break
                idx = rng.integers(0, len(current_comps))
                old_comp = current_comps[idx]
                # Sample a new comparator different from the old one
                new_comp = all_comps[rng.integers(0, len(all_comps))]
                attempts = 0
                while new_comp == old_comp and attempts < 10:
                    new_comp = all_comps[rng.integers(0, len(all_comps))]
                    attempts += 1
                
                if new_comp != old_comp:
                    current_comps[idx] = new_comp
                    if _is_sorted_network_vectorized(current_comps, bits):
                        best_comps = current_comps
                        improved = True
                        break
                    else:
                        current_comps[idx] = old_comp
        
        # Phase 3: Occasionally, try a temporary insertion to escape local minima
        if not improved and rng.random() < 0.05 and len(best_comps) < 80:
            new_comp = all_comps[rng.integers(0, len(all_comps))]
            if new_comp not in best_comps:
                idx = rng.integers(0, len(best_comps) + 1)
                candidate = best_comps[:]
                candidate.insert(idx, new_comp)
                if _is_sorted_network_vectorized(candidate, bits):
                    # Accept the larger network; hope we can reduce further
                    best_comps = candidate
                    best_len = len(best_comps)
    
    return best_comps

# EVOLVE-BLOCK-END
