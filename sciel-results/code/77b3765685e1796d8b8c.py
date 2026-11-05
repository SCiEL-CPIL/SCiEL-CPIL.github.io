"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18

# EVOLVE-BLOCK-START

def _apply_network(net, bits):
    """Apply a sorting network to a list of bits. Returns sorted list."""
    vals = list(bits)
    for i, j in net:
        if vals[i] > vals[j]:
            vals[i], vals[j] = vals[j], vals[i]
    return vals

def _is_sorted(vals):
    for k in range(len(vals) - 1):
        if vals[k] > vals[k+1]:
            return False
    return True

def _verify_network(net):
    """Verify that a network sorts all 2^N Boolean inputs."""
    for mask in range(1 << N):
        bits = [(mask >> i) & 1 for i in range(N)]
        if not _is_sorted(_apply_network(net, bits)):
            return False
    return True

def _depth(net):
    """Compute parallel depth of a network."""
    depth = [0] * N
    for i, j in net:
        d = max(depth[i], depth[j]) + 1
        depth[i] = d
        depth[j] = d
    return max(depth) if depth else 0

def _odd_even_transposition(n):
    """Odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(n)
            for i in range(phase % 2, n - 1, 2)]

def _bitonic_merge(n):
    """Bitonic merge network for n wires."""
    if n <= 1:
        return []
    half = n // 2
    result = [(i, i + half) for i in range(half)]
    left = _bitonic_merge(half)
    right = _bitonic_merge(n - half)
    right_shifted = [(i + half, j + half) for i, j in right]
    return result + left + right_shifted

def _bitonic_sort(n):
    """Bitonic sort network for n wires."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    half = n // 2
    left = _bitonic_sort(half)
    right = _bitonic_sort(n - half)
    # Shift right network
    right_shifted = [(i + half, j + half) for i, j in right]
    # Reverse right to make descending
    right_reversed = [(j, i) for i, j in right]
    right_reversed_shifted = [(i + half, j + half) for i, j in right_reversed]
    merge = _bitonic_merge(n)
    return left + right_reversed_shifted + merge

def _merge_sort_network(n):
    """Classic merge sort network."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    half = n // 2
    left = _merge_sort_network(half)
    right = _merge_sort_network(n - half)
    right_shifted = [(i + half, j + half) for i, j in right]
    # For simplicity, use a basic merge: compare (i, half+i) then recursively merge
    # This is not optimal but correct
    merge = []
    for i in range(half):
        merge.append((i, half + i))
    # Then sort the result
    # For correctness, we'll just do a full sort of the merged array
    # But that's expensive. Let's use a simpler approach:
    # Actually, let's just return a valid but potentially non-optimal network
    # For now, use odd-even transposition on the whole thing as fallback
    # Better: use bitonic merge
    merge = _bitonic_merge(n)
    return left + right_shifted + merge

def _build_initial_network():
    """Build an initial sorting network using bitonic sort."""
    return _bitonic_sort(N)

def _local_search(net, max_time=50.0):
    """Local search to improve the network."""
    start_time = time.time()
    best_net = list(net)
    best_score = len(best_net)
    
    while time.time() - start_time < max_time:
        improved = False
        
        # Try removing each comparator
        for idx in range(len(best_net)):
            candidate = best_net[:idx] + best_net[idx+1:]
            if len(candidate) < best_score and _verify_network(candidate):
                best_net = candidate
                best_score = len(best_net)
                improved = True
                break
        
        if not improved:
            # Try replacing one comparator with another
            for idx in range(len(best_net)):
                i, j = best_net[idx]
                # Try swapping with other possible comparators
                for a in range(N):
                    for b in range(a + 1, N):
                        if (a, b) == (i, j):
                            continue
                        candidate = list(best_net)
                        candidate[idx] = (a, b)
                        if len(candidate) == best_score and _verify_network(candidate):
                            best_net = candidate
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            # Try adding a comparator
            for a in range(N):
                for b in range(a + 1, N):
                    if (a, b) in best_net:
                        continue
                    candidate = list(best_net)
                    candidate.append((a, b))
                    if _verify_network(candidate):
                        best_net = candidate
                        best_score = len(best_net)
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            # No improvement found in this pass
            break
    
    return best_net

def _supercharged_local_search(net, max_time=50.0):
    """Supercharged local search with Move/Swap/Replace operations."""
    start_time = time.time()
    best_net = list(net)
    best_score = len(best_net)
    
    while time.time() - start_time < max_time:
        improved = False
        
        # Try removing each comparator
        for idx in range(len(best_net)):
            candidate = best_net[:idx] + best_net[idx+1:]
            if len(candidate) < best_score and _verify_network(candidate):
                best_net = candidate
                best_score = len(best_net)
                improved = True
                break
        
        if not improved:
            # Try moving a comparator to a different position
            for idx in range(len(best_net)):
                for new_idx in range(len(best_net)):
                    if idx == new_idx:
                        continue
                    candidate = list(best_net)
                    comp = candidate.pop(idx)
                    candidate.insert(new_idx, comp)
                    if len(candidate) == best_score and _verify_network(candidate):
                        best_net = candidate
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            # Try replacing one comparator with another
            for idx in range(len(best_net)):
                i, j = best_net[idx]
                # Try swapping with other possible comparators
                for a in range(N):
                    for b in range(a + 1, N):
                        if (a, b) == (i, j):
                            continue
                        candidate = list(best_net)
                        candidate[idx] = (a, b)
                        if len(candidate) == best_score and _verify_network(candidate):
                            best_net = candidate
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            # Try adding a comparator
            for a in range(N):
                for b in range(a + 1, N):
                    if (a, b) in best_net:
                        continue
                    candidate = list(best_net)
                    candidate.append((a, b))
                    if _verify_network(candidate):
                        best_net = candidate
                        best_score = len(best_net)
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            # No improvement found in this pass
            break
    
    return best_net

def construct():
    """Construct a sorting network for N wires."""
    # Start with bitonic sort
    net = _build_initial_network()
    
    # Apply supercharged local search with time limit
    # Keep a safety margin before the 90s limit
    max_search_time = 50.0
    net = _supercharged_local_search(net, max_time=max_search_time)
    
    return net

# EVOLVE-BLOCK-END
