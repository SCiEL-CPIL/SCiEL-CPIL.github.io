"""Optimized sorting network search: fast verification + aggressive greedy removal + swap local search."""
import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 85  # Safety margin before 90s kill

# Precompute all 2^N boolean inputs
_ALL_INPUTS = np.arange(1 << N, dtype=np.int32)
def _make_wires():
    """Create the full input matrix (2^N x N) of boolean values."""
    wires = np.zeros((1 << N, N), dtype=np.int8)
    for i in range(N):
        wires[:, i] = (_ALL_INPUTS >> i) & 1
    return wires

def _apply_comparator(wires, i, j):
    """Apply a single comparator in-place on the wires matrix."""
    wi = wires[:, i]
    wj = wires[:, j]
    # min goes to i, max goes to j
    min_val = np.minimum(wi, wj)
    max_val = np.maximum(wi, wj)
    wires[:, i] = min_val
    wires[:, j] = max_val

def _is_sorted(wires):
    """Check if the wires matrix is sorted (non-decreasing along wires)."""
    if np.any(wires[:, :-1] > wires[:, 1:]):
        return False
    return True

def _verify_network(net):
    """
    Verify that the comparator network sorts all 2^N boolean inputs.
    Returns True if the network is a valid sorting network.
    """
    if len(net) > MAX_COMPARATORS:
        return False
    wires = _make_wires()
    for i, j in net:
        _apply_comparator(wires, i, j)
    return _is_sorted(wires)

def _verify_network_from_state(wires, start_idx, net):
    """
    Verify the network starting from a given wires state, applying net[start_idx:].
    Returns True if the final state is sorted.
    """
    for i, j in net[start_idx:]:
        _apply_comparator(wires, i, j)
    return _is_sorted(wires)

def _odd_even_transposition():
    """Generate the odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def _batcher_odd_merge(n, depth=0):
    """Generate a Batcher odd-even merge network for n wires."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    half = n // 2
    net = []
    # Recursively sort halves
    left = _batcher_odd_merge(half)
    right = _batcher_odd_merge(n - half)
    for i, j in left:
        net.append((i, j))
    for i, j in right:
        net.append((i + half, j + half))
    # Odd-even merge
    if n == 4:
        net.extend([(0, 2), (1, 3), (1, 2)])
    elif n == 8:
        net.extend([(0, 4), (1, 5), (2, 6), (3, 7)])
        net.extend([(1, 4), (2, 5), (3, 6)])
        net.extend([(1, 3), (2, 4), (3, 5)])
        net.extend([(2, 3), (4, 5), (5, 6)])
        net.extend([(3, 4)])
    else:
        # General odd-even merge
        for i in range(half):
            net.append((i, i + half))
        # Recursively merge odd and even subsequences
        if half > 1:
            odd_idx = list(range(1, n, 2))
            even_idx = list(range(0, n, 2))
            # This is a simplification; for small n we handle explicitly
            # For now, just add adjacent comparators as a fallback
            for i in range(0, n - 1, 2):
                net.append((i, i + 1))
            for i in range(1, n - 1, 2):
                net.append((i, i + 1))
    return net

def _bitonic_sorting_network(n):
    """Generate a bitonic sorting network for n wires (power of 2)."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    half = n // 2
    net = []
    # Sort first half ascending
    left = _bitonic_sorting_network(half)
    for i, j in left:
        net.append((i, j))
    # Sort second half descending
    right = _bitonic_sorting_network(half)
    for i, j in right:
        net.append((i + half, j + half))
    # Reverse the right half to make it descending
    # Actually, for bitonic, we need: sort left asc, sort right desc, then merge
    # Let's use a simpler approach: just use odd-even transposition for correctness
    return _odd_even_transposition()

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Start with odd-even transposition network
    net = _odd_even_transposition()
    best_net = net[:]
    best_count = len(net)
    
    # Phase 1: Aggressive greedy removal
    # Try removing comparators in multiple passes, both forward and backward
    for pass_num in range(20):
        if time.time() > deadline - 5:
            break
        
        changed = False
        # Try removing from each position
        for idx in range(len(net)):
            if time.time() > deadline - 3:
                break
            
            new_net = net[:idx] + net[idx+1:]
            
            if _verify_network(new_net):
                net = new_net
                best_net = net[:]
                best_count = len(net)
                changed = True
                break  # Restart from beginning
        
        if not changed:
            # Try reverse order
            for idx in range(len(net) - 1, -1, -1):
                if time.time() > deadline - 3:
                    break
                
                new_net = net[:idx] + net[idx+1:]
                
                if _verify_network(new_net):
                    net = new_net
                    best_net = net[:]
                    best_count = len(net)
                    changed = True
                    break
        
        if not changed:
            break
    
    # Phase 2: Local search with swaps
    if time.time() < deadline - 10:
        attempts = 0
        max_attempts = 2000
        while time.time() < deadline - 3 and attempts < max_attempts:
            attempts += 1
            if len(net) == 0:
                break
            
            # Pick a random comparator to replace
            idx = random.randint(0, len(net) - 1)
            i, j = net[idx]
            
            # Try a random different comparator
            ni = random.randint(0, N - 2)
            nj = random.randint(ni + 1, N - 1)
            if (ni, nj) == (i, j):
                continue
            
            new_net = net[:]
            new_net[idx] = (ni, nj)
            
            if _verify_network(new_net):
                net = new_net
                best_net = net[:]
                best_count = len(net)
    
    # Phase 3: Try removing pairs of comparators simultaneously
    if time.time() < deadline - 15:
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            if len(net) < 2:
                break
            
            idx1 = random.randint(0, len(net) - 1)
            idx2 = random.randint(0, len(net) - 1)
            if idx1 == idx2:
                continue
            
            # Remove both
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1
            new_net = net[:idx2] + net[idx2+1:idx1] + net[idx1+1:]
            
            if _verify_network(new_net):
                net = new_net
                best_net = net[:]
                best_count = len(net)
    
    # Phase 4: Try adding then removing (if we have time)
    if time.time() < deadline - 10 and best_count > 70:
        for _ in range(30):
            if time.time() > deadline - 5:
                break
            if len(net) >= MAX_COMPARATORS:
                break
            
            # Add a random comparator
            ni = random.randint(0, N - 2)
            nj = random.randint(ni + 1, N - 1)
            new_net = net[:] + [(ni, nj)]
            
            if _verify_network(new_net):
                # Now try to remove more comparators
                temp_net = new_net
                temp_changed = True
                while temp_changed:
                    if time.time() > deadline - 3:
                        break
                    temp_changed = False
                    for idx in range(len(temp_net)):
                        if time.time() > deadline - 3:
                            break
                        candidate = temp_net[:idx] + temp_net[idx+1:]
                        if _verify_network(candidate):
                            temp_net = candidate
                            temp_changed = True
                            break
                    
                    if not temp_changed:
                        for idx in range(len(temp_net) - 1, -1, -1):
                            if time.time() > deadline - 3:
                                break
                            candidate = temp_net[:idx] + temp_net[idx+1:]
                            if _verify_network(candidate):
                                temp_net = candidate
                                temp_changed = True
                                break
                
                if len(temp_net) < best_count:
                    net = temp_net
                    best_net = net[:]
                    best_count = len(net)
    
    return best_net
