"""Combined approach: fast vectorized verification + greedy removal + local search."""
import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 50  # Safety margin before 60s/90s kill

def _depth(net):
    """Compute the parallel depth of a comparator network."""
    if not net:
        return 0
    layer = [0] * N
    for i, j in net:
        layer[i] = max(layer[i], layer[j]) + 1
    return max(layer) if layer else 0

def _is_sorted_network(net):
    """
    Verify that the comparator network sorts all 2^N boolean inputs.
    Uses the 0-1 principle with vectorized NumPy operations.
    """
    n = len(net)
    if n > MAX_COMPARATORS:
        return False
    if n == 0:
        return True
    
    # Generate all 2^N boolean inputs as a matrix
    all_masks = np.arange(1 << N, dtype=np.int32)
    wires = np.zeros((1 << N, N), dtype=np.int8)
    for i in range(N):
        wires[:, i] = (all_masks >> i) & 1
    
    # Apply comparators using vectorized min/max
    for i, j in net:
        wi = wires[:, i]
        wj = wires[:, j]
        min_val = np.minimum(wi, wj)
        max_val = np.maximum(wi, wj)
        wires[:, i] = min_val
        wires[:, j] = max_val
    
    # Check if sorted: wires[:, k] <= wires[:, k+1] for all k
    if np.any(wires[:, :-1] > wires[:, 1:]):
        return False
    return True

def _odd_even_transposition():
    """Generate the odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Start with odd-even transposition network (153 comparators for N=18)
    net = _odd_even_transposition()
    best_net = net[:]
    best_count = len(net)
    
    # Phase 1: Greedy removal - try removing comparators one at a time
    # Iterate until no more removals are possible or time runs out
    changed = True
    while changed:
        if time.time() > deadline - 3:
            break
        changed = False
        
        # Try removing from end to start (often less critical comparators are at the end)
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline - 3:
                return best_net
            
            new_net = net[:idx] + net[idx+1:]
            
            if _is_sorted_network(new_net):
                net = new_net
                best_net = net[:]
                best_count = len(net)
                changed = True
                break  # Restart with new network
        
        # If no change from end-to-start, try start-to-end
        if not changed:
            for idx in range(len(net)):
                if time.time() > deadline - 3:
                    return best_net
                
                new_net = net[:idx] + net[idx+1:]
                
                if _is_sorted_network(new_net):
                    net = new_net
                    best_net = net[:]
                    best_count = len(net)
                    changed = True
                    break
    
    # Phase 2: Local search - try replacing individual comparators
    if time.time() < deadline - 5:
        attempts = 0
        max_attempts = 500
        while time.time() < deadline - 2 and attempts < max_attempts:
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
            
            if _is_sorted_network(new_net):
                net = new_net
                best_net = net[:]
                best_count = len(net)
    
    # Phase 3: Try adding and then removing (if we have time and count allows)
    if time.time() < deadline - 10 and best_count > 70:
        # Sometimes adding a comparator allows removing multiple others
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            if len(net) >= MAX_COMPARATORS:
                break
            
            # Add a random comparator
            ni = random.randint(0, N - 2)
            nj = random.randint(ni + 1, N - 1)
            new_net = net[:] + [(ni, nj)]
            
            if _is_sorted_network(new_net):
                # Now try to remove more comparators from this larger network
                temp_net = new_net
                temp_changed = True
                while temp_changed:
                    if time.time() > deadline - 3:
                        break
                    temp_changed = False
                    for idx in range(len(temp_net) - 1, -1, -1):
                        if time.time() > deadline - 3:
                            break
                        candidate = temp_net[:idx] + temp_net[idx+1:]
                        if _is_sorted_network(candidate):
                            temp_net = candidate
                            temp_changed = True
                            break
                
                if len(temp_net) < best_count:
                    net = temp_net
                    best_net = net[:]
                    best_count = len(net)
    
    return best_net
