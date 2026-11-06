import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
# Set deadline well below 60s to ensure we return before kill
DEADLINE_SECONDS = 50

def _count_comparators(net):
    return len(net)

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
    Uses the 0-1 principle.
    """
    n = len(net)
    if n > MAX_COMPARATORS:
        return False
    if n == 0:
        return True
    
    # Generate all 2^N boolean inputs
    # We can generate the matrix more efficiently
    all_masks = np.arange(1 << N, dtype=np.int32)
    wires = np.zeros((1 << N, N), dtype=np.int8)
    for i in range(N):
        wires[:, i] = (all_masks >> i) & 1
    
    # Apply comparators
    # To speed this up, we can avoid creating temporary arrays if we are careful,
    # but np.minimum/maximum are very fast.
    for i, j in net:
        wi = wires[:, i]
        wj = wires[:, j]
        # min to i, max to j
        # We need to be careful with in-place operations if i and j overlap? No, they are distinct indices.
        # But we need the original values for both.
        min_val = np.minimum(wi, wj)
        max_val = np.maximum(wi, wj)
        wires[:, i] = min_val
        wires[:, j] = max_val
    
    # Check if sorted: wires[:, k] <= wires[:, k+1]
    # Vectorized check
    if np.any(wires[:, :-1] > wires[:, 1:]):
        return False
    return True

def _odd_even_transposition():
    """Generate the odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Start with odd-even transposition network
    net = _odd_even_transposition()
    best_net = net[:]
    best_count = len(net)
    
    # Initial check (should be true)
    # _is_sorted_network(net)
    
    # Phase 1: Greedy removal
    # We iterate until no more removals are possible or time runs out.
    # We try removing comparators from the end first, as they are often redundant.
    # Then we try from the beginning.
    
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        
        # Try removing from end to start
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline:
                return best_net
            
            # Create new network without idx
            new_net = net[:idx] + net[idx+1:]
            
            # Verify
            if _is_sorted_network(new_net):
                net = new_net
                best_net = net[:]
                best_count = len(net)
                changed = True
                break # Restart the loop with the new network
        
        # If no change from end-to-start, try start-to-end
        if not changed:
            for idx in range(len(net)):
                if time.time() > deadline:
                    return best_net
                
                new_net = net[:idx] + net[idx+1:]
                
                if _is_sorted_network(new_net):
                    net = new_net
                    best_net = net[:]
                    best_count = len(net)
                    changed = True
                    break
    
    # Phase 2: Random pair removal (if time remains)
    if time.time() < deadline - 5:
        # Try removing pairs of comparators
        # This is more expensive, so we limit attempts
        attempts = 0
        while time.time() < deadline - 2 and attempts < 200:
            if len(net) < 2:
                break
            idx1, idx2 = random.sample(range(len(net)), 2)
            # Ensure idx1 < idx2 for slicing logic simplicity, though set logic works
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1
            
            new_net = net[:idx1] + net[idx1+1:idx2] + net[idx2+1:]
            
            if _is_sorted_network(new_net):
                net = new_net
                best_net = net[:]
                best_count = len(net)
                attempts = 0 # Reset attempts on success
            else:
                attempts += 1
                
    return best_net
