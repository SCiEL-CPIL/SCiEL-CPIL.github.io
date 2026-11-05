"""Size-efficient sorting networks on 18 wires."""
import numpy as np
import time

N = 18

# EVOLVE-BLOCK-START

def _apply_network(net, x):
    """Apply network to array x (in-place). x is length N."""
    for i, j in net:
        if x[i] > x[j]:
            x[i], x[j] = x[j], x[i]

def _verify_network(net):
    """Verify network sorts all 2^N boolean inputs. Returns True if valid."""
    # Use bit-parallel: represent 2^N inputs as an array of shape (N, 2^N)
    # where column c is the c-th input's bits.
    total = 1 << N
    # Build the input matrix
    # For each wire k, the k-th bit of input c
    idx = np.arange(total, dtype=np.uint8)
    # We need to check if after applying the network, the output is sorted
    # i.e., for each column, the bits are non-increasing (or we can check sorted ascending)
    
    # Actually, let's think differently. A sorting network sorts any input.
    # For boolean inputs, sorting means all 1s come before all 0s (if sorting descending)
    # or all 0s come before all 1s (if sorting ascending).
    # The compare-exchange writes min to i and max to j, so it sorts in ascending order.
    # So for boolean input, after sorting, we should have 0s followed by 1s.
    
    # Let's represent the state as a 2D array: state[wire] = bit value for all inputs
    # state has shape (N, total)
    # Initialize: for wire k, state[k] = (idx >> k) & 1
    state = np.zeros((N, total), dtype=np.uint8)
    for k in range(N):
        state[k] = (idx >> k) & 1
    
    for i, j in net:
        # compare-exchange: min to i, max to j
        # For boolean: min(a,b) = a AND b, max(a,b) = a OR b
        new_i = state[i] & state[j]
        new_j = state[i] | state[j]
        state[i] = new_i
        state[j] = new_j
    
    # Check if sorted: for each column, bits should be non-decreasing (0s then 1s)
    # i.e., state[0] <= state[1] <= ... <= state[N-1]
    for k in range(N - 1):
        if np.any(state[k] > state[k+1]):
            return False
    return True

def _depth(net):
    """Compute the parallel depth of the network."""
    if not net:
        return 0
    # depth[i] = the depth at which wire i last had a comparator
    depth = [0] * N
    for i, j in net:
        d = max(depth[i], depth[j]) + 1
        depth[i] = d
        depth[j] = d
    return max(depth)

def _odd_even_transposition():
    """Generate the odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def construct():
    """Construct a sorting network with as few comparators as possible."""
    start_time = time.time()
    deadline = start_time + 75  # Safety margin: 90s limit, 75s deadline
    
    # Start with the seed odd-even transposition network
    best_net = _odd_even_transposition()
    best_len = len(best_net)
    
    # Try to optimize by removing redundant comparators
    # Strategy: iteratively try removing each comparator and check validity
    # If valid, keep it removed; otherwise restore.
    
    net = list(best_net)
    
    # First pass: try removing each comparator
    i = 0
    while i < len(net) and time.time() < deadline:
        # Try removing net[i]
        candidate = net[:i] + net[i+1:]
        if _verify_network(candidate):
            net = candidate
            # Don't increment i, since the list shifted
            best_len = len(net)
            best_net = list(net)
        else:
            i += 1
    
    # Second pass: try removing each comparator again (might find more after first pass)
    i = 0
    while i < len(net) and time.time() < deadline:
        candidate = net[:i] + net[i+1:]
        if _verify_network(candidate):
            net = candidate
            best_len = len(net)
            best_net = list(net)
        else:
            i += 1
    
    # Third pass
    i = 0
    while i < len(net) and time.time() < deadline:
        candidate = net[:i] + net[i+1:]
        if _verify_network(candidate):
            net = candidate
            best_len = len(net)
            best_net = list(net)
        else:
            i += 1
    
    # Fourth pass
    i = 0
    while i < len(net) and time.time() < deadline:
        candidate = net[:i] + net[i+1:]
        if _verify_network(candidate):
            net = candidate
            best_len = len(net)
            best_net = list(net)
        else:
            i += 1
    
    # Fifth pass
    i = 0
    while i < len(net) and time.time() < deadline:
        candidate = net[:i] + net[i+1:]
        if _verify_network(candidate):
            net = candidate
            best_len = len(net)
            best_net = list(net)
        else:
            i += 1
    
    # Sixth pass
    i = 0
    while i < len(net) and time.time() < deadline:
        candidate = net[:i] + net[i+1:]
        if _verify_network(candidate):
            net = candidate
            best_len = len(net)
            best_net = list(net)
        else:
            i += 1
    
    # Seventh pass
    i = 0
    while i < len(net) and time.time() < deadline:
        candidate = net[:i] + net[i+1:]
        if _verify_network(candidate):
            net = candidate
            best_len = len(net)
            best_net = list(net)
        else:
            i += 1
    
    # Eighth pass
    i = 0
    while i < len(net) and time.time() < deadline:
        candidate = net[:i] + net[i+1:]
        if _verify_network(candidate):
            net = candidate
            best_len = len(net)
            best_net = list(net)
        else:
            i += 1
    
    # Ninth pass
    i = 0
    while i < len(net) and time.time() < deadline:
        candidate = net[:i] + net[i+1:]
        if _verify_network(candidate):
            net = candidate
            best_len = len(net)
            best_net = list(net)
        else:
            i += 1
    
    # Tenth pass
    i = 0
    while i < len(net) and time.time() < deadline:
        candidate = net[:i] + net[i+1:]
        if _verify_network(candidate):
            net = candidate
            best_len = len(net)
            best_net = list(net)
        else:
            i += 1
    
    return best_net

# EVOLVE-BLOCK-END
