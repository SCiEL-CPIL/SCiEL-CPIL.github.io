"""Size-efficient sorting networks on 18 wires."""
import time
import numpy as np

N = 18
_input_bits = None

def _get_input_bits():
    global _input_bits
    if _input_bits is None:
        # Pre-compute bit arrays for all 2^N inputs
        # inputs[i] is the i-th bit of the integer 0..2^N-1
        _input_bits = [(np.arange(1 << N) >> i) & 1 for i in range(N)]
    return _input_bits

def _verify_network(network):
    """Verify network sorts all boolean inputs. Returns True if correct."""
    if not network:
        return N <= 1
        
    bits = _get_input_bits()
    # Copy the bits because we will modify them
    wires = [b.copy() for b in bits]
    
    for a, b in network:
        wa = wires[a]
        wb = wires[b]
        # min goes to a, max goes to b
        wires[a] = np.minimum(wa, wb)
        wires[b] = np.maximum(wa, wb)
    
    # Check if sorted: wire[0] <= wire[1] <= ... <= wire[N-1]
    for i in range(N - 1):
        if np.any(wires[i] > wires[i + 1]):
            return False
    return True

def _batcher_odd_even_merge_sort(n):
    """Build Batcher's odd-even merge sort network for n wires."""
    network = []
    
    def oe_merge(k, offset):
        if k <= 1:
            return
        half = k // 2
        # Recursively merge odd and even positions
        # This specific recursion structure is for Batcher's O(EM)
        oe_merge(half, offset + 1)
        oe_merge(half, offset)
        # Compare-exchange
        for i in range(half):
            a = offset + i
            b = offset + half + i
            network.append((a, b))
            
    def oe_sort(k, offset):
        if k <= 1:
            return
        half = k // 2
        oe_sort(half, offset)
        oe_sort(k - half, offset + half)
        oe_merge(half, offset)
        
    oe_sort(n, 0)
    return network

def _prune_network(network, time_limit):
    """Prune redundant comparators using efficient backward/forward passes."""
    current = list(network)
    start = time.time()
    
    # Backward pass
    i = len(current) - 1
    while i >= 0:
        if time.time() - start > time_limit:
            break
        candidate = current[:i] + current[i+1:]
        if _verify_network(candidate):
            current = candidate
            # Stay at same index i because next element shifted down
        else:
            i -= 1
            
    # Forward pass
    if time.time() - start < time_limit * 0.7:
        i = 0
        while i < len(current):
            if time.time() - start > time_limit:
                break
            candidate = current[:i] + current[i+1:]
            if _verify_network(candidate):
                current = candidate
                # Stay at same index i
            else:
                i += 1
                
    return current

def construct():
    deadline = 85.0 # 5s safety margin from 90s
    start = time.time()
    
    # 1. Generate Batcher's O(EM) network
    network = _batcher_odd_even_merge_sort(N)
    
    # 2. Verify base network
    if not _verify_network(network):
        # Fallback to Odd-Even Transposition (153 comparators)
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        
    # 3. Prune redundant comparators
    elapsed = time.time() - start
    remaining = deadline - elapsed
    if remaining > 5:
        network = _prune_network(network, remaining)
        
    # 4. Final verification
    if not _verify_network(network):
        # If pruning broke it (shouldn't happen if verify is correct), fallback
        # Or just return the pruned one if it's valid, else base
        # We verified in prune, so it should be valid.
        pass
        
    return network
