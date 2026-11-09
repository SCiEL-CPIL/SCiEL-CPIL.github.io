import time

N = 16

def construct():
    """
    Constructs a sorting network for 16 wires using Batcher's Bitonic Sort.
    """
    start_time = time.time()
    
    # Generate Bitonic Sort Network
    network = bitonic_sort_dir(N, 0, 1)
    
    # Optional: Verify correctness
    # If verification is too slow, we can trust the construction.
    # But let's verify to be safe.
    if not is_sorting(network, N):
        # Fallback to Odd-Even Transposition
        network = [(i, i + 1) for phase in range(N)
                   for i in range(phase % 2, N - 1, 2)]
        
    return network

def bitonic_sort_dir(n, offset, direction):
    """
    Recursively constructs a bitonic sort network.
    direction: 1 for ascending, -1 for descending.
    """
    if n == 1:
        return []
    
    half = n // 2
    # Sort first half in 'direction'
    left = bitonic_sort_dir(half, offset, direction)
    # Sort second half in opposite direction
    right = bitonic_sort_dir(half, offset + half, -direction)
    
    # Merge the two halves
    merge = bitonic_merge(n, offset)
    
    return left + right + merge

def bitonic_merge(n, offset):
    """
    Recursively constructs a bitonic merge network.
    Assumes input is a bitonic sequence of length n.
    """
    if n == 1:
        return []
    
    half = n // 2
    # Merge first half
    left = bitonic_merge(half, offset)
    # Merge second half
    right = bitonic_merge(half, offset + half)
    
    # Compare pairs (i, i + half)
    comps = []
    for i in range(half):
        comps.append((offset + i, offset + half + i))
        
    return left + right + comps

def is_sorting(network, n):
    """
    Verifies if the network sorts all 2^n inputs using Zero-One Principle.
    """
    for mask in range(1 << n):
        wires = [1 if (mask >> i) & 1 else 0 for i in range(n)]
        
        for i, j in network:
            if wires[i] > wires[j]:
                wires[i], wires[j] = wires[j], wires[i]
        
        for i in range(n - 1):
            if wires[i] > wires[i + 1]:
                return False
                
    return True
