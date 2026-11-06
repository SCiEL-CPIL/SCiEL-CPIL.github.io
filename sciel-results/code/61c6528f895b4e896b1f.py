import time
import numpy as np

N = 16

def generate_batcher_bitonic(n):
    # Returns list of comparators for Batcher's Bitonic Sort
    comparators = []
    
    def bitonic_merge(start, size, reverse=False):
        # Merge two bitonic sequences of length size/2 starting at start
        # Assumes first half is ascending, second half is descending (if not reverse)
        # Actually, standard bitonic merge:
        # 1. Compare (start+i, start+size-1-i) for i in 0..size/2-1
        # 2. Recursively merge top half and bottom half
        
        if size == 1:
            return
            
        # Step 1: n/2 comparators
        for i in range(size // 2):
            comparators.append((start + i, start + size - 1 - i))
            
        # Step 2: Recurse
        # Top half: start .. start+size/2-1
        # Bottom half: start+size/2 .. start+size-1
        
        # The top half is sorted in ascending order
        # The bottom half is sorted in descending order
        # We need to merge them.
        # Bitonic merge on top half (ascending + descending -> sorted)
        bitonic_merge(start, size // 2, False)
        # Bitonic merge on bottom half (ascending + descending -> sorted)
        # Wait, the bottom half after step 1 is also bitonic?
        # Yes, the result of the first step creates two bitonic sequences.
        # The top half is ascending, bottom half is descending?
        # No, the standard bitonic merge splits into two bitonic sequences.
        # We recursively merge them.
        
        # Actually, the recursive call should be on the two halves.
        # The top half is a bitonic sequence (ascending then descending? No, just ascending if we are careful?)
        # Let's check the property.
        # If we have a bitonic sequence of length n, we split into two halves of n/2.
        # After the first round of comparators, each half is bitonic.
        # We then recursively sort each half.
        
        bitonic_merge(start + size // 2, size // 2, False)
        
    def bitonic_sort(start, size, reverse=False):
        if size == 1:
            return
        bitonic_sort(start, size // 2, reverse)
        bitonic_sort(start + size // 2, size // 2, not reverse)
        bitonic_merge(start, size)

    bitonic_sort(0, n)
    return comparators

def verify(network):
    # Generate all 2^N inputs
    # Use vectorized numpy
    n_inputs = 1 << N
    # Create array of shape (n_inputs, N)
    # Each row is a binary representation of 0..2^N-1
    # We can use np.array([list(map(int, bin(i)[2:].zfill(N))) for i in range(n_inputs)])
    # But this is slow in Python.
    # Better:
    inputs = np.zeros((n_inputs, N), dtype=int)

    for i in range(N):
        inputs[:, i] = (np.arange(n_inputs) >> i) & 1
        
    # Simulate network
    for i, j in network:
        a = inputs[:, i]
        b = inputs[:, j]
        min_val = a & b
        max_val = a | b
        inputs[:, i] = min_val
        inputs[:, j] = max_val
        
    # Check if sorted
    # A sequence is sorted if x[0] <= x[1] <= ... <= x[N-1]
    # For boolean inputs, this means 0...0 1...1
    # So we just need to check if the array is non-decreasing along axis 1
    diffs = np.diff(inputs, axis=1)
    return np.all(diffs >= 0)

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    # Initial network
    net = generate_batcher_bitonic(N)
    
    if not verify(net):
        # Fallback to odd-even transposition if bitonic fails (should not happen)
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        
    best_net = net[:]
    best_len = len(net)
    
    import random
    random.seed(42)
    
    while time.time() < deadline:
        # Try to remove a random comparator
        if len(net) == 0:
            break
            
        idx = random.randint(0, len(net) - 1)
        test_net = net[:]
        test_net.pop(idx)
        
        if verify(test_net):
            net = test_net
            if len(net) < best_len:
                best_net = net[:]
                best_len = len(net)
        else:
            # Optionally, try swapping with another comparator to unlock removals
            # This is more complex, so we'll stick to removals for now
            pass
            
    return best_net
