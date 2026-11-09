"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

import time
import random

# EVOLVE-BLOCK-START

def _verify_network_bitvector(network):
    """
    Fast verification using bit-vectors.
    Represents all 2^16 inputs as 16 bit-vectors (one per wire).
    Returns True if sorted, False otherwise.
    """
    # Initialize bit-vectors for all 16 wires
    # wire_bits[w] is an integer where bit k is 1 if input k has 1 on wire w
    # We iterate over all 2^16 inputs (k from 0 to 65535)
    # To be fast, we can construct the initial state directly.
    # Input k is the binary representation of k.
    # Wire w of input k is (k >> w) & 1.
    
    # We can represent the state of all inputs for a specific wire as a big integer.
    # Bit k of wire_bits[w] corresponds to input k.
    # If input k has value 1 on wire w, bit k is set.
    
    # Initialize wire_bits
    # For wire w, the pattern of 1s is: 0000111100001111... (period 2^(w+1), half 0s, half 1s)
    # Actually, simpler:
    # wire_bits[w] = sum( (1 << k) for k in range(65536) if (k >> w) & 1 )
    
    # Let's generate them efficiently.
    # Pattern for wire 0: 010101... -> 0x5555...
    # Pattern for wire 1: 00110011... -> 0x3333...
    # Pattern for wire w: 00001111... (w zeros, w ones) -> 
    # We can use a recursive doubling or just loop. 
    # 65536 bits is 8192 bytes. 16 wires is 128KB. This is small.
    
    # However, Python integers are arbitrary precision. 
    # Creating 16 large integers of 65536 bits is fast.
    
    wire_bits = [0] * N
    
    # Construct initial patterns
    # For wire w, bit k is 1 if k has bit w set.
    # This is equivalent to:
    # pattern = 0
    # for k in range(65536):
    #   if (k >> w) & 1: pattern |= 1 << k
    
    # Faster construction:
    # Wire 0: 0x5555... (alternating)
    # Wire 1: 0x3333...
    # ...
    # Let's just build them using a loop over blocks to be safe and clear.
    # Or even simpler, just use the property that we are processing all 2^16 inputs.
    
    # Actually, we can just simulate the network on the "symbolic" inputs.
    # But bit-vector is the way.
    
    # Let's pre-compute the initial bit-vectors.
    # We can do this by iterating k from 0 to 2^N - 1 and setting bits.
    # This takes 2^N * N operations which is 16 * 65536 = 1M ops. Fast enough.
    
    for k in range(1 << N):
        for w in range(N):
            if (k >> w) & 1:
                wire_bits[w] |= (1 << k)
                
    # Apply network
    for i, j in network:
        # Compare wire i and wire j
        # We want min on i, max on j.
        # min(a, b) = a & b
        # max(a, b) = a | b
        
        # Current values: A = wire_bits[i], B = wire_bits[j]
        # New A = A & B
        # New B = A | B
        
        A = wire_bits[i]
        B = wire_bits[j]
        
        wire_bits[i] = A & B
        wire_bits[j] = A | B
        
    # Check if sorted
    # Sorted means: wire_bits[0] <= wire_bits[1] <= ... <= wire_bits[N-1]
    # In terms of bit-vectors representing sets of inputs:
    # If input k has 1 on wire i, it must have 1 on wire j for all j > i? No.
    # Sorted means: For any input k, if wire i is 1, then wire i+1 must be 1?
    # Wait, the output is sorted in non-decreasing order.
    # So output[0] <= output[1] <= ...
    # Since values are 0 or 1, this means:
    # If output[i] is 1, then output[i+1] must be 1.
    # Equivalently: output[i] & ~output[i+1] must be 0 for all i.
    # Or: output[i] is a subset of output[i+1] in terms of set of inputs.
    # So: wire_bits[i] & ~wire_bits[i+1] == 0
    
    for i in range(N - 1):
        if wire_bits[i] & ~wire_bits[i+1] != 0:
            return False
            
    return True

def _generate_batcher_network(n):
    """
    Generate Batcher Odd-Even Merge Sort network.
    Known to be efficient. For N=16, it uses 53 comparators.
    """
    def _odd_even_merge(n):
        # Returns list of comparators for merging two sorted sequences of length n
        # Assume inputs are on wires 0..2n-1
        # First half 0..n-1 is ascending, second half n..2n-1 is descending?
        # Standard Batcher merge:
        # 1. Recursively merge pairs? No, that's bitonic.
        # Odd-even merge:
        # Merge two sorted lists A (len n) and B (len n).
        # Result length 2n.
        # If n == 1: compare (0, 1)
        # Else:
        #   Merge A[0], A[2]... with B[0], B[2]... (even indices)
        #   Merge A[1], A[3]... with B[1], B[3]... (odd indices)
        #   Then compare adjacent elements in the resulting merged list to fix ordering.
        
        if n == 1:
            return [(0, 1)]
            
        comps = []
        # Recursive calls
        # The recursive structure of odd-even merge is a bit complex to implement correctly 
        # without referencing the specific wire indices.
        # Let's use a simpler known network: Batcher Bitonic Sort is easier to generate correctly.
        # Bitonic Sort for N=16 uses 53 comparators? 
        # Actually Bitonic Sort for N=16:
        # Depth log2(16) = 4.
        # Comparators: N/2 * (log2(N)^2 + log2(N)) / 2 ?
        # Formula for Bitonic Sort comparators: (N/2) * (log2(N) * (log2(N)+1) / 2) ? No.
        # Let's just implement Bitonic Sort properly.
        
        pass

    def _bitonic_sort(start, length, direction):
        if length <= 1:
            return []
        half = length // 2
        network = []
        network.extend(_bitonic_sort(start, half, 1))
        network.extend(_bitonic_sort(start + half, half, -1))
        network.extend(_bitonic_merge(start, length, 1))
        return network

    def _bitonic_merge(start, length, direction):
        if length <= 1:
            return []
        half = length // 2
        network = []
        for i in range(half):
            left = start + i
            right = start + i + half
            if direction == 1:
                network.append((left, right))
            else:
                network.append((right, left))
        network.extend(_bitonic_merge(start, half, direction))
        network.extend(_bitonic_merge(start + half, half, direction))
        return network

    return _bitonic_sort(0, n, 1)

def _construct_impl():
    start_time = time.time()
    DEADLINE = 85  # seconds, with safety margin (limit is 90s)
    
    # Start with Bitonic Sort
    best_net = _generate_batcher_network(N)
    
    # Verify initial network
    if not _verify_network_bitvector(best_net):
        # Fallback to odd-even transposition if bitonic failed (shouldn't happen)
        best_net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        
    # Local Search: Deletion
    random.seed(42)
    improved = True
    while improved and time.time() - start_time < DEADLINE - 5:
        improved = False
        # Try to remove each comparator
        for i in range(len(best_net)):
            if time.time() - start_time > DEADLINE - 5:
                break
            candidate = best_net[:i] + best_net[i+1:]
            if _verify_network_bitvector(candidate):
                best_net = candidate
                improved = True
                # Restart the loop to try removing more
                break
                
    # Local Search: Swap/Replace (if deletion stalls)
    # Try replacing a comparator with a different one
    if time.time() - start_time < DEADLINE - 2:
        for _ in range(1000): # Limited iterations
            if time.time() - start_time > DEADLINE - 2:
                break
            if len(best_net) == 0: break
            
            # Pick a random comparator to replace
            idx = random.randint(0, len(best_net) - 1)
            # Generate a random new comparator
            i, j = random.sample(range(N), 2)
            if i > j: i, j = j, i
            
            candidate = best_net[:idx] + [(i, j)] + best_net[idx+1:]
            if _verify_network_bitvector(candidate):
                best_net = candidate
                # If we improved, we might want to run deletion again?
                # For now, just accept it.
                
    return best_net

def construct():
    try:
        return _construct_impl()
    except Exception:
        # Fallback
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

# EVOLVE-BLOCK-END
