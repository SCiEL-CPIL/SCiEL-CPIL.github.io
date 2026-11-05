"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin before 60s kill
    
    def odd_even_transposition():
        comparators = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                comparators.append((i, i + 1))
        return comparators
    
    # Pre-compute bit masks for initialization
    # We represent the state of all 2^N inputs for each wire as an array of uint64s.
    # Number of uint64s needed = 2^N / 64 = 2^18 / 2^6 = 2^12 = 4096.
    NUM_WORDS = (1 << N) // 64
    words_per_wire = NUM_WORDS
    
    def init_wires():
        """
        Initialize the wires such that wire i has bit k set if the k-th input has bit i set.
        Input k is an integer from 0 to 2^N - 1.
        Wire i corresponds to the i-th bit of the input.
        So, wire i has 1s at positions k where (k >> i) & 1 == 1.
        """
        wires = [np.zeros(words_per_wire, dtype=np.uint64) for _ in range(N)]
        
        # For each wire i, we need to set bits in the uint64 words.
        # This is a bit tricky to vectorize perfectly without loops, but N is small.
        # Alternative: Generate all 2^N bits for each wire? No, that's what we're avoiding.
        # Let's construct the patterns.
        # For wire i, the pattern of 0s and 1s repeats every 2^(i+1) bits.
        # Block of 2^i zeros, block of 2^i ones.
        
        for i in range(N):
            w = wires[i]
            block_len = 1 << i
            # We have NUM_WORDS * 64 bits total.
            # We can fill this by repeating the pattern.
            # Pattern: 0...0 (block_len), 1...1 (block_len)
            # Total bits = 2^18.
            # Number of blocks = 2^18 / (2 * block_len) = 2^(18 - i - 1).
            
            # We can create a base pattern for one word (64 bits) if 64 is a multiple of the period?
            # Period is 2^(i+1).
            # If i+1 <= 6, 2^(i+1) divides 64.
            # If i+1 > 6, the pattern spans multiple words.
            
            # Simpler approach: Use numpy to generate the full bitstring for one wire?
            # 2^18 bits = 32KB. Doing this for 18 wires is 576KB.
            # Generating indices might be slower.
            
            # Let's use the property that wire i is the i-th bit of the index.
            # We can generate the indices 0..2^N-1 and extract bits?
            # That was the slow way.
            
            # Let's stick to the bitwise construction.
            # We want an array of uint64s.
            # Bit b (0..63) in word w corresponds to global bit index (w*64 + b).
            # We want this bit to be 1 if ((w*64 + b) >> i) & 1 == 1.
            
            # This is equivalent to checking if the i-th bit of the global index is 1.
            # We can precompute a mask for each word?
            
            # Actually, for small N, we can just do:
            # indices = np.arange(1 << N)
            # bits = (indices >> i) & 1
            # Then pack into uint64s.
            # This takes O(2^N) time per wire, which is 262k ops. 18 wires -> 4.7M ops.
            # This is fast enough in NumPy if done once.
            
            # Let's do it this way for correctness and simplicity, then cache it.
            pass
            
        # Optimization: Do this initialization once and store the initial state.
        # But `verify_network` needs to reset state for each call.
        # So we should pre-compute the initial wire states as a list of arrays.
        pass
        
        # Let's refine init_wires to be fast.
        # We will pre-compute `initial_wires` outside of verify_network.
        return wires

    # Pre-compute initial wires
    initial_wires = [None] * N
    # Generate indices
    # To avoid creating a huge array of int64, we can use uint32 since N=18.
    # But we need to pack into uint64.
    # Let's just use the loop method for initialization, it's only done once.
    total_inputs = 1 << N
    # We can generate the bits for all wires at once?
    # indices = np.arange(total_inputs, dtype=np.uint32)
    # for i in range(N):
    #     bits_i = (indices >> i) & 1
    #     # Pack bits_i into uint64s
    #     # This requires reshaping and viewing or manual packing.
    #     # Manual packing:
    #     # bits_i is 1D array of 0/1 of length 2^N.
    #     # Reshape to (NUM_WORDS, 64)
    #     # Sum with weights 2^0, 2^1, ... 2^63
    #     weights = 1 << np.arange(64, dtype=np.uint64)
    #     packed = (bits_i.reshape(NUM_WORDS, 64) * weights).sum(axis=1)
    #     initial_wires[i] = packed
    
    indices = np.arange(total_inputs, dtype=np.uint32)
    weights = 1 << np.arange(64, dtype=np.uint64)
    for i in range(N):
        bits_i = (indices >> i) & 1
        # bits_i is uint32 array. Convert to uint64 for multiplication.
        bits_i_64 = bits_i.astype(np.uint64)
        packed = (bits_i_64.reshape(NUM_WORDS, 64) * weights).sum(axis=1)
        initial_wires[i] = packed

    def verify_network(comparators):
        """
        Fast verification using bit-parallelism.
        """
        # Copy initial wires
        wires = [w.copy() for w in initial_wires]
        
        for (i, j) in comparators:
            wi = wires[i]
            wj = wires[j]
            # min to i, max to j
            # min(a, b) = a & b
            # max(a, b) = a | b
            wires[i] = wi & wj
            wires[j] = wi | wj
        
        # Check sorted: for all k, wires[k] & ~wires[k+1] == 0
        for k in range(N - 1):
            if np.any(wires[k] & ~wires[k + 1]):
                return False
        return True
    
    # Start with odd-even transposition
    best_network = odd_even_transposition()
    best_count = len(best_network)
    
    # Phase 1: Greedy removal
    current = list(best_network)
    
    changed = True
    while changed and time.time() < deadline - 2.0:
        changed = False
        for idx in range(len(current) - 1, -1, -1):
            if time.time() > deadline - 2.0:
                break
            test = current[:idx] + current[idx + 1:]
            if verify_network(test):
                current = test
                best_network = list(current)
                best_count = len(current)
                changed = True
                break  # Restart since indices shifted
    
    # Phase 2: Local search (limited)
    random.seed(42)
    
    for iteration in range(1000):
        if time.time() > deadline - 1.0:
            break
        if not current:
            break
        
        idx = random.randint(0, len(current) - 1)
        old_i, old_j = current[idx]
        
        # Try a few random replacements
        for _ in range(5):
            if time.time() > deadline - 1.0:
                break
            
            new_i = random.randint(0, N - 1)
            new_j = random.randint(0, N - 1)
            if new_i > new_j:
                new_i, new_j = new_j, new_i
            if new_i == new_j:
                continue
            if (new_i, new_j) == (old_i, old_j):
                continue
            
            test = list(current)
            test[idx] = (new_i, new_j)
            
            if verify_network(test):
                current = test
                best_network = list(current)
                best_count = len(current)
                
                # Quick removal pass
                removed = True
                while removed and time.time() < deadline - 1.0:
                    removed = False
                    for ridx in range(len(current) - 1, -1, -1):
                        if time.time() > deadline - 1.0:
                            break
                        test2 = current[:ridx] + current[ridx + 1:]
                        if verify_network(test2):
                            current = test2
                            best_network = list(current)
                            best_count = len(current)
                            removed = True
                            break
                break
    
    # Final verification
    if not verify_network(best_network):
        best_network = odd_even_transposition()
    
    return best_network
# EVOLVE-BLOCK-END
