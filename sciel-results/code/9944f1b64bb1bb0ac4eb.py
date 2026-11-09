import time
import numpy as np
import random

N = 18
FULL_SIZE = 1 << N

# Pre-generate full inputs as bit-parallel arrays
# inputs[k] is an integer where bit i is 1 if input i has 1 in position k
# We need to simulate the network on all 2^N inputs simultaneously.
# Let's represent the state of the network for all inputs as a list of N integers.
# state[i] is an integer of 2^N bits. Bit k of state[i] is the value on wire i for input k.

def _init_state():
    state = []
    for i in range(N):
        # Input k is the integer k. Bit (N-1-i) of k is the value for wire i.
        # We need to construct an integer M_i such that bit k of M_i is ((k >> (N-1-i)) & 1).
        # This is equivalent to a pattern of 0s and 1s.
        # Pattern for bit b: 0...0 1...1 0...0 1...1 ...
        # Period is 2^(b+1). Half 0s, half 1s.
        # Let's build it.
        val = 0
        period = 1 << (i + 1)  # Since bit index is N-1-i, let's be careful.
        # Wire i corresponds to bit position (N-1-i) in the input integer k.
        # So for wire i, the bit is 1 when k has bit (N-1-i) set.
        # Let's just loop through k? No, that's slow.
        # Use bitwise tricks.
        # For bit pos p, the mask is sum(1 << k for k in range(2^N) if (k >> p) & 1).
        # This is a known pattern.
        p = N - 1 - i
        # Generate mask
        mask = 0
        # We can build this recursively or with shifts.
        # For p=0: 0101...
        # For p=1: 0011...
        # For p=2: 00001111...
        
        # Efficient construction:
        # block_size = 1 << p
        # pattern = (1 << block_size) - 1
        # repeat pattern for 2^(N-p) times, alternating? No.
        # The bit is 1 for half the block, 0 for the other half?
        # Actually, for bit p, the value is 1 for k in [2^p, 2^(p+1)-1], [3*2^p, 4*2^p-1], etc.
        # So it's 0 for first 2^p, 1 for next 2^p, 0 for next, 1 for next...
        
        block = 1 << p
        # Create a block of 1s of length block
        ones_block = (1 << block) - 1
        # We need to place this block at positions block, 3*block, 5*block...
        # Total number of blocks is 2^(N-p-1) pairs of (0,1) blocks?
        # Total length 2^N. Number of 2^(p+1) periods is 2^(N-p-1).
        # In each period, first half 0, second half 1.
        
        # Let's construct the integer by shifting and ORing.
        # This might be slow if done naively for N=18 (262k bits).
        # But we only do it once.
        
        # Alternative: Use Python's big int arithmetic.
        # Let's just use a loop for the initial generation, it's only 18 * 2^18/2 operations?
        # No, 18 integers of 2^18 bits.
        # Let's use a faster method:
        # mask = 0
        # for k in range(FULL_SIZE):
        #     if (k >> p) & 1:
        #         mask |= (1 << k)
        # This is 18 * 2^18 / 2 ~ 2M operations. Acceptable.
        
        # Even faster:
        # Use a recursive pattern.
        # def gen_mask(p, n):
        #    if p == n: return 0
        #    half = 1 << (n - p - 1)
        #    block0 = gen_mask(p, n-1) # No, this is getting complicated.
        
        # Let's stick to the simple loop for initialization, it's fast enough in Python if optimized?
        # Actually, numpy might be faster for generating the bits, but we need big ints.
        
        # Let's try a bitwise construction.
        # For bit p:
        # The sequence of bits is 0...0 1...1 0...0 1...1
        # Length of 0s and 1s is 2^p.
        # We can form this by:
        # base = (1 << (1 << p)) - 1  # A block of 1s of length 2^p
        # We need to place this base at offsets 2^(p+1)*k + 2^p for k=0...
        
        # Let's just use the loop. It runs once.
        m = 0
        # Optimization: Iterate over blocks of 1s
        block_len = 1 << p
        # Number of 1-blocks is 2^(N-p-1)
        # The k-th 1-block starts at index (2*k+1) * block_len
        for k in range(1 << (N - p - 1)):
            start = (2 * k + 1) * block_len
            m |= ((1 << block_len) - 1) << start
        state.append(m)
    return state

INITIAL_STATE = _init_state()

def _verify_bitparallel(network):
    state = INITIAL_STATE[:]
    for i, j in network:
        if i > j:
            i, j = j, i
        a = state[i]
        b = state[j]
        # min(a,b) = a & b
        # max(a,b) = a | b
        state[i] = a & b
        state[j] = a | b
    
    # Check if sorted:
    # For each adjacent pair (k, k+1), we need state[k] <= state[k+1] bit-wise?
    # No, "sorted" means for every input k, the sequence of bits across wires is 0...01...1.
    # So for each wire i, state[i] should be a subset of state[i+1]?
    # i.e., state[i] & ~state[i+1] should be 0.
    # This means if wire i is 1, wire i+1 must be 1.
    # This is equivalent to state[i] <= state[i+1] in integer comparison?
    # Not exactly integer comparison, but bitwise subset.
    # If state[i] has a 1 where state[i+1] has a 0, it's not sorted.
    # So we check: for all i, (state[i] & ~state[i+1]) == 0.
    
    for i in range(N - 1):
        if (state[i] & ~state[i+1]) != 0:
            return False
    return True

def _bitonic_merge(left, right):
    # left and right are lists of comparators
    # Merges two sorted sequences (one ascending, one descending) into one ascending sequence.
    # Standard bitonic merge.
    # n = len(left) + len(right)
    # This implementation assumes the input to the merge is a bitonic sequence.
    # The function returns a list of comparators for the merge.
    # We need to know the wire indices.
    # Let's assume the bitonic sequence is on wires [start, end).
    # The merge operates on a block of size 2^k.
    # This helper is tricky to generalize without knowing the wire mapping.
    # Let's implement a specific Bitonic Sort for N=18.
    
    # A standard recursive bitonic sort:
    # def bitonic_sort(n):
    #   if n <= 1: return []
    #   left = bitonic_sort(n//2)
    #   right = bitonic_sort(n//2)
    #   # Interleave? No, bitonic sort works on blocks.
    #   # This is complex. Let's use a known small network or a simpler heuristic seed.
    #   # Actually, for N=18, we can use a "Bitonic" construction.
    
    # Let's use a simpler, robust seed:
    # A "Knuth" style or just a slightly optimized Odd-Even Transposition?
    # Or better: A known good network for 18?
    # Best known is 77.
    # Let's try to construct a Bitonic Sort network properly.
    
    # Bitonic Sort on N wires:
    # 1. Split into two halves.
    # 2. Sort left half ascending, right half descending.
    # 3. Merge them.
    
    # We need a helper to generate comparators for a block of wires [s, e).
    
    def _bitonic_merge_block(s, e):
        # Merges a bitonic sequence on wires [s, e) into an ascending sequence.
        # The bitonic sequence consists of an ascending part and a descending part.
        # The split point is the middle.
        # The comparators are:
        # For d in range(log2(e-s), 0, -1):
        #   for i in range(s, e - 2^d, 2^d):
        #     for j in range(2^d - 1, -1, -1):
        #       compare (i + j, i + 2^d + j)
        
        length = e - s
        # length must be a power of 2 for standard bitonic merge?
        # If not, we can pad or use a different merge.
        # For N=18, it's not a power of 2.
        # We can decompose 18 into 8 + 8 + 2? Or 16 + 2?
        # Let's use a recursive approach that handles non-powers of 2 by splitting as evenly as possible.
        
        if length <= 1:
            return []
        
        # Split into two halves
        mid = s + length // 2
        # The first half is ascending, the second is descending?
        # No, the input to merge is bitonic.
        # Standard bitonic sort:
        # Sort left half ascending, right half descending.
        # Then merge.
        
        # This function `_bitonic_merge_block` assumes the input is ALREADY bitonic.
        # We need a function `_bitonic_sort_block(s, e)` that sorts [s, e).
        
        pass # Placeholder, will implement below

    return [] # Dummy

def _bitonic_sort_block(s, e):
    length = e - s
    if length <= 1:
        return []
    
    # Split
    mid = s + length // 2
    
    # Sort left half ascending
    left_comps = _bitonic_sort_block(s, mid)
    
    # Sort right half descending
    # To sort descending, we can sort ascending and then reverse the wire order?
    # Or just swap the compare-exchange directions?
    # compare-exchange (i, j) puts min at i, max at j.
    # To sort descending, we want max at i, min at j.
    # This is equivalent to swapping i and j in the comparator? No.
    # If we use compare-exchange (j, i) where j < i, it puts min at j, max at i.
    # So if we generate a sorting network for ascending and then map wire k to wire (e-1-k), we get descending?
    # Let's just generate ascending for the right block and then "reverse" the logic?
    # Actually, a simpler way:
    # Generate ascending sort for [mid, e).
    # Then, to make it descending, we can just swap the outputs of every comparator?
    # No, the comparators are fixed.
    # If we have a network that sorts [mid, e) ascending, and we want it descending,
    # we can just use the same network but interpret the result?
    # No, the network is a list of (i, j).
    # If we run the ascending network on [mid, e), it sorts ascending.
    # We want descending.
    # We can achieve this by swapping the indices in the comparators?
    # If we have comp (u, v) with u < v, it sorts ascending.
    # If we use (v, u) with v > u? The interface requires i < j.
    # So we can't just swap.
    # However, if we permute the wires:
    # Let P be the permutation that reverses the order of wires in [mid, e).
    # If we apply P, sort ascending, then apply P^-1, we get descending?
    # No, that just sorts the permuted values.
    
    # Correct approach for Bitonic Sort:
    # 1. Sort left half [s, mid) ascending.
    # 2. Sort right half [mid, e) descending.
    #    How to get descending?
    #    Generate ascending sort for [mid, e).
    #    Then, for each comparator (u, v) in that list, replace it with (v, u)?
    #    But we must maintain u < v for the interface?
    #    The interface says: "ordered list of integer pairs (i, j), 0 <= i < j < N".
    #    And "writes min to wire i and max to j".
    #    So (i, j) always sorts ascending (min at i, max at j).
    #    To get a descending sort on [mid, e), we can:
    #    - Generate a standard ascending sorting network for [mid, e).
    #    - Then, reverse the order of the wires in the output?
    #    - No, we need the network to produce descending order on the wires [mid, e).
    #    - Trick: If we have a network that sorts ascending, and we swap the roles of min and max?
    #    - We can't change the comparator definition.
    #    - But we can permute the wires.
    #    - Let's map wire k in [mid, e) to wire (e - 1 - (k - mid)) = (e + mid - 1 - k).
    #    - This is a reversal.
    #    - If we apply this reversal, sort ascending, and apply reversal again, do we get descending?
    #    - Let x be the input. Let R be reversal.
    #    - Sort(R(x)) is ascending.
    #    - R(Sort(R(x))) is descending?
    #    - Yes.
    #    - So, to get a descending sort network on [mid, e):
    #      1. Generate ascending sort network for [mid, e).
    #      2. For each comparator (u, v) in the network, map u -> u' and v -> v' using the reversal.
    #      3. Ensure u' < v' for the interface.
    #      4. The resulting network sorts descending.
    
    right_comps_asc = _bitonic_sort_block(mid, e)
    right_comps_desc = []
    for u, v in right_comps_asc:
        # Map to reversed indices
        # Original range [mid, e).
        # Reversed index for u: e - 1 - (u - mid) = e + mid - 1 - u
        # Reversed index for v: e + mid - 1 - v
        u_rev = e + mid - 1 - u
        v_rev = e + mid - 1 - v
        # Ensure u_rev < v_rev
        if u_rev < v_rev:
            right_comps_desc.append((u_rev, v_rev))
        else:
            right_comps_desc.append((v_rev, u_rev))
    
    # Now we have left ascending, right descending.
    # Merge them.
    # The merge step for bitonic sort:
    # The combined sequence [s, e) is bitonic.
    # We need to merge it into ascending.
    # The standard bitonic merge uses comparators (s + j, s + length/2 + j) for j in ...
    # But this requires length to be a power of 2?
    # If length is not a power of 2, the standard bitonic merge doesn't work directly.
    # However, we can use a "generalized" bitonic merge or just use a simple merge?
    # For N=18, we can split 18 into 8 and 10? Or 9 and 9?
    # If we split 18 into 9 and 9, we need a merge for size 9?
    # Bitonic merge works for powers of 2.
    # We can pad to 32? No, that increases size.
    
    # Alternative: Use a known efficient network for 18?
    # Or use a simpler seed:
    # Let's use a "Batcher" network if possible, or just a good heuristic.
    
    # Given the complexity, let's use a simpler, robust seed:
    # A "Knuth" network or just a slightly perturbed Odd-Even Transposition?
    # Actually, a "Bitonic" network for N=16 is 120 comps.
    # For N=18, we can take a Bitonic 16 network and add 2 wires?
    # Or just use a generic "Merge Sort" network?
    
    # Let's try a different seed:
    # "Odd-Even Merge Sort" network.
    # Or just stick with a good heuristic:
    # Start with Odd-Even Transposition (153).
    # It's safe and simple.
    
    return left_comps + right_comps_desc + _bitonic_merge_block(s, e)

def _bitonic_merge_block(s, e):
    length = e - s
    if length <= 1:
        return []
    
    # This implementation assumes length is a power of 2.
    # If not, we need a different merge.
    # For now, if length is not power of 2, we fallback to a simple merge?
    # Or we can just use the fact that for N=18, we can use a specific construction.
    
    # Let's assume we use a power-of-2 based construction by padding?
    # No, let's just use a simpler seed to ensure correctness and speed.
    
    # Seed: Odd-Even Transposition
    # It's 153 comparators.
    # It's guaranteed to sort.
    # We will optimize from there.
    
    return [] # Dummy, we won't use bitonic sort if it's too complex to implement correctly for non-power-of-2 quickly.

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85  # 5s margin
    
    # Seed network: Odd-Even Transposition
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    best_network = network[:]
    best_size = len(network)
    
    # Verify seed
    if not _verify_bitparallel(network):
        # Should not happen
        return best_network
        
    current_network = network[:]
    
    # Local Search
    # 1. Greedy Removal
    # 2. Swap (Remove one, Add one)
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try Removals
        # Try removing each comparator in random order
        indices = list(range(len(current_network)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline: break
            
            new_network = current_network[:idx] + current_network[idx+1:]
            
            if _verify_bitparallel(new_network):
                current_network = new_network
                best_network = current_network[:]
                best_size = len(current_network)
                improved = True
                break # Restart search from new state
        
        if time.time() > deadline: break
        
        # 2. If no removal worked, try Swaps
        if not improved:
            # Try swapping a comparator with a random new one
            # Pick a random existing comparator to remove
            idx_remove = random.randint(0, len(current_network) - 1)
            
            # Pick a random new comparator to add
            # Avoid duplicates
            # To speed up, we can just pick random i, j and check if in set
            comp_set = set(current_network)
            new_comp = None
            attempts = 0
            while new_comp is None and attempts < 100:
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                if (i, j) not in comp_set:
                    new_comp = (i, j)
                attempts += 1
            
            if new_comp is None:
                # No new comparator found, break
                break
            
            new_network = current_network[:idx_remove] + current_network[idx_remove+1:] + [new_comp]
            
            if _verify_bitparallel(new_network):
                current_network = new_network
                best_network = current_network[:]
                best_size = len(current_network)
                improved = True
        
        if not improved:
            # Stuck. Try a random mutation to escape local minimum.
            # Replace a random comparator with a random other one.
            # Or just break if we've been stuck for a while?
            # Let's do one random mutation and continue.
            
            idx_mut = random.randint(0, len(current_network) - 1)
            comp_set = set(current_network)
            new_comp = None
            attempts = 0
            while new_comp is None and attempts < 100:
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                if (i, j) not in comp_set:
                    new_comp = (i, j)
                attempts += 1
            
            if new_comp is None:
                break
                
            # Mutate
            mutated_network = current_network[:]
            mutated_network[idx_mut] = new_comp
            
            # Check if mutation is valid (it might not be, but we'll try to repair)
            # Actually, we just set it as current and try to remove comps?
            # If it's invalid, we can't use it.
            # So we only accept mutation if it sorts?
            # That's a hill climb.
            
            if _verify_bitparallel(mutated_network):
                current_network = mutated_network
                # Don't update best_network unless it's smaller?
                # It's the same size.
                # But it might lead to a smaller network later.
            else:
                # If mutation fails, we are stuck.
                # Try a different mutation?
                # Or break.
                break
                
    return best_network
# EVOLVE-BLOCK-END
