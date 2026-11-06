"""Rewritten program: search for a smaller sorting network on 17 wires."""
import numpy as np
from itertools import combinations
import time

N = 17

# Precompute all 2^17 boolean inputs as bitmasks
# We represent a state as a 17-bit integer where bit i is wire i.
# But for efficiency in simulation, we can use a list of 17 ints or a tuple.
# However, 131,072 states * 71 comparators is ~9.3M operations, which is fast in Python if optimized.
# Let's use a list of integers (0 or 1) for each wire, or better, a single integer bitmask.
# Bit i of the mask represents wire i.
# Compare-exchange (i, j):
#   bi = (mask >> i) & 1
#   bj = (mask >> j) & 1
#   if bi > bj: swap them (set bit i to 0, bit j to 1)
#   else: no change (since we want min at i, max at j; if bi<=bj, it's already correct)
#   Wait: min goes to i, max to j.
#   If bi=1, bj=0: min=0, max=1. So i gets 0, j gets 1. Swap.
#   If bi=0, bj=1: min=0, max=1. i gets 0, j gets 1. No swap (already correct).
#   If bi=0, bj=0: no change.
#   If bi=1, bj=1: no change.
#   So, swap if and only if bit i is 1 and bit j is 0.

def apply_comparator(mask, i, j):
    bi = (mask >> i) & 1
    bj = (mask >> j) & 1
    if bi and not bj:
        # Swap: set i to 0, j to 1
        mask ^= (1 << i)
        mask ^= (1 << j)
    return mask

def is_sorted(mask):
    # Check if mask is sorted: all 0s then all 1s.
    # i.e., there is no index k < l such that bit k is 1 and bit l is 0.
    # Equivalently, the mask looks like 00...011...1.
    # We can check if (mask & (mask + 1)) == 0? No.
    # A sorted mask of length N has the form: lower k bits are 0, upper N-k bits are 1.
    # So mask must be of the form ((1 << (N-k)) - 1) << k for some k.
    # Or simply: check if for all i<j, bit i <= bit j.
    # Faster: count trailing zeros, then check if the rest are all 1s.
    # Or: mask must be a "suffix" of 1s.
    # Let's just check: 
    # sorted_masks = [0] + [((1 << (N - k)) - 1) << k for k in range(1, N+1)]
    # But checking membership in a set is O(1).
    # Precompute the set of sorted masks.
    sorted_masks = set()
    for k in range(N + 1):
        mask = (1 << (N - k)) - 1 if k < N else 0
        sorted_masks.add(mask << k)

def construct():
    # We will perform a local search / hill climbing to minimize the number of comparators.
    # Start with a known valid network (e.g., odd-even transposition or AKS-like or just a simple one).
    # The seed odd-even transposition has 136 comparators.
    # Let's start with that, then try to remove comparators or replace them.
    
    # Precompute sorted masks for quick check
    sorted_masks = set()
    for k in range(N + 1):
        mask = (1 << (N - k)) - 1 if k < N else 0
        sorted_masks.add(mask << k)
        # k zeros, N-k ones
        mask = ((1 << (N - k)) - 1) << k
        sorted_masks.add(mask)
    
    # Generate all 2^N input masks
    all_masks = list(range(1 << N))
    
    # Initial network: Odd-even transposition
    # For N wires, N phases. Phase p: compare (p%2, p%2+2, ...), (p%2+1, p%2+3, ...) etc?
    # Standard odd-even transposition:
    # Phase 0: (0,1), (2,3), (4,5), ...
    # Phase 1: (1,2), (3,4), (5,6), ...
    # ...
    # Phase N-1: depends on parity.
    # Actually, for N=17, we need 17 phases?
    # Let's just use a simple bubble sort network:
    # For i in range(N):
    #   for j in range(N-1-i):
    #     compare(j, j+1)
    # This is O(N^2) comparators, which is 17*16/2 = 136? No, bubble sort network is N*(N-1)/2 comparators?
    # No, bubble sort network has N phases, each with decreasing number of comparators.
    # Total comparators = N*(N-1)/2 = 17*16/2 = 136.
    
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
    
    # Verify initial network
    def verify(network):
        for mask in all_masks:
            m = mask
            for (i, j) in network:
                m = apply_comparator(m, i, j)
            if m not in sorted_masks:
                return False
        return True
    
    # We want to minimize len(network).
    # Strategy: Try to remove each comparator and see if the network is still valid.
    # If it is, remove it. Repeat until no more can be removed.
    # Then, try to replace a pair of comparators with a single one? Or move them?
    # Let's start with removal.
    
    improved = True
    while improved:
        improved = False
        new_network = list(network)
        for idx in range(len(new_network) - 1, -1, -1):
            # Try removing new_network[idx]
            candidate = new_network[:idx] + new_network[idx+1:]
            # Verify candidate
            valid = True
            for mask in all_masks:
                m = mask
                for (i, j) in candidate:
                    m = apply_comparator(m, i, j)
                if m not in sorted_masks:
                    valid = False
                    break
            if valid:
                network = candidate
                improved = True
                # Restart loop because indices changed
                break
    
    # Now try to optimize further: 
    # Maybe we can replace two comparators with one? Unlikely to be simple.
    # Or maybe we can start from a different seed?
    # The task says "best known network has 71 comparators". 
    # My current approach starts from 136 and removes. It might get stuck in a local minimum.
    # Let's add a randomization step: shuffle the network, then remove, then maybe shuffle again?
    # Or use simulated annealing?
    
    # Given the 60s time limit, and 131k masks, each verification takes ~131k * len(network) ops.
    # If len(network) is ~100, that's 13M ops. In Python, that might take ~1-2 seconds.
    # So we can do maybe 10-30 full verifications per second.
    # The removal loop above does one verification per comparator removed.
    # If we remove 100 comparators, that's 100 verifications = 100-200 seconds. Too slow!
    
    # We need a faster verification or a smarter search.
    # Alternative: Use the zero-one principle but check only a subset? No, must be correct.
    # Alternative: Optimize the verification loop in Python.
    
    # Let's rewrite verify to be faster.
    # Instead of iterating all masks, we can precompute the effect of the network on all masks?
    # No, the network changes.
    
    # Let's try a different approach: 
    # Start with a known good network of ~71 comparators? 
    # I don't have one hardcoded.
    
    # Let's try to speed up the removal process.
    # We can check if a comparator is redundant by checking if its effect is already achieved by the rest of the network.
    # But that's hard.
    
    # Let's just do a simple hill climbing with random restarts and removal.
    # To fit in 60s, we must be efficient.
    
    # Let's try to generate a random network and then remove?
    # Random networks are rarely valid.
    
    # Let's stick to the seed and remove, but optimize the verification.
    # We can use NumPy to vectorize the verification?
    # 131,072 masks. We can represent them as a 131,072 x 17 binary matrix.
    # For each comparator (i, j), we can update the matrix in vectorized fashion.
    
    # Let's implement a vectorized verifier.
    
    def verify_vectorized(network):
        # Create a matrix of all masks
        # masks = np.arange(1 << N)
        # bits = (masks[:, None] >> np.arange(N)) & 1
        # This is 131072 x 17.
        # For each comparator (i, j):
        #   bi = bits[:, i]
        #   bj = bits[:, j]
        #   swap_mask = (bi == 1) & (bj == 0)
        #   bits[swap_mask, i] = 0
        #   bits[swap_mask, j] = 1
        # At the end, check if each row is sorted.
        # A row is sorted if it is of the form 0...01...1.
        # We can check if for all i<j, bits[:, i] <= bits[:, j].
        # Or check if the row matches one of the sorted patterns.
        
        masks = np.arange(1 << N, dtype=np.uint32)
        bits = (masks[:, None] >> np.arange(N, dtype=np.uint8)) & 1
        
        for (i, j) in network:
            bi = bits[:, i]
            bj = bits[:, j]
            swap_mask = (bi == 1) & (bj == 0)
            if np.any(swap_mask):
                bits[swap_mask, i] = 0
                bits[swap_mask, j] = 1
        
        # Check if sorted
        # For each row, check if it is sorted.
        # A row is sorted if bits[:, k] is non-decreasing in k.
        # We can check if for all k, bits[:, k] >= bits[:, k+1]? No, 0 then 1.
        # So bits[:, k] <= bits[:, k+1] is NOT correct. 0,1 is sorted. 1,0 is not.
        # So we need bits[:, k] <= bits[:, k+1] to be FALSE for 1,0.
        # Actually, sorted means 0s then 1s.
        # So if we see a 1 followed by a 0, it's not sorted.
        # Check: for any k, if bits[:, k] == 1 and bits[:, k+1] == 0, then not sorted.
        
        # Let's check: 
        # diffs = bits[:, 1:] - bits[:, :-1]
        # If any diff is -1, then it's not sorted.
        # Because 1->0 is -1. 0->1 is 1. 0->0 is 0. 1->1 is 0.
        
        diffs = bits[:, 1:].astype(np.int8) - bits[:, :-1].astype(np.int8)
        # If any row has a -1 in diffs, it's not sorted.
        return not np.any(diffs == -1)
    
    # Now, let's try the removal again with vectorized verification.
    # Each verification is O(N * 2^N) operations in NumPy, which is fast.
    # 17 * 131072 ~ 2.2M ops per comparator? No, per network.
    # The loop over comparators is in Python, but the inner ops are vectorized.
    # If network has 100 comparators, 100 vectorized ops.
    # This should be fast enough.
    
    # Let's restart the removal process with the vectorized verifier.
    
    # Re-initialize network
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
    
    improved = True
    while improved:
        improved = False
        new_network = list(network)
        for idx in range(len(new_network) - 1, -1, -1):
            candidate = new_network[:idx] + new_network[idx+1:]
            if verify_vectorized(candidate):
                network = candidate
                improved = True
                break
    
    return network

# We need to ensure the function is named construct and returns a list of tuples.
# The code above does that.

# Let's add a fallback if the vectorized verification is too slow or fails.
# But it should be fine.

# One more optimization: 
# We can try to swap the order of comparators to enable further removals?
# For now, let's just do removal.

# To potentially beat 136 significantly, we might need a better starting point or more aggressive search.
# But given the constraints, this is a solid improvement over the seed if it removes any comparators.
# The seed is 136. If we remove even 1, we beat it.

# Let's also try a random restart: 
# Shuffle the network, verify, if valid, try to remove.
# But shuffling a valid network usually makes it invalid.
# So, let's just do the removal from the seed.

# Final check: The task says "Keep the function names and signatures of the seed program."
# Seed program has `construct()` returning a list of integer pairs.
# My code does that.

# Let's wrap it in the EVOLVE-BLOCK as required by the prompt structure? 
# The prompt says "Rewrite the program from scratch... Output the COMPLETE program in a single python code block."
# And "Keep the function names and signatures... EXACTLY as in the current program above".
# The current program has `N = 17` and `def construct():`.
# I will include `N = 17` and `def construct():`.

# I will remove the EVOLVE-BLOCK markers as they are for the internal system, and just provide the code.

# Wait, the prompt says "Output the COMPLETE program in a single
