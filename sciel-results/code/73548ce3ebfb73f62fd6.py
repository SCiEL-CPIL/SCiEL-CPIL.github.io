import time
import random
import numpy as np

N = 18

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit

    # 1. Generate an initial valid network (Odd-Even Transposition)
    # This is O(N^2) comparators, which is 17 * 18 = 306 for N=18.
    # Wait, odd-even transposition is N phases, each with N/2 comparators.
    # Total N^2 / 2 = 162 for N=18.
    # Let's just use a simple bubble sort network which is also valid.
    # Or better, use a known good network if possible, but we can't hardcode 77 easily.
    # Let's start with the odd-even transposition network.
    def gen_oe_network():
        net = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                net.append((i, i + 1))
        return net

    net = gen_oe_network()
    
    # Helper to check if network is valid using bitmask simulation
    # This is the fastest way to verify sorting networks.
    def is_valid(net):
        # 2^18 = 262144 inputs.
        # We can represent the state of the network as a set of bitmasks?
        # No, standard verification:
        # For each input mask (0 to 2^N - 1):
        #   Apply comparators
        #   Check if output is sorted (non-decreasing)
        
        # Optimization: We can simulate all inputs in parallel using numpy?
        # Or just use Python sets/bit operations.
        # 262,144 inputs * ~100 comparators * 18 wires.
        # In Python, this might be slow if done naively.
        
        # Better verification:
        # A sorting network is valid iff it sorts all 2^N boolean inputs.
        # We can track the "signature" of the network.
        # Actually, let's just do the brute force check but optimized.
        
        # We can represent the wires as a tuple of bits.
        # But doing this for 262k inputs in a loop in Python is ~0.1s per check if optimized.
        # Let's try to be efficient.
        
        # Alternative: Use the "zero-principle" or just simulate.
        # Let's use a fast simulator.
        
        # We'll represent the input as an integer bitmask.
        # State: list of N integers (0 or 1).
        
        # To speed up, we can check if the network is "reduced" or just valid.
        # For hill climbing, we need to check validity after each removal.
        # 262,144 inputs.
        
        # Let's precompute the comparator effect? No, it depends on the state.
        
        # Let's try a numpy-based vectorized check if possible, or just a tight loop.
        # A tight loop in Python for 262k * 100 steps is ~26 million operations.
        # That might take ~1-2 seconds in pure Python.
        # If we have 85 seconds, we can do ~40-80 checks.
        # That's enough to remove a few comparators.
        
        # Let's implement a fast check.
        
        # Optimization: If the network is valid, it maps every input to a sorted output.
        # The set of possible outputs is small (only sorted bitmasks).
        # Sorted bitmasks are of the form: 0...01...1.
        # There are N+1 such masks.
        
        # We can check if the network outputs are in the set of sorted masks.
        
        sorted_masks = set()
        for k in range(N + 1):
            mask = (1 << k) - 1  # k ones at the bottom? No, min to wire i, max to j.
            # Wait, the problem says: "min to wire i and max to j".
            # Usually sorting networks sort in non-decreasing order.
            # So wire 0 has the smallest, wire N-1 has the largest.
            # For boolean inputs, the sorted outputs are:
            # 00...0 (all 0)
            # 10...0 (one 1 at the end? No, min to i, max to j.
            # If i < j, and we put min in i, max in j.
            # If the array is sorted non-decreasing, wire 0 is min, wire N-1 is max.
            # So for boolean, the sorted arrays are:
            # 0 0 ... 0
            # 0 0 ... 1
            # ...
            # 1 1 ... 1
            # Wait, if min goes to i and max to j, and i < j, then the value at i is <= value at j.
            # This implies the network sorts in non-decreasing order from left to right?
            # Let's check the odd-even transposition network.
            # Standard odd-even transposition sorts in non-decreasing order.
            # So wire 0 is the smallest, wire N-1 is the largest.
            # For boolean inputs, the sorted outputs are bitmasks where all 0s are on the left and all 1s are on the right.
            # i.e., 0...01...1.
            
            mask = 0
            for j in range(N):
                if j >= N - k: # Last k bits are 1
                    mask |= (1 << j)
            sorted_masks.add(mask)
            
        # Now, for each input mask, simulate and check if output is in sorted_masks.
        
        # To make this fast, we can use a list of wires.
        # But 262k inputs * 100 comparators is heavy.
        
        # Let's try to optimize the simulation.
        # We can process inputs in batches?
        
        # Actually, there is a known trick: 
        # A network is a sorting network iff it sorts all 2^N inputs.
        # We can just do it.
        
        # Let's write a fast simulator.
        
        # We'll use a list of integers for the wires.
        # For each input:
        #   wires = [(input >> i) & 1 for i in range(N)]
        #   for (i, j) in net:
        #       if wires[i] > wires[j]:
        #           wires[i], wires[j] = wires[j], wires[i]
        #   out_mask = 0
        #   for i in range(N):
        #       out_mask |= (wires[i] << i)
        #   if out_mask not in sorted_masks: return False
        # return True
        
        # This is ~262k * 100 * 18 = 470 million operations in Python. Too slow.
        
        # We need a faster verification.
        
        # Idea: Use numpy to simulate all inputs at once?
        # Inputs: shape (2^N, N)
        # Comparators: apply to all rows.
        # This is memory intensive: 262144 * 18 * 8 bytes = ~37 MB. Acceptable.
        # Time: 262144 * 100 comparators * vectorized ops.
        # Vectorized numpy ops are fast.
        
        # Let's implement numpy verification.
        
        # Generate all inputs:
        # np.arange(2**N).reshape(-1, 1) >> np.arange(N) & 1
        # This creates a (2**N, N) array of 0s and 1s.
        
        # For each comparator (i, j):
        #   a = inputs[:, i]
        #   b = inputs[:, j]
        #   # min to i, max to j
        #   inputs[:, i] = np.minimum(a, b)
        #   inputs[:, j] = np.maximum(a, b)
        
        # At the end, check if each row is sorted.
        # A row is sorted if it is non-decreasing.
        # diff = np.diff(inputs, axis=1)
        # if np.any(diff < 0): return False
        
        # This should be fast.
        
        # Let's test this approach.
        
        # However, generating the input array takes time.
        # We can cache it?
        # Yes, the input array is constant.
        
        # Let's create a global cached input array.
        
        return True # Placeholder, will implement below

    # Let's implement the numpy verification properly.
    
    # Global cache for inputs
    _input_cache = None
    def get_inputs():
        global _input_cache
        if _input_cache is None:
            # 2^18 inputs, 18 wires
            # Use uint8 to save memory
            n_inputs = 1 << N
            # Create array of shape (n_inputs, N)
            # We can use broadcasting
            # np.arange(n_inputs)[:, None] >> np.arange(N)[None, :] & 1
            # This might be slow to generate once, but we only do it once.
            _input_cache = (np.arange(1 << N, dtype=np.uint32)[:, None] >> np.arange(N, dtype=np.uint32)[None, :]) & 1
            _input_cache = _input_cache.astype(np.uint8)
        return _input_cache

    def is_valid_fast(net):
        inputs = get_inputs().copy() # Copy because we modify in place
        for i, j in net:
            a = inputs[:, i]
            b = inputs[:, j]
            # min to i, max to j
            inputs[:, i] = np.minimum(a, b)
            inputs[:, j] = np.maximum(a, b)
        
        # Check if sorted: non-decreasing
        # diff = np.diff(inputs, axis=1)
        # if np.any(diff < 0): return False
        # Faster: check if inputs[:, :-1] <= inputs[:, 1:]
        if np.any(inputs[:, :-1] > inputs[:, 1:]):
            return False
        return True

    # Initial check
    if not is_valid_fast(net):
        return net # Should not happen

    # Hill Climbing
    # Try to remove comparators.
    # We can try removing each comparator and check validity.
    # To speed up, we can randomize the order or try all.
    # Since we have ~85s, and each check takes ~0.1-0.5s (numpy), we can do ~100-500 checks.
    
    # Strategy:
    # Loop while time < deadline:
    #   Try to remove a random comparator.
    #   If valid, keep it removed.
    #   Else, put it back.
    
    # To be more effective, we can try removing comparators that are "redundant".
    # But random removal is a good start.
    
    # Let's use a more systematic approach:
    # Iterate through the network, try to remove each comparator.
    # If removal is valid, remove it.
    # This is a greedy reduction.
    
    # We can repeat this process multiple times if time permits.
    
    # Let's implement greedy reduction.
    
    def reduce_network(net, deadline):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for idx in range(len(net) - 1, -1, -1): # Try from end to start
                # Try removing net[idx]
                if time.time() > deadline:
                    break
                test_net = net[:idx] + net[idx+1:]
                if is_valid_fast(test_net):
                    net = test_net
                    changed = True
                    # Restart the loop or continue?
                    # If we remove one, the indices change.
                    # It's safer to restart the loop or just continue with the new list.
                    # Since we are iterating backwards, if we remove idx, the next idx-1 is still valid.
                    # But the list is shorter.
                    # Let's just break and restart the outer while loop to be safe and simple.
                    break
        return net

    # Run greedy reduction
    net = reduce_network(net, deadline)
    
    # If we have time left, we can try random swaps or other moves.
    # But for now, greedy reduction is a solid strategy.
    
    # One more pass?
    if time.time() < deadline - 5:
        net = reduce_network(net, deadline)
        
    return net
# EVOLVE-BLOCK-END
