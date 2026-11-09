"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a sorting network for N=18 wires.
    Strategy:
    1. Start with a known good baseline (e.g., Odd-Even Transposition or a pre-computed optimal-ish network).
    2. Use a greedy local search / mutation strategy to reduce the number of comparators.
    3. Verify correctness using the 0-1 principle (checking all 2^18 Boolean inputs).
    4. Implement a deadline guard to ensure we return within the time limit.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5 seconds safety margin before 90s kill

    # --- Helper Functions ---
    
    def apply_comparator(state, i, j):
        """Apply a compare-exchange on wires i and j of the state."""
        if state[i] > state[j]:
            state[i], state[j] = state[j], state[i]
        return state

    def verify_network(comparators):
        """
        Verify if the network sorts all 2^N Boolean inputs.
        Returns True if valid, False otherwise.
        """
        # Optimization: We can check all 2^N inputs. For N=18, this is 262,144.
        # Each check takes O(C * N) roughly, but we can do it efficiently.
        # Actually, for each input vector, we simulate the network.
        # 262,144 * (number of comparators) operations.
        # If we have ~100 comparators, that's ~26M ops. Python is slow for this.
        
        # Better approach: Use the 0-1 principle but vectorize or optimize the inner loop.
        # Even better: Check if the network is a sorting network by checking all 0-1 sequences.
        
        # Let's try to optimize the verification.
        # We can pre-compute the effect of the network on all inputs? No, too much memory.
        
        # Standard approach in Python for N=18:
        # Loop through all 2^18 masks.
        # Represent state as a list of 0s and 1s.
        # This is likely too slow in pure Python if done naively for many iterations.
        
        # Alternative: Use a faster verification method.
        # Since we are evolving, we need a fast verifier.
        # Let's use the fact that a sorting network is correct iff it sorts all 0-1 inputs.
        # We can represent the 0-1 inputs as integers 0..2^N-1.
        
        # Optimization: 
        # For each input x in range(2**N):
        #   state = [ (x >> i) & 1 for i in range(N) ]
        #   for i, j in comparators:
        #       if state[i] > state[j]: swap
        #   if state != sorted(state): return False
        #   # Actually, since it's 0-1, sorted is [0]*k + [1]*(N-k). 
        #   # So we just need to check if the number of 1s is preserved (it is) and they are at the end?
        #   # No, the output must be exactly [0,0,...,1,1,...,1].
        #   # So check if state is sorted.
        
        # This loop is 262,144 iterations. Inside, we do C swaps.
        # If C=100, 26M operations. In Python, this might take ~1-5 seconds.
        # If we do this in the inner loop of an evolution, it's too slow.
        
        # We need a faster verifier or a smarter search.
        # Maybe we can check a subset first? No, correctness requires all.
        
        # Let's try to speed up the verifier using numpy?
        # Or maybe we can use the "sorting network" property differently.
        
        # Actually, for the purpose of this task, let's first generate a good network
        # using a known algorithm (like Batcher's Odd-Merge or Bitonic) and then try to optimize.
        # If we start with a valid network, we can try to remove comparators or swap them.
        
        # Let's implement a faster verifier using bit manipulation tricks or just optimized Python.
        
        # Trick: 
        # Instead of simulating the whole network for each input, we can track the "signature" or use a different approach.
        # But the 0-1 principle is the standard way.
        
        # Let's write a reasonably fast verifier.
        
        n_inputs = 1 << N
        # Pre-compute the sorted target for each popcount? No, the target is always 0s then 1s.
        # So for a given input with k ones, the target is [0]*(N-k) + [1]*k.
        
        # We can pre-compute the targets? 
        # targets = {k: [0]*(N-k) + [1]*k for k in range(N+1)}
        
        # Let's try to vectorize the simulation for a batch of inputs?
        # It's complex. Let's stick to a tight Python loop first and see if it's fast enough.
        
        # Optimization: Use a local variable for comparators.
        comps = comparators
        n = N
        
        for x in range(n_inputs):
            state = [(x >> i) & 1 for i in range(n)]
            for i, j in comps:
                if state[i] > state[j]:
                    state[i], state[j] = state[j], state[i]
            # Check if sorted
            # Since it's 0/1, it's sorted if it's non-decreasing.
            for i in range(n - 1):
                if state[i] > state[i + 1]:
                    return False
        return True

    # --- Initial Network Generation ---
    
    # Let's use Batcher's Bitonic Sort Network as a starting point.
    # It is O(N log^2 N) comparators.
    # For N=18, we can pad to 32? Or use a recursive construction.
    # Actually, Odd-Even Transposition is N^2/2 = 162 comparators for N=18.
    # Bitonic for 18: 
    # We can use a standard bitonic sort network generator.
    
    def bitonic_network(n):
        """Generate a bitonic sorting network for n wires."""
        # This is a recursive definition.
        # Bitonic(n) = Bitonic(n/2) || Bitonic(n/2) || Merge(n)
        # Merge(n) is the bitonic merge.
        
        # For simplicity and speed, let's just use Odd-Even Transposition as the seed.
        # It's simple and valid.
        # Seed: 153 comparators (as per prompt, though N^2/2 is 162, maybe they count differently or it's a specific variant).
        # The prompt says "seed odd-even transposition network uses 153 comparators".
        # Let's generate that.
        
        # Odd-Even Transposition:
        # Phase k (0 to N-1):
        #   If k is even: compare (0,1), (2,3), ...
        #   If k is odd: compare (1,2), (3,4), ...
        
        comps = []
        for phase in range(n):
            for i in range(phase % 2, n - 1, 2):
                comps.append((i, i + 1))
        return comps

    # Start with Odd-Even Transposition
    current_network = bitonic_network(N)
    
    # Verify the seed to be sure
    if not verify_network(current_network):
        # Fallback to a simpler, definitely correct network if something went wrong
        # (Though OET should be correct)
        return current_network

    best_network = current_network
    best_count = len(current_network)
    
    # --- Local Search / Optimization ---
    
    # Strategy:
    # 1. Try to remove comparators one by one. If the network remains valid, keep the removal.
    # 2. Try to swap comparators (change order) or replace them with other pairs to see if we can reduce count or enable further removals.
    #    Removing is the primary goal for score.
    
    # Since verify_network is expensive (~1-5s), we can only do a few iterations.
    # Let's try a greedy removal approach.
    
    # We need to be careful with the time budget.
    # Let's estimate the time for one verification.
    # If it's too slow, we might only be able to do a few checks.
    
    # Let's try to optimize the verifier further.
    # Can we use the fact that if we remove a comparator, we only need to check if the network still sorts?
    # Yes, that's what verify_network does.
    
    # Let's try to remove comparators from the end first, as they might be redundant?
    # Or randomly?
    
    # Let's try a simple hill climbing:
    # While time permits:
    #   Pick a random comparator index
    #   Remove it
    #   Verify
    #   If valid, keep it
    #   Else, restore it
    
    # But random removal might get stuck.
    # A better approach: Try to remove each comparator in order, from last to first?
    # Or try all possible removals in parallel? No, sequential is easier.
    
    # Let's try to remove comparators from the network.
    # We'll iterate through the comparators and try to remove them.
    
    # To speed up verification, we can use a "fingerprint" or just accept the cost.
    # Let's assume verify_network takes ~2 seconds.
    # In 85 seconds, we can do ~40 verifications.
    
    # Let's try to be smarter.
    # We can try to remove multiple comparators at once? No, one by one is safer.
    
    # Let's try a different strategy: 
    # Use a known better network as the seed if possible.
    # But we don't have one hardcoded.
    
    # Let's stick with OET and try to remove comparators.
    
    # To improve the chance of finding a better network, we can also try to "shuffle" the order of comparators?
    # Changing the order doesn't change the count, so it doesn't help the score directly,
    # but it might enable further removals.
    
    # Let's focus on removal.
    
    # We'll try to remove comparators from the current best network.
    # We'll do this in a loop, trying to remove one comparator at a time.
    # We'll prioritize removing comparators that are "later" in the network, as they might be less critical?
    # Or just iterate through them.
    
    # Let's try to remove comparators in a random order to escape local minima?
    # Or a systematic order?
    
    # Let's try systematic removal from the end.
    
    i = len(current_network) - 1
    while i >= 0 and time.time() < deadline:
        # Try to remove comparator at index i
        trial_network = current_network[:i] + current_network[i+1:]
        
        # Verify
        if verify_network(trial_network):
            current_network = trial_network
            best_network = current_network
            best_count = len(current_network)
            # Don't decrement i, because the next comparator is now at index i
            # (since we removed one, the list is shorter)
            # Actually, if we remove index i, the element at i+1 moves to i.
            # So we should check the new index i again.
        else:
            i -= 1
    
    # If we have time, we can try to remove more by restarting the removal process?
    # Or try random removals?
    
    # Let's try a few random removal attempts if time permits.
    # But we need to be careful not to degrade the network.
    # We only keep a removal if it's valid.
    
    # Let's do a few random restarts of the removal process?
    # No, we should stick with the best found so far.
    
    # Let's try to remove comparators from the beginning as well?
    # The systematic pass from the end might have missed some.
    
    # Let's do another pass from the beginning.
    i = 0
    while i < len(current_network) and time.time() < deadline:
        trial_network = current_network[:i] + current_network[i+1:]
        if verify_network(trial_network):
            current_network = trial_network
            best_network = current_network
            best_count = len(current_network)
            # Don't increment i
        else:
            i += 1
            
    return best_network

# EVOLVE-BLOCK-END
