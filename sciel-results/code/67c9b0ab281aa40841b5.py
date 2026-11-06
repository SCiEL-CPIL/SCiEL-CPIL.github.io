"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a sorting network for N=17 wires.
    Strategy: Use a known efficient construction (Batcher's odd-even merge sort
    or a hybrid) and then attempt local search to reduce comparator count.
    
    However, given the 60s time limit and the complexity of finding a network
    significantly better than 136 (odd-even transposition) within that time,
    we will use a robust, efficient base network and perform limited optimization.
    
    The best known for 17 is 71, but that's hard to find quickly.
    A good baseline is Batcher's odd-even merge sort, which for N=16 is 85 comparators.
    For N=17, we can use a recursive construction or simply use a slightly larger power of 2
    and ignore the extra wire, but that's wasteful.
    
    Let's try to construct a network using a simple, efficient algorithm that is guaranteed
    to sort, and then apply a heuristic search to remove redundant comparators.
    
    Actually, the safest bet within the time limit is to return a known good network
    that is significantly better than the seed. 
    Odd-even merge sort for N=17:
    We can decompose 17 into 16+1.
    Sort the first 16 using a 16-wire network (85 comparators).
    Then merge the 16-sorted list with the 1st element.
    Merging a sorted list of size 16 with a single element requires 16 comparators (to find where it fits).
    Total: 85 + 16 = 101.
    
    Wait, the seed is 136. 101 is better.
    Can we do better?
    Batcher's odd-even merge sort for N=17 might be constructed recursively.
    Let's implement a function to generate Batcher's odd-even merge sort network.
    
    Alternatively, we can use a simple local search starting from the odd-even transposition
    network to remove redundant comparators.
    
    Given the constraints, let's implement a fast local search on top of a good initial network.
    The initial network will be Batcher's odd-even merge sort for N=17.
    
    Let's define a function to create the Batcher network.
    """
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s margin
    
    # Helper: Check if a network is valid on all boolean inputs
    def is_valid_network(net):
        # net is a list of (i, j) pairs
        # Use numpy for speed
        # Generate all 2^17 boolean inputs? That's 131072. 
        # We can simulate in parallel using numpy bit operations.
        # Represent the 131072 inputs as a matrix of shape (131072, 17) of bits.
        # But 131072 * 17 is ~2.2M elements, which is fine.
        
        # Actually, we can just check a subset or use a faster method.
        # For local search, we need fast validation.
        # Let's use a representative set of inputs: all permutations of 0/1? No, too many.
        # The zero-one principle says we only need to check boolean inputs.
        # There are 2^17 = 131072 inputs.
        # We can simulate the network on all of them at once using vectorized operations.
        
        # Create all boolean inputs
        # Input k has bits corresponding to binary representation of k?
        # No, each wire can be 0 or 1 independently.
        # So we need a matrix of shape (2^N, N) where row i is the i-th boolean vector.
        
        # Efficient way to generate all boolean vectors:
        # We can use np.arange(2**N) and then bit shift.
        # But 2**17 is 131072.
        
        # Let's try to simulate the network on all inputs at once.
        # State: (2^N, N) array of 0s and 1s.
        # Comparator (i, j): for each row, if state[k, i] > state[k, j], swap them.
        
        # This might be memory heavy but doable.
        # 131072 * 17 * 1 byte = ~2.2 MB.
        
        # Generate all inputs
        # inputs = np.zeros((2**N, N), dtype=np.uint8)
        # for bit in range(N):
        #     inputs[:, bit] = (np.arange(2**N) >> bit) & 1
        
        # This is slow in Python loop. Let's use a trick.
        # Actually, we can just use a smaller set of "critical" inputs if we are doing local search,
        # but for final validation we must be sure.
        # For the search, we can use a random sample of boolean inputs to quickly reject bad moves.
        # For the final network, we do a full check.
        
        # Let's implement a fast check on a random subset first.
        # If it passes the subset, we do the full check.
        
        # For now, let's implement the full check but efficiently.
        # We will generate the inputs once and reuse them? No, they change if we modify the network?
        # No, the inputs are fixed. The network changes.
        # So we can pre-generate the inputs matrix.
        
        # But we are inside construct(), so we can't easily cache across calls unless we use a global.
        # Let's use a global cache for the inputs matrix.
        
        global _BOOL_INPUTS
        if '_BOOL_INPUTS' not in globals():
            # Generate all 2^N boolean vectors
            # This takes a bit of time and memory.
            # 131072 x 17
            _BOOL_INPUTS = np.zeros((2**N, N), dtype=np.uint8)
            for bit in range(N):
                _BOOL_INPUTS[:, bit] = (np.arange(2**N) >> bit) & 1
        
        # Simulate the network
        state = _BOOL_INPUTS.copy()
        
        for i, j in net:
            # Comparator: min to i, max to j
            # If state[:, i] > state[:, j], swap
            # We can do this with numpy operations
            # swap_mask = state[:, i] > state[:, j]
            # state[:, i], state[:, j] = np.where(swap_mask, state[:, j], state[:, i]), np.where(swap_mask, state[:, i], state[:, j])
            
            # More efficient:
            # left = state[:, i]
            # right = state[:, j]
            # new_left = np.minimum(left, right)
            # new_right = np.maximum(left, right)
            # state[:, i] = new_left
            # state[:, j] = new_right
            
            # This is 4 array accesses and 2 writes per comparator.
            # With ~100 comparators, this is fast.
            
            left = state[:, i]
            right = state[:, j]
            state[:, i] = np.minimum(left, right)
            state[:, j] = np.maximum(left, right)
            
            # Check deadline
            if time.time() > deadline:
                return False # Timeout, assume invalid to be safe? Or return current best?
                # Actually, if we timeout during validation, we should stop search.
        
        # Check if sorted: for each row, state[:, k] <= state[:, k+1] for all k
        # sorted_mask = all(state[:, k] <= state[:, k+1] for k in range(N-1))
        # We can vectorize this too.
        # diffs = np.diff(state, axis=1)
        # return np.all(diffs >= 0)
        
        diffs = np.diff(state, axis=1)
        return np.all(diffs >= 0)
    
    # Helper: Generate Batcher's Odd-Even Merge Sort network
    def generate_batcher_network(n):
        """
        Generates Batcher's odd-even merge sort network for n wires.
        This is a recursive construction.
        """
        if n == 1:
            return []
        
        # Split n into two halves
        half = n // 2
        rem = n % 2
        
        # Sort the two halves
        # Left half: wires 0 to half-1
        # Right half: wires half to n-1
        # If rem is 1, the last wire is in the right half?
        # Standard odd-even merge sort splits into floor(n/2) and ceil(n/2).
        # Let's use:
        # n0 = n // 2
        # n1 = n - n0
        
        n0 = n // 2
        n1 = n - n0
        
        net0 = generate_batcher_network(n0)
        net1 = generate_batcher_network(n1)
        
        # Merge the two sorted halves
        # The merge phase of odd-even merge sort.
        # This is complex to implement correctly from scratch.
        
        # Given the complexity and time, let's use a simpler, known good network.
        # We can use the "AKS" or just a simple bubble sort network (which is O(N^2) comparators).
        # Bubble sort network: N(N-1)/2 comparators. For N=17, that's 136.
        # That's the same as the seed!
        
        # We need something better.
        # Let's use a precomputed network for N=17 if we can find one.
        # Or, we can use a heuristic search starting from the bubble sort network.
        
        # Let's implement a simple local search:
        # 1. Start with bubble sort network (136 comparators).
        # 2. Try to remove each comparator one by one.
        # 3. If the network remains valid, keep it removed.
        # 4. Repeat until no more can be removed.
        
        # This is a greedy deletion algorithm. It's fast and often works well.
        
        # Initial network: Bubble sort (odd-even transposition is similar but parallel)
        # Let's use the standard "brick wall" or bubble sort network:
        # For i in 0..N-2:
        #   For j in 0..N-2-i:
        #     (j, j+1)
        
        # Actually, the seed is odd-even transposition, which is parallel bubble sort.
        # Let's start with that.
        
        current_net = [(i, i + 1) for phase in range(N)
                       for i in range(phase % 2, N - 1, 2)]
        
        # Local search: Try to remove comparators
        # We need a fast validity check.
        # The full boolean check is expensive if done repeatedly.
        # Let's use a random sample of boolean inputs for the search.
        
        # Sample 1000 random boolean inputs
        sample_size = 1000
        sample_inputs = np.random.randint(0, 2, (sample_size, N), dtype=np.uint8)
        
        def is_valid_sample(net):
            state = sample_inputs.copy()
            for i, j in net:
                left = state[:, i]
                right = state[:, j]
                state[:, i] = np.minimum(left, right)
                state[:, j] = np.maximum(left, right)
            diffs = np.diff(state, axis=1)
            return np.all(diffs >= 0)
        
        # Greedy deletion
        # Try to remove each comparator
        # We iterate through the list and try to remove each one.
        # If removal keeps the network valid (on sample), we remove it.
        # We repeat this process until no more can be removed.
        
        # To make it faster, we can randomize the order of removal attempts.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle the indices to try different orders
            indices = list(range(len(current_net)))
            random.shuffle(indices)
            
            for idx in indices:
                if idx >= len(current_net):
                    continue
                
                # Try to remove current_net[idx]
                # Create a new network without this comparator
                # To avoid O(N^2) copying, we can just simulate the removal?
                # No, we need to check validity.
                # We can temporarily remove it, check, and put it back.
                
                removed = current_net.pop(idx)
                
                # Check validity on sample
                if is_valid_sample(current_net):
                    # Keep it removed
                    improved = True
                else:
                    # Put it back
                    current_net.insert(idx, removed)
                
                # Check deadline
                if time.time() > deadline:
                    break
            
            if time.time() > deadline:
                break
        
        # Final validation on all boolean inputs
        if is_valid_network(current_net):
            return current_net
        else:
            # If the local search broke it (shouldn't happen if sample is good, but just in case)
            # Return the original seed network
            return [(i, i + 1) for phase in range(N)
                    for i in range(phase % 2, N - 1, 2)]

# EVOLVE-BLOCK-END
