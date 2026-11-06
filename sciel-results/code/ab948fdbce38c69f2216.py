import numpy as np
import time
import random

N = 17
NUM_INPUTS = 1 << N

# Precompute the input matrix for validation
# shape: (NUM_INPUTS, N), dtype uint8
# input[k, j] is the value on wire j for the k-th Boolean input
# We can generate this efficiently.
# The k-th input is the binary representation of k.
# We want a matrix where row k is the bits of k.
# np.array([list(map(int, bin(i)[2:].zfill(N))) for i in range(NUM_INPUTS)]) is too slow in pure python loops.
# Better:
# indices = np.arange(NUM_INPUTS)
# bits = (indices[:, None] >> np.arange(N)[None, :]) & 1
# This creates a (NUM_INPUTS, N) array.
INPUTS = None

def construct():
    global INPUTS
    if INPUTS is None:
        indices = np.arange(NUM_INPUTS, dtype=np.uint32)
        # (indices[:, None] >> np.arange(N)[None, :]) & 1
        # np.arange(N) is (N,)
        # indices[:, None] is (NUM_INPUTS, 1)
        # result is (NUM_INPUTS, N)
        INPUTS = (indices[:, None] >> np.arange(N)[None, :]) & 1
        INPUTS = INPUTS.astype(np.uint8)

    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    # 1. Generate seed network: Odd-Even Transposition
    # N phases, each with N/2 comparators
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))
    
    def is_valid(network):
        """
        Check if the network sorts all inputs.
        network: list of (i, j)
        """
        if not network:
            # Empty network is only valid if N=1, which is not the case here
            return N == 1
        
        state = INPUTS.copy()
        
        for i, j in network:
            # min and max for sorting
            # state[:, i] and state[:, j]
            # min_val = np.minimum(state[:, i], state[:, j])
            # max_val = np.maximum(state[:, i], state[:, j])
            # state[:, i] = min_val
            # state[:, j] = max_val
            
            # In-place to save memory? No, state is a copy.
            # We can do:
            min_val = np.minimum(state[:, i], state[:, j])
            max_val = np.maximum(state[:, i], state[:, j])
            state[:, i] = min_val
            state[:, j] = max_val

        # Check if sorted: state[:, k] <= state[:, k+1] for all k
        # This is equivalent to: no 1s on the left of 0s in any row.
        # Or simply: np.all(state[:, :-1] <= state[:, 1:])
        # To be faster, we can check if any row is NOT sorted.
        # A row is not sorted if there exists k such that state[k, i] > state[k, i+1].
        # Since values are 0/1, this means state[k, i]==1 and state[k, i+1]==0.
        # So we check if there is any 1 in state[:, :-1] where state[:, 1:] is 0.
        # i.e., np.any((state[:, :-1] == 1) & (state[:, 1:] == 0))
        
        if np.any((state[:, :-1] == 1) & (state[:, 1:] == 0)):
            return False
        return True

    def reduce_network(network):
        """
        Greedily remove redundant comparators.
        """
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            # Iterate through comparators. 
            # To be more effective, we can shuffle the order we try removals, 
            # or just go through them. Going through them is deterministic and fast.
            # However, removing one might make another removable.
            # We restart the scan from the beginning whenever a removal happens.
            for idx in range(len(network)):
                if time.time() > deadline:
                    return network
                
                # Create a candidate network without network[idx]
                # To avoid slicing overhead, we can just skip it in the validation?
                # No, is_valid takes a list. Slicing is O(N), which is small (100-150).
                candidate = network[:idx] + network[idx+1:]
                
                if is_valid(candidate):
                    network = candidate
                    changed = True
                    break # Restart scan
        
        return network

    # Initial reduction
    best_network = reduce_network(seed_network[:])
    best_cost = len(best_network)
    
    current_network = best_network[:]
    current_cost = best_cost
    
    random.seed(42)
    
    # Search Loop
    # We use a "Grow and Shrink" strategy.
    # 1. Take current network.
    # 2. Add k random comparators at random positions.
    # 3. Reduce the new network.
    # 4. If the result is smaller, accept.
    
    # We also try "Swap" strategy:
    # 1. Take current network.
    # 2. Replace one comparator with a random one.
    # 3. Reduce.
    # 4. If smaller, accept.
    
    while time.time() < deadline:
        mutation_type = random.random()
        
        if mutation_type < 0.7:
            # Grow and Shrink
            # Number of additions: 1 to 3
            num_adds = random.choice([1, 1, 2, 2, 3])
            candidate = current_network[:]
            
            for _ in range(num_adds):
                # Generate random comparator
                i = random.randint(0, N - 1)
                j = random.randint(0, N - 1)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                # Insert at random position
                pos = random.randint(0, len(candidate))
                candidate.insert(pos, (i, j))
            
            # Reduce
            optimized = reduce_network(candidate)
            
        else:
            # Swap and Shrink
            if len(current_network) == 0:
                continue
                
            candidate = current_network[:]
            idx = random.randint(0, len(candidate) - 1)
            
            # Generate random replacement
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            candidate[idx] = (i, j)
            
            # Reduce
            optimized = reduce_network(candidate)
        
        if len(optimized) < current_cost:
            current_network = optimized
            current_cost = len(optimized)
            if current_cost < best_cost:
                best_network = current_network[:]
                best_cost = current_cost
                # Optional: print progress if needed, but keep quiet for scoring
                # print(f"New best: {best_cost}")
                
    return best_network
