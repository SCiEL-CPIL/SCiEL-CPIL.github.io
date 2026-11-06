import time
import numpy as np

N = 18
TOTAL_INPUTS = 1 << N
DEADLINE = 85.0

def construct():
    start_time = time.time()
    
    # 1. Precompute Inputs
    # Shape: (262144, 18)
    all_inputs = np.zeros((TOTAL_INPUTS, N), dtype=np.uint8)
    for i in range(N):
        all_inputs[:, i] = (np.arange(TOTAL_INPUTS) >> i) & 1
        
    # Precompute all possible comparators
    # comparator (i, j) -> mask where i > j (needs swapping)
    # We will use this to quickly check if a comparator is "useful" or to simulate.
    
    def apply_comparator(state, i, j):
        # state: (M, N)
        # min to i, max to j
        # if state[:, i] > state[:, j], swap
        # mask: state[:, i] > state[:, j]
        mask = state[:, i] > state[:, j]
        # swap values at i and j where mask is True
        # We can do this with np.where or by swapping the columns and then restoring?
        # Faster: 
        # temp = state[:, i].copy()
        # state[:, i] = np.where(mask, state[:, j], state[:, i])
        # state[:, j] = np.where(mask, temp, state[:, j])
        # Even faster with bitwise ops?
        # state[:, i] &= ~mask | state[:, j]? No.
        # Let's stick to the vectorized where, it's quite fast.
        
        # Optimization: if we know the state is boolean, we can use:
        # state[:, i] = np.minimum(state[:, i], state[:, j])
        # state[:, j] = np.maximum(state[:, i], state[:, j]) -- CAREFUL: state[:, i] is already updated!
        # So:
        # min_val = np.minimum(state[:, i], state[:, j])
        # max_val = np.maximum(state[:, i], state[:, j])
        # state[:, i] = min_val
        # state[:, j] = max_val
        
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
        return state

    def is_valid(network):
        state = all_inputs.copy()
        for i, j in network:
            apply_comparator(state, i, j)
        # Check sorted: state[:, k] <= state[:, k+1] for all k
        # This is equivalent to: no inversion exists.
        # Inversion: state[:, i] > state[:, j] for i < j.
        # We can check adjacent pairs only? No, sorting requires global order.
        # But if adjacent pairs are sorted, the whole array is sorted.
        # So check state[:, :-1] <= state[:, 1:]
        return np.all(state[:, :-1] <= state[:, 1:])

    # 2. Initial Network: Odd-Even Transposition
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
            
    # Verify initial
    if not is_valid(network):
        return network
        
    # 3. Greedy Removal (Fast Pass)
    # Try to remove comparators in a specific order to get a good baseline quickly.
    # We'll do a few passes.
    changed = True
    passes = 0
    while changed and passes < 10 and time.time() < DEADLINE - 20:
        changed = False
        passes += 1
        # Iterate backwards to avoid index shifting issues, or use a set of indices.
        # Backwards is safe for deletion.
        for idx in range(len(network) - 1, -1, -1):
            if time.time() > DEADLINE - 20:
                break
            # Try removing network[idx]
            new_network = network[:idx] + network[idx+1:]
            if is_valid(new_network):
                network = new_network
                changed = True
                # Don't break, keep removing if possible.
                
    best_network = network
    best_size = len(network)
    
    # 4. Advanced Local Search: Simulated Annealing with Bitwise Verification
    # To speed up verification, we can use a bitmask representation of the state?
    # No, the state is 262144 x 18.
    # However, we can precompute the "effect" of a comparator on the *unsortedness*.
    # Let's stick to the full simulation but optimize the loop.
    
    # Optimization: Instead of copying the entire all_inputs array every time is_valid is called,
    # we can maintain a state and apply/revert changes?
    # Reverting is hard.
    # Alternative: Use a faster verification.
    # The main cost is the loop over comparators and the numpy ops.
    
    # Let's try a different search: 
    # 1. Identify "critical" comparators.
    # 2. Randomly swap or replace non-critical ones.
    
    # Actually, the previous attempt at SA was slow because is_valid is O(N * 2^N * C).
    # C ~ 100. 2^N = 262144. N=18.
    # 100 * 262144 * 18 ops ~ 470 million ops. In Python/Numpy this takes ~0.5s - 1s.
    # So we can only do ~50-100 iterations in 60s. That's not enough for SA.
    
    # We need a faster validity check.
    # Idea: Use a smaller set of test inputs?
    # The task says "correctness is certified over all 262,144 Boolean inputs".
    # But during search, we can use a subset to guide, and only verify full set when we find a candidate.
    # Let's pick a random subset of 1024 inputs for fast checks.
    
    # 5. Hybrid Search with Subsampled Verification
    # Fast check: Use 1024 random inputs.
    # Full check: Use all 262144 inputs.
    
    # Generate subsample
    subsample_size = 1024
    subsample_indices = np.random.choice(TOTAL_INPUTS, subsample_size, replace=False)
    subsample_inputs = all_inputs[subsample_indices]
    
    def is_valid_fast(network):
        state = subsample_inputs.copy()
        for i, j in network:
            apply_comparator(state, i, j)
        return np.all(state[:, :-1] <= state[:, 1:])
        
    def is_valid_full(network):
        return is_valid(network)

    # Restart SA with fast checks
    current_network = best_network
    current_size = best_size
    
    T = 1.0
    T_min = 0.01
    alpha = 0.99
    
    no_improve_count = 0
    
    # Precompute all possible comparators for random moves
    all_comparators = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    while time.time() < DEADLINE:
        if no_improve_count > 200:
            # Perturb: Add a random comparator
            i = np.random.randint(0, N-1)
            j = np.random.randint(i+1, N)
            new_net = current_network + [(i, j)]
            current_network = new_net
            current_size = len(new_net)
            no_improve_count = 0
            T = max(T * 0.5, T_min)
            
        # Choose a move
        move_type = np.random.choice(['remove', 'swap', 'replace'], p=[0.5, 0.3, 0.2])
        
        if move_type == 'remove':
            if len(current_network) == 0:
                continue
            idx = np.random.randint(0, len(current_network))
            new_network = current_network[:idx] + current_network[idx+1:]
            new_size = len(new_network)
            
            if is_valid_fast(new_network):
                # Verify with full set
                if is_valid_full(new_network):
                    current_network = new_network
                    current_size = new_size
                    no_improve_count = 0
                    if current_size < best_size:
                        best_network = current_network
                        best_size = current_size
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
                
        elif move_type == 'swap':
            if len(current_network) < 2:
                continue
            i1, i2 = np.random.choice(len(current_network), 2, replace=False)
            # Swap
            temp = current_network[i1]
            current_network[i1] = current_network[i2]
            current_network[i2] = temp
            
            if is_valid_fast(current_network):
                if is_valid_full(current_network):
                    no_improve_count = 0
                else:
                    # Revert
                    current_network[i2] = current_network[i1]
                    current_network[i1] = temp
                    no_improve_count += 1
            else:
                # Revert
                current_network[i2] = current_network[i1]
                current_network[i1] = temp
                no_improve_count += 1
                
        elif move_type == 'replace':
            if len(current_network) == 0:
                continue
            idx = np.random.randint(0, len(current_network))
            # Replace with random comparator
            i = np.random.randint(0, N-1)
            j = np.random.randint(i+1, N)
            new_network = current_network.copy()
            new_network[idx] = (i, j)
            
            if is_valid_fast(new_network):
                if is_valid_full(new_network):
                    current_network = new_network
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
                
        # Cool down
        T *= alpha
        
    return best_network
