# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 16

def construct():
    start_time = time.time()
    deadline = 88.0  # Safety margin

    # 1. Generate Batcher's Bitonic Sort Network (80 comparators)
    def generate_batcher(n):
        if n == 2:
            return [(0, 1)]
        
        half = n // 2
        left_net = generate_batcher(half)
        right_net = generate_batcher(half)
        right_net_remapped = [(i + half, j + half) for i, j in right_net]
        
        merge_net = []
        d = n // 2
        while d > 0:
            for i in range(0, n, 2 * d):
                for j in range(i, i + d):
                    merge_net.append((j, j + d))
            d //= 2
            
        return left_net + right_net + merge_net

    network = generate_batcher(N)
    
    # 2. Prepare Inputs for Verification
    # Full set of 65536 inputs for final check
    indices = np.arange(1 << N, dtype=np.uint32)
    inputs_full = np.zeros((1 << N, N), dtype=np.uint8)
    for j in range(N):
        inputs_full[:, j] = (indices >> j) & 1
        
    # Subset of inputs for fast checking during search
    # We use a random subset of ~2048 inputs to speed up iterations
    num_fast_inputs = 2048
    perm = np.random.permutation(1 << N)
    inputs_fast = inputs_full[perm[:num_fast_inputs]]

    def is_sorting_fast(net):
        """Check sorting on the fast subset. Returns True if sorted on all fast inputs."""
        state = inputs_fast.copy()
        for i, j in net:
            min_val = np.minimum(state[:, i], state[:, j])
            max_val = np.maximum(state[:, i], state[:, j])
            state[:, i] = min_val
            state[:, j] = max_val
        # Check sorted: state[:, k] <= state[:, k+1] for all k
        return not np.any(state[:, :-1] > state[:, 1:])

    def is_sorting_full(net):
        """Check sorting on the full set of 65536 inputs."""
        state = inputs_full.copy()
        for i, j in net:
            min_val = np.minimum(state[:, i], state[:, j])
            max_val = np.maximum(state[:, i], state[:, j])
            state[:, i] = min_val
            state[:, j] = max_val
        return not np.any(state[:, :-1] > state[:, 1:])

    # Verify initial Batcher network
    if not is_sorting_full(network):
        # Fallback to Odd-Even if Batcher fails (should not)
        network = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                network.append((i, i + 1))
        if not is_sorting_full(network):
            return network

    best_network = network[:]
    best_len = len(network)
    current_network = network[:]
    current_len = len(network)
    
    # Precompute all possible comparators
    all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
    comp_set = set(all_comps)
    
    # Simulated Annealing with Fast Check
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    steps = 0
    while time.time() < deadline and T > min_T:
        steps += 1
        
        # Choose a move
        move_type = random.choice(['remove', 'add', 'swap', 'replace'])
        
        new_net = current_network[:]
        
        if move_type == 'remove':
            if len(new_net) > 1:
                idx = random.randint(0, len(new_net) - 1)
                new_net.pop(idx)
            else:
                continue
        elif move_type == 'add':
            # Add a random comparator not in the list
            added = False
            for _ in range(10):
                c = random.choice(all_comps)
                if c not in new_net:
                    new_net.append(c)
                    added = True
                    break
            if not added:
                continue
        elif move_type == 'swap':
            if len(new_net) > 1:
                i = random.randint(0, len(new_net) - 1)
                j = random.randint(0, len(new_net) - 1)
                if i != j:
                    new_net[i], new_net[j] = new_net[j], new_net[i]
                else:
                    continue
            else:
                continue
        elif move_type == 'replace':
            if len(new_net) > 0:
                idx = random.randint(0, len(new_net) - 1)
                for _ in range(10):
                    c = random.choice(all_comps)
                    if c not in new_net:
                        new_net[idx] = c
                        break
                else:
                    continue

        # Fast Check
        if is_sorting_fast(new_net):
            new_len = len(new_net)
            # Accept if better or with probability
            if new_len < current_len:
                current_network = new_net
                current_len = new_len
                if current_len < best_len:
                    best_len = current_len
                    best_network = current_network[:]
            else:
                delta_cost = new_len - current_len
                if delta_cost == 0:
                    current_network = new_net
                    current_len = new_len
                else:
                    if random.random() < np.exp(-delta_cost / T):
                        current_network = new_net
                        current_len = new_len
        # If fast check fails, reject (do not update current)
        
        T *= cooling

    # Final greedy removal pass on best_network with FAST check first, then FULL check for acceptance
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for idx in range(len(best_network) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            candidate = best_network[:idx] + best_network[idx+1:]
            # Fast check first
            if is_sorting_fast(candidate):
                # Full check to be sure
                if is_sorting_full(candidate):
                    best_network = candidate
                    changed = True
                    break # Restart loop to try next removal

    # Final verification of the best network
    if is_sorting_full(best_network):
        return best_network
    else:
        # If best failed full check (should be rare if fast check is decent, but possible), 
        # fallback to the initial valid network or try a few more random restarts?
        # For safety, return the initial Batcher network which we know is valid.
        return network

# EVOLVE-BLOCK-END
