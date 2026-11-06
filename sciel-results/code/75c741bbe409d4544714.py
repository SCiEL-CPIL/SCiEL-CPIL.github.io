"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
DEADLINE = 85.0  # seconds, safety margin before 90s kill

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Precompute all Boolean inputs as bitmasks
    all_inputs = np.arange(1 << N, dtype=np.int64)
    
    # Precompute sorted values: for each input, the sorted bitmask
    # sorted(x) has the lowest (N - popcount(x)) bits as 0 and highest popcount(x) bits as 1
    popcounts = np.array([bin(x).count('1') for x in range(1 << N)], dtype=np.int32)
    sorted_values = np.array([((1 << N) - 1) ^ ((1 << (N - k)) - 1) for k in range(N + 1)], dtype=np.int64)
    # For an input with k ones, sorted output is: top k bits are 1, bottom (N-k) bits are 0
    # sorted_output = ((1 << k) - 1) << (N - k)
    sorted_map = np.zeros(1 << N, dtype=np.int64)
    for k in range(N + 1):
        mask = (popcounts == k)
        sorted_map[mask] = ((1 << k) - 1) << (N - k)
    
    def apply_network(inputs, comparators):
        """Apply a list of comparators to an array of bitmasks."""
        for (i, j) in comparators:
            mask_i = (inputs >> i) & 1
            mask_j = (inputs >> j) & 1
            swap_mask = mask_j & (~mask_i)
            inputs = inputs ^ (swap_mask << i) ^ (swap_mask << j)
        return inputs
        
    def verify_network(network):
        """Verify if the network sorts all inputs."""
        if not network:
            return False
        result = apply_network(all_inputs.copy(), network)
        return np.array_equal(result, sorted_map)
    
    # Batcher's odd-even merge sort
    def batcher_odd_even_merge(left, right, comparators):
        """Merge two sorted lists (each of length n) into one sorted list of length 2n.
        left and right are lists of wire indices."""
        n = len(left)
        if n == 1:
            comparators.append((left[0], right[0]))
            return
        # Odd-even merge
        # Step 1: compare left[i] with right[i] for i in 0, 2, 4, ...
        for i in range(0, n, 2):
            comparators.append((left[i], right[i]))
        # Step 2: compare left[i] with right[i] for i in 1, 3, 5, ...
        for i in range(1, n, 2):
            comparators.append((left[i], right[i]))
        # Step 3: recursively merge the even and odd parts
        even_left = [left[i] for i in range(0, n, 2)]
        even_right = [right[i] for i in range(0, n, 2)]
        odd_left = [left[i] for i in range(1, n, 2)]
        odd_right = [right[i] for i in range(1, n, 2)]
        batcher_odd_even_merge(even_left, even_right, comparators)
        batcher_odd_even_merge(odd_left, odd_right, comparators)
    
    def batcher_sort(n):
        """Generate Batcher's odd-even merge sort network for n wires."""
        comparators = []
        _batcher_sort_recursive(list(range(n)), comparators)
        return comparators
    
    def _batcher_sort_recursive(wires, comparators):
        n = len(wires)
        if n <= 1:
            return
        mid = n // 2
        left = wires[:mid]
        right = wires[mid:]
        _batcher_sort_recursive(left, comparators)
        _batcher_sort_recursive(right, comparators)
        # Merge
        if len(left) != len(right):
            # For unequal sizes, pad the smaller one
            # Actually, Batcher's merge requires equal sizes.
            # For N=18, we need to handle this.
            # Let's just use odd-even transposition for now as a fallback.
            pass
        else:
            batcher_odd_even_merge(left, right, comparators)
    
    # Since Batcher's merge requires equal sizes, and 18 is even, we can split into 9 and 9.
    # But 9 is odd, so we need to handle that.
    # Let's just use a different approach: use the known fact that Batcher's network for 18 wires
    # can be constructed by splitting into two 9-wire networks and merging.
    # But 9-wire Batcher's network requires further splitting.
    # This is getting complicated. Let's just use the odd-even transposition network as the start,
    # but apply a more aggressive optimization.
    
    # Actually, let's just use the odd-even transposition network as the start.
    initial_network = [(i, i + 1) for phase in range(N)
                       for i in range(phase % 2, N - 1, 2)]
    
    best_network = list(initial_network)
    best_count = len(best_network)
    
    # Verify initial network
    if not verify_network(best_network):
        return best_network
    
    rng = random.Random(42)
    
    def greedy_remove(network, max_passes=10):
        """Try to remove comparators one by one, checking validity."""
        net = list(network)
        for _ in range(max_passes):
            if time.time() - start_time > DEADLINE - 2:
                break
            changed = False
            # Try removing from the end to avoid index shifting
            for idx in range(len(net) - 1, -1, -1):
                if time.time() - start_time > DEADLINE - 2:
                    break
                candidate = net[:idx] + net[idx+1:]
                if verify_network(candidate):
                    net = candidate
                    changed = True
            if not changed:
                break
        return net
    
    # Phase 1: Initial Greedy Removal
    best_network = greedy_remove(best_network, max_passes=5)
    best_count = len(best_network)
    
    # Phase 2: Simulated Annealing
    def random_move(net):
        """Generate a random move: remove, add, or swap."""
        move_type = rng.choice(['remove', 'swap', 'add'])
        net = list(net)
        if move_type == 'remove' and len(net) > 1:
            idx = rng.randint(0, len(net) - 1)
            net.pop(idx)
        elif move_type == 'swap' and len(net) > 1:
            i = rng.randint(0, len(net) - 1)
            j = rng.randint(0, len(net) - 1)
            net[i], net[j] = net[j], net[i]
        elif move_type == 'add':
            i = rng.randint(0, N - 2)
            j = rng.randint(i + 1, N - 1)
            idx = rng.randint(0, len(net))
            net.insert(idx, (i, j))
        return net
    
    # Simulated annealing
    current_network = list(best_network)
    current_count = len(current_network)
    T = 10.0  # initial temperature
    cooling_rate = 0.99
    
    while time.time() - start_time < DEADLINE - 5:
        if T < 0.1:
            T = 10.0  # reheat
            current_network = list(best_network)
            current_count = len(current_network)
        
        new_network = random_move(current_network)
        
        if len(new_network) > MAX_COMPARATORS:
            T *= cooling_rate
            continue
        
        if not verify_network(new_network):
            T *= cooling_rate
            continue
        
        new_count = len(new_network)
        delta = new_count - current_count
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_network = new_network
            current_count = new_count
            
            if current_count < best_count:
                best_network = list(current_network)
                best_count = current_count
        
        T *= cooling_rate
    
    return best_network
# EVOLVE-BLOCK-END
