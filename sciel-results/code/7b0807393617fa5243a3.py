"""Direct construction of a size-efficient sorting network on 18 wires using Optimized Simulated Annealing."""
N = 18
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    # Pre-compute bit masks for verification
    total_inputs = 1 << N  # 262144
    NUM_WORDS = total_inputs // 64  # 4096

    indices = np.arange(total_inputs, dtype=np.uint32)
    weights = 1 << np.arange(64, dtype=np.uint64)

    initial_wires = [None] * N
    for i in range(N):
        bits_i = (indices >> i) & 1
        bits_i_64 = bits_i.astype(np.uint64)
        packed = (bits_i_64.reshape(NUM_WORDS, 64) * weights).sum(axis=1)
        initial_wires[i] = packed

    # Stack into a single 2D array for faster copying and operations
    initial_wires_2d = np.vstack(initial_wires)

    def verify_network(comparators):
        """
        Fast verification using bit-parallelism on a 2D array.
        Returns True if the network sorts all Boolean inputs.
        """
        # Copy the 2D array. This is faster than copying a list of 1D arrays.
        wires = initial_wires_2d.copy()

        for (i, j) in comparators:
            wi = wires[i]
            wj = wires[j]
            wires[i] = wi & wj
            wires[j] = wi | wj

        # Check sorted: for all k, wires[k] is subset of wires[k+1]
        # wires[:-1] & ~wires[1:] should be all zeros
        for k in range(N - 1):
            if np.any(wires[k] & ~wires[k + 1]):
                return False
        return True

    def greedy_remove(comparators, time_limit=None):
        """
        Iteratively remove redundant comparators.
        Scans the list and removes any comparator that doesn't break sorting.
        """
        current = list(comparators)
        changed = True
        while changed:
            changed = False
            if time_limit and time.time() > time_limit:
                break
            n = len(current)
            if n == 0:
                break
            
            # Try removing from random positions, but prioritize ends?
            # Actually, just shuffling and trying each one is robust.
            # To speed up, we can try a subset if n is large.
            # But n is usually < 150 here. 150 verifications is fast.
            positions = list(range(n))
            random.shuffle(positions)
            
            for idx in positions:
                if idx >= len(current):
                    continue
                test = current[:idx] + current[idx + 1:]
                if verify_network(test):
                    current = test
                    changed = True
                    # Restart the loop to re-shuffle and try again on the new list
                    break

        return current

    def odd_even_transposition():
        comparators = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                comparators.append((i, i + 1))
        return comparators

    def bubble_sort_network():
        comparators = []
        for i in range(N - 1):
            for j in range(N - 1 - i):
                comparators.append((j, j + 1))
        return comparators

    def shell_sort_network():
        comparators = []
        gaps = [9, 4, 2, 1]
        for gap in gaps:
            for i in range(gap, N):
                comparators.append((i - gap, i))
        return comparators

    def random_dense_network(num_comparators=150):
        comparators = []
        for _ in range(num_comparators):
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i != j:
                if i > j:
                    i, j = j, i
                comparators.append((i, j))
        random.shuffle(comparators)
        return comparators

    def simulated_annealing(initial_net, t_max=1000, t_min=1.0, alpha=0.995, max_iter=20000, deadline=None):
        """
        Simulated Annealing to minimize comparator count.
        """
        current = list(initial_net)
        current_energy = len(current)
        best = list(current)
        best_energy = current_energy
        
        t = t_max
        random.seed(42) # Fixed seed for reproducibility within restart
        
        for i in range(max_iter):
            if deadline and time.time() > deadline:
                break
            
            r = random.random()
            candidate = list(current)
            
            if r < 0.5:
                # Remove (Higher probability as we want to minimize size)
                if len(candidate) == 0:
                    continue
                idx = random.randint(0, len(candidate) - 1)
                candidate.pop(idx)
            elif r < 0.7:
                # Insert
                u = random.randint(0, N - 1)
                v = random.randint(0, N - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                pos = random.randint(0, len(candidate))
                candidate.insert(pos, (u, v))
            elif r < 0.9:
                # Swap
                if len(candidate) < 2:
                    continue
                i1, i2 = random.sample(range(len(candidate)), 2)
                candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
            else:
                # Shift (Move a comparator to a new position)
                if len(candidate) < 2:
                    continue
                i1 = random.randint(0, len(candidate) - 1)
                i2 = random.randint(0, len(candidate) - 1)
                if i1 == i2:
                    continue
                comp = candidate.pop(i1)
                candidate.insert(i2, comp)

            if not verify_network(candidate):
                continue # Invalid, reject

            new_energy = len(candidate)
            
            if new_energy < current_energy:
                # Always accept improvement
                current = candidate
                current_energy = new_energy
                if new_energy < best_energy:
                    best = list(current)
                    best_energy = new_energy
            else:
                # Accept with probability exp(-delta_e / t)
                delta_e = new_energy - current_energy
                if delta_e > 0:
                    prob = np.exp(-delta_e / t)
                    if random.random() < prob:
                        current = candidate
                        current_energy = new_energy
            
            t *= alpha
            if t < t_min:
                break
        
        return best

    # Generate starting candidates
    candidates = []
    candidates.append(("odd_even", odd_even_transposition()))
    candidates.append(("bubble", bubble_sort_network()))
    candidates.append(("shell", shell_sort_network()))
    
    # Add some random dense starts
    for i in range(3):
        random.seed(100 + i)
        candidates.append((f"random_{i}", random_dense_network(150)))

    best_network = None
    best_count = float('inf')

    # Run Prune + SA on each candidate
    for name, net in candidates:
        if time.time() > deadline - 5.0:
            break
        
        # First, try to prune the initial network to get a better starting point for SA
        pruned = greedy_remove(net, time_limit=start_time + 5.0) # Quick prune
        if len(pruned) < best_count:
            best_count = len(pruned)
            best_network = list(pruned)
        
        # Run SA from the pruned state
        sa_result = simulated_annealing(
            pruned, 
            t_max=500, 
            t_min=0.5, 
            alpha=0.998, 
            max_iter=10000, 
            deadline=deadline - 5.0
        )
        
        if len(sa_result) < best_count:
            best_count = len(sa_result)
            best_network = list(sa_result)
            
        # Final greedy remove on the SA result
        final_pruned = greedy_remove(best_network, time_limit=deadline - 2.0)
        if len(final_pruned) < best_count:
            best_count = len(final_pruned)
            best_network = list(final_pruned)

    if best_network is None:
        # Fallback
        best_network = odd_even_transposition()
        best_network = greedy_remove(best_network, time_limit=deadline - 1.0)

    # Final verification
    if not verify_network(best_network):
        # Should not happen, but safety
        fallback = odd_even_transposition()
        fallback = greedy_remove(fallback, time_limit=deadline - 1.0)
        return fallback

    return best_network
