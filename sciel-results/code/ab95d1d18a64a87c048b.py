"""Direct construction of a size-efficient sorting network on 18 wires using Simulated Annealing."""
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

    # Pre-compute negations for faster sorted check
    neg_initial = [~w for w in initial_wires]

    def verify_network(comparators):
        """
        Fast verification using bit-parallelism.
        Returns True if the network sorts all Boolean inputs.
        """
        wires = [w.copy() for w in initial_wires]

        for (i, j) in comparators:
            wi = wires[i]
            wj = wires[j]
            wires[i] = wi & wj
            wires[j] = wi | wj

        # Check sorted: for all k, wires[k] is subset of wires[k+1]
        # Equivalent to: no element in wires[k] is missing from wires[k+1]
        for k in range(N - 1):
            if np.any(wires[k] & ~wires[k + 1]):
                return False
        return True

    def greedy_remove(comparators, time_limit=None, thorough=False):
        """
        Iteratively remove redundant comparators.
        If thorough=True, try all positions each pass (slower but more complete).
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
            
            if thorough or n <= 30:
                positions = list(range(n))
                random.shuffle(positions)
            else:
                positions = random.sample(range(n), min(15, n))

            for idx in positions:
                if idx >= len(current):
                    continue
                test = current[:idx] + current[idx + 1:]
                if verify_network(test):
                    current = test
                    changed = True
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

    def odd_even_merge_sort_network():
        """
        Odd-even merge sort network - typically more efficient than transposition.
        """
        def merge(a, b):
            """Merge two sorted lists of comparator pairs."""
            result = []
            if not a:
                return b
            if not b:
                return a
            i, j = 0, 0
            while i < len(a) and j < len(b):
                if a[i][1] <= b[j][1]:
                    result.append(a[i])
                    i += 1
                else:
                    result.append(b[j])
                    j += 1
            result.extend(a[i:])
            result.extend(b[j:])
            return result

        def merge_sort(n):
            """Generate comparators for odd-even merge sort on n wires."""
            if n == 1:
                return []
            if n == 2:
                return [(0, 1)]
            
            half = n // 2
            left = merge_sort(half)
            right = merge_sort(n - half)
            # Shift right comparators
            right = [(i + half, j + half) for (i, j) in right]
            
            # Merge the two halves
            merged = merge(left, right)
            return merged

        return merge_sort(N)

    def random_dense_network(num_comparators=200):
        """Generate a random dense network and shuffle."""
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
        random.seed(42)
        
        for i in range(max_iter):
            if deadline and time.time() > deadline:
                break
            
            r = random.random()
            candidate = list(current)
            
            if r < 0.45:
                # Remove
                if len(candidate) == 0:
                    continue
                idx = random.randint(0, len(candidate) - 1)
                candidate.pop(idx)
                delta_e = -1
            elif r < 0.60:
                # Insert
                u = random.randint(0, N - 1)
                v = random.randint(0, N - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                pos = random.randint(0, len(candidate))
                candidate.insert(pos, (u, v))
                delta_e = 1
            elif r < 0.80:
                # Swap
                if len(candidate) < 2:
                    continue
                i1, i2 = random.sample(range(len(candidate)), 2)
                candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
                delta_e = 0
            else:
                # Replace
                if len(candidate) == 0:
                    continue
                idx = random.randint(0, len(candidate) - 1)
                u = random.randint(0, N - 1)
                v = random.randint(0, N - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                candidate[idx] = (u, v)
                delta_e = 0

            if not verify_network(candidate):
                continue

            new_energy = len(candidate)
            
            if new_energy < current_energy:
                current = candidate
                current_energy = new_energy
                if new_energy < best_energy:
                    best = list(current)
                    best_energy = new_energy
            else:
                if random.random() < np.exp(-(new_energy - current_energy) / t):
                    current = candidate
                    current_energy = new_energy
            
            t *= alpha
            if t < t_min:
                break
        
        return best

    def polish_remove(comparators, time_limit=None):
        """
        Exhaustive removal: try removing each comparator one by one,
        and if removable, remove it and repeat.
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
            
            # Try all positions in order
            for idx in range(n):
                if time_limit and time.time() > time_limit:
                    break
                test = current[:idx] + current[idx + 1:]
                if verify_network(test):
                    current = test
                    changed = True
                    break

        return current

    # Generate starting candidates
    candidates = []
    candidates.append(("odd_even_trans", odd_even_transposition()))
    candidates.append(("bubble", bubble_sort_network()))
    candidates.append(("odd_even_merge", odd_even_merge_sort_network()))
    
    # Add random dense starts
    for i in range(3):
        random.seed(100 + i)
        candidates.append((f"random_{i}", random_dense_network(150)))

    best_network = None
    best_count = float('inf')

    # Phase 1: Quick prune each candidate
    pruned_candidates = []
    for name, net in candidates:
        if time.time() > deadline - 15.0:
            break
        pruned = greedy_remove(net, time_limit=start_time + 8.0)
        pruned_candidates.append((name, pruned))
        if len(pruned) < best_count:
            best_count = len(pruned)
            best_network = list(pruned)

    # Phase 2: Run SA on each pruned candidate
    for name, pruned in pruned_candidates:
        if time.time() > deadline - 10.0:
            break
        
        sa_result = simulated_annealing(
            pruned, 
            t_max=500, 
            t_min=0.5, 
            alpha=0.998, 
            max_iter=15000, 
            deadline=deadline - 8.0
        )
        
        if len(sa_result) < best_count:
            best_count = len(sa_result)
            best_network = list(sa_result)

    # Phase 3: Final polish with exhaustive removal
    if best_network is not None:
        final_pruned = polish_remove(best_network, time_limit=deadline - 2.0)
        if len(final_pruned) < best_count:
            best_count = len(final_pruned)
            best_network = list(final_pruned)

    if best_network is None:
        best_network = odd_even_transposition()
        best_network = greedy_remove(best_network, time_limit=deadline - 1.0)

    # Final verification
    if not verify_network(best_network):
        fallback = odd_even_transposition()
        fallback = greedy_remove(fallback, time_limit=deadline - 1.0)
        return fallback

    return best_network
