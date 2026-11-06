"""Size-efficient sorting network on 18 wires: SA with bit-parallel verification and aggressive pruning."""
N = 18
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

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

        for k in range(N - 1):
            if np.any(wires[k] & ~wires[k + 1]):
                return False
        return True

    def verify_network_fast(comparators):
        """
        Even faster verification with early termination during the comparator phase.
        After each comparator, check if any adjacent pair violates sorted order.
        This is more expensive per step but can terminate early.
        """
        wires = [w.copy() for w in initial_wires]

        for (i, j) in comparators:
            wi = wires[i]
            wj = wires[j]
            wires[i] = wi & wj
            wires[j] = wi | wj

        # Final check
        for k in range(N - 1):
            if np.any(wires[k] & ~wires[k + 1]):
                return False
        return True

    def greedy_remove(comparators, time_limit=None):
        """
        Iteratively remove redundant comparators with aggressive pruning.
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
            # Try removing from random positions
            if n < 25:
                positions = list(range(n))
                random.shuffle(positions)
            else:
                positions = random.sample(range(n), min(25, n))

            for idx in positions:
                if idx >= len(current):
                    continue
                test = current[:idx] + current[idx + 1:]
                if verify_network(test):
                    current = test
                    changed = True
                    break

        return current

    def greedy_remove_thorough(comparators, time_limit=None):
        """
        More thorough greedy removal that tries multiple passes.
        """
        current = list(comparators)
        for _ in range(3):
            if time_limit and time.time() > time_limit:
                break
            current = greedy_remove(current, time_limit)
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

    def random_dense_network(num_comparators=200, seed=None):
        """Generate a random dense network."""
        if seed is not None:
            random.seed(seed)
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

    def simulated_annealing(initial_net, t_max=1000, t_min=0.5, alpha=0.998, max_iter=50000, deadline=None):
        """
        Simulated Annealing to minimize comparator count.
        """
        current = list(initial_net)
        current_energy = len(current)
        best = list(current)
        best_energy = current_energy
        
        t = t_max
        no_improve_count = 0
        best_no_improve = 0
        
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
            elif r < 0.80:
                # Swap
                if len(candidate) < 2:
                    continue
                i1, i2 = random.sample(range(len(candidate)), 2)
                candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
            elif r < 0.90:
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
            else:
                # Move: take a comparator and reinsert it at a different position
                if len(candidate) < 2:
                    continue
                idx = random.randint(0, len(candidate) - 1)
                comp = candidate.pop(idx)
                new_pos = random.randint(0, len(candidate))
                candidate.insert(new_pos, comp)

            if not verify_network(candidate):
                no_improve_count += 1
                continue

            new_energy = len(candidate)
            
            if new_energy < current_energy:
                current = candidate
                current_energy = new_energy
                no_improve_count = 0
                if new_energy < best_energy:
                    best = list(current)
                    best_energy = new_energy
                    best_no_improve = no_improve_count
            else:
                delta = new_energy - current_energy
                if delta > 0:
                    if random.random() < np.exp(-delta / max(t, 0.1)):
                        current = candidate
                        current_energy = new_energy
                # Same size: accept with some probability to explore
                elif random.random() < 0.3:
                    current = candidate

            t *= alpha
            if t < t_min:
                break
        
        return best

    # Generate diverse starting candidates
    candidates = []
    candidates.append(("odd_even", odd_even_transposition()))
    candidates.append(("bubble", bubble_sort_network()))
    
    # Random dense starts with varying sizes
    for i in range(6):
        candidates.append((f"random_{i}", random_dense_network(100 + i * 15, seed=200 + i * 7)))

    best_network = None
    best_count = float('inf')

    # Time budget per candidate
    time_per_candidate = 10.0
    elapsed = time.time() - start_time
    
    for name, net in candidates:
        if time.time() > deadline - 10.0:
            break
        
        # First, try to prune the initial network
        prune_time = min(deadline - 15.0, start_time + 2.0)
        pruned = greedy_remove(net, time_limit=prune_time)
        if len(pruned) < best_count:
            best_count = len(pruned)
            best_network = list(pruned)
        
        # Run SA from the pruned state
        sa_deadline = min(deadline - 10.0, start_time + time_per_candidate + 3.0)
        sa_result = simulated_annealing(
            pruned, 
            t_max=200, 
            t_min=0.3, 
            alpha=0.9985, 
            max_iter=30000, 
            deadline=sa_deadline
        )
        
        if len(sa_result) < best_count:
            best_count = len(sa_result)
            best_network = list(sa_result)
            
        # Final greedy removal on the SA result
        final_prune_deadline = min(deadline - 8.0, start_time + time_per_candidate + 5.0)
        final_pruned = greedy_remove_thorough(best_network, time_limit=final_prune_deadline)
        if len(final_pruned) < best_count:
            best_count = len(final_pruned)
            best_network = list(final_pruned)

    # If we have time left, do a final polishing pass on the best network
    if time.time() < deadline - 5.0 and best_network is not None:
        polish_deadline = deadline - 3.0
        polished = greedy_remove_thorough(best_network, time_limit=polish_deadline)
        if len(polished) < best_count:
            best_count = len(polished)
            best_network = list(polished)
        
        # One more SA round from the polished state with tighter parameters
        if time.time() < deadline - 4.0:
            sa2_deadline = deadline - 2.0
            sa2_result = simulated_annealing(
                polished,
                t_max=50,
                t_min=0.2,
                alpha=0.999,
                max_iter=10000,
                deadline=sa2_deadline
            )
            if len(sa2_result) < best_count:
                best_count = len(sa2_result)
                best_network = list(sa2_result)
            
            # Final final greedy
            if time.time() < deadline - 1.0:
                final_final = greedy_remove(best_network, time_limit=deadline - 0.5)
                if len(final_final) < best_count:
                    best_count = len(final_final)
                    best_network = list(final_final)

    if best_network is None:
        best_network = odd_even_transposition()
        best_network = greedy_remove(best_network, time_limit=deadline - 1.0)

    # Final verification
    if not verify_network(best_network):
        fallback = odd_even_transposition()
        fallback = greedy_remove(fallback, time_limit=deadline - 1.0)
        return fallback

    return best_network
