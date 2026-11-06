import time
import random
import math
import numpy as np

N = 17
NUM_INPUTS = 1 << N
NUM_WORDS = (NUM_INPUTS + 63) // 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 48.0  # Safety margin for 60s limit (evaluator kills at 90s, but let's be safe)

    # 1. Generate the seed network: Odd-Even Transposition
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))

    # 2. Precompute initial bitmasks as numpy uint64 arrays
    def build_initial_masks():
        masks = []
        for j in range(N):
            arr = np.zeros(NUM_WORDS, dtype=np.uint64)
            for w in range(NUM_WORDS):
                word = 0
                base = w * 64
                for bit in range(64):
                    k = base + bit
                    if k < NUM_INPUTS and ((k >> j) & 1):
                        word |= (1 << bit)
                arr[w] = word
            masks.append(arr)
        return masks

    initial_masks = build_initial_masks()

    # Cache for is_valid to avoid recomputation
    valid_cache = {}

    def is_valid(network):
        """Check if network sorts all boolean inputs using numpy vectorization."""
        key = tuple(network)
        if key in valid_cache:
            return valid_cache[key]
        
        state = [m.copy() for m in initial_masks]
        for i, j in network:
            new_i = np.bitwise_and(state[i], state[j])
            new_j = np.bitwise_or(state[i], state[j])
            state[i] = new_i
            state[j] = new_j
        
        # Check sorted: for each adjacent pair, state[k] & ~state[k+1] should be 0
        for k in range(N - 1):
            if np.any(np.bitwise_and(state[k], np.invert(state[k + 1]))):
                valid_cache[key] = False
                return False
        
        valid_cache[key] = True
        return True

    def clear_cache():
        valid_cache.clear()

    best_network = seed_network[:]
    if not is_valid(best_network):
        return best_network

    # Greedy removal with numpy verification
    def try_remove_all(current_net, max_rounds=3):
        if time.time() > deadline:
            return current_net
        removed = True
        rounds = 0
        while removed and rounds < max_rounds:
            if time.time() > deadline:
                break
            removed = False
            rounds += 1
            n = len(current_net)
            indices = list(range(n))
            random.shuffle(indices)
            for idx in indices:
                if time.time() > deadline:
                    break
                candidate = current_net[:idx] + current_net[idx + 1:]
                if is_valid(candidate):
                    current_net = candidate
                    removed = True
                    break
        return current_net

    # Compute depth of each comparator in the network
    def compute_depths(network):
        depth = [0] * N
        depths = []
        for i, j in network:
            d = max(depth[i], depth[j]) + 1
            depths.append(d)
            depth[i] = d
            depth[j] = d
        return depths

    # Multi-comparator insertion: insert 2-3 comparators at once
    def multi_insert(current_net, num_new=2):
        n = len(current_net)
        candidate = current_net[:]
        depths = compute_depths(candidate)

        for _ in range(num_new):
            if time.time() > deadline:
                return candidate
            # Choose distance with bias toward medium-long
            r = random.random()
            if r < 0.2:
                dist = random.randint(2, 4)
            elif r < 0.5:
                dist = random.randint(5, 8)
            elif r < 0.8:
                dist = random.randint(9, 12)
            else:
                dist = random.randint(13, 16)

            new_i = random.randint(0, N - 1 - dist)
            new_j = new_i + dist

            # Depth-aware position
            if candidate:
                pos = random.randint(0, len(candidate))
                if random.random() < 0.3:
                    pos = random.randint(max(0, len(candidate) - 5), len(candidate))
            else:
                pos = 0

            candidate.insert(pos, (new_i, new_j))

        return candidate

    # Simulated annealing with multi-insertion
    def anneal_with_multi_insert(current_net, seed_val, time_limit):
        rng = random.Random(seed_val)
        t_start = time.time()
        temperature = 6.0
        cooling_rate = 0.9985
        min_temp = 0.01
        best_local = current_net[:]
        best_len = len(best_local)

        while time.time() - t_start < time_limit and temperature > min_temp:
            if time.time() > deadline:
                break

            neighbor = current_net[:]
            n = len(neighbor)
            strategy = rng.random()

            if strategy < 0.35:
                # Multi-comparator insertion (2-3 at once)
                num_new = rng.choice([2, 2, 3])
                neighbor = multi_insert(current_net, num_new)
                neighbor = try_remove_all(neighbor, max_rounds=1)

            elif strategy < 0.50:
                # Single add + optimize
                r = rng.random()
                if r < 0.3:
                    dist = rng.randint(2, 5)
                elif r < 0.6:
                    dist = rng.randint(6, 9)
                else:
                    dist = rng.randint(10, 16)
                new_i = rng.randint(0, N - 1 - dist)
                new_j = new_i + dist
                pos = rng.randint(0, len(neighbor))
                neighbor.insert(pos, (new_i, new_j))
                neighbor = try_remove_all(neighbor, max_rounds=1)

            elif strategy < 0.65:
                # Remove a random comparator
                if n > 1:
                    idx = rng.randint(0, n - 1)
                    neighbor.pop(idx)

            elif strategy < 0.78:
                # Move a comparator to a different position
                if n > 2:
                    i = rng.randint(0, n - 1)
                    j = rng.randint(0, n - 1)
                    if i != j:
                        comp = neighbor.pop(i)
                        if i < j:
                            j -= 1
                        neighbor.insert(j, comp)

            elif strategy < 0.88:
                # Replace a comparator with a nearby wire pair
                if n > 0:
                    i = rng.randint(0, n - 1)
                    old_i, old_j = neighbor[i]
                    new_i = max(0, min(N - 2, old_i + rng.randint(-3, 3)))
                    new_j = max(new_i + 1, min(N - 1, old_j + rng.randint(-2, 2)))
                    if new_i < new_j:
                        neighbor[i] = (new_i, new_j)

            else:
                # Reorder a segment
                if n > 3:
                    start = rng.randint(0, n - 4)
                    end = rng.randint(start + 2, n - 1)
                    segment = neighbor[start:end + 1]
                    rng.shuffle(segment)
                    neighbor[start:end + 1] = segment

            # Check validity
            if not is_valid(neighbor):
                temperature *= cooling_rate
                continue

            # Light cleanup
            neighbor = try_remove_all(neighbor, max_rounds=1)

            # Acceptance decision
            current_cost = len(current_net)
            neighbor_cost = len(neighbor)
            delta = neighbor_cost - current_cost

            if delta <= 0:
                current_net = neighbor
                if len(current_net) < best_len:
                    best_local = current_net[:]
                    best_len = len(best_local)
            else:
                if temperature > 0:
                    prob = math.exp(-delta / temperature)
                    if rng.random() < prob:
                        current_net = neighbor

            temperature *= cooling_rate

        return best_local

    # Phase 1: Initial greedy removal
    current_network = try_remove_all(best_network, max_rounds=5)
    if len(current_network) < len(best_network):
        best_network = current_network[:]

    # Phase 2: Multiple annealing runs with different seeds
    remaining = deadline - time.time()
    num_runs = 4
    per_run_time = remaining * 0.8 / num_runs

    for run in range(num_runs):
        if time.time() > deadline:
            break
        seed_val = 1000 + run * 7919
        start_net = best_network[:]
        if run > 0 and len(best_network) > 10:
            n = len(start_net)
            for _ in range(min(3, n // 5)):
                idx = random.randint(0, n - 1)
                start_net.pop(idx)
                n -= 1
            if not is_valid(start_net):
                start_net = best_network[:]

        result = anneal_with_multi_insert(start_net, seed_val, per_run_time)
        if len(result) < len(best_network):
            best_network = result[:]

    # Phase 3: Final aggressive cleanup
    if time.time() < deadline:
        for _ in range(10):
            if time.time() > deadline:
                break
            final_network = try_remove_all(best_network, max_rounds=5)
            if len(final_network) < len(best_network):
                best_network = final_network[:]

    return best_network
# EVOLVE-BLOCK-END
