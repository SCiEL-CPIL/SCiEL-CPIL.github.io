import time
import random
import math

N = 17
NUM_INPUTS = 1 << N
FULL_MASK = (1 << NUM_INPUTS) - 1

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin (90s wall-clock limit)

    # 1. Generate the seed network: Odd-Even Transposition
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))

    # 2. Precompute initial state masks (bit-vector over all 2^N inputs)
    #    state[j] is a bitmask of length 2^N; bit k = 1 if input k has a 1 on wire j
    initial_state = [0] * N
    for j in range(N):
        mask = 0
        for k in range(NUM_INPUTS):
            if (k >> j) & 1:
                mask |= (1 << k)
        initial_state[j] = mask

    # 3. Fast validity check: simulate full network, then check sorted order
    def is_valid(network):
        state = initial_state[:]
        for i, j in network:
            a = state[i]
            b = state[j]
            state[i] = a & b
            state[j] = a | b
        # Sorted iff state[k] ⊆ state[k+1] for all k
        for k in range(N - 1):
            if (state[k] & ~state[k + 1]) != 0:
                return False
        return True

    # 4. Prefix validity check: simulate only network[:pos], return state.
    #    Used to cheaply evaluate mutations that only change network[pos:].
    def simulate_prefix(network, pos):
        state = initial_state[:]
        for i in range(pos):
            a_i, a_j = network[i]
            a = state[a_i]
            b = state[a_j]
            state[a_i] = a & b
            state[a_j] = a | b
        return state

    # 5. Full removal pass: repeatedly try removing each comparator
    #    Using full is_valid (safe, correct).
    def try_remove_all(network, time_limit):
        removed = True
        while removed and time.time() < time_limit:
            removed = False
            n = len(network)
            indices = list(range(n))
            random.shuffle(indices)
            for idx in indices:
                if time.time() > time_limit:
                    break
                candidate = network[:idx] + network[idx + 1:]
                if is_valid(candidate):
                    network = candidate
                    removed = True
                    break
        return network

    # 6. Cheap removal pass using prefix checks:
    #    For position idx, simulate prefix up to idx, then check if removing
    #    network[idx] keeps the suffix valid. This avoids re-simulating prefix.
    def try_remove_all_fast(network, time_limit):
        removed = True
        while removed and time.time() < time_limit:
            removed = False
            n = len(network)
            indices = list(range(n))
            random.shuffle(indices)
            for idx in indices:
                if time.time() > time_limit:
                    break
                # Simulate prefix up to idx
                state = simulate_prefix(network, idx)
                # Apply comparator at idx
                i, j = network[idx]
                a = state[i]
                b = state[j]
                new_i = a & b
                new_j = a | b
                # Now simulate suffix (idx+1 .. end) with the comparator applied
                # (i.e., as if the comparator IS present, to get final state)
                # But we want to check if it's valid WITHOUT the comparator.
                # So: simulate suffix starting from state with comparator NOT applied.
                # That means we just simulate network[idx+1:] from `state`.
                state2 = state[:]
                for k in range(idx + 1, n):
                    c_i, c_j = network[k]
                    a2 = state2[c_i]
                    b2 = state2[c_j]
                    state2[c_i] = a2 & b2
                    state2[c_j] = a2 | b2
                # Check sorted
                valid = True
                for k in range(N - 1):
                    if (state2[k] & ~state2[k + 1]) != 0:
                        valid = False
                        break
                if valid:
                    network = network[:idx] + network[idx + 1:]
                    removed = True
                    break
        return network

    # 7. Long-distance insertion: insert a comparator with large j-i at a random position
    def insert_long_distance(network, time_limit):
        # Prefer long-distance comparators
        candidates = []
        for i in range(N):
            for j in range(i + 1, N):
                dist = j - i
                # Weight by distance: long-distance comparators are more useful
                weight = dist * dist
                for _ in range(weight):
                    candidates.append((i, j))
        if not candidates:
            return network
        new_comp = random.choice(candidates)
        pos = random.randint(0, len(network))
        candidate = network[:pos] + [new_comp] + network[pos:]
        if is_valid(candidate):
            return candidate
        return None

    # 8. Grow and shrink: add random comparators, then remove redundancies
    def grow_and_shrink(network, num_adds, time_limit):
        candidate = network[:]
        for _ in range(num_adds):
            new_i = random.randint(0, N - 1)
            new_j = random.randint(0, N - 1)
            if new_i > new_j:
                new_i, new_j = new_j, new_i
            if new_i == new_j:
                continue
            pos = random.randint(0, len(candidate))
            candidate.insert(pos, (new_i, new_j))
        if not is_valid(candidate):
            return None
        return try_remove_all_fast(candidate, time_limit)

    # 9. Targeted swap: replace one comparator with a long-distance one
    def targeted_swap(network, time_limit):
        if len(network) == 0:
            return None
        idx = random.randint(0, len(network) - 1)
        old_i, old_j = network[idx]
        # Generate long-distance replacement candidates
        candidates = []
        for new_i in range(N):
            for new_j in range(new_i + 1, N):
                if (new_i, new_j) == (old_i, old_j):
                    continue
                dist = new_j - new_i
                weight = dist * dist
                for _ in range(weight):
                    candidates.append((new_i, new_j))
        random.shuffle(candidates)
        for new_i, new_j in candidates[:15]:
            if time.time() > time_limit:
                return None
            candidate = network[:idx] + [(new_i, new_j)] + network[idx + 1:]
            if is_valid(candidate):
                return try_remove_all_fast(candidate, time_limit)
        return None

    # 10. Multi-removal: try removing 2 comparators at once
    def try_remove_two(network, time_limit):
        if len(network) < 2:
            return None
        i1 = random.randint(0, len(network) - 1)
        i2 = random.randint(0, len(network) - 2)
        if i2 >= i1:
            i2 += 1
        if i1 < i2:
            candidate = network[:i1] + network[i1 + 1:i2] + network[i2 + 1:]
        else:
            candidate = network[:i2] + network[i2 + 1:i1] + network[i1 + 1:]
        if is_valid(candidate):
            return try_remove_all_fast(candidate, time_limit)
        return None

    # 11. Mutation dispatcher
    def mutate(network, time_limit):
        r = random.random()
        if r < 0.30:
            # Long-distance insertion (key improvement)
            return insert_long_distance(network, time_limit)
        elif r < 0.50:
            # Grow and shrink
            num_adds = random.choice([1, 1, 2, 2, 3])
            return grow_and_shrink(network, num_adds, time_limit)
        elif r < 0.70:
            # Targeted swap with long-distance preference
            return targeted_swap(network, time_limit)
        elif r < 0.85:
            # Multi-removal
            return try_remove_two(network, time_limit)
        else:
            # Grow by 2, then shrink
            return grow_and_shrink(network, 2, time_limit)

    # 12. Main search
    random.seed(42)
    current_network = seed_network[:]
    current_cost = len(current_network)
    best_network = current_network[:]
    best_cost = current_cost

    T = 5.0
    T_min = 0.01
    cooling_rate = 0.995

    # First, aggressively reduce the seed network
    current_network = try_remove_all_fast(current_network, deadline)
    current_cost = len(current_network)
    if current_cost < best_cost:
        best_network = current_network[:]
        best_cost = current_cost

    iteration = 0
    while time.time() < deadline:
        iteration += 1
        sub_deadline = min(deadline, time.time() + 0.5)  # Per-iteration budget

        # 1. Aggressively reduce current network
        optimized = try_remove_all_fast(current_network, sub_deadline)
        if len(optimized) < current_cost:
            current_network = optimized
            current_cost = len(optimized)
            if current_cost < best_cost:
                best_network = current_network[:]
                best_cost = current_cost

        if time.time() > deadline:
            break

        # 2. Mutation
        new_net = mutate(current_network, sub_deadline)
        if new_net is not None:
            new_cost = len(new_net)
            delta = new_cost - current_cost
            if delta <= 0:
                accept = True
            else:
                accept_prob = math.exp(-delta / T)
                accept = random.random() < accept_prob
            if accept:
                current_network = new_net
                current_cost = new_cost
                if current_cost < best_cost:
                    best_network = current_network[:]
                    best_cost = current_cost

        # 3. Cool down
        T = max(T * cooling_rate, T_min)

    return best_network
# EVOLVE-BLOCK-END
