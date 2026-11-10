import time
import random
import sys

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 57.0  # Safety margin before 60s limit

    # Precompute all 4-APs
    # A 4-AP is (a, a+d, a+2d, a+3d) for d >= 1, a >= 1, a+3d <= N
    all_aps = []
    max_d = (N - 1) // 3
    for d in range(1, max_d + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            all_aps.append((a, a + d, a + 2 * d, a + 3 * d))

    num_aps = len(all_aps)

    # For each element x, store the other elements in each 4-AP containing x
    # as a bitmask for fast checking
    # constraints[x] = list of masks, where mask has bits set for the other 3 elements
    constraints = [[] for _ in range(N + 1)]
    for ap in all_aps:
        a, b, c, d = ap
        # For element a: others are b, c, d
        constraints[a].append((1 << (b - 1)) | (1 << (c - 1)) | (1 << (d - 1)))
        # For element b: others are a, c, d
        constraints[b].append((1 << (a - 1)) | (1 << (c - 1)) | (1 << (d - 1)))
        # For element c: others are a, b, d
        constraints[c].append((1 << (a - 1)) | (1 << (b - 1)) | (1 << (d - 1)))
        # For element d: others are a, b, c
        constraints[d].append((1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1)))

    def can_add(x, current_mask):
        """Check if x can be added to the set represented by current_mask"""
        for mask in constraints[x]:
            if (current_mask & mask) == mask:
                return False
        return True

    def greedy_fill(order, current_mask):
        """Greedily add elements from order that can be added"""
        for x in order:
            if not (current_mask & (1 << (x - 1))):
                if can_add(x, current_mask):
                    current_mask |= (1 << (x - 1))
        return current_mask

    def count_bits(mask):
        """Count number of set bits in mask"""
        count = 0
        while mask:
            mask &= mask - 1
            count += 1
        return count

    # Generate diverse initial solutions
    initial_solutions = []

    # 1. Base-4 {0,1} construction
    base4_mask = 0
    for i in range(1, N + 1):
        n = i
        valid = True
        while n > 0:
            if n % 4 > 1:
                valid = False
                break
            n //= 4
        if valid:
            base4_mask |= (1 << (i - 1))
    initial_solutions.append(base4_mask)

    # 2. Greedy from low to high
    mask = 0
    for x in range(1, N + 1):
        if can_add(x, mask):
            mask |= (1 << (x - 1))
    initial_solutions.append(mask)

    # 3. Greedy from high to low
    mask = 0
    for x in range(N, 0, -1):
        if can_add(x, mask):
            mask |= (1 << (x - 1))
    initial_solutions.append(mask)

    # 4. Greedy starting from base-4, then add more
    mask = base4_mask
    for x in range(1, N + 1):
        if not (mask & (1 << (x - 1))):
            if can_add(x, mask):
                mask |= (1 << (x - 1))
    initial_solutions.append(mask)

    # 5. Randomized greedy with multiple seeds
    for seed in range(50):
        if time.time() > deadline - 20:
            break
        rng = random.Random(seed)
        order = list(range(1, N + 1))
        rng.shuffle(order)
        mask = 0
        for x in order:
            if can_add(x, mask):
                mask |= (1 << (x - 1))
        initial_solutions.append(mask)

    # Find best initial solution
    best_mask = max(initial_solutions, key=count_bits)
    best_size = count_bits(best_mask)

    # Local search: 1-exchange (remove one, greedy refill)
    all_elements = list(range(1, N + 1))
    rng = random.Random(12345)

    for iteration in range(50000):
        if time.time() > deadline - 5:
            break

        current_mask = best_mask

        # Pick a random element to remove
        elements_in_set = []
        m = current_mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            elements_in_set.append(idx + 1)
            m ^= lsb

        if not elements_in_set:
            break

        remove_elem = rng.choice(elements_in_set)
        current_mask &= ~(1 << (remove_elem - 1))

        # Greedy refill: try to add all possible elements
        for c in all_elements:
            if not (current_mask & (1 << (c - 1))):
                if can_add(c, current_mask):
                    current_mask |= (1 << (c - 1))

        new_size = count_bits(current_mask)

        if new_size > best_size:
            best_size = new_size
            best_mask = current_mask

    # Local search: 2-exchange (remove two, greedy refill)
    for iteration in range(30000):
        if time.time() > deadline - 3:
            break

        current_mask = best_mask

        # Pick two random elements to remove
        elements_in_set = []
        m = current_mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            elements_in_set.append(idx + 1)
            m ^= lsb

        if len(elements_in_set) < 2:
            break

        rem1, rem2 = rng.sample(elements_in_set, 2)
        current_mask &= ~(1 << (rem1 - 1))
        current_mask &= ~(1 << (rem2 - 1))

        # Greedy refill
        for c in all_elements:
            if not (current_mask & (1 << (c - 1))):
                if can_add(c, current_mask):
                    current_mask |= (1 << (c - 1))

        new_size = count_bits(current_mask)

        if new_size > best_size:
            best_size = new_size
            best_mask = current_mask

    # Simulated annealing style: accept equal or slightly worse moves
    current_mask = best_mask
    current_size = best_size
    temperature = 2.0
    cooling_rate = 0.9995

    for sa_iter in range(100000):
        if time.time() > deadline - 1:
            break

        elements_in_set = []
        m = current_mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            elements_in_set.append(idx + 1)
            m ^= lsb

        if not elements_in_set:
            break

        remove_elem = rng.choice(elements_in_set)
        trial_mask = current_mask & ~(1 << (remove_elem - 1))

        # Greedy refill
        for c in all_elements:
            if not (trial_mask & (1 << (c - 1))):
                if can_add(c, trial_mask):
                    trial_mask |= (1 << (c - 1))

        new_size = count_bits(trial_mask)
        delta = new_size - current_size

        if delta >= 0 or rng.random() < temperature:
            current_mask = trial_mask
            current_size = new_size
            if new_size > best_size:
                best_size = new_size
                best_mask = current_mask

        temperature *= cooling_rate

    # Convert best_mask to sorted list
    result = []
    m = best_mask
    while m:
        lsb = m & (-m)
        idx = lsb.bit_length() - 1
        result.append(idx + 1)
        m ^= lsb

    return sorted(result)
