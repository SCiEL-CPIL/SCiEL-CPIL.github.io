import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 87.0  # safety margin before 90s

    n_max = 256

    # Precompute conflict masks: bit j set in mask[i] means i and j conflict
    conflict_mask = [0] * (n_max + 1)
    for a in range(1, n_max + 1):
        for b in range(a + 1, n_max + 1):
            if (a * b) % (a + b) == 0:
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)

    def popcount(x):
        return bin(x).count("1")

    def can_add(mask, x):
        return (mask & conflict_mask[x]) == 0

    # Greedy completion: add all possible elements
    def greedy_complete(mask):
        for i in range(1, n_max + 1):
            if not (mask & (1 << i)):
                if can_add(mask, i):
                    mask |= (1 << i)
        return mask

    # Multi-start greedy to find good initial solution
    all_elements = list(range(1, n_max + 1))
    evens = list(range(2, n_max + 1, 2))
    odds = list(range(1, n_max + 1, 2))

    random.seed(42)
    best_mask = 0
    best_size = 0

    # Start from all odds (known good base)
    base_mask = 0
    for i in odds:
        base_mask |= (1 << i)
    best_mask = greedy_complete(base_mask)
    best_size = popcount(best_mask)

    # Try multiple random greedy orderings
    for _ in range(80):
        cur = 0
        shuffled = all_elements[:]
        random.shuffle(shuffled)
        for e in shuffled:
            if can_add(cur, e):
                cur |= (1 << e)
        cur = greedy_complete(cur)
        sz = popcount(cur)
        if sz > best_size:
            best_size = sz
            best_mask = cur

    # Also try starting from random subsets of various sizes
    for trial_size in [64, 96, 128, 160]:
        for _ in range(20):
            cur = 0
            sample = random.sample(all_elements, min(trial_size, n_max))
            for e in sample:
                if can_add(cur, e):
                    cur |= (1 << e)
            cur = greedy_complete(cur)
            sz = popcount(cur)
            if sz > best_size:
                best_size = sz
                best_mask = cur

    # Simulated Annealing
    current_mask = best_mask
    current_size = best_size

    T = 5.0
    T_min = 0.001
    cooling_rate = 0.998
    steps = 0

    # Precompute element lists for faster random selection
    while time.time() < deadline - 3.0 and T > T_min:
        steps += 1
        r = random.random()
        new_mask = current_mask
        delta = 0

        if r < 0.35:
            # Add a random element
            x = random.randint(1, n_max)
            if not (current_mask & (1 << x)):
                if can_add(current_mask, x):
                    new_mask = current_mask | (1 << x)
                    delta = 1
                else:
                    # Try to find a valid addition by scanning
                    added = False
                    for i in range(1, n_max + 1):
                        if not (current_mask & (1 << i)) and can_add(current_mask, i):
                            new_mask = current_mask | (1 << i)
                            delta = 1
                            added = True
                            break
                    if not added:
                        # No valid add, do a swap instead
                        x_in = random.randint(1, n_max)
                        if current_mask & (1 << x_in):
                            y = random.randint(1, n_max)
                            if not (current_mask & (1 << y)):
                                mask_without = current_mask & ~(1 << x_in)
                                if can_add(mask_without, y):
                                    new_mask = mask_without | (1 << y)
                                    delta = 0
        elif r < 0.65:
            # Remove a random element
            x = random.randint(1, n_max)
            if current_mask & (1 << x):
                new_mask = current_mask & ~(1 << x)
                delta = -1
        else:
            # Swap: remove x, add y
            x = random.randint(1, n_max)
            if current_mask & (1 << x):
                y = random.randint(1, n_max)
                if not (current_mask & (1 << y)):
                    mask_without = current_mask & ~(1 << x)
                    if can_add(mask_without, y):
                        new_mask = mask_without | (1 << y)
                        delta = 0

        new_size = popcount(new_mask)

        if new_size > current_size:
            current_mask = new_mask
            current_size = new_size
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
        elif new_size < current_size:
            prob = math.exp(delta / T)
            if random.random() < prob:
                current_mask = new_mask
                current_size = new_size
        else:
            # Same size: accept to explore
            current_mask = new_mask

        T *= cooling_rate

        # Periodic greedy completion
        if steps % 50 == 0:
            completed = greedy_complete(current_mask)
            if popcount(completed) > current_size:
                current_mask = completed
                current_size = popcount(current_mask)
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask

    # Final local search: 2-opt swaps on best_mask
    final_mask = best_mask
    final_size = best_size
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        members = [i for i in range(1, n_max + 1) if final_mask & (1 << i)]
        non_members = [i for i in range(1, n_max + 1) if not (final_mask & (1 << i))]
        for a in members:
            if time.time() > deadline - 2.0:
                break
            for b in non_members:
                mask_without_a = final_mask & ~(1 << a)
                if can_add(mask_without_a, b):
                    new_mask = mask_without_a | (1 << b)
                    new_size = popcount(new_mask)
                    if new_size > final_size:
                        final_mask = new_mask
                        final_size = new_size
                        improved = True
                        break
            if improved:
                break

    # Final greedy completion
    final_mask = greedy_complete(final_mask)
    final_size = popcount(final_mask)

    # Convert to sorted list
    result = []
    for i in range(1, n_max + 1):
        if final_mask & (1 << i):
            result.append(i)

    return result
