import time
import random
import numpy as np

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s buffer for 90s limit

    # Precompute all 4-APs and their element indices (0-indexed)
    ap_list = []  # list of tuples of 4 indices
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            ap_list.append((a - 1, b - 1, c - 1, e - 1))

    num_aps = len(ap_list)

    # For each element, store list of AP indices it belongs to
    elem_to_aps = [[] for _ in range(N)]
    for i, ap in enumerate(ap_list):
        for idx in ap:
            elem_to_aps[idx].append(i)

    # For each AP, store the 4 elements
    # ap_elements[i] = (e1, e2, e3, e4)

    # Conflict: if we have elements x, y, z from an AP, we cannot add the 4th
    # For each element x, and each AP containing x, the other 3 elements form a "conflict set"
    # If all 3 are in S, we cannot add x.

    # Build conflict masks using numpy for speed
    # conflicts[x] = list of bitmasks (as integers) representing the other 3 elements
    conflicts = [[] for _ in range(N)]
    for i, ap in enumerate(ap_list):
        for pos in range(4):
            x = ap[pos]
            others = [ap[j] for j in range(4) if j != pos]
            mask = 0
            for o in others:
                mask |= (1 << o)
            conflicts[x].append(mask)

    def can_add(x, S):
        """Check if element x can be added to set S without creating a 4-AP."""
        for mask in conflicts[x]:
            if (S & mask) == mask:
                return False
        return True

    def count_bits(S):
        return bin(S).count('1')

    def greedy_fill(S, order):
        """Add elements from order to S if they don't create a 4-AP."""
        for x in order:
            if not (S & (1 << x)):
                if can_add(x, S):
                    S |= (1 << x)
        return S

    # Initial greedy constructions
    best_S = 0
    best_size = 0

    # Try natural order
    base_order = list(range(N))
    S_try = greedy_fill(0, base_order)
    size_try = count_bits(S_try)
    if size_try > best_size:
        best_size = size_try
        best_S = S_try

    # Try several random orders
    for _ in range(100):
        random.shuffle(base_order)
        S_try = greedy_fill(0, base_order)
        size_try = count_bits(S_try)
        if size_try > best_size:
            best_size = size_try
            best_S = S_try

    current_S = best_S
    current_size = best_size

    # Iterated Local Search with swap moves
    # A swap move: remove one element from S, then greedily fill to see if we get a larger set
    # Also try periodic full passes

    iterations = 0
    while time.time() < deadline:
        iterations += 1

        # Determine perturbation strategy
        # Option 1: Remove k random elements and refill
        # Option 2: Try a swap move (remove one, try to add others)
        # Option 3: Periodic full greedy pass in random order

        strategy = random.random()
        if strategy < 0.5:
            # Remove k random elements and refill
            k = random.randint(1, 5)
            elements_in_set = [i for i in range(N) if (current_S & (1 << i))]
            if len(elements_in_set) < k:
                k = max(1, len(elements_in_set))
            if k == 0:
                continue

            removed = random.sample(elements_in_set, k)
            S_temp = current_S
            for idx in removed:
                S_temp &= ~(1 << idx)

            # Run multiple greedy fills
            best_local_S = S_temp
            best_local_size = count_bits(S_temp)

            num_fills = 15
            for _ in range(num_fills):
                order = list(range(N))
                random.shuffle(order)
                S_filled = greedy_fill(S_temp, order)
                size_filled = count_bits(S_filled)
                if size_filled > best_local_size:
                    best_local_size = size_filled
                    best_local_S = S_filled

            if best_local_size > best_size:
                best_size = best_local_size
                best_S = best_local_S

            current_S = best_local_S
            current_size = best_local_size

        elif strategy < 0.8:
            # Swap move: remove one element, try to add elements that were previously blocked
            elements_in_set = [i for i in range(N) if (current_S & (1 << i))]
            if not elements_in_set:
                continue

            # Pick a random element to remove
            rem_idx = random.choice(elements_in_set)
            S_temp = current_S & ~(1 << rem_idx)

            # Now try to add any element that can be added
            # First, try a greedy fill in random order
            best_local_S = S_temp
            best_local_size = count_bits(S_temp)

            num_fills = 10
            for _ in range(num_fills):
                order = list(range(N))
                random.shuffle(order)
                S_filled = greedy_fill(S_temp, order)
                size_filled = count_bits(S_filled)
                if size_filled > best_local_size:
                    best_local_size = size_filled
                    best_local_S = S_filled

            if best_local_size > best_size:
                best_size = best_local_size
                best_S = best_local_S

            current_S = best_local_S
            current_size = best_local_size

        else:
            # Periodic full greedy pass: try to add any element that can be added,
            # in a random order, multiple times
            S_temp = current_S
            best_local_S = S_temp
            best_local_size = count_bits(S_temp)

            for _ in range(20):
                order = list(range(N))
                random.shuffle(order)
                S_filled = greedy_fill(S_temp, order)
                size_filled = count_bits(S_filled)
                if size_filled > best_local_size:
                    best_local_size = size_filled
                    best_local_S = S_filled
                    S_temp = S_filled  # Continue from improved state

            if best_local_size > best_size:
                best_size = best_local_size
                best_S = best_local_S

            current_S = best_local_S
            current_size = best_local_size

    # Return the best set found
    result = []
    for i in range(N):
        if best_S & (1 << i):
            result.append(i + 1)

    return sorted(result)
