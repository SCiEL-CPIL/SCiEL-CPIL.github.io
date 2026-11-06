"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    N = 256
    squares = set()
    for k in range(2, 23):
        squares.add(k * k)

    individually_invalid = set()
    for a in range(1, N + 1):
        if 2 * a in squares:
            individually_invalid.add(a)

    valid_nodes = [i for i in range(1, N + 1) if i not in individually_invalid]
    valid_set = set(valid_nodes)
    n = len(valid_nodes)
    node_to_idx = {node: i for i, node in enumerate(valid_nodes)}

    # Build conflict bitsets
    conflict_bits = [0] * n
    for i, node in enumerate(valid_nodes):
        for s in squares:
            j = s - node
            if 1 <= j <= N and j != node and j in valid_set:
                j_idx = node_to_idx[j]
                conflict_bits[i] |= (1 << j_idx)

    def greedy_min_degree():
        """Greedy min-degree independent set."""
        remaining = (1 << n) - 1
        selected_bits = 0
        while remaining:
            best_node = -1
            best_degree = n + 1
            for i in range(n):
                if (remaining >> i) & 1:
                    deg = bin(conflict_bits[i] & remaining).count('1')
                    if deg < best_degree:
                        best_degree = deg
                        best_node = i
            selected_bits |= (1 << best_node)
            remaining &= ~(1 << best_node)
            remaining &= ~conflict_bits[best_node]
        return selected_bits

    def greedy_add_pass(bits, size, time_limit):
        """Repeatedly add any non-conflicting node until no more can be added."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(n):
                if (bits >> i) & 1:
                    continue
                if conflict_bits[i] & bits:
                    continue
                bits |= (1 << i)
                size += 1
                improved = True
        return bits, size

    def fast_2swap(bits, size, time_limit):
        """Try 1-in/2-out and 1-in/1-out swaps using bitsets for speed."""
        for _ in range(50):
            if time.time() > time_limit:
                break
            improved = False
            unselected_mask = ((1 << n) - 1) & ~bits
            # Iterate over unselected nodes
            um = unselected_mask
            while um:
                a = (um & -um).bit_length() - 1
                um &= um - 1
                if (bits >> a) & 1:
                    continue
                conf_with_a = conflict_bits[a] & bits
                conf_count = bin(conf_with_a).count('1')
                if conf_count == 0:
                    # Can just add
                    bits |= (1 << a)
                    size += 1
                    improved = True
                    continue
                if conf_count == 1:
                    # 1-in/1-out swap
                    b = (conf_with_a & -conf_with_a).bit_length() - 1
                    candidate = (bits & ~(1 << b)) | (1 << a)
                    # Try greedy add
                    temp, temp_size = greedy_add_pass(candidate, size, time.time() + 0.05)
                    if temp_size > size:
                        bits = temp
                        size = temp_size
                        improved = True
                        continue
                elif conf_count == 2:
                    # 1-in/2-out swap
                    # Get the two conflicting nodes
                    c1 = (conf_with_a & -conf_with_a).bit_length() - 1
                    rest = conf_with_a & ~(1 << c1)
                    c2 = (rest & -rest).bit_length() - 1
                    removed = (1 << c1) | (1 << c2)
                    candidate = (bits & ~removed) | (1 << a)
                    temp, temp_size = greedy_add_pass(candidate, size - 1, time.time() + 0.05)
                    if temp_size > size:
                        bits = temp
                        size = temp_size
                        improved = True
                        continue
                elif conf_count == 3:
                    # Try removing 2 of the 3 conflicting
                    c1 = (conf_with_a & -conf_with_a).bit_length() - 1
                    rest1 = conf_with_a & ~(1 << c1)
                    c2 = (rest1 & -rest1).bit_length() - 1
                    rest2 = rest1 & ~(1 << c2)
                    c3 = (rest2 & -rest2).bit_length() - 1
                    # Try each pair
                    for pair in [(c1, c2), (c1, c3), (c2, c3)]:
                        removed = (1 << pair[0]) | (1 << pair[1])
                        candidate = (bits & ~removed) | (1 << a)
                        temp, temp_size = greedy_add_pass(candidate, size - 1, time.time() + 0.03)
                        if temp_size > size:
                            bits = temp
                            size = temp_size
                            improved = True
                            break
                    if improved:
                        continue
            if not improved:
                break
        return bits, size

    # Multiple greedy starts with different orderings
    best_solution_bits = greedy_min_degree()
    best_size = bin(best_solution_bits).count('1')

    # Greedy by natural order
    remaining = (1 << n) - 1
    selected_bits = 0
    for i in range(n):
        if (remaining >> i) & 1 and not (conflict_bits[i] & selected_bits):
            selected_bits |= (1 << i)
            remaining &= ~conflict_bits[i]
    if bin(selected_bits).count('1') > best_size:
        best_solution_bits = selected_bits
        best_size = bin(selected_bits).count('1')

    # Greedy by reverse order
    remaining = (1 << n) - 1
    selected_bits = 0
    for i in range(n - 1, -1, -1):
        if (remaining >> i) & 1 and not (conflict_bits[i] & selected_bits):
            selected_bits |= (1 << i)
            remaining &= ~conflict_bits[i]
    if bin(selected_bits).count('1') > best_size:
        best_solution_bits = selected_bits
        best_size = bin(selected_bits).count('1')

    # Greedy by degree (ascending)
    degrees = [bin(conflict_bits[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda x: degrees[x])
    remaining = (1 << n) - 1
    selected_bits = 0
    for i in order:
        if (remaining >> i) & 1 and not (conflict_bits[i] & selected_bits):
            selected_bits |= (1 << i)
            remaining &= ~conflict_bits[i]
    if bin(selected_bits).count('1') > best_size:
        best_solution_bits = selected_bits
        best_size = bin(selected_bits).count('1')

    # Improve best with greedy add
    best_solution_bits, best_size = greedy_add_pass(best_solution_bits, best_size, deadline - 15.0)

    # Improve with 2-swap local search
    best_solution_bits, best_size = fast_2swap(best_solution_bits, best_size, deadline - 5.0)

    # Final greedy add pass
    best_solution_bits, best_size = greedy_add_pass(best_solution_bits, best_size, deadline - 2.0)

    # Extract result
    result = []
    for i in range(n):
        if (best_solution_bits >> i) & 1:
            result.append(valid_nodes[i])

    result.sort()

    # Verify
    valid = True
    for i in range(len(result)):
        for j in range(i, len(result)):
            s = result[i] + result[j]
            r = math.isqrt(s)
            if r * r == s:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback: use a simple valid set
        result = [i for i in range(1, N + 1) if 2 * i not in squares and (i % 4 == 1 or i % 4 == 3)]
        # Verify this fallback
        valid2 = True
        for i in range(len(result)):
            for j in range(i, len(result)):
                s = result[i] + result[j]
                r = math.isqrt(s)
                if r * r == s:
                    valid2 = False
                    break
            if not valid2:
                break
        if not valid2 or not result:
            result = [1]

    return result
# EVOLVE-BLOCK-END
