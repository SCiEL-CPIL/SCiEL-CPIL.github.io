"""Combined approach: Simulated Annealing + greedy add + structured local search for max independent set."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill

    N = 256

    # Build squares up to 2*N
    squares = set()
    k = 2
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    # Individually invalid: 2*a is a perfect square
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

    def popcount(x):
        return bin(x).count('1')

    def greedy_min_degree():
        """Greedy min-degree independent set."""
        remaining = (1 << n) - 1
        selected_bits = 0
        while remaining:
            best_node = -1
            best_degree = n + 1
            for i in range(n):
                if (remaining >> i) & 1:
                    deg = popcount(conflict_bits[i] & remaining)
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

    def simulated_annealing(mask, size, steps, t_start, t_end, time_limit):
        """Simulated annealing with single-vertex moves."""
        current_mask = mask
        current_size = size
        t = t_start
        cooling = (t_end / t_start) ** (1.0 / max(steps, 1))

        for _ in range(steps):
            if time.time() > time_limit:
                break

            # Pick a random unselected vertex
            unselected = [i for i in range(n) if not (current_mask >> i) & 1]
            if not unselected:
                break
            v = random.choice(unselected)

            neighbors_in_set = conflict_bits[v] & current_mask
            n_neighbors = popcount(neighbors_in_set)

            if n_neighbors == 0:
                trial_mask = current_mask | (1 << v)
                trial_size = current_size + 1
                delta = -1
            else:
                trial_mask = (current_mask & ~neighbors_in_set) | (1 << v)
                trial_size = current_size - n_neighbors + 1
                delta = n_neighbors - 1

            if delta <= 0:
                current_mask = trial_mask
                current_size = trial_size
            else:
                if t > 0 and random.random() < math.exp(-delta / t):
                    current_mask = trial_mask
                    current_size = trial_size

            t *= cooling

        return current_mask, current_size

    def greedy_polish(mask, size, time_limit):
        """Final greedy add pass."""
        return greedy_add_pass(mask, size, time_limit)

    random.seed(42)

    # Phase 1: Greedy min-degree initial solution
    best_mask = greedy_min_degree()
    best_size = popcount(best_mask)

    # Phase 2: Greedy add pass
    best_mask, best_size = greedy_add_pass(best_mask, best_size, deadline - 50.0)

    # Phase 3: Simulated Annealing from greedy solution
    if time.time() < deadline - 20.0:
        sa_mask, sa_size = simulated_annealing(
            best_mask, best_size,
            steps=80000,
            t_start=50.0,
            t_end=0.01,
            time_limit=deadline - 15.0
        )
        if sa_size > best_size:
            best_mask = sa_mask
            best_size = sa_size

    # Phase 4: Greedy polish after SA
    best_mask, best_size = greedy_add_pass(best_mask, best_size, deadline - 12.0)

    # Phase 5: Multiple random restarts with SA
    restart_limit = deadline - 10.0
    for restart in range(30):
        if time.time() > restart_limit:
            break
        # Random greedy initialization
        random_order = list(range(n))
        random.shuffle(random_order)
        init_mask = 0
        for i in random_order:
            if (conflict_bits[i] & init_mask) == 0:
                init_mask |= (1 << i)
        init_size = popcount(init_mask)

        # SA on this init
        sa_mask, sa_size = simulated_annealing(
            init_mask, init_size,
            steps=40000,
            t_start=30.0,
            t_end=0.01,
            time_limit=deadline - 8.0
        )
        if sa_size > best_size:
            best_mask = sa_mask
            best_size = sa_size

    # Phase 6: Final greedy polish
    if time.time() < deadline - 5.0:
        best_mask, best_size = greedy_add_pass(best_mask, best_size, deadline - 4.0)

    # Phase 7: Structured swap local search (from current program's approach)
    if time.time() < deadline - 3.0:
        unselected = [i for i in range(n) if not ((best_mask >> i) & 1)]
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            for a in unselected:
                if (best_mask >> a) & 1:
                    continue
                conflicting = [b for b in range(n) if (best_mask >> b) & 1 and (conflict_bits[a] >> b) & 1]
                if len(conflicting) <= 1:
                    continue
                removed_bits = 0
                for b in conflicting:
                    removed_bits |= (1 << b)
                candidate_bits = (best_mask & ~removed_bits) | (1 << a)
                temp_bits = candidate_bits
                temp_size = popcount(candidate_bits)
                for c in unselected:
                    if c == a or (removed_bits >> c) & 1:
                        continue
                    if (temp_bits >> c) & 1:
                        continue
                    if conflict_bits[c] & temp_bits:
                        continue
                    temp_bits |= (1 << c)
                    temp_size += 1
                if temp_size > best_size:
                    best_mask = temp_bits
                    best_size = temp_size
                    unselected = [i for i in range(n) if not ((best_mask >> i) & 1)]
                    improved = True
                    break

    # Phase 8: Final greedy add
    if time.time() < deadline - 2.0:
        best_mask, best_size = greedy_add_pass(best_mask, best_size, deadline - 1.5)

    # Extract result
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
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
        result = []
        for i in range(n):
            if (greedy_min_degree() >> i) & 1:
                result.append(valid_nodes[i])
        result.sort()

    if not result:
        return [1]

    return result
# EVOLVE-BLOCK-END
