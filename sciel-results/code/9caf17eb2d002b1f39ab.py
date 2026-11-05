import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s hard kill

    N = 256
    elements = list(range(1, N + 1))

    # Build conflict graph as Python int bitmasks
    conflict = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                conflict[a] |= (1 << b)
                conflict[b] |= (1 << a)

    def popcount(x):
        return bin(x).count('1')

    def is_valid(mask):
        for e in range(1, N + 1):
            if mask & (1 << e):
                if conflict[e] & mask & ~(1 << e):
                    return False
        return True

    def can_add(e, mask):
        return (conflict[e] & mask) == 0

    def local_search_add(mask):
        improved = True
        while improved:
            improved = False
            for e in range(1, N + 1):
                if not (mask & (1 << e)) and can_add(e, mask):
                    mask |= (1 << e)
                    improved = True
        return mask

    def greedy_build(order, start_mask=0):
        result = start_mask
        for e in order:
            if can_add(e, result):
                result |= (1 << e)
        return result

    def spectral_scores():
        """
        Compute eigenvector-like centrality scores using power iteration
        on the non-conflict adjacency (i.e., we want vertices with few conflicts).
        Actually, for independent set, we want vertices that have FEW conflicts.
        So we use the complement: adjacency where edge exists if NO conflict.
        Then high eigenvector centrality = many non-neighbor connections = good.
        """
        # Build non-conflict adjacency matrix (excluding self)
        # For vertex e, non_conflict[e] is bitmask of vertices that do NOT conflict with e
        # and are not e itself
        non_conflict = [0] * (N + 1)
        for e in range(1, N + 1):
            # All vertices except e
            all_except_e = ((1 << (N + 1)) - 1) & ~(1 << e)
            # Remove conflicts
            non_conflict[e] = all_except_e & ~conflict[e]

        # Power iteration on non-conflict adjacency
        # Start with uniform vector
        scores = [1.0] * (N + 1)
        scores[0] = 0.0

        for iteration in range(50):
            new_scores = [0.0] * (N + 1)
            for e in range(1, N + 1):
                # Sum scores of non-conflicting neighbors
                s = 0.0
                nc = non_conflict[e]
                while nc:
                    lb = nc & (-nc)
                    v = lb.bit_length() - 1
                    s += scores[v]
                    nc ^= lb
                new_scores[e] = s
            # Normalize
            max_val = max(new_scores[1:]) if max(new_scores[1:]) > 0 else 1.0
            for e in range(1, N + 1):
                new_scores[e] /= max_val
            scores = new_scores

        return scores

    def simulated_annealing(mask, max_time):
        """
        Simulated annealing to improve the independent set.
        Moves: add a vertex, remove a vertex, or swap (remove one, add one).
        """
        current_mask = mask
        current_size = popcount(current_mask)
        best_mask = current_mask
        best_size = current_size

        temperature = 10.0
        cooling = 0.995
        rng = random.Random(12345)

        while time.time() < max_time and temperature > 0.01:
            # Choose a random move
            move_type = rng.random()
            if move_type < 0.3:
                # Try to add a random vertex
                candidates = [e for e in range(1, N + 1) if not (current_mask & (1 << e))]
                if not candidates:
                    temperature *= cooling
                    continue
                e = rng.choice(candidates)
                if can_add(e, current_mask):
                    new_mask = current_mask | (1 << e)
                    new_size = current_size + 1
                    # Always accept if it improves
                    current_mask = new_mask
                    current_size = new_size
                    if current_size > best_size:
                        best_mask = current_mask
                        best_size = current_size
                else:
                    # Reject (would be invalid)
                    pass
            elif move_type < 0.6:
                # Try to remove a random vertex and add two others
                if current_size < 2:
                    temperature *= cooling
                    continue
                e = rng.choice([v for v in range(1, N + 1) if current_mask & (1 << v)])
                candidate = current_mask & ~(1 << e)
                addable = [f for f in range(1, N + 1) if not (candidate & (1 << f)) and can_add(f, candidate)]
                if len(addable) >= 2:
                    f, g = rng.sample(addable, 2)
                    if can_add(g, candidate | (1 << f)):
                        new_mask = candidate | (1 << f) | (1 << g)
                        new_size = current_size - 1 + 2
                        if new_size > current_size or rng.random() < math.exp((new_size - current_size) / temperature):
                            current_mask = new_mask
                            current_size = new_size
                            if current_size > best_size:
                                best_mask = current_mask
                                best_size = current_size
            else:
                # Try 1-for-1 swap
                if current_size < 1:
                    temperature *= cooling
                    continue
                e = rng.choice([v for v in range(1, N + 1) if current_mask & (1 << v)])
                candidate = current_mask & ~(1 << e)
                addable = [f for f in range(1, N + 1) if not (candidate & (1 << f)) and can_add(f, candidate)]
                if addable:
                    f = rng.choice(addable)
                    new_mask = candidate | (1 << f)
                    new_size = current_size
                    # Same size, accept with probability based on temperature
                    if rng.random() < math.exp(0 / temperature):  # Always accept same-size for diversity
                        current_mask = new_mask
                        current_size = new_size
                        if current_size > best_size:
                            best_mask = current_mask
                            best_size = current_size

            temperature *= cooling

        # Final polish: try to add any remaining vertices
        current_mask = local_search_add(current_mask)
        current_size = popcount(current_mask)
        if current_size > best_size:
            best_mask = current_mask
            best_size = current_size

        return best_mask

    # Strategy 1: Start with all odds (known valid independent set)
    odds = [e for e in range(1, N + 1) if e % 2 == 1]
    evens = [e for e in range(1, N + 1) if e % 2 == 0]

    odds_mask = 0
    for e in odds:
        odds_mask |= (1 << e)

    # Greedily add evens in order of increasing conflict count
    evens_sorted = sorted(evens, key=lambda e: popcount(conflict[e]))
    mask = greedy_build(evens_sorted, odds_mask)
    mask = local_search_add(mask)

    best_mask = mask
    best_size = popcount(mask)

    # Strategy 2: Spectral-guided greedy
    if time.time() < deadline - 30.0:
        scores = spectral_scores()
        # Sort all elements by spectral score (descending)
        all_elements_sorted = sorted(range(1, N + 1), key=lambda e: -scores[e])
        m = greedy_build(all_elements_sorted, 0)
        m = local_search_add(m)
        s = popcount(m)
        if s > best_size:
            best_mask = m
            best_size = s

        # Also try starting from odds and adding by spectral score
        evens_by_score = sorted(evens, key=lambda e: -scores[e])
        m = greedy_build(evens_by_score, odds_mask)
        m = local_search_add(m)
        s = popcount(m)
        if s > best_size:
            best_mask = m
            best_size = s

    # Strategy 3: Multiple randomized greedy restarts
    rng = random.Random(42)
    for trial in range(300):
        if time.time() > deadline - 35.0:
            break
        shuffled = evens[:]
        rng.shuffle(shuffled)
        m = greedy_build(shuffled, odds_mask)
        m = local_search_add(m)
        s = popcount(m)
        if s > best_size:
            best_mask = m
            best_size = s

    # Strategy 4: Simulated annealing on the best solution
    if time.time() < deadline - 10.0:
        sa_result = simulated_annealing(best_mask, deadline - 5.0)
        sa_size = popcount(sa_result)
        if sa_size > best_size:
            best_mask = sa_result
            best_size = sa_size

    # Strategy 5: Final local search polish
    if time.time() < deadline - 2.0:
        polished = local_search_add(best_mask)
        if popcount(polished) > best_size:
            best_mask = polished
            best_size = popcount(polished)

    # Convert to list
    result = [e for e in range(1, N + 1) if best_mask & (1 << e)]

    if not result:
        result = [1]

    # Final validation
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            if (a * b) % (a + b) == 0:
                result.pop(j)
                j -= 1

    return result if result else [1]
