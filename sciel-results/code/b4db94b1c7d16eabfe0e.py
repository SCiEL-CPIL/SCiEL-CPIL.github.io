<<<<<<< SEARCH
def _simulated_annealing(pt_list, perm, deadline, max_iter=100000, seed=0, cost_matrix=None):
    """Simulated annealing with insertion moves, biased by cost matrix if available."""
    rng = random.Random(seed)
    n = len(perm)
    if n <= 1:
        return perm, _makespan_fast(pt_list, perm)
    current = list(perm)
    current_ms = _makespan_fast(pt_list, current)
    best = list(current)
    best_ms = current_ms

    T = max(current_ms * 0.05, 1.0)
    T_min = max(current_ms * 0.0005, 0.01)
    cooling = 0.9995

    for _ in range(max_iter):
        if time.time() > deadline:
            break
        i = rng.randint(0, n - 1)
        job = current[i]
        rest = current[:i] + current[i+1:]
        if cost_matrix is not None:
            j = rng.randint(0, n - 1)
            if rng.random() < 0.3:
                best_j = 0
                best_cost = float('inf')
                for jj in range(n):
                    if jj == i or jj == i + 1:
                        continue
                    pred = current[jj-1] if jj > 0 else current[0]
                    c = cost_matrix[job][pred]
                    if c < best_cost:
                        best_cost = c
                        best_j = jj
                if rng.random() < 0.5:
                    j = best_j
        else:
            j = rng.randint(0, n - 1)
        if j == i or j == i + 1:
            j = (j + 1) % n
        test = rest[:j] + [job] + rest[j:]
        new_ms = _makespan_fast(pt_list, test)
        delta = new_ms - current_ms
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current = test
            current_ms = new_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best = list(current)
        T *= cooling
        if T < T_min:
            T = T_min
    return best, best_ms
=======
def _simulated_annealing(pt_list, perm, deadline, max_iter=100000, seed=0, cost_matrix=None):
    """Simulated annealing with mixed insertion and or-opt moves."""
    rng = random.Random(seed)
    n = len(perm)
    if n <= 1:
        return perm, _makespan_fast(pt_list, perm)
    current = list(perm)
    current_ms = _makespan_fast(pt_list, current)
    best = list(current)
    best_ms = current_ms

    T = max(current_ms * 0.03, 1.0)
    T_min = max(current_ms * 0.001, 0.01)
    cooling = 0.9997

    for _ in range(max_iter):
        if time.time() > deadline:
            break
        if rng.random() < 0.7:
            # Insertion move
            i = rng.randint(0, n - 1)
            job = current[i]
            rest = current[:i] + current[i+1:]
            if cost_matrix is not None and rng.random() < 0.4:
                best_j = 0
                best_cost = float('inf')
                for jj in range(n):
                    if jj == i or jj == i + 1:
                        continue
                    pred = current[jj-1] if jj > 0 else current[0]
                    c = cost_matrix[job][pred]
                    if c < best_cost:
                        best_cost = c
                        best_j = jj
                j = best_j
            else:
                j = rng.randint(0, n - 1)
            if j == i or j == i + 1:
                j = (j + 1) % n
            test = rest[:j] + [job] + rest[j:]
        else:
            # Or-opt move
            block_size = rng.choice([1, 2, 3])
            if n > block_size:
                i = rng.randint(0, n - block_size)
                block = current[i:i+block_size]
                rest = current[:i] + current[i+block_size:]
                j = rng.randint(0, len(rest))
                if j == i or j == i + 1:
                    j = (j + 1) % (len(rest) + 1)
                test = rest[:j] + block + rest[j:]
            else:
                continue

        new_ms = _makespan_fast(pt_list, test)
        delta = new_ms - current_ms
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            current = test
            current_ms = new_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best = list(current)
        T *= cooling
        if T < T_min:
            T = T_min
    return best, best_ms
>>>>>>> REPLACE

