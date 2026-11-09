def _fast_local_search(U, V, W, n, m, p, rank, time_limit, seed=42):
    """
    Fast local search with incremental energy updates.
    """
    rng = random.Random(seed)
    start_time = time.time()
    deadline = start_time + time_limit
    
    allowed_vals = [-2, -1, 0, 1, 2]
    
    # Precompute constraint indices
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                constraints.append((i, j, k))
    
    # Compute current sums for each constraint
    sums = {}
    for i, j, k in constraints:
        s = 0
        for r in range(rank):
            s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
        sums[(i, j, k)] = s
    
    energy = sum((s - 1) ** 2 for s in sums.values())
    
    best_energy = energy
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    temperature = 1.0
    cooling = 0.97
    
    for iteration in range(50000):
        if time.time() > deadline:
            break
        
        if energy == 0:
            return U, V, W
        
        if energy < best_energy:
            best_energy = energy
            best_U = [row[:] for row in U]
            best_V = [row[:] for row in V]
            best_W = [row[:] for row in W]
        
        # Pick a random element to change
        choice = rng.choice(['U', 'V', 'W'])
        r = rng.randint(0, rank - 1)
        
        if choice == 'U':
            row_idx = rng.randint(0, n * m - 1)
            i = row_idx // m
            j = row_idx % m
            old_val = U[row_idx][r]
            new_val = rng.choice(allowed_vals)
            if new_val == old_val:
                continue
            # Update affected constraints: all k
            delta_energy = 0
            for k in range(p):
                key = (i, j, k)
                old_s = sums[key]
                new_s = old_s + (new_val - old_val) * V[j * p + k][r] * W[k * n + i][r]
                delta_energy += (new_s - 1) ** 2 - (old_s - 1) ** 2
                sums[key] = new_s
            U[row_idx][r] = new_val
            energy += delta_energy
        elif choice == 'V':
            row_idx = rng.randint(0, m * p - 1)
            j = row_idx // p
            k = row_idx % p
            old_val = V[row_idx][r]
            new_val = rng.choice(allowed_vals)
            if new_val == old_val:
                continue
            delta_energy = 0
            for i in range(n):
                key = (i, j, k)
                old_s = sums[key]
                new_s = old_s + (new_val - old_val) * U[i * m + j][r] * W[k * n + i][r]
                delta_energy += (new_s - 1) ** 2 - (old_s - 1) ** 2
                sums[key] = new_s
            V[row_idx][r] = new_val
            energy += delta_energy
        else:
            row_idx = rng.randint(0, p * n - 1)
            k = row_idx // n
            i = row_idx % n
            old_val = W[row_idx][r]
            new_val = rng.choice(allowed_vals)
            if new_val == old_val:
                continue
            delta_energy = 0
            for j in range(m):
                key = (i, j, k)
                old_s = sums[key]
                new_s = old_s + (new_val - old_val) * U[i * m + j][r] * V[j * p + k][r]
                delta_energy += (new_s - 1) ** 2 - (old_s - 1) ** 2
                sums[key] = new_s
            W[row_idx][r] = new_val
            energy += delta_energy
        
        # Accept/reject
        if energy >= 0 and (delta_energy <= 0 or rng.random() < np.exp(-delta_energy / max(temperature, 0.01))):
            pass
        else:
            # Revert
            if choice == 'U':
                U[row_idx][r] = old_val
                for k in range(p):
                    key = (i, j, k)
                    sums[key] -= (new_val - old_val) * V[j * p + k][r] * W[k * n + i][r]
            elif choice == 'V':
                V[row_idx][r] = old_val
                for i in range(n):
                    key = (i, j, k)
                    sums[key] -= (new_val - old_val) * U[i * m + j][r] * W[k * n + i][r]
            else:
                W[row_idx][r] = old_val
                for j in range(m):
                    key = (i, j, k)
                    sums[key] -= (new_val - old_val) * U[i * m + j][r] * V[j * p + k][r]
            energy -= delta_energy
        
        temperature *= cooling
    
    if best_energy == 0:
        return best_U, best_V, best_W
    return None
