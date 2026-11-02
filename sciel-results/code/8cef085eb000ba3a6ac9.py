<<<<<<< SEARCH
def _pt_mc_search(initial_states, temps, max_steps, rng):
    """
    Parallel Tempering Monte Carlo search with structure-aware moves.
    initial_states: list of arrays (N, 3)
    temps: list of temperatures (floats)
    max_steps: maximum number of MC steps
    rng: numpy random generator
    """
    n_replicas = len(temps)
    states = [s.copy() for s in initial_states]
    energies = [_compute_energy(s) for s in states]
    
    best_energy = min(energies)
    best_state = states[energies.index(best_energy)].copy()
    
    # Precompute index pairs for overlap checks
    idx_i, idx_j = np.triu_indices(N, k=1)
    
    for step in range(max_steps):
        if _remaining_time() < 5.0:
            break
        
        # Randomly pick a replica to move
        i = rng.integers(0, n_replicas)
        T = temps[i]
        
        new_state = states[i].copy()
        
        # Choose move type based on temperature
        move_type = rng.integers(0, 4)
        
        if move_type == 0:
            # Random position jitter (exploration)
            scale = 0.008 * T
            new_state[:, 0] += rng.normal(0, scale, N)
            new_state[:, 1] += rng.normal(0, scale, N)
        elif move_type == 1:
            # Random radius change
            idx = rng.integers(0, N)
            delta_r = rng.normal(0, 0.005 * T)
            new_state[idx, 2] = max(0.01, new_state[idx, 2] + delta_r)
        elif move_type == 2:
            # Push apart overlapping circles
            x, y, r = new_state[:, 0], new_state[:, 1], new_state[:, 2]
            dx = x[idx_i] - x[idx_j]
            dy = y[idx_i] - y[idx_j]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(np.maximum(dist_sq, 1e-12))
            overlaps = r[idx_i] + r[idx_j] - dist
            # Find overlapping pairs
            overlap_mask = overlaps > 0
            if np.any(overlap_mask):
                # Pick a random overlapping pair
                overlap_indices = np.where(overlap_mask)[0]
                pick = overlap_indices[rng.integers(0, len(overlap_indices))]
                ii, jj = idx_i[pick], idx_j[pick]
                d = dist[pick]
                if d > 1e-9:
                    # Push apart along the line connecting centers
                    push = overlaps[pick] * 0.5 * rng.uniform(0.5, 1.5)
                    ux, uy = dx[pick]/d, dy[pick]/d
                    new_state[ii, 0] += ux * push
                    new_state[ii, 1] += uy * push
                    new_state[jj, 0] -= ux * push
                    new_state[jj, 1] -= uy * push
        else:
            # Grow a circle if it has room
            idx = rng.integers(0, N)
            x_i, y_i, r_i = new_state[idx, 0], new_state[idx, 1], new_state[idx, 2]
            # Check max possible radius
            max_r = min(x_i, 1-x_i, y_i, 1-y_i)
            # Check distance to other circles
            other_r = np.delete(new_state[:, 2], idx)
            other_x = np.delete(new_state[:, 0], idx)
            other_y = np.delete(new_state[:, 1], idx)
            d2 = (other_x - x_i)**2 + (other_y - y_i)**2
            d = np.sqrt(np.maximum(d2, 1e-12))
            max_r = min(max_r, np.min(d - other_r) if len(other_r) > 0 else max_r)
            max_r = max(0.01, max_r)
            if r_i < max_r:
                new_r = min(max_r, r_i + rng.uniform(0.001, 0.01) * T)
                new_state[idx, 2] = new_r
        
        # Clip positions and radii
        new_state[:, 0] = np.clip(new_state[:, 0], -0.05, 1.05)
        new_state[:, 1] = np.clip(new_state[:, 1], -0.05, 1.05)
        new_state[:, 2] = np.maximum(new_state[:, 2], 0.005)
        
        # Calculate new energy
        new_energy = _compute_energy(new_state)
        
        # Metropolis acceptance
        delta_e = new_energy - energies[i]
        if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
            states[i] = new_state
            energies[i] = new_energy
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_state = new_state.copy()
        
        # Attempt swap with adjacent replica
        if n_replicas > 1:
            j = rng.integers(0, n_replicas - 1)
            k = j + 1
            
            inv_t_j = 1.0 / temps[j]
            inv_t_k = 1.0 / temps[k]
            
            delta_swap = (energies[j] - energies[k]) * (inv_t_k - inv_t_j)
            
            if delta_swap < 0 or rng.random() < np.exp(-delta_swap):
                states[j], states[k] = states[k], states[j]
                energies[j], energies[k] = energies[k], energies[j]
                
                if energies[j] < best_energy:
                    best_energy = energies[j]
                    best_state = states[j].copy()
                if energies[k] < best_energy:
                    best_energy = energies[k]
                    best_state = states[k].copy()
    
    return best_state
=======
def _pt_mc_search(initial_states, temps, max_steps, rng):
    """
    Parallel Tempering Monte Carlo search with structure-aware moves.
    Uses sum of radii as the objective (higher is better), with penalty for violations.
    """
    n_replicas = len(temps)
    states = [s.copy() for s in initial_states]
    # Use negative sum of radii + penalty as energy (lower is better)
    energies = [_compute_energy(s, penalty_weight=1000.0) for s in states]
    
    best_energy = min(energies)
    best_state = states[energies.index(best_energy)].copy()
    
    # Precompute index pairs for overlap checks
    idx_i, idx_j = np.triu_indices(N, k=1)
    
    # Track step counter for periodic SLSQP refinement
    slsqp_interval = max(1, max_steps // 5)
    
    for step in range(max_steps):
        if _remaining_time() < 5.0:
            break
        
        # Randomly pick a replica to move
        i = rng.integers(0, n_replicas)
        T = temps[i]
        
        new_state = states[i].copy()
        
        # Choose move type based on temperature (weighted toward useful moves)
        roll = rng.random()
        if roll < 0.35:
            # Random position jitter (exploration)
            scale = 0.006 * T
            new_state[:, 0] += rng.normal(0, scale, N)
            new_state[:, 1] += rng.normal(0, scale, N)
        elif roll < 0.55:
            # Random radius change
            idx = rng.integers(0, N)
            delta_r = rng.normal(0, 0.004 * T)
            new_state[idx, 2] = max(0.005, new_state[idx, 2] + delta_r)
        elif roll < 0.80:
            # Push apart overlapping circles
            x, y, r = new_state[:, 0], new_state[:, 1], new_state[:, 2]
            dx = x[idx_i] - x[idx_j]
            dy = y[idx_i] - y[idx_j]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(np.maximum(dist_sq, 1e-12))
            overlaps = r[idx_i] + r[idx_j] - dist
            overlap_mask = overlaps > 0
            if np.any(overlap_mask):
                overlap_indices = np.where(overlap_mask)[0]
                pick = overlap_indices[rng.integers(0, len(overlap_indices))]
                ii, jj = idx_i[pick], idx_j[pick]
                d = dist[pick]
                if d > 1e-9:
                    push = overlaps[pick] * 0.5 * rng.uniform(0.5, 1.5)
                    ux, uy = dx[pick]/d, dy[pick]/d
                    new_state[ii, 0] += ux * push
                    new_state[ii, 1] += uy * push
                    new_state[jj, 0] -= ux * push
                    new_state[jj, 1] -= uy * push
        else:
            # Grow a circle if it has room
            idx = rng.integers(0, N)
            x_i, y_i, r_i = new_state[idx, 0], new_state[idx, 1], new_state[idx, 2]
            max_r = min(x_i, 1-x_i, y_i, 1-y_i)
            other_r = np.delete(new_state[:, 2], idx)
            other_x = np.delete(new_state[:, 0], idx)
            other_y = np.delete(new_state[:, 1], idx)
            d2 = (other_x - x_i)**2 + (other_y - y_i)**2
            d = np.sqrt(np.maximum(d2, 1e-12))
            if len(other_r) > 0:
                max_r = min(max_r, np.min(d - other_r))
            max_r = max(0.005, max_r)
            if r_i < max_r:
                new_r = min(max_r, r_i + rng.uniform(0.001, 0.008) * T)
                new_state[idx, 2] = new_r
        
        # Clip positions and radii
        new_state[:, 0] = np.clip(new_state[:, 0], 0.0, 1.0)
        new_state[:, 1] = np.clip(new_state[:, 1], 0.0, 1.0)
        new_state[:, 2] = np.maximum(new_state[:, 2], 0.005)
        
        # Calculate new energy
        new_energy = _compute_energy(new_state, penalty_weight=1000.0)
        
        # Metropolis acceptance
        delta_e = new_energy - energies[i]
        if delta_e < 0 or (T > 0 and rng.random() < np.exp(-delta_e / T)):
            states[i] = new_state
            energies[i] = new_energy
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_state = new_state.copy()
        
        # Periodic SLSQP refinement on best replica
        if step % slsqp_interval == 0 and _remaining_time() > 10.0:
            try:
                slsqp_bounds = [(0, 1), (0, 1), (0.005, 0.45)] * N
                res = minimize(_slsqp_objective, best_state.flatten(), method='SLSQP',
                              bounds=slsqp_bounds,
                              constraints=_slsqp_constraints(best_state.flatten()),
                              options={'maxiter': 50, 'ftol': 1e-12, 'eps': 1e-6})
                if _validate(res.x.reshape(N, 3)):
                    slsqp_score = np.sum(res.x[2::3])
                    current_best_score = -best_energy + 1000.0 * 0  # approximate
                    if slsqp_score > np.sum(best_state[:, 2]):
                        best_state = res.x.reshape(N, 3)
                        best_energy = _compute_energy(best_state, penalty_weight=1000.0)
                        # Update best replica
                        best_rep_idx = np.argmin(energies)
                        states[best_rep_idx] = best_state.copy()
                        energies[best_rep_idx] = best_energy
            except Exception:
                pass
        
        # Attempt swap with adjacent replica
        if n_replicas > 1:
            j = rng.integers(0, n_replicas - 1)
            k = j + 1
            
            inv_t_j = 1.0 / temps[j]
            inv_t_k = 1.0 / temps[k]
            
            delta_swap = (energies[j] - energies[k]) * (inv_t_k - inv_t_j)
            
            if delta_swap < 0 or rng.random() < np.exp(-max(0, delta_swap)):
                states[j], states[k] = states[k], states[j]
                energies[j], energies[k] = energies[k], energies[j]
                
                if energies[j] < best_energy:
                    best_energy = energies[j]
                    best_state = states[j].copy()
                if energies[k] < best_energy:
                    best_energy = energies[k]
                    best_state = states[k].copy()
    
    return best_state
>>>>>>> REPLACE
