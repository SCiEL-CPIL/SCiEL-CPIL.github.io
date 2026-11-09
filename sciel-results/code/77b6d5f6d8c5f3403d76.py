<<<<<<< SEARCH
def _simulate_annealing(centers, angles, max_iters=5000, T_start=1.0, T_end=0.001):
    """Simulated annealing to optimize packing."""
    best_side, _, _, _ = _min_enclosing_side(centers, angles)
    best_centers = list(centers)
    best_angles = list(angles)
    
    n = len(centers)
    current_penalty = _hex_overlap_penalty(centers, angles)
    current_side = best_side
    current_cost = current_side + 100 * current_penalty
    
    for it in range(max_iters):
        T = T_start * (T_end / T_start) ** (it / max_iters)
        
        # Random perturbation
        new_centers = [list(c) for c in centers]
        new_angles = list(angles)
        
        # Move one hexagon
        idx = random.randint(0, n - 1)
        new_centers[idx][0] += random.gauss(0, 0.5)
        new_centers[idx][1] += random.gauss(0, 0.5)
        new_angles[idx] = (new_angles[idx] + random.gauss(0, 10)) % 360
        
        # Convert to tuples for evaluation
        new_centers_tuples = [(c[0], c[1]) for c in new_centers]
        
        new_side, _, _, _ = _min_enclosing_side(new_centers_tuples, new_angles)
        new_penalty = _hex_overlap_penalty(new_centers_tuples, new_angles)
        new_cost = new_side + 100 * new_penalty
        
        delta = new_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
            centers = new_centers_tuples
            angles = new_angles
            current_side = new_side
            current_penalty = new_penalty
            current_cost = new_cost
            
            if current_cost < best_side + 100 * 0:  # if no overlap
                if current_penalty < 1e-6:
                    best_side = current_side
                    best_centers = list(centers)
                    best_angles = list(angles)
        
        # Also try moving the whole configuration
        if it % 10 == 0:
            new_centers2 = [(c[0] + random.gauss(0, 0.3), c[1] + random.gauss(0, 0.3)) for c in centers]
            new_side2, _, _, _ = _min_enclosing_side(new_centers2, angles)
            new_penalty2 = _hex_overlap_penalty(new_centers2, new_angles)
            new_cost2 = new_side2 + 100 * new_penalty2
            delta2 = new_cost2 - current_cost
            if delta2 < 0 or random.random() < math.exp(-delta2 / max(T, 1e-10)):
                centers = new_centers2
                current_side = new_side2
                current_penalty = new_penalty2
                current_cost = new_cost2
                if current_penalty < 1e-6 and current_side < best_side:
                    best_side = new_side2
                    best_centers = list(centers)
                    best_angles = list(angles)
    
    return best_side, best_centers, best_angles
=======
def _simulate_annealing(centers, angles, max_iters=5000, T_start=1.0, T_end=0.001):
    """Simulated annealing to optimize packing. Strictly rejects overlaps."""
    # Verify initial state is valid
    if not _hex_overlap_check(centers, angles):
        return _min_enclosing_side(centers, angles)[0], centers, angles
        
    best_side, _, _, _ = _min_enclosing_side(centers, angles)
    best_centers = list(centers)
    best_angles = list(angles)
    
    n = len(centers)
    current_side = best_side
    current_cost = current_side
    
    for it in range(max_iters):
        T = T_start * (T_end / T_start) ** (it / max_iters)
        
        # Random perturbation
        new_centers = [list(c) for c in centers]
        new_angles = list(angles)
        
        # Move one hexagon
        idx = random.randint(0, n - 1)
        # Smaller step size to maintain validity more often
        new_centers[idx][0] += random.gauss(0, 0.2)
        new_centers[idx][1] += random.gauss(0, 0.2)
        new_angles[idx] = (new_angles[idx] + random.gauss(0, 5)) % 360
        
        # Convert to tuples for evaluation
        new_centers_tuples = [(c[0], c[1]) for c in new_centers]
        
        # STRICT OVERLAP CHECK: If overlap, reject immediately
        if not _hex_overlap_check(new_centers_tuples, new_angles):
            continue
            
        new_side, _, _, _ = _min_enclosing_side(new_centers_tuples, new_angles)
        new_cost = new_side
        
        delta = new_cost - current_cost
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
            centers = new_centers_tuples
            angles = new_angles
            current_side = new_side
            current_cost = new_cost
            
            if current_side < best_side:
                best_side = current_side
                best_centers = list(centers)
                best_angles = list(angles)
        
        # Also try moving the whole configuration (small step)
        if it % 20 == 0:
            new_centers2 = [(c[0] + random.gauss(0, 0.1), c[1] + random.gauss(0, 0.1)) for c in centers]
            # Check overlap for global move
            if _hex_overlap_check(new_centers2, angles):
                new_side2, _, _, _ = _min_enclosing_side(new_centers2, angles)
                new_cost2 = new_side2
                delta2 = new_cost2 - current_cost
                if delta2 < 0 or random.random() < math.exp(-delta2 / max(T, 1e-10)):
                    centers = new_centers2
                    current_side = new_side2
                    current_cost = new_cost2
                    if current_side < best_side:
                        best_side = current_side
                        best_centers = list(centers)
                        best_angles = list(angles)
    
    return best_side, best_centers, best_angles
>>>>>>> REPLACE

