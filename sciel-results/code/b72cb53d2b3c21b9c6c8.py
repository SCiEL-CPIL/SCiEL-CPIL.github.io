import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    grid_sq = (grid ** 2).sum(axis=1)
    
    # Pre-compute all 4-combinations for fast access
    all_quads = list(itertools.combinations(range(N), 4))
    quad_indices = np.array(all_quads, dtype=np.int64)  # shape (num_quads, 4)
    num_quads = len(quad_indices)
    
    # Pre-compute the quad data for fast determinant checks
    # For a quad (i1,i2,i3,i4), we need to check if adding point p makes 5 co-spherical
    # The 5 points are co-spherical if and only if the determinant of the 5x5 matrix is 0
    # But we can use a more efficient check
    
    def check_point_valid(selected, p_idx):
        """Check if p_idx can be added to selected without creating 5 co-spherical points."""
        m = len(selected)
        if m < 4:
            return True
        
        p = grid[p_idx]
        p_sq = grid_sq[p_idx]
        
        # For each quad in selected, check if it forms a valid 5-set with p
        for i in range(m - 3):
            for j in range(i + 1, m - 2):
                for k in range(j + 1, m - 1):
                    for l in range(k + 1, m):
                        q1 = grid[selected[i]]
                        q2 = grid[selected[j]]
                        q3 = grid[selected[k]]
                        q4 = grid[selected[l]]
                        
                        # Form the 4x4 matrix with rows (q - p, |q|^2 - |p|^2)
                        mat = np.array([
                            [q1[0]-p[0], q1[1]-p[1], q1[2]-p[2], grid_sq[selected[i]] - p_sq],
                            [q2[0]-p[0], q2[1]-p[1], q2[2]-p[2], grid_sq[selected[j]] - p_sq],
                            [q3[0]-p[0], q3[1]-p[1], q3[2]-p[2], grid_sq[selected[k]] - p_sq],
                            [q4[0]-p[0], q4[1]-p[1], q4[2]-p[2], grid_sq[selected[l]] - p_sq]
                        ], dtype=np.float64)
                        
                        det = np.linalg.det(mat)
                        if abs(det) < 0.5:
                            return False
        return True
    
    def fast_check_point_valid(selected, p_idx):
        """Vectorized check for whether p_idx can be added to selected."""
        m = len(selected)
        if m < 4:
            return True
        
        p = grid[p_idx]
        p_sq = grid_sq[p_idx]
        sel_pts = grid[selected]
        sel_sq = grid_sq[selected]
        
        # For each quad in selected, check if it forms a valid 5-set with p
        # Use vectorized computation
        combos = list(itertools.combinations(range(m), 4))
        if not combos:
            return True
        
        combos_arr = np.array(combos, dtype=np.int64)
        
        # Compute differences and squared differences
        diff = sel_pts - p  # shape (m, 3)
        sq_diff = sel_sq - p_sq  # shape (m,)
        
        # For each combo, form the 4x4 matrix
        valid = True
        batch_size = 5000
        for start in range(0, len(combos_arr), batch_size):
            batch = combos_arr[start:start + batch_size]
            if len(batch) == 0:
                break
            
            # Extract the relevant rows
            sub_diff = diff[batch]  # shape (len(batch), 4, 3)
            sub_sq = sq_diff[batch]  # shape (len(batch), 4)
            
            # Form the matrices
            mat = np.concatenate([sub_diff, sub_sq[:, :, None]], axis=2)  # shape (len(batch), 4, 4)
            
            # Compute determinants
            dets = np.linalg.det(mat)
            
            if np.any(np.abs(dets) < 0.5):
                valid = False
                break
        
        return valid
    
    def simulate_annealing(init_points, time_budget):
        """Simulated annealing to maximize the number of points."""
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + time_budget
        
        # State: list of selected point indices
        current = list(init_points)
        current_set = set(current)
        
        # Energy function: number of violations (5 co-spherical points)
        def count_violations(points):
            if len(points) < 5:
                return 0
            
            violations = 0
            pts = grid[points]
            sq = grid_sq[points]
            
            # For each 5-set, check if co-spherical
            for combo in itertools.combinations(range(len(points)), 5):
                p0 = pts[combo[0]]
                p0_sq = sq[combo[0]]
                
                mat = np.array([
                    [pts[combo[i]][0]-p0[0], pts[combo[i]][1]-p0[1], pts[combo[i]][2]-p0[2], sq[combo[i]] - p0_sq]
                    for i in range(1, 5)
                ], dtype=np.float64)
                
                det = np.linalg.det(mat)
                if abs(det) < 0.5:
                    violations += 1
            
            return violations
        
        def try_move(points, point_set, move_type):
            """Try a random move and return the new state or None."""
            if move_type == 'add':
                candidates = [i for i in range(N) if i not in point_set]
                if not candidates:
                    return None, None, None
                p = candidates[rng.integers(len(candidates))]
                new_points = points + [p]
                new_set = point_set | {p}
                return new_points, new_set, 'add', p
            elif move_type == 'remove':
                if not points:
                    return None, None, None
                idx = rng.integers(len(points))
                p = points[idx]
                new_points = points[:idx] + points[idx+1:]
                new_set = point_set - {p}
                return new_points, new_set, 'remove', p
            elif move_type == 'swap':
                if not points:
                    return None, None, None
                candidates = [i for i in range(N) if i not in point_set]
                if not candidates:
                    return None, None, None
                idx = rng.integers(len(points))
                p_rem = points[idx]
                p_add = candidates[rng.integers(len(candidates))]
                new_points = points[:idx] + [p_add] + points[idx+1:]
                new_set = (point_set - {p_rem}) | {p_add}
                return new_points, new_set, 'swap', (p_rem, p_add)
            return None, None, None
        
        # Temperature schedule
        T_start = 10.0
        T_end = 0.001
        alpha = 0.9995  # Cooling rate
        
        T = T_start
        best_points = current.copy()
        best_violations = count_violations(current)
        
        while time.time() < deadline:
            # Choose a move type
            r = rng.random()
            if r < 0.4:
                move_type = 'add'
            elif r < 0.4 + 0.2:
                move_type = 'remove'
            else:
                move_type = 'swap'
            
            new_points, new_set, move_info, move_data = try_move(current, current_set, move_type)
            if new_points is None:
                T *= alpha
                continue
            
            # Compute energy difference
            old_violations = count_violations(current)
            new_violations = count_violations(new_points)
            
            # Also consider the size of the set as part of the objective
            old_score = len(current) - 10 * old_violations
            new_score = len(new_points) - 10 * new_violations
            
            delta = new_score - old_score
            
            # Accept or reject
            if delta > 0 or rng.random() < np.exp(delta / T):
                current = new_points
                current_set = new_set
                
                if new_violations == 0 and len(current) > len(best_points):
                    best_points = current.copy()
                    best_violations = new_violations
            
            T *= alpha
            if T < T_end:
                T = T_end
        
        return best_points
    
    def greedy_fill(order, check_fn=fast_check_point_valid):
        """Greedily fill points in the given order."""
        selected = []
        for idx in order:
            if check_fn(selected, idx):
                selected.append(idx)
        return selected
    
    def polish(selected, time_budget=5.0):
        """Polish the solution by trying to add more points."""
        current = list(selected)
        current_set = set(current)
        start_time = time.time()
        
        # Try multiple orderings
        orderings = [
            list(range(N)),
            list(range(N-1, -1, -1)),
            sorted(range(N), key=lambda i: grid[i].sum()),
            sorted(range(N), key=lambda i: np.linalg.norm(grid[i] - 3.5)),
            sorted(range(N), key=lambda i: -grid[i].sum()),
        ]
        
        for order in orderings:
            if time.time() - start_time > time_budget:
                break
            for idx in order:
                if idx in current_set:
                    continue
                if fast_check_point_valid(current, idx):
                    current.append(idx)
                    current_set.add(idx)
        
        return current
    
    # Main search
    best_result = []
    start_time = time.time()
    deadline = start_time + 600 - 30  # Safety margin
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Quick greedy fills
    while time.time() < deadline - 120:
        r = rng.random()
        if r < 0.25:
            order = rng.permutation(N)
        elif r < 0.5:
            center = np.array([3.5, 3.5, 3.5])
            dists = np.linalg.norm(grid - center, axis=1)
            noise = rng.random(N) * 3.0
            order = np.argsort(dists + noise)
        elif r < 0.75:
            sums = grid.sum(axis=1)
            noise = rng.random(N) * 3.0
            order = np.argsort(sums + noise)
        else:
            weights = rng.random(N)
            order = np.argsort(weights)
        
        current = greedy_fill(order)
        current = polish(current, time_budget=2.0)
        
        if len(current) > len(best_result):
            best_result = current.copy()
    
    # Phase 2: Simulated annealing on best solutions
    if time.time() < deadline - 60:
        sa_budget = min(30, deadline - time.time() - 60)
        sa_result = simulate_annealing(best_result, sa_budget)
        sa_result = polish(sa_result, time_budget=3.0)
        if len(sa_result) > len(best_result):
            best_result = sa_result.copy()
    
    # Phase 3: More simulated annealing rounds
    while time.time() < deadline - 30:
        sa_budget = min(15, deadline - time.time() - 30)
        if sa_budget <= 5:
            break
        sa_result = simulate_annealing(best_result, sa_budget)
        sa_result = polish(sa_result, time_budget=2.0)
        if len(sa_result) > len(best_result):
            best_result = sa_result.copy()
    
    # Final polish
    if len(best_result) > 0 and time.time() < deadline - 5:
        best_result = polish(best_result, deadline - time.time())
    
    return grid[best_result]
