import numpy as np
import time

def moving_sofa_2d() -> np.ndarray:
    # Grid definition
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    x_grid, y_grid = np.meshgrid(xs, ys)
    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()
    n_points = len(x_flat)
    
    # Time management
    start_time = time.time()
    time_limit = 575.0  # Safety margin before 630s kill
    
    def _make_densified(path):
        """Build the full densified path: start + path + end, with linear interpolation."""
        full = [(0.0, 0.0, 0.0)]
        for p in path:
            full.append((float(p[0]), float(p[1]), float(p[2])))
        full.append((-2.0, 0.0, 90.0))
        dens = []
        for i in range(len(full) - 1):
            p1 = full[i]
            p2 = full[i+1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            for k in range(m):
                t = (k + 1) / m
                x = p1[0] + (p2[0] - p1[0]) * t
                y = p1[1] + (p2[1] - p1[1]) * t
                a = p1[2] + (p2[2] - p1[2]) * t
                dens.append((x, y, a))
        dens.append(full[-1])
        return dens
    
    def _placement_mask(cx, cy, ca_deg):
        """Return boolean mask of points inside the corridor at placement (cx, cy, ca_deg)."""
        ca = np.deg2rad(ca_deg)
        cos_a = np.cos(ca)
        sin_a = np.sin(ca)
        
        dx = x_flat - cx
        dy = y_flat - cy
        
        # R(-a) rotation
        u = dx * cos_a + dy * sin_a
        v = -dx * sin_a + dy * cos_a
        
        in_arm1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        in_arm2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        
        return in_arm1 | in_arm2
    
    def evaluate_path(path):
        """path: list of (x, y, a). Returns area score."""
        dens = _make_densified(path)
        mask = np.ones(n_points, dtype=bool)
        
        for (cx, cy, ca_deg) in dens:
            in_corridor = _placement_mask(cx, cy, ca_deg)
            mask &= in_corridor
            
            if not np.any(mask):
                return 0.0
        
        return np.sum(mask) * 0.0001
    
    def evaluate_path_fast(path, prev_masks=None):
        """
        Fast evaluation using cached masks where possible.
        prev_masks: dict mapping index -> mask, or None.
        Returns (score, masks_dict).
        """
        dens = _make_densified(path)
        mask = np.ones(n_points, dtype=bool)
        masks_dict = {}
        
        for i, (cx, cy, ca_deg) in enumerate(dens):
            if prev_masks is not None and i in prev_masks:
                in_corridor = prev_masks[i]
            else:
                in_corridor = _placement_mask(cx, cy, ca_deg)
            masks_dict[i] = in_corridor
            mask &= in_corridor
            
            if not np.any(mask):
                return 0.0, masks_dict
        
        return np.sum(mask) * 0.0001, masks_dict
    
    # Initial path: hand-tuned starting point
    initial_path = [
        (-0.1, 0.1, 15.0),
        (-0.2, 0.2, 20.0),
        (-0.3, 0.3, 34.0),
        (-0.3, 0.3, 40.0),
        (-0.4, 0.3, 50.0),
        (-0.5, 0.2, 55.0),
        (-0.6, 0.2, 75.0),
        (-0.7, 0.3, 86.0),
    ]
    
    best_score, best_masks = evaluate_path_fast(initial_path)
    best_path = initial_path[:]
    current_path = initial_path[:]
    current_masks = best_masks.copy()
    
    # Local search with adaptive step sizes and multi-start
    np.random.seed(42)
    epsilon_xy = 0.15
    epsilon_a = 3.0
    no_improve = 0
    max_no_improve = 25
    
    # Multi-start: try several different initial paths
    initial_paths = [
        initial_path,
        [
            (-0.05, 0.05, 10.0),
            (-0.15, 0.15, 25.0),
            (-0.25, 0.25, 35.0),
            (-0.35, 0.25, 45.0),
            (-0.45, 0.15, 60.0),
            (-0.55, 0.25, 75.0),
            (-0.65, 0.35, 88.0),
        ],
        [
            (-0.15, 0.15, 12.0),
            (-0.25, 0.25, 22.0),
            (-0.35, 0.35, 38.0),
            (-0.45, 0.35, 48.0),
            (-0.55, 0.25, 58.0),
            (-0.65, 0.35, 72.0),
            (-0.75, 0.45, 85.0),
        ],
    ]
    
    for init_idx, init_path in enumerate(initial_paths):
        if time.time() - start_time > time_limit:
            break
        
        # Evaluate this initial path
        s, m = evaluate_path_fast(init_path)
        if s > best_score:
            best_score = s
            best_path = init_path[:]
            best_masks = m.copy()
        
        current_path = init_path[:]
        current_masks = m.copy()
        current_score = s
        
        # Local search
        epsilon_xy_local = 0.2
        epsilon_a_local = 4.0
        no_improve_local = 0
        max_no_improve_local = 20
        
        while time.time() - start_time < time_limit:
            # Pick random control point
            idx = np.random.randint(len(current_path))
            p = current_path[idx]
            
            # Perturb all three coordinates
            new_x = p[0] + np.random.normal(0, epsilon_xy_local)
            new_y = p[1] + np.random.normal(0, epsilon_xy_local)
            new_a = p[2] + np.random.normal(0, epsilon_a_local)
            
            # Constrain angle to [0, 90]
            new_a = np.clip(new_a, 0.0, 90.0)
            
            new_p = (new_x, new_y, new_a)
            test_path = current_path[:]
            test_path[idx] = new_p
            
            # Fast evaluation with cached masks
            score, masks = evaluate_path_fast(test_path, current_masks)
            
            if score > current_score:
                current_score = score
                current_path = test_path
                current_masks = masks
                no_improve_local = 0
                
                if score > best_score:
                    best_score = score
                    best_path = current_path[:]
                    best_masks = current_masks.copy()
            else:
                no_improve_local += 1
                if no_improve_local >= max_no_improve_local:
                    # Shrink step sizes
                    epsilon_xy_local *= 0.8
                    epsilon_a_local *= 0.8
                    no_improve_local = 0
                    # If step sizes get too small, try a restart from best
                    if epsilon_xy_local < 0.02 and epsilon_a_local < 0.5:
                        current_path = best_path[:]
                        s2, m2 = evaluate_path_fast(current_path)
                        current_score = s2
                        current_masks = m2
                        epsilon_xy_local = 0.3
                        epsilon_a_local = 5.0
    
    # Final polish: high-precision local search around best path
    if time.time() - start_time < time_limit - 60:
        current_path = best_path[:]
        current_score = best_score
        current_masks = best_masks.copy()
        
        epsilon_xy_fine = 0.05
        epsilon_a_fine = 1.0
        no_improve_fine = 0
        max_no_improve_fine = 30
        
        while time.time() - start_time < time_limit - 10:
            idx = np.random.randint(len(current_path))
            p = current_path[idx]
            
            new_x = p[0] + np.random.normal(0, epsilon_xy_fine)
            new_y = p[1] + np.random.normal(0, epsilon_xy_fine)
            new_a = p[2] + np.random.normal(0, epsilon_a_fine)
            
            new_a = np.clip(new_a, 0.0, 90.0)
            
            new_p = (new_x, new_y, new_a)
            test_path = current_path[:]
            test_path[idx] = new_p
            
            score, masks = evaluate_path_fast(test_path, current_masks)
            
            if score > current_score:
                current_score = score
                current_path = test_path
                current_masks = masks
                no_improve_fine = 0
                
                if score > best_score:
                    best_score = score
                    best_path = current_path[:]
                    best_masks = current_masks.copy()
            else:
                no_improve_fine += 1
                if no_improve_fine >= max_no_improve_fine:
                    epsilon_xy_fine *= 0.7
                    epsilon_a_fine *= 0.7
                    no_improve_fine = 0
                    if epsilon_xy_fine < 0.005 and epsilon_a_fine < 0.1:
                        break
    
    return np.array(best_path)
