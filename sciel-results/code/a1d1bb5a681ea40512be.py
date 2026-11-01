"""Circle packing in perimeter-4 rectangle: Simulated Annealing with hard constraints."""
import numpy as np
import time

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-9):
    """Check if packing is valid."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _greedy_expand(x, y, r, w, h, time_budget):
    """Iteratively expand each circle to its maximum possible radius."""
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    improved = True
    rounds = 0
    while improved and rounds < 100:
        if time.time() > time_budget:
            break
        improved = False
        rounds += 1
        for i in range(N):
            # Max radius from walls
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            # Max radius from other circles
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)
            if r_max > r[i] + 1e-8:
                r[i] = r_max
                improved = True
    return x, y, r


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    best_value = 0.0
    best_result = None

    # Multi-start Simulated Annealing
    rng = np.random.default_rng(42)
    
    # Define a set of diverse initial aspect ratios
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.85, 1.15, 0.75, 1.25, 0.7]
    
    for w0 in aspect_ratios:
        h0 = 2.0 - w0
        if h0 < 0.3 or w0 < 0.3:
            continue
        
        # Initialize positions in a grid
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        for i in range(N):
            row, col = divmod(i, side)
            x[i] = (col + 0.5) / side * w0
            y[i] = (row + 0.5) / side * h0
            # Initial radius slightly smaller than cell size to ensure validity
            r[i] = min(w0 / side, h0 / side) * 0.4
            
        # Ensure initial state is valid
        if not _is_valid(x, y, r, w0, h0):
            # Shrink if invalid
            scale = 0.5
            r *= scale
            if not _is_valid(x, y, r, w0, h0):
                continue

        current_x = x.copy()
        current_y = y.copy()
        current_r = r.copy()
        current_w = w0
        current_h = h0
        
        # Expand radii initially
        current_x, current_y, current_r = _greedy_expand(current_x, current_y, current_r, current_w, current_h, deadline - 5)
        current_val = np.sum(current_r)
        
        if current_val > best_value:
            best_value = current_val
            best_result = np.column_stack((current_x, current_y, current_r))
            best_w = current_w

        # Simulated Annealing Loop
        T_start = 0.05
        T_end = 0.001
        T = T_start
        step_size = 0.05
        
        # We will run SA for a fixed number of steps or until time runs out
        # Each SA run should take roughly 10-15 seconds
        sa_deadline = start_time + (deadline - start_time) / len(aspect_ratios)
        
        n_steps = 20000
        
        for step in range(n_steps):
            if time.time() > sa_deadline:
                break
            
            # Pick a random circle
            i = rng.integers(0, N)
            
            # Generate a move
            # 1. Move position
            # 2. Change aspect ratio (less frequent)
            # 3. Nudge radius (handled by expansion later, but we can try to move to allow larger radius)
            
            move_type = rng.random()
            
            if move_type < 0.8:
                # Move position
                dx = rng.normal(0, step_size)
                dy = rng.normal(0, step_size)
                new_x = current_x[i] + dx
                new_y = current_y[i] + dy
                
                # Check hard constraints for new position
                if new_x < current_r[i] or new_x > current_w - current_r[i]:
                    continue
                if new_y < current_r[i] or new_y > current_h - current_r[i]:
                    continue
                
                valid = True
                for j in range(N):
                    if i == j:
                        continue
                    ddx = new_x - current_x[j]
                    ddy = new_y - current_y[j]
                    if ddx*ddx + ddy*ddy < (current_r[i] + current_r[j])**2 - 1e-9:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Calculate energy change (sum of radii)
                # Since we don't change radii immediately, energy change is 0 unless we can expand
                # To make SA effective, we evaluate the "potential" by expanding radii after the move
                # This is expensive. Instead, we accept the move if it's "neutral" or if we allow some overlap?
                # No, hard constraints.
                # Strategy: Accept move if it doesn't reduce the *maximum possible* sum?
                # Simpler: Just accept valid moves. Then expand.
                
                # To allow escaping local minima, we need to accept moves that might temporarily
                # not allow expansion, or we need to perturb radii down?
                # Actually, if we keep radii fixed, moving a circle usually doesn't change sum(r).
                # The value only increases when we expand.
                # So the "energy" is -sum(r).
                # If we move a circle, sum(r) stays same.
                # We need to allow moves that might "waste" space but allow future larger circles?
                # This is hard with fixed radii.
                
                # Alternative: Perturb radii slightly down to allow moves, then expand.
                # Let's try: Move position. If valid, accept.
                # Then expand. If sum(r) increases, keep. If not, revert?
                # This is essentially hill climbing with random restarts.
                
                # Let's try a different move: Perturb radius down slightly, move, then expand.
                # This allows "unpacking" and "repacking".
                
                # For now, let's just accept valid moves and expand.
                # If the expansion yields a better result, we keep it.
                # If not, we revert.
                # This is deterministic hill climbing on the expanded state.
                
                # To make it SA, we need to accept worse states.
                # But with hard constraints, "worse" means smaller sum(r).
                # We can't have a state with smaller sum(r) if we always expand to max.
                # So we must allow states that are NOT maximally expanded.
                
                # Revised Strategy:
                # State: (x, y, r) where r is NOT necessarily maximal.
                # Move: Perturb x, y, r.
                # Constraint: Validity of (x, y, r).
                # Energy: -sum(r).
                # After a move, we do NOT immediately expand.
                # We accept based on T.
                # Periodically (or at the end), we expand and update best.
                
                # This requires the state to be valid at all times.
                
                # Let's implement this.
                
                # Perturb r slightly down to allow moves?
                # No, just move x,y.
                
                # Actually, if we just move x,y, sum(r) is constant.
                # We need to perturb r to change the energy.
                # Let's perturb r[i] by a small delta.
                
                delta_r = rng.normal(0, 0.005)
                new_r = current_r[i] + delta_r
                if new_r < 0.01:
                    continue
                if new_r > min(new_x, current_w - new_x, new_y, current_h - new_y):
                    # Too big for walls
                    new_r = min(new_x, current_w - new_x, new_y, current_h - new_y) * 0.99
                    if new_r < 0.01:
                        continue
                
                # Check collisions with new r
                valid = True
                for j in range(N):
                    if i == j:
                        continue
                    ddx = new_x - current_x[j]
                    ddy = new_y - current_y[j]
                    if ddx*ddx + ddy*ddy < (new_r + current_r[j])**2 - 1e-9:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Calculate energy change
                old_sum = np.sum(current_r)
                new_sum = np.sum(current_r) - current_r[i] + new_r
                delta_e = old_sum - new_sum # We want to minimize -sum, so delta_e = -(new_sum - old_sum) = old_sum - new_sum
                
                # Acceptance
                if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
                    current_x[i] = new_x
                    current_y[i] = new_y
                    current_r[i] = new_r
                    
            elif move_type < 0.95:
                # Change aspect ratio
                dw = rng.normal(0, 0.02)
                new_w = current_w + dw
                new_h = 2.0 - new_w
                if new_w < 0.3 or new_h < 0.3:
                    continue
                
                # Scale positions to fit new box? Or just check if they fit?
                # If we change W,H, existing circles might go out of bounds.
                # We need to scale x,y or reject if out of bounds.
                # Let's scale x,y proportionally.
                scale_x = new_w / current_w
                scale_y = new_h / current_h
                
                new_x = current_x * scale_x
                new_y = current_y * scale_y
                # Radii remain same? If we scale box, relative positions change.
                # If we scale box, circles might overlap or go out.
                # This move is risky. Let's skip aspect ratio change for now to keep it simple.
                pass
            
            # Cooling
            T = T_start * (T_end / T_start) ** (step / n_steps)
            
            # Periodic expansion and best update
            if step % 100 == 0:
                # Expand radii to max
                temp_x, temp_y, temp_r = _greedy_expand(current_x, current_y, current_r, current_w, current_h, deadline - 5)
                if _is_valid(temp_x, temp_y, temp_r, current_w, current_h):
                    val = np.sum(temp_r)
                    if val > best_value:
                        best_value = val
                        best_result = np.column_stack((temp_x, temp_y, temp_r))
                        best_w = current_w
                        # Update current state to the expanded one to continue from there
                        current_x = temp_x
                        current_y = temp_y
                        current_r = temp_r

    # Final polish
    if best_result is not None and time.time() < deadline - 5:
        x = best_result[:, 0].copy()
        y = best_result[:, 1].copy()
        r = best_result[:, 2].copy()
        w = best_w
        h = 2.0 - w
        
        x, y, r = _greedy_expand(x, y, r, w, h, deadline - 1)
        if _is_valid(x, y, r, w, h):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = np.column_stack((x, y, r))

    # Fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i+0.5)/side, (j+0.5)/side, 0.5/side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Final safety shrink
    x = best_result[:, 0]
    y = best_result[:, 1]
    r = best_result[:, 2]
    w = best_w
    h = 2.0 - w
    
    for _ in range(5):
        r_trial = r * (1 - 1e-10)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break
            
    return np.column_stack((x, y, r))
