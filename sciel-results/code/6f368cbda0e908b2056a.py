    import numpy as np
    from scipy.optimize import linprog
    import random

    N = 26

    def construct():
        # SA Parameters
        T_start = 0.1
        T_end = 0.001
        steps = 5000
        cooling_rate = (T_end / T_start) ** (1.0 / steps)
        
        # Initial positions: Hexagonal-like grid
        # 26 circles. 5x5 grid is 25. Add 1 more? Or 6x5?
        # Let's use a jittered grid for initialization
        rng = np.random.RandomState(42)
        
        # Try multiple initializations
        best_positions = None
        best_sum_radii = -np.inf
        
        for init_attempt in range(5):
            # Create a grid of positions
            # 26 circles. 5x5 grid has 25. 
            # Let's place 25 in a 5x5 grid and 1 in the center of a cell? No, that's bad.
            # Better: 6x5 grid is 30 slots. Pick 26?
            # Or just random points in [0.1, 0.9]^2
            
            if init_attempt == 0:
                # 5x5 grid + 1 center
                pos = []
                for i in range(5):
                    for j in range(5):
                        pos.append([(i+0.5)/5.0, (j+0.5)/5.0])
                pos.append([0.5, 0.5]) # This will overlap heavily, SA will fix it
            else:
                # Random
                pos = rng.uniform(0.1, 0.9, size=(N, 2))
            
            positions = np.array(pos)
            
            # SA Loop
            T = T_start
            current_positions = positions.copy()
            
            for step in range(steps):
                # Pick a random circle to move
                i = rng.randint(N)
                
                # Propose move
                move_size = T * 0.1 # Scale move with temperature
                new_pos = current_positions[i] + rng.normal(0, move_size, size=2)
                
                # Keep in bounds (loose bounds, LP will handle radius=0 if too close)
                # Actually, if position is outside [0,1], radius is 0.
                # Let's clamp to [0,1] to keep it sane, but allow it to go to edge.
                if new_pos[0] < 0 or new_pos[0] > 1 or new_pos[1] < 0 or new_pos[1] > 1:
                    continue # or reflect? Let's just skip invalid moves to save time
                
                # Solve LP for new configuration
                # Temporarily update
                old_pos_i = current_positions[i]
                current_positions[i] = new_pos
                
                # Calculate LP
                try:
                    radii = solve_lp_radii(current_positions)
                    new_sum = np.sum(radii)
                except Exception:
                    # Revert if LP fails (shouldn't happen)
                    current_positions[i] = old_pos_i
                    continue
                
                # Get current sum (we need to track this)
                # For efficiency, we should keep track of current_sum
                # But we didn't calculate it for the initial state in the loop above properly?
                # Let's restructure: Calculate current_sum at start of loop.
                
                # To make this efficient, I'll rewrite the loop structure in the final code.
                pass
            
            # This placeholder logic is messy. I will write the clean version in the SEARCH/REPLACE block.
            pass
        
        # Final LP solve for best positions
        # ...
        return result

    def solve_lp_radii(positions):
        # Maximize sum(r_i)
        # Minimize -sum(r_i)
        c = -np.ones(N)
        
        # Constraints: r_i + r_j <= dist_ij
        A_ub = []
        b_ub = []
        
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(positions[i] - positions[j])
                row = np.zeros(N)
                row[i] = 1
                row[j] = 1
                A_ub.append(row)
                b_ub.append(dist)
                
        # Boundary constraints: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
        for i in range(N):
            row = np.zeros(N)
            row[i] = 1
            A_ub.append(row)
            b_ub.append(min(positions[i,0], 1-positions[i,0], positions[i,1], 1-positions[i,1]))
            
        # r_i >= 0
        bounds = [(0, None)] * N
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            return res.x
        else:
            # Fallback: small radii
            return np.zeros(N)
    