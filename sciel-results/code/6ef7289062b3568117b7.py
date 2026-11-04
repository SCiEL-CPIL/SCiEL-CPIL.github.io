<<<<<<< SEARCH
    def polish_solution(x_sol):
        """Final high-precision polish pass"""
        w_best = x_sol[0]
        h_best = 2.0 - w_best
        xs_best = x_sol[1:N+1].copy()
        ys_best = x_sol[N+1:2*N+1].copy()
        rs_best = x_sol[2*N+1:3*N+1].copy()
        
        # Multiple passes to resolve collisions and ensure validity
        for _ in range(50):
            changed = False
            
            # Ensure within bounds
            for i in range(N):
                max_r = min(xs_best[i], w_best - xs_best[i], ys_best[i], h_best - ys_best[i])
                if rs_best[i] > max_r:
                    rs_best[i] = max_r
                    changed = True
            
            # Resolve overlaps iteratively
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs_best[i] - xs_best[j]
                    dy = ys_best[i] - ys_best[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    min_d = rs_best[i] + rs_best[j]
                    if d < min_d - 1e-12:
                        # Shrink both radii proportionally
                        if d > 1e-10:
                            factor = d / (min_d * 1.001)
                            rs_best[i] *= factor
                            rs_best[j] *= factor
                        else:
                            # Move apart if very close
                            if dx == 0 and dy == 0:
                                xs_best[i] += 0.01
                                xs_best[j] -= 0.01
                        changed = True
            
            if not changed:
                break
        
        # Final validation
        for i in range(N):
            rs_best[i] = min(rs_best[i], xs_best[i], w_best - xs_best[i], ys_best[i], h_best - ys_best[i])
            if rs_best[i] < 0:
                rs_best[i] = 0
        
        return np.column_stack((xs_best, ys_best, rs_best))
=======
    def polish_solution(x_sol):
        """Final high-precision polish pass with tighter convergence"""
        w_best = x_sol[0]
        h_best = 2.0 - w_best
        xs_best = x_sol[1:N+1].copy()
        ys_best = x_sol[N+1:2*N+1].copy()
        rs_best = x_sol[2*N+1:3*N+1].copy()
        
        # Multiple passes to resolve collisions and ensure validity
        for it in range(100):
            changed = False
            
            # Ensure within bounds
            for i in range(N):
                max_r = min(xs_best[i], w_best - xs_best[i], ys_best[i], h_best - ys_best[i])
                if rs_best[i] > max_r:
                    rs_best[i] = max_r
                    changed = True
            
            # Resolve overlaps iteratively with gentler shrinkage
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs_best[i] - xs_best[j]
                    dy = ys_best[i] - ys_best[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    min_d = rs_best[i] + rs_best[j]
                    if d < min_d - 1e-10:
                        # Try to move apart first, then shrink if needed
                        if d > 1e-8:
                            # Move apart proportionally
                            overlap = min_d - d
                            move_x = (dx / d) * overlap * 0.5
                            move_y = (dy / d) * overlap * 0.5
                            xs_best[i] += move_x
                            ys_best[i] += move_y
                            xs_best[j] -= move_x
                            ys_best[j] -= move_y
                        else:
                            # Shrink radii
                            factor = d / (min_d * 1.0001)
                            rs_best[i] *= factor
                            rs_best[j] *= factor
                        changed = True
            
            # Re-check bounds after moving
            for i in range(N):
                xs_best[i] = np.clip(xs_best[i], 0, w_best)
                ys_best[i] = np.clip(ys_best[i], 0, h_best)
                max_r = min(xs_best[i], w_best - xs_best[i], ys_best[i], h_best - ys_best[i])
                if rs_best[i] > max_r:
                    rs_best[i] = max_r
                    changed = True
            
            if not changed:
                break
        
        # Final validation
        for i in range(N):
            rs_best[i] = min(rs_best[i], xs_best[i], w_best - xs_best[i], ys_best[i], h_best - ys_best[i])
            if rs_best[i] < 0:
                rs_best[i] = 0
        
        return np.column_stack((xs_best, ys_best, rs_best))
>>>>>>> REPLACE

