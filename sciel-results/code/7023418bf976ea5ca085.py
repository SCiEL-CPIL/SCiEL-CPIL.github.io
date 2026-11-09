import numpy as np
import time
import math

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    time_limit = 590.0  # Safety margin for 600s limit
    
    # Helper to check validity
    def is_valid(cx, cy, s, tol=1e-7):
        if not np.all(np.isfinite(cx)) or not np.all(np.isfinite(cy)) or not np.all(np.isfinite(s)):
            return False
        if np.any(s <= 0):
            return False
        for i in range(n):
            if cx[i] - s[i]/2 < -tol or cx[i] + s[i]/2 > 1 + tol:
                return False
            if cy[i] - s[i]/2 < -tol or cy[i] + s[i]/2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j])/2.0 - tol
                if abs(cx[i] - cx[j]) < t and abs(cy[i] - cy[j]) < t:
                    return False
        return True

    # Helper to compute score
    def score(cx, cy, s):
        return np.sum(s)

    # Initial deterministic construction
    # Strategy: 4 large squares in corners, 6 smaller in gaps?
    # Let's try a simple grid first to ensure a valid start, then refine.
    # A 3x3 grid of 9 squares + 1 in center?
    # Or 4 large (0.4), 6 small (0.2)?
    # Let's construct a specific layout known to be valid and decent.
    # Layout:
    # 4 squares of side 0.4 in corners: (0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)
    # Remaining space is a cross.
    # Center square of side 0.4? No, overlaps with corners?
    # Distance between (0.2,0.2) and (0.8,0.2) is 0.6. Sum of halves 0.4. 0.6 > 0.4. OK.
    # Distance between (0.2,0.2) and (0.2,0.8) is 0.6. OK.
    # Can we fit a central square?
    # If we place 4 squares of side 0.4, they occupy [0,0.4]x[0,0.4], [0.6,1]x[0,0.4], etc.
    # The central region [0.4,0.6]x[0.4,0.6] is free?
    # Square at (0.2, 0.2) covers x in [0, 0.4].
    # Square at (0.8, 0.2) covers x in [0.6, 1.0].
    # So x in [0.4, 0.6] is free between them.
    # Same for y.
    # So a square of side 0.2 centered at (0.5, 0.5) fits perfectly in the middle.
    # That's 5 squares. We need 10.
    # We can split the remaining spaces.
    
    # Let's try a different initial layout: 10 squares in a 2x5 grid?
    # Side 0.2. Sum = 2.0.
    # Or 4 large, 6 medium.
    
    # Let's use a simple valid grid to start, then let the optimizer work.
    # 4x4 grid is 16 squares. We need 10.
    # Let's place 10 squares in a 3x3 grid (9) + 1 extra?
    # Simpler: Start with 10 squares of side 0.3 in a grid?
    # 0.3 * 3 = 0.9 < 1.0.
    # Centers:
    # 0.15, 0.5, 0.85
    # This gives 9 squares. Add one more?
    
    # Let's try to maximize by construction first.
    # Known good configuration:
    # 4 squares of side 0.4 in corners.
    # 6 squares of side 0.2 in the remaining strips?
    # Top strip: y in [0.4, 1]. Height 0.6.
    # Left part of top strip: x in [0, 0.4]. Width 0.4.
    # We can fit two squares of side 0.4? No, height is 0.6, width 0.4.
    # Two squares of side 0.4 side by side? No, width 0.4.
    # One square of side 0.4 at (0.2, 0.6)?
    # Let's just use a robust random initialization with multiple restarts and local search.
    
    best_cx = None
    best_cy = None
    best_s = None
    best_val = -np.inf
    
    rng = np.random.default_rng(42)
    
    # Local Refinement Function
    def refine(cx, cy, s):
        # Coordinate descent
        # Try moving each square in x, y, and scaling s
        # Step size decreases over time
        for iter in range(200):
            improved = False
            for i in range(n):
                # Try scaling up s
                # Max possible s for square i given others?
                # Simple: try small increases
                for delta_s in [0.01, 0.005, 0.001, 0.0001]:
                    new_s = s.copy()
                    new_s[i] += delta_s
                    if new_s[i] > 1.0: continue
                    if is_valid(cx, cy, new_s):
                        if np.sum(new_s) > np.sum(s) + 1e-9:
                            s = new_s
                            improved = True
                            break
                # Try moving x
                for delta_x in [0.01, -0.01, 0.005, -0.005, 0.001, -0.001]:
                    new_cx = cx.copy()
                    new_cx[i] += delta_x
                    if is_valid(new_cx, cy, s):
                        if np.sum(s) > np.sum(s) + 1e-9: # Score doesn't change with move, but might allow future scaling
                            # We only accept moves if they enable scaling? 
                            # For now, accept valid moves to explore space, but prefer scaling.
                            # Actually, just accepting valid moves can wander.
                            # Let's only accept moves if they allow a subsequent scale up?
                            # That's complex. Let's just do random valid moves to escape local minima?
                            # No, let's stick to scaling up only for score improvement.
                            # But we might be stuck.
                            # Let's try "pushing" squares apart to create space.
                            pass
                # Try moving y
                for delta_y in [0.01, -0.01, 0.005, -0.005, 0.001, -0.001]:
                    new_cy = cy.copy()
                    new_cy[i] += delta_y
                    if is_valid(cx, new_cy, s):
                        pass
            
            # If no improvement in scaling, try random perturbation to escape local minima?
            # Or try a larger scale jump?
            # For simplicity, if no improvement, break or continue with smaller steps?
            # Let's add a simple "shake" if stuck for a few iterations.
            
            if not improved:
                # Try a random valid perturbation to change the landscape
                if rng.random() < 0.1:
                    idx = rng.integers(0, n)
                    dx = rng.uniform(-0.05, 0.05)
                    dy = rng.uniform(-0.05, 0.05)
                    new_cx = cx.copy()
                    new_cy = cy.copy()
                    new_cx[idx] += dx
                    new_cy[idx] += dy
                    if is_valid(new_cx, new_cy, s):
                        cx = new_cx
                        cy = new_cy
            
            if time.time() - start_time > time_limit - 10:
                break
                
        return cx, cy, s

    # Multiple restarts with different initial layouts
    layouts = []
    
    # Layout 1: Grid 3x3 + 1
    cx1 = np.array([0.2, 0.5, 0.8, 0.2, 0.5, 0.8, 0.2, 0.5, 0.8, 0.5])
    cy1 = np.array([0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8, 0.5])
    s1 = np.full(10, 0.3)
    if is_valid(cx1, cy1, s1):
        layouts.append((cx1, cy1, s1))
        
    # Layout 2: 4 large corners, 6 small
    cx2 = np.array([0.2, 0.8, 0.2, 0.8, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    cy2 = np.array([0.2, 0.2, 0.8, 0.8, 0.5, 0.2, 0.8, 0.2, 0.8, 0.5])
    # 4 large (0.4), 6 small (0.2)
    s2 = np.array([0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
    # Check validity of Layout 2
    # Corners: (0.2,0.2) s=0.4 -> [0,0.4]x[0,0.4]
    # (0.8,0.2) s=0.4 -> [0.6,1.0]x[0.6,1.0] NO, y is 0.2. [0.6,1.0]x[0,0.4]
    # (0.2,0.8) s=0.4 -> [0,0.4]x[0.6,1.0]
    # (0.8,0.8) s=0.4 -> [0.6,1.0]x[0.6,1.0]
    # Small at (0.5, 0.5) s=0.2 -> [0.4,0.6]x[0.4,0.6]
    # Small at (0.5, 0.2) s=0.2 -> [0.4,0.6]x[0.1,0.3]
    #   Overlap with (0.2,0.2)? x: [0,0.4] vs [0.4,0.6]. Touching. OK.
    #   Overlap with (0.8,0.2)? x: [0.6,1.0] vs [0.4,0.6]. Touching. OK.
    # Small at (0.5, 0.8) s=0.2 -> [0.4,0.6]x[0.7,0.9]
    # Small at (0.2, 0.5) s=0.2 -> [0.1,0.3]x[0.4,0.6]
    # Small at (0.8, 0.5) s=0.2 -> [0.7,0.9]x[0.4,0.6]
    # Small at (0.5, 0.5) s=0.2 -> [0.4,0.6]x[0.4,0.6]
    #   Overlap with (0.2,0.5)? x: [0.1,0.3] vs [0.4,0.6]. OK.
    #   Overlap with (0.8,0.5)? x: [0.7,0.9] vs [0.4,0.6]. OK.
    #   Overlap with (0.5,0.2)? y: [0.1,0.3] vs [0.4,0.6]. OK.
    #   Overlap with (0.5,0.8)? y: [0.7,0.9] vs [0.4,0.6]. OK.
    #   Overlap with corners?
    #   (0.2,0.2) x[0,0.4] y[0,0.4]. Small (0.5,0.5) x[0.4,0.6] y[0.4,0.6]. Touching at corner. OK.
    # All seem valid.
    if is_valid(cx2, cy2, s2):
        layouts.append((cx2, cy2, s2))
        
    # Random layouts
    for _ in range(20):
        cx_r = rng.uniform(0.1, 0.9, n)
        cy_r = rng.uniform(0.1, 0.9, n)
        s_r = rng.uniform(0.1, 0.4, n)
        if is_valid(cx_r, cy_r, s_r):
            layouts.append((cx_r, cy_r, s_r))
            
    if not layouts:
        # Fallback grid
        k = 4
        side = 1.0/k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        s = np.full(n, side)
        return np.stack([cx, cy], axis=1), s

    # Refine each layout
    for cx0, cy0, s0 in layouts:
        if time.time() - start_time > time_limit - 10:
            break
        cx, cy, s = refine(cx0.copy(), cy0.copy(), s0.copy())
        val = score(cx, cy, s)
        if val > best_val:
            best_val = val
            best_cx = cx
            best_cy = cy
            best_s = s
            
    if best_cx is None:
        # Should not happen
        k = 4
        side = 1.0/k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        s = np.full(n, side)
        return np.stack([cx, cy], axis=1), s

    centers = np.stack([best_cx, best_cy], axis=1)
    sides = best_s
    
    return centers, sides

if __name__ == "__main__":
    c, s = squares_in_square10()
    print(c)
    print(s)
    print(np.sum(s))
