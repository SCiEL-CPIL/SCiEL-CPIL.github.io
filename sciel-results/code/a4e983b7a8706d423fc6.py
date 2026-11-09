import numpy as np
import time
from scipy.optimize import minimize
import sys

def squares_in_square10() -> tuple:
    """
    Optimizes the placement of 10 squares in a unit square to maximize the sum of side lengths.
    Uses a penalty-based local optimization approach with multiple restarts.
    """
    n = 10
    # Deadline guard: 630s limit, aim to finish in 600s
    deadline = time.time() + 590
    
    def unpack(x):
        """Unpack the flat vector into centers and sides."""
        cx = x[0::3]
        cy = x[1::3]
        s = x[2::3]
        return cx, cy, s

    def objective(x):
        """
        Objective function to minimize.
        Minimizes negative sum of sides plus penalties for violations.
        """
        cx, cy, s = unpack(x)
        
        # Penalty weight
        w = 100.0
        
        # 1. Boundary penalties
        pen = 0.0
        for i in range(n):
            # Left
            if cx[i] - s[i]/2 < 0:
                pen += (cx[i] - s[i]/2)**2
            # Right
            if cx[i] + s[i]/2 > 1:
                pen += (cx[i] + s[i]/2 - 1)**2
            # Bottom
            if cy[i] - s[i]/2 < 0:
                pen += (cy[i] - s[i]/2)**2
            # Top
            if cy[i] + s[i]/2 > 1:
                pen += (cy[i] + s[i]/2 - 1)**2
        
        # 2. Overlap penalties
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                req_x = (s[i] + s[j]) / 2
                req_y = (s[i] + s[j]) / 2
                
                # If they overlap in both x and y, penalize
                if dx < req_x and dy < req_y:
                    # Penalty is the squared depth of penetration
                    # Using a soft form to help gradient
                    pen_x = max(0, req_x - dx)
                    pen_y = max(0, req_y - dy)
                    pen += pen_x**2 + pen_y**2
        
        # Maximize sum of sides -> Minimize -sum(sides)
        score = -np.sum(s)
        
        return score + w * pen

    def is_valid(cx, cy, s):
        """Check if the configuration is strictly valid within tolerance."""
        tol = 1e-7
        # Bounds
        for i in range(n):
            if cx[i] - s[i]/2 < -tol or cx[i] + s[i]/2 > 1 + tol:
                return False
            if cy[i] - s[i]/2 < -tol or cy[i] + s[i]/2 > 1 + tol:
                return False
            if s[i] <= 0:
                return False
        
        # Overlaps
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                t_x = (s[i] + s[j]) / 2 - tol
                t_y = (s[i] + s[j]) / 2 - tol
                if dx < t_x and dy < t_y:
                    return False
        return True

    def get_score(cx, cy, s):
        return np.sum(s)

    # Initial guesses
    # 1. Uniform grid (3x3 + 1)
    # 2. Random valid-ish starts
    starts = []
    
    # Start 1: Grid 3x3 with one square removed and resized? 
    # Just use the baseline grid as a start, but perturb it.
    k = 3
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx0 = (cols + 0.5) * side
    cy0 = (rows + 0.5) * side
    s0 = np.full(n, side)
    # The 10th square in a 3x3 grid would be at (0,0) if we wrap, but we have 9 spots.
    # For n=10, the baseline puts the 10th at (0, 1) which is out of bounds if k=3?
    # Baseline code: cols = idx % 3, rows = idx // 3.
    # idx=9: col=0, row=3. cy = 3.5 * 1/3 = 1.16 > 1. Invalid baseline!
    # Wait, the baseline code in the prompt:
    # k = ceil(sqrt(10)) = 4.
    # side = 0.25.
    # idx=9: col=1, row=2. cx=0.75, cy=1.25? No.
    # rows = 9//4 = 2. cy = (2+0.5)*0.25 = 0.625.
    # So baseline is a 4x4 grid using 10 cells.
    
    cx_base = np.array([(i % 4 + 0.5) * 0.25 for i in range(10)])
    cy_base = np.array([(i // 4 + 0.5) * 0.25 for i in range(10)])
    s_base = np.array([0.25] * 10)
    x_base = np.concatenate([cx_base, cy_base, s_base]) # Wrong interleaving for unpack
    # unpack expects [cx1, cy1, s1, cx2, cy2, s2, ...]
    x_base = np.zeros(30)
    for i in range(10):
        x_base[3*i] = cx_base[i]
        x_base[3*i+1] = cy_base[i]
        x_base[3*i+2] = s_base[i]
    starts.append(x_base)
    
    # Start 2: Two large squares in bottom row, smaller ones above?
    # Let's try a random start that is roughly valid
    np.random.seed(42)
    for _ in range(5):
        cx_r = np.random.uniform(0.1, 0.9, n)
        cy_r = np.random.uniform(0.1, 0.9, n)
        s_r = np.random.uniform(0.1, 0.4, n)
        x_r = np.zeros(30)
        for i in range(10):
            x_r[3*i] = cx_r[i]
            x_r[3*i+1] = cy_r[i]
            x_r[3*i+2] = s_r[i]
        starts.append(x_r)
        
    # Start 3: Specific heuristic: 4 big, 6 small
    # 2x2 grid of side 0.4 (width 0.8)
    # 3x2 grid of side 0.2 (width 0.6, height 0.4)
    # Total height 0.8+0.4=1.2 > 1. Scale down.
    # Let's just add a few more random starts
    
    best_x = None
    best_score = -np.inf
    best_cx = None
    best_cy = None
    best_s = None
    
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.01, 1.0)] * 10 # cx, cy, s for each square
    
    for start in starts:
        if time.time() > deadline - 60:
            break
            
        try:
            res = minimize(objective, start, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-10})
            
            if res.success or res.fun < best_score:
                cx, cy, s = unpack(res.x)
                if is_valid(cx, cy, s):
                    score = get_score(cx, cy, s)
                    if score > best_score:
                        best_score = score
                        best_x = res.x
                        best_cx = cx
                        best_cy = cy
                        best_s = s
        except Exception:
            continue

    # If no valid solution found from optimization, fallback to a known valid simple packing
    # e.g. 4x4 grid (side 0.25)
    if best_cx is None:
        k = 4
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        best_cx = (cols + 0.5) * side
        best_cy = (rows + 0.5) * side
        best_s = np.full(n, side)
    else:
        # Final polish: ensure exact validity by clamping
        # The optimizer might leave tiny violations. We can try to project them back.
        # For simplicity, if it's valid, we return it.
        # If not, we might need to shrink slightly.
        if not is_valid(best_cx, best_cy, best_s):
             # Fallback to grid
             k = 4
             side = 1.0 / k
             idx = np.arange(n)
             cols = idx % k
             rows = idx // k
             best_cx = (cols + 0.5) * side
             best_cy = (rows + 0.5) * side
             best_s = np.full(n, side)

    centers = np.stack([best_cx, best_cy], axis=1).astype(np.float64)
    sides = best_s.astype(np.float64)
    
    return centers, sides

if __name__ == "__main__":
    c, s = squares_in_square10()
    print(f"Sum of sides: {np.sum(s)}")
