import numpy as np
import time
import math
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s safety margin
    
    def get_delaunay_neighbors(positions):
        """
        Get Delaunay neighbors for each point.
        Returns a list of lists of neighbor indices.
        """
        try:
            from scipy.spatial import Delaunay
            tri = Delaunay(positions)
            neighbors = [[] for _ in range(N)]
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(3):
                        if i != j:
                            a = simplex[i]
                            b = simplex[j]
                            if b not in neighbors[a]:
                                neighbors[a].append(b)
            return neighbors
        except Exception:
            # Fallback: all neighbors (complete graph)
            return [list(range(N)) for _ in range(N)]

    def objective_and_grad(params):
        """
        Objective: minimize negative sum of radii + penalty for violations.
        params: (3*N,) array [x_0..x_{N-1}, y_0..y_{N-1}, r_0..r_{N-1}]
        """
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        # Compute distances
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)
        
        # Objective value: -sum(r) + penalty
        obj = -np.sum(r)
        
        # Penalty terms
        # 1. Boundary violations
        # x - r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
        v_left = np.maximum(0, r - x)
        v_right = np.maximum(0, x + r - 1)
        v_bottom = np.maximum(0, r - y)
        v_top = np.maximum(0, y + r - 1)
        
        pen_boundary = np.sum(v_left**2) + np.sum(v_right**2) + np.sum(v_bottom**2) + np.sum(v_top**2)
        
        # 2. Overlap violations (only for Delaunay neighbors to reduce cost, but for gradient we need all? 
        # Actually, if we use all pairs, it's O(N^2). N=26 is small. Let's use all pairs for correctness.)
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        viol = np.maximum(0, r_sum - dist)
        # Zero out diagonal
        viol = viol * (1 - np.eye(N))
        pen_overlap = np.sum(viol**2)
        
        total_penalty = pen_boundary + pen_overlap
        obj += 100.0 * total_penalty
        
        # Gradient
        grad = np.zeros(3*N)
        
        # Gradient of -sum(r)
        grad[2*N:3*N] -= 1.0
        
        # Gradient of boundary penalties
        # d/dx (r-x)^2 = -2(r-x) if r>x
        grad[:N] += -2 * v_left
        # d/dx (x+r-1)^2 = 2(x+r-1) if x+r>1
        grad[:N] += 2 * v_right
        # d/dy (r-y)^2 = -2(r-y) if r>y
        grad[N:2*N] += -2 * v_bottom
        # d/dy (y+r-1)^2 = 2(y+r-1) if y+r>1
        grad[N:2*N] += 2 * v_top
        
        # d/dr (r-x)^2 = 2(r-x) if r>x
        grad[2*N:3*N] += 2 * v_left
        # d/dr (x+r-1)^2 = 2(x+r-1) if x+r>1
        grad[2*N:3*N] += 2 * v_right
        # d/dr (r-y)^2 = 2(r-y) if r>y
        grad[2*N:3*N] += 2 * v_bottom
        # d/dr (y+r-1)^2 = 2(y+r-1) if y+r>1
        grad[2*N:3*N] += 2 * v_top
        
        # Gradient of overlap penalties
        # f_ij = (r_i + r_j - d_ij)^2
        # d/dx_i = 2 * viol_ij * (-(x_i - x_j) / d_ij)
        # d/dr_i = 2 * viol_ij
        
        # We need to sum contributions from all j != i
        # This is O(N^2)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                v = viol[i, j]
                if v > 0:
                    # Contribution to grad[i]
                    # d/dx_i: 2 * v * (-(x_i - x_j) / d_ij)
                    # Note: viol is symmetric, so we can just compute for i<j and add to both?
                    # No, let's just do the full sum.
                    # Actually, the term is (r_i+r_j - d_ij).
                    # d/dx_i (r_i+r_j - d_ij)^2 = 2(r_i+r_j - d_ij) * (-d/dx_i d_ij)
                    # d/dx_i d_ij = (x_i - x_j) / d_ij
                    # So d/dx_i = 2 * v * (-(x_i - x_j) / d_ij) = 2 * v * (x_j - x_i) / d_ij
                    
                    d_ij = dist[i, j]
                    if d_ij > 1e-9:
                        grad[i] += 2 * v * (x[j] - x[i]) / d_ij
                        grad[N+i] += 2 * v * (y[j] - y[i]) / d_ij
                        grad[2*N+i] += 2 * v
                        
        # Scale penalty gradient
        grad *= 1.0 # The 100.0 is in the objective, so gradient should be scaled too?
        # Wait, I added 100 * penalty to obj. So grad should be 100 * grad_penalty.
        # Let's re-calculate.
        # obj = -sum(r) + 100 * (pen_boundary + pen_overlap)
        # grad = -1 (for r) + 100 * (grad_boundary + grad_overlap)
        
        # My current grad calculation:
        # grad[2*N:3*N] -= 1.0 (Correct)
        # grad_boundary parts are currently unscaled.
        # grad_overlap parts are currently unscaled.
        
        # So I need to multiply the penalty gradients by 100.
        
        # Let's redo the gradient accumulation properly.
        grad = np.zeros(3*N)
        grad[2*N:3*N] -= 1.0
        
        # Boundary
        grad[:N] += 100.0 * (-2 * v_left + 2 * v_right)
        grad[N:2*N] += 100.0 * (-2 * v_bottom + 2 * v_top)
        grad[2*N:3*N] += 100.0 * (2 * v_left + 2 * v_right + 2 * v_bottom + 2 * v_top)
        
        # Overlap
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                v = viol[i, j]
                if v > 0:
                    d_ij = dist[i, j]
                    if d_ij > 1e-9:
                        grad[i] += 100.0 * 2 * v * (x[j] - x[i]) / d_ij
                        grad[N+i] += 100.0 * 2 * v * (y[j] - y[i]) / d_ij
                        grad[2*N+i] += 100.0 * 2 * v
        
        return obj, grad

    def is_valid(x, eps=1e-8):
        x_pos = x[:, 0]
        y_pos = x[:, 1]
        r = x[:, 2]
        
        if np.any(x_pos - r < -eps) or np.any(y_pos - r < -eps):
            return False
        if np.any(x_pos + r > 1 + eps) or np.any(y_pos + r > 1 + eps):
            return False
            
        dx = x_pos[:, np.newaxis] - x_pos[np.newaxis, :]
        dy = y_pos[:, np.newaxis] - y_pos[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2)
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        viol = np.maximum(0, r_sum - dist)
        viol = viol * mask
        
        if np.any(viol > eps):
            return False
            
        return True

    best_x = None
    best_score = 0.0

    # Initial guess: Grid
    side = 5
    positions = []
    for i in range(side):
        for j in range(side):
            positions.append([(i + 0.5) / side, (j + 0.5) / side])
    positions.append([0.5, 0.5])
    positions = np.array(positions)
    radii = np.ones(N) * 0.08
    
    x0 = np.zeros(3*N)
    x0[:N] = positions[:, 0]
    x0[N:2*N] = positions[:, 1]
    x0[2*N:3*N] = radii

    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))

    # Run SLSQP
    try:
        res = minimize(
            lambda p: objective_and_grad(p)[0],
            x0,
            jac=lambda p: objective_and_grad(p)[1],
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        
        if res.success or res.fun < 1.0: # If it converged or is good enough
            best_x = res.x.reshape(N, 3)
            # Check validity
            if is_valid(best_x, eps=1e-7):
                best_score = np.sum(best_x[:, 2])
            else:
                # Try to fix by shrinking radii slightly
                # Or just accept if penalty is low?
                # For now, let's try to polish
                pass
    except Exception as e:
        pass

    # If SLSQP didn't yield a valid solution, try a local polish from the result
    if best_x is not None:
        # Verify again
        if not is_valid(best_x, eps=1e-7):
            # Shrink radii to make valid
            # This is a simple heuristic
            for _ in range(100):
                if time.time() > deadline:
                    break
                valid = is_valid(best_x, eps=1e-8)
                if valid:
                    break
                # Find max violation
                x_pos = best_x[:, 0]
                y_pos = best_x[:, 1]
                r = best_x[:, 2]
                dx = x_pos[:, np.newaxis] - x_pos[np.newaxis, :]
                dy = y_pos[:, np.newaxis] - y_pos[np.newaxis, :]
                dist = np.sqrt(dx**2 + dy**2)
                r_sum = r[:, np.newaxis] + r[np.newaxis, :]
                mask = np.triu(np.ones((N, N), dtype=bool), k=1)
                viol = np.maximum(0, r_sum - dist)
                viol = viol * mask
                max_viol = np.max(viol)
                if max_viol > 0:
                    # Shrink all radii by a small amount
                    best_x[:, 2] *= (1 - 0.01)
                # Also check boundaries
                if np.any(x_pos - r < 0) or np.any(y_pos - r < 0) or np.any(x_pos + r > 1) or np.any(y_pos + r > 1):
                    best_x[:, 2] *= 0.95

        if is_valid(best_x, eps=1e-7):
            best_score = np.sum(best_x[:, 2])
        else:
            best_x = None

    # Fallback if no valid solution found
    if best_x is None:
        x = np.zeros((N, 3))
        for i in range(N):
            row = i // 5
            col = i % 5
            x[i, 0] = (col + 0.5) / 5
            x[i, 1] = (row + 0.5) / 5
            x[i, 2] = 0.5 / 5 - 0.01
        best_x = x
        best_score = np.sum(best_x[:, 2])

    return best_x
