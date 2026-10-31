"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import linprog
from scipy.spatial.distance import cdist
import time

N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
from scipy.spatial.distance import cdist
import time

def _contact_graph(positions, radii, tol=1e-6):
    """Build contact graph: edge (i,j) if dist(i,j) <= r_i + r_j + tol."""
    n = len(positions)
    dists = cdist(positions, positions)
    graph = []
    for i in range(n):
        for j in range(i + 1, n):
            if dists[i, j] <= radii[i] + radii[j] + tol:
                graph.append((i, j))
    return graph

def _solve_lp(positions, graph, n):
    """
    Given fixed positions and a contact graph, solve the LP to maximize sum of radii
    subject to:
      r_i + r_j >= dist(i,j) for all (i,j) in graph (non-overlap)
      r_i + r_j >= dist(i,j) for all pairs (to be safe, use all pairs? No, just graph is enough if graph is correct)
      r_i >= 0
      r_i + x_i <= 1 (left wall)
      r_i + x_i >= 0 -> r_i >= -x_i (but x_i >= 0 so r_i >= 0 covers this if we also have r_i >= 0)
      Actually: x_i - r_i >= 0 and x_i + r_i <= 1
      y_i - r_i >= 0 and y_i + r_i <= 1
      
    Wait, the constraints are:
      r_i >= 0
      x_i - r_i >= 0  => -x_i + r_i <= 0
      x_i + r_i <= 1
      y_i - r_i >= 0  => -y_i + r_i <= 0
      y_i + r_i <= 1
      
    For each pair (i,j) in graph:
      r_i + r_j >= dist(i,j)  => -r_i - r_j <= -dist(i,j)
    
    Objective: maximize sum(r_i) => minimize -sum(r_i)
    """
    n = len(positions)
    # Variables: r_0, ..., r_{n-1}
    # Objective: minimize -sum(r_i)
    c = -np.ones(n)
    
    A_ub = []
    b_ub = []
    
    # Wall constraints
    for i in range(n):
        # -x_i + r_i <= 0
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(positions[i, 0])
        
        # x_i + r_i <= 1
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0 - positions[i, 0])
        
        # -y_i + r_i <= 0
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(positions[i, 1])
        
        # y_i + r_i <= 1
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0 - positions[i, 1])
    
    # Pair constraints from contact graph
    for (i, j) in graph:
        d = np.linalg.norm(positions[i] - positions[j])
        row = np.zeros(n)
        row[i] = -1.0
        row[j] = -1.0
        A_ub.append(row)
        b_ub.append(-d)
    
    bounds = [(0, None)] * n
    
    if len(A_ub) == 0:
        return np.zeros(n)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
        else:
            # Fallback: simple estimate
            return np.maximum(0, np.minimum(
                positions[:, 0], 1 - positions[:, 0],
                positions[:, 1], 1 - positions[:, 1]
            ) * 0.5)
    except Exception:
        return np.maximum(0, np.minimum(
            positions[:, 0], 1 - positions[:, 0],
            positions[:, 1], 1 - positions[:, 1]
        ) * 0.5)

def _improve_positions(positions, radii, n, max_iter=5000, lr=0.01):
    """
    Gradient-based position optimization with fixed radii initially, 
    then alternating with LP radius updates.
    Uses a penalty method for non-overlap.
    """
    positions = positions.copy()
    radii = radii.copy()
    
    for it in range(max_iter):
        # Compute forces
        forces = np.zeros_like(positions)
        
        # Wall repulsion
        for i in range(n):
            # Left wall
            if positions[i, 0] - radii[i] < 0:
                forces[i, 0] += (0 - (positions[i, 0] - radii[i])) * 100
            # Right wall
            if positions[i, 0] + radii[i] > 1:
                forces[i, 0] -= (positions[i, 0] + radii[i] - 1) * 100
            # Bottom wall
            if positions[i, 1] - radii[i] < 0:
                forces[i, 1] += (0 - (positions[i, 1] - radii[i])) * 100
            # Top wall
            if positions[i, 1] + radii[i] > 1:
                forces[i, 1] -= (positions[i, 1] + radii[i] - 1) * 100
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                d = positions[i] - positions[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    d = np.array([0.001, 0.001])
                    dist = np.linalg.norm(d)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    force_mag = overlap * 1000 / dist
                    forces[i] += d * force_mag
                    forces[j] -= d * force_mag
        
        # Update positions
        positions += lr * forces
        # Clip to [0,1]
        positions = np.clip(positions, 0, 1)
        
        # Periodically update radii via LP
        if it % 50 == 0:
            graph = _contact_graph(positions, radii)
            new_radii = _solve_lp(positions, graph, n)
            # Only accept if it improves
            if np.sum(new_radii) > np.sum(radii) * 0.99:
                radii = new_radii
    
    # Final LP solve
    graph = _contact_graph(positions, radii)
    radii = _solve_lp(positions, graph, n)
    
    return positions, radii

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_positions = None
    best_radii = None
    best_score = -1
    
    # Multiple random restarts
    np.random.seed(42)
    n_restarts = 20
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Initialize positions randomly
        positions = np.random.rand(N, 2)
        
        # Initial radii: small
        radii = np.full(N, 0.02)
        
        # Run improvement
        positions, radii = _improve_positions(positions, radii, N, max_iter=2000, lr=0.005)
        
        # Validate
        valid = True
        for i in range(N):
            if positions[i, 0] - radii[i] < -1e-8 or positions[i, 0] + radii[i] > 1 + 1e-8:
                valid = False
                break
            if positions[i, 1] - radii[i] < -1e-8 or positions[i, 1] + radii[i] > 1 + 1e-8:
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < radii[i] + radii[j] - 1e-8:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_positions = positions.copy()
                best_radii = radii.copy()
        
        # If we have time, do a finer polish on the best so far
        if best_positions is not None and time.time() < deadline - 30:
            positions, radii = _improve_positions(best_positions, best_radii, N, max_iter=3000, lr=0.002)
            # Validate
            valid = True
            for i in range(N):
                if positions[i, 0] - radii[i] < -1e-8 or positions[i, 0] + radii[i] > 1 + 1e-8:
                    valid = False
                    break
                if positions[i, 1] - radii[i] < -1e-8 or positions[i, 1] + radii[i] > 1 + 1e-8:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i + 1, N):
                        dist = np.linalg.norm(positions[i] - positions[j])
                        if dist < radii[i] + radii[j] - 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
            if valid:
                score = np.sum(radii)
                if score > best_score:
                    best_score = score
                    best_positions = positions.copy()
                    best_radii = radii.copy()
    
    if best_positions is None:
        # Fallback: grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    result = np.column_stack([best_positions, best_radii])
    return result

# EVOLVE-BLOCK-END
