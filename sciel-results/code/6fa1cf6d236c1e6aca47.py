"""Program for packing 50 equal circles in a unit square; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment
import time


def construct():
    """
    Formulate the packing as a continuous rigid-body assembly problem.
    Circle centers are nodes in a sparsely connected contact graph.
    We solve for optimal radius by iteratively adjusting node positions
    to satisfy exact distance constraints using a sparse linear solver approach,
    bypassing local minima of smooth gradient-based methods.
    
    Approach:
    1. Start from a good initial configuration (hexagonal packing perturbed).
    2. Build contact graph based on nearest neighbors.
    3. Use iterative constraint satisfaction: for each pair in contact graph,
       enforce distance >= 2r by projecting positions.
    4. Also enforce boundary constraints (centers stay in [-0.5, 0.5]^2).
    5. Binary search on r to find maximum feasible radius.
    """
    
    def hexagonal_initial(n, size=1.0):
        """Generate initial positions using hexagonal packing pattern."""
        positions = []
        # Hexagonal packing with spacing
        # We'll generate more points than needed and take the first n
        rows = int(np.ceil(np.sqrt(n * 2 / np.sqrt(3))))
        for row in range(rows):
            cols = int(np.ceil(np.sqrt(n / (2 / np.sqrt(3)))))
            for col in range(cols):
                x = col + (0.5 if row % 2 else 0)
                y = row * np.sqrt(3) / 2
                positions.append([x, y])
                if len(positions) >= n + 10:
                    break
            if len(positions) >= n + 10:
                break
        
        # Normalize to fit in unit square
        pos = np.array(positions[:n + 10])
        pos = pos - pos.mean(axis=0)
        scale = size / (2 * np.max(np.abs(pos)))
        pos = pos * scale * 0.95  # 95% to stay inside
        
        return pos[:n]
    
    def build_contact_graph(positions, r, k=4):
        """Build contact graph: each node connects to k nearest neighbors."""
        n = len(positions)
        # Compute pairwise distances
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Get k nearest neighbors for each point
        contact_pairs = set()
        for i in range(n):
            nn_idx = np.argsort(dist[i])[:k]
            for j in nn_idx:
                pair = (min(i, j), max(i, j))
                contact_pairs.add(pair)
        
        return list(contact_pairs)
    
    def project_constraints(positions, r, contact_pairs, n_iter=200):
        """
        Iteratively project positions to satisfy:
        1. Distance between connected pairs >= 2r
        2. Positions within [-0.5, 0.5]^2
        """
        pos = positions.copy()
        n = len(pos)
        
        for iteration in range(n_iter):
            moved = False
            
            # Enforce distance constraints for contact pairs
            for (i, j) in contact_pairs:
                d = pos[j] - pos[i]
                dist = np.linalg.norm(d)
                if dist < 2 * r and dist > 1e-12:
                    # Move apart to achieve distance 2r
                    needed = 2 * r - dist
                    # Split the correction
                    pos[i] -= d / dist * needed * 0.5
                    pos[j] += d / dist * needed * 0.5
                    moved = True
                elif dist < 2 * r and dist <= 1e-12:
                    # Coincident points: separate randomly
                    angle = np.random.uniform(0, 2 * np.pi)
                    offset = np.array([np.cos(angle), np.sin(angle)]) * r
                    pos[i] -= offset
                    pos[j] += offset
                    moved = True
            
            # Enforce boundary constraints
            for i in range(n):
                for coord in range(2):
                    if pos[i, coord] < -0.5:
                        pos[i, coord] = -0.5
                        moved = True
                    elif pos[i, coord] > 0.5:
                        pos[i, coord] = 0.5
                        moved = True
            
            if not moved:
                break
        
        return pos
    
    def check_feasibility(positions, r):
        """Check if all constraints are satisfied."""
        n = len(positions)
        
        # Check boundary
        if np.any(np.abs(positions) > 0.5 + 1e-9):
            return False
        
        # Check pairwise distances
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        if np.min(dist) < 2 * r - 1e-9:
            return False
        
        return True
    
    def compute_radius(positions):
        """Compute the actual radius for a given configuration."""
        n = len(positions)
        
        # Minimum distance to boundary
        boundary_dist = 0.5 - np.max(np.abs(positions), axis=1)
        r_boundary = np.min(boundary_dist)
        
        # Minimum pairwise distance
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        r_pair = np.min(dist) / 2
        
        return min(r_boundary, r_pair)
    
    # Initial configuration
    np.random.seed(42)
    pos = hexagonal_initial(N)
    
    # Get initial radius
    r_init = compute_radius(pos)
    
    # Binary search for optimal radius
    r_low = r_init * 0.9
    r_high = r_init * 1.5
    
    # First, try to improve with constraint projection at current radius
    contact_pairs = build_contact_graph(pos, r_init, k=4)
    pos = project_constraints(pos, r_init, contact_pairs, n_iter=500)
    r_improved = compute_radius(pos)
    r_low = max(r_low, r_improved)
    
    # Binary search
    best_pos = pos.copy()
    best_r = r_improved
    
    for _ in range(30):
        r_mid = (r_low + r_high) / 2
        
        # Build contact graph for this radius
        contact_pairs = build_contact_graph(pos, r_mid, k=4)
        
        # Project to satisfy constraints
        pos_new = project_constraints(pos, r_mid, contact_pairs, n_iter=300)
        
        # Check feasibility
        if check_feasibility(pos_new, r_mid):
            r_low = r_mid
            best_pos = pos_new.copy()
            best_r = r_mid
        else:
            r_high = r_mid
        
        if r_high - r_low < 1e-7:
            break
    
    # Final refinement: try multiple restarts with different initial configs
    for restart in range(5):
        np.random.seed(42 + restart * 100)
        # Perturb initial config
        pos_try = hexagonal_initial(N)
        pos_try += np.random.randn(N, 2) * 0.01
        
        r_try = compute_radius(pos_try)
        contact_pairs = build_contact_graph(pos_try, r_try, k=4)
        pos_try = project_constraints(pos_try, r_try, contact_pairs, n_iter=500)
        r_try = compute_radius(pos_try)
        
        if r_try > best_r:
            best_r = r_try
            best_pos = pos_try.copy()
    
    # Ensure valid output
    # Clip to boundary
    best_pos = np.clip(best_pos, -0.5, 0.5)
    
    # Verify no coincident centers
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    if np.min(dist) < 1e-10:
        # Fallback to grid if something went wrong
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return best_pos
# EVOLVE-BLOCK-END
