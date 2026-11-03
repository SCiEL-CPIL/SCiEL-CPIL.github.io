"""Structural change: Delaunay triangulation-based max-min edge optimization with targeted bottleneck forces."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.spatial import Delaunay

_DEADLINE = 120.0  # Safety margin below 150s kill


def _compute_radius(centers):
    """Return the largest common radius for given centers in [-0.5,0.5]^2."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Boundary constraint
    boundary = np.min(np.minimum(
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ))
    if boundary < 0:
        return 0.0
    # Pairwise distances - full check for accuracy
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    return min(boundary, min_pair / 2.0)


def _initial_hex_grid(n_points):
    """Generate initial hexagonal lattice configuration."""
    # Hexagonal packing: rows offset by half spacing
    # Estimate rows needed
    # For hex packing in square [-0.5, 0.5]^2
    # Spacing s, row height = s * sqrt(3)/2
    # We want ~200 points
    
    best_points = None
    best_count = 0
    
    # Try various spacings
    for s in np.linspace(0.04, 0.12, 50):
        row_h = s * np.sqrt(3) / 2
        # How many rows fit?
        n_rows = int(1.0 / row_h) + 1
        # Points per row
        n_per_row = int(1.0 / s) + 1
        
        points = []
        for i in range(n_rows):
            y = -0.5 + row_h * (i + 0.5)
            if y > 0.5 - s/2:
                continue
            offset = 0 if i % 2 == 0 else s / 2
            for j in range(n_per_row + 1):
                x = -0.5 + s * (j + 0.5) + offset
                if x <= 0.5 - s/2:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) >= n_points and len(points) > best_count:
            # Keep the n_points closest to center for symmetry
            center_dist = np.sum(points ** 2, axis=1)
            idx = np.argsort(center_dist)[:n_points]
            points = points[idx]
            best_points = points
            best_count = n_points
            # Don't break - we might find a better spacing
            if len(points) == n_points:
                break
    
    if best_points is None or len(best_points) < n_points:
        # Fallback: simple grid
        m = int(np.ceil(np.sqrt(n_points)))
        x_coords = np.linspace(-0.45, 0.45, m)
        y_coords = np.linspace(-0.45, 0.45, m)
        points = np.array([(x, y) for y in y_coords for x in x_coords])
        if len(points) > n_points:
            center_dist = np.sum(points ** 2, axis=1)
            idx = np.argsort(center_dist)[:n_points]
            points = points[idx]
        elif len(points) < n_points:
            extra = np.random.uniform(-0.45, 0.45, size=(n_points - len(points), 2))
            points = np.vstack([points, extra])
        best_points = points[:n_points]
    
    return best_points


def _delaunay_bottleneck_forces(centers, min_edge_target, num_bottleneck_edges=50):
    """
    Compute forces based on Delaunay triangulation.
    Focus repulsive forces on the shortest edges (bottlenecks).
    """
    try:
        tri = Delaunay(centers)
    except Exception:
        # Fallback: return zero forces if triangulation fails
        return np.zeros_like(centers)
    
    forces = np.zeros_like(centers)
    
    # Get all edges from triangles
    simplices = tri.simplices
    edges = set()
    for simplex in simplices:
        for i in range(3):
            j = (i + 1) % 3
            edge = tuple(sorted([simplices[i], simplices[j]]))
            edges.add(edge)
    
    edges = list(edges)
    if not edges:
        return forces
    
    # Compute edge lengths
    edge_indices = np.array(edges)
    p1 = centers[edge_indices[:, 0]]
    p2 = centers[edge_indices[:, 1]]
    edge_vecs = p2 - p1
    edge_lengths = np.sqrt(np.sum(edge_vecs ** 2, axis=1))
    
    # Find the shortest edges (bottlenecks)
    n_bottleneck = min(num_bottleneck_edges, len(edges))
    bottleneck_idx = np.argsort(edge_lengths)[:n_bottleneck]
    
    # Apply strong repulsive forces along bottleneck edges
    for idx in bottleneck_idx:
        i, j = edges[idx]
        vec = centers[j] - centers[i]
        dist = edge_lengths[idx]
        if dist < 1e-10:
            continue
        # Force magnitude inversely proportional to distance
        force_mag = min_edge_target / (dist ** 2 + 1e-12)
        unit_vec = vec / dist
        forces[i] -= force_mag * unit_vec
        forces[j] += force_mag * unit_vec
    
    # Add weak global uniformity pressure: push points away from local density
    # Use a simple approximation: for each point, sum vectors to its Delaunay neighbors
    # weighted by 1/d^2, then subtract the mean to center the force
    neighbor_forces = np.zeros_like(centers)
    for i, j in edges:
        vec = centers[j] - centers[i]
        dist = np.sqrt(np.sum(vec ** 2))
        if dist < 1e-10:
            continue
        f = vec / (dist ** 2 + 1e-12) * 0.01
        neighbor_forces[i] += f
        neighbor_forces[j] -= f
    
    forces += neighbor_forces
    
    # Boundary repulsion: keep points inside [-0.5, 0.5]^2
    margin = 0.03
    bound = 0.5 - margin
    for axis in range(2):
        forces[:, axis] += np.where(centers[:, axis] > bound, 
                                     -0.5 * (centers[:, axis] - bound), 0)
        forces[:, axis] += np.where(centers[:, axis] < -bound, 
                                     -0.5 * (centers[:, axis] + bound), 0)
    
    return forces


def _delaunay_relax(centers, max_iters=5000, learning_rate=0.005):
    """
    Iterative optimization using Delaunay triangulation to identify bottlenecks.
    Uses adaptive learning rate and momentum.
    """
    centers = centers.copy()
    r_best = _compute_radius(centers)
    
    momentum = np.zeros_like(centers)
    beta = 0.9  # Momentum coefficient
    
    # Adaptive learning rate
    lr = learning_rate
    no_improve_count = 0
    max_no_improve = 100
    
    for it in range(max_iters):
        if time.time() > _DEADLINE:
            break
        
        # Compute current min distance for target
        # Use Delaunay to quickly estimate min edge
        try:
            tri = Delaunay(centers)
            simplices = tri.simplices
            min_edge = np.inf
            for simplex in simplices:
                for i in range(3):
                    j = (i + 1) % 3
                    d = np.linalg.norm(centers[simplex[i]] - centers[simplex[j]])
                    if d < min_edge:
                        min_edge = d
            # Target: slightly above current min to push it up
            min_edge_target = min_edge * 1.05 + 0.001
        except Exception:
            min_edge_target = 0.07  # Fallback estimate
        
        # Compute forces
        forces = _delaunay_bottleneck_forces(centers, min_edge_target, num_bottleneck_edges=30)
        
        # Update with momentum
        momentum = beta * momentum + (1 - beta) * forces
        new_centers = centers + lr * momentum
        
        # Clip to square
        new_centers[:, 0] = np.clip(new_centers[:, 0], -0.5, 0.5)
        new_centers[:, 1] = np.clip(new_centers[:, 1], -0.5, 0.5)
        
        # Evaluate
        r_new = _compute_radius(new_centers)
        
        if r_new > r_best:
            centers = new_centers
            r_best = r_new
            no_improve_count = 0
            # Slightly increase lr on improvement
            lr = min(lr * 1.05, 0.02)
        else:
            no_improve_count += 1
            # Decrease lr on stagnation
            lr = max(lr * 0.95, 0.001)
            
            # If stuck for too long, add random perturbation
            if no_improve_count > max_no_improve:
                perturbation = np.random.normal(0, 0.005, size=centers.shape)
                new_centers = centers + perturbation
                new_centers[:, 0] = np.clip(new_centers[:, 0], -0.49, 0.49)
                new_centers[:, 1] = np.clip(new_centers[:, 1], -0.49, 0.49)
                r_pert = _compute_radius(new_centers)
                if r_pert > r_best:
                    centers = new_centers
                    r_best = r_pert
                no_improve_count = 0
                lr = learning_rate  # Reset lr
        
        # Every 500 iterations, do a full pairwise check to ensure accuracy
        if it % 500 == 0 and it > 0:
            r_check = _compute_radius(centers)
            if r_check > r_best:
                r_best = r_check
    
    return centers, r_best


def construct():
    start_time = time.time()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # 1. Generate initial hexagonal grid
    centers = _initial_hex_grid(N)
    
    # 2. Delaunay-based relaxation
    centers, r_best = _delaunay_relax(centers, max_iters=4000, learning_rate=0.005)
    
    # 3. Multi-start: Try different initializations if time permits
    for i in range(5):
        if time.time() > _DEADLINE - 15:
            break
        
        # Different random perturbations of the best solution
        np.random.seed(100 + i * 7)
        perturbed = centers + np.random.normal(0, 0.008, size=centers.shape)
        perturbed[:, 0] = np.clip(perturbed[:, 0], -0.49, 0.49)
        perturbed[:, 1] = np.clip(perturbed[:, 1], -0.49, 0.49)
        
        # Re-relax with lower learning rate for refinement
        centers2, r2 = _delaunay_relax(perturbed, max_iters=2000, learning_rate=0.003)
        
        if r2 > r_best:
            centers = centers2
            r_best = r2
    
    # 4. Final fine-tuning pass with very small steps
    if time.time() < _DEADLINE - 5:
        centers, r_final = _delaunay_relax(centers, max_iters=1000, learning_rate=0.001)
        if r_final > r_best:
            centers = centers
            r_best = r_final
    
    # Final validation
    r_final = _compute_radius(centers)
    if r_final <= 0:
        # Fallback: simple grid
        m = int(np.ceil(np.sqrt(N)))
        x_coords = np.linspace(-0.45, 0.45, m)
        y_coords = np.linspace(-0.45, 0.45, m)
        grid = np.array([(x, y) for y in y_coords for x in x_coords])
        centers = grid[:N]
    
    return centers
# EVOLVE-BLOCK-END
