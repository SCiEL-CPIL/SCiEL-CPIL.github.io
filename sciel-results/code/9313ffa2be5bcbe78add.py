"""Improved packing of 50 equal circles in a unit circle using PSO + local refinement."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _score(pts):
    """Compute the score: min of half min pairwise distance and min distance to boundary."""
    if pts.shape != (N, 2):
        return -1.0
    if not np.all(np.isfinite(pts)):
        return -1.0
    # Check all points are inside unit circle
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    if np.any(norms >= 1.0):
        return -1.0
    # Boundary distance
    boundary_dist = np.min(1.0 - norms)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(boundary_dist, min_pair)


def _init_hexagonal():
    """Generate a hexagonal lattice configuration as initial guess."""
    # Use a hexagonal packing pattern
    points = []
    # Center point
    points.append([0.0, 0.0])
    # Rings of hexagonal lattice
    for ring in range(1, 10):
        for i in range(6 * ring):
            angle = 2 * np.pi * i / (6 * ring)
            r = ring * 0.15  # approximate spacing
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            if x * x + y * y < 0.9:
                points.append([x, y])
        if len(points) >= N:
            break
    points = np.array(points[:N])
    return points


def _pso_optimize(points, max_iter=200, n_particles=20, deadline=None):
    """Particle Swarm Optimization to improve the packing."""
    start_time = time.time()
    points = points.copy()
    
    # Initialize particles
    particles = np.zeros((n_particles, N, 2))
    velocities = np.zeros((n_particles, N, 2))
    for i in range(n_particles):
        # Start from the given points with small random perturbation
        particles[i] = points + np.random.randn(N, 2) * 0.05
    
    # Compute initial scores
    scores = np.array([_score(particles[i]) for i in range(n_particles)])
    pbest = particles.copy()
    pbest_scores = scores.copy()
    gbest_idx = np.argmax(scores)
    gbest = particles[gbest_idx].copy()
    gbest_score = scores[gbest_idx]
    
    # PSO parameters
    w = 0.7  # inertia
    c1 = 1.5  # cognitive
    c2 = 1.5  # social
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        for i in range(n_particles):
            # Update velocity
            r1 = np.random.rand(N, 2)
            r2 = np.random.rand(N, 2)
            velocities[i] = (w * velocities[i] +
                            c1 * r1 * (pbest[i] - particles[i]) +
                            c2 * r2 * (gbest - particles[i]))
            
            # Update position
            particles[i] = particles[i] + velocities[i]
            
            # Clamp to unit circle (with margin)
            norms = np.sqrt(np.sum(particles[i] ** 2, axis=1))
            for j in range(N):
                if norms[j] > 0.95:
                    scale = 0.95 / norms[j]
                    particles[i][j] *= scale
            
            # Evaluate
            s = _score(particles[i])
            if s > pbest_scores[i]:
                pbest[i] = particles[i].copy()
                pbest_scores[i] = s
            if s > gbest_score:
                gbest = particles[i].copy()
                gbest_score = s
        
        # Adaptively reduce inertia
        w = max(0.4, w * 0.99)
    
    return gbest, gbest_score


def _local_refinement(points, max_iter=100, deadline=None):
    """Local refinement using gradient-like ascent on the score function."""
    start_time = time.time()
    points = points.copy()
    current_score = _score(points)
    
    # Simple iterative improvement: for each point, try small moves
    step_size = 0.02
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            # Try small random perturbations
            for _ in range(10):
                delta = np.random.randn(2) * step_size
                new_pt = points.copy()
                new_pt[i] = points[i] + delta
                
                # Check boundary
                norm = np.sqrt(np.sum(new_pt[i] ** 2))
                if norm > 0.98:
                    continue
                
                new_score = _score(new_pt)
                if new_score > current_score + 1e-9:
                    points = new_pt
                    current_score = new_score
                    improved = True
                    break
        
        if not improved:
            # Reduce step size
            step_size *= 0.8
            if step_size < 0.001:
                break
    
    return points, current_score


def construct():
    """Construct the optimal packing of 50 equal circles in a unit circle."""
    deadline = time.time() + 120.0  # Leave 30s safety margin
    
    # Start with hexagonal lattice
    points = _init_hexagonal()
    
    # Ensure we have exactly N points
    if len(points) < N:
        # Add random points
        while len(points) < N:
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, 0.8)
            points.append([r * np.cos(angle), r * np.sin(angle)])
    points = points[:N]
    points = np.array(points)
    
    # PSO optimization
    points, score = _pso_optimize(points, max_iter=100, n_particles=15, deadline=deadline - 30)
    
    # Local refinement
    if time.time() < deadline - 10:
        points, score = _local_refinement(points, max_iter=50, deadline=deadline - 5)
    
    return points
# EVOLVE-BLOCK-END
