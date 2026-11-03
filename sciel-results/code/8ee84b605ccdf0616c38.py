"""Improved program for the 200-circle packing task using Delaunay-based optimization."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist


def _compute_radius(centres):
    """Compute the maximum common radius for given centres."""
    if len(centres) < 2:
        min_dist_to_border = np.min(0.5 - np.abs(centres))
        return float(min_dist_to_border)
    
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    min_dist_to_border = np.min(0.5 - np.abs(centres))
    
    return float(min(min_pair_dist / 2.0, min_dist_to_border))


def _get_delaunay_edges(tri):
    """Extract unique edges from a Delaunay triangulation."""
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                edge = (min(simplex[i], simplex[j]), max(simplex[i], simplex[j]))
                edges.add(edge)
    return list(edges)


def _find_shortest_edge(centres, edges):
    """Find the shortest edge among the given edges."""
    min_dist = np.inf
    min_edge = None
    for i, j in edges:
        d = np.linalg.norm(centres[i] - centres[j])
        if d < min_dist:
            min_dist = d
            min_edge = (i, j)
    return min_edge, min_dist


def _try_move_point(centres, idx, direction, step, best_radius):
    """Try moving point idx in the given direction by step. Return new centres if improved."""
    new_centres = centres.copy()
    new_centres[idx] += direction * step
    new_centres[idx] = np.clip(new_centres[idx], -0.5, 0.5)
    
    new_radius = _compute_radius(new_centres)
    if new_radius > best_radius + 1e-12:
        return new_centres, new_radius
    return None, best_radius


def _optimize_with_flips(centres, max_iters=1000, step_size=0.005):
    """
    Optimize circle positions using Delaunay-based edge analysis.
    At each step:
    1. Build Delaunay triangulation
    2. Find the shortest edge
    3. Move the two endpoints apart
    4. Also check boundary constraints
    5. Repeat
    """
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    for it in range(max_iters):
        try:
            tri = Delaunay(centres)
        except Exception:
            break
        
        edges = _get_delaunay_edges(tri)
        min_edge, min_dist = _find_shortest_edge(centres, edges)
        if min_edge is None:
            break
        
        i, j = min_edge
        
        dist_to_border = 0.5 - np.abs(centres)
        min_border_dist = np.min(dist_to_border)
        min_border_idx = np.unravel_index(np.argmin(dist_to_border), dist_to_border.shape)[0]
        
        if min_dist / 2.0 < min_border_dist:
            direction = centres[i] - centres[j]
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                direction = np.random.randn(2)
                direction /= np.linalg.norm(direction)
            else:
                direction /= norm
            
            improved = False
            for step_mult in [1.0, 0.5, 0.25]:
                step = step_size * step_mult
                new_i, r_i = _try_move_point(centres, i, direction, step, best_radius)
                if new_i is not None:
                    new_j, r_j = _try_move_point(new_i, j, -direction, step, r_i)
                    if new_j is not None and r_j > best_radius + 1e-12:
                        best_radius = r_j
                        best_centres = new_j.copy()
                        improved = True
                        break
            
            if not improved:
                for idx, dir_sign in [(i, 1), (j, -1)]:
                    step = step_size * 0.5
                    new_c, r_new = _try_move_point(centres, idx, dir_sign * direction, step, best_radius)
                    if new_c is not None:
                        best_radius = r_new
                        best_centres = new_c.copy()
                        improved = True
                        break
            
            if not improved:
                for idx in [i, j]:
                    rand_dir = np.random.randn(2)
                    rand_dir /= np.linalg.norm(rand_dir)
                    step = step_size * 0.3
                    new_c, r_new = _try_move_point(centres, idx, rand_dir, step, best_radius)
                    if new_c is not None:
                        best_radius = r_new
                        best_centres = new_c.copy()
                        improved = True
                        break
        else:
            idx = min_border_idx
            pos = centres[idx]
            direction = np.zeros(2)
            for k in range(2):
                if pos[k] > 0:
                    direction[k] = -1.0
                elif pos[k] < 0:
                    direction[k] = 1.0
                else:
                    direction[k] = np.random.choice([-1, 1])
            
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                direction = np.random.randn(2)
                direction /= np.linalg.norm(direction)
            else:
                direction /= norm
            
            improved = False
            for step_mult in [1.0, 0.5, 0.25]:
                step = step_size * step_mult
                new_c, r_new = _try_move_point(centres, idx, direction, step, best_radius)
                if new_c is not None:
                    best_radius = r_new
                    best_centres = new_c.copy()
                    improved = True
                    break
            
            if not improved:
                rand_dir = np.random.randn(2)
                rand_dir /= np.linalg.norm(rand_dir)
                step = step_size * 0.3
                new_c, r_new = _try_move_point(centres, idx, rand_dir, step, best_radius)
                if new_c is not None:
                    best_radius = r_new
                    best_centres = new_c.copy()
        
        centres = best_centres.copy()
        
        if it % 100 == 99:
            step_size *= 0.95
            if step_size < 1e-7:
                break
    
    return best_centres, best_radius


def construct():
    """
    Construct centres of 200 equal circles inside the unit square [-1/2, 1/2]^2
    to maximize the common radius.
    """
    np.random.seed(42)
    
    m = int(np.ceil(np.sqrt(N)))
    points = []
    row = 0
    while len(points) < N:
        if row % 2 == 0:
            n_in_row = m
            xs = np.linspace(-0.5 + 0.5/m, 0.5 - 0.5/m, n_in_row)
        else:
            n_in_row = m - 1
            if n_in_row <= 0:
                n_in_row = 1
            xs = np.linspace(-0.5 + 1.0/m, 0.5 - 1.0/m, n_in_row)
        
        y = -0.5 + (row + 0.5) * (1.0 / m)
        for x in xs:
            if len(points) < N:
                points.append((x, y))
        row += 1
    
    centres = np.array(points[:N])
    centres = np.clip(centres, -0.5 + 0.01, 0.5 - 0.01)
    
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    for restart in range(5):
        if restart == 0:
            current = centres.copy()
        else:
            current = best_centres + np.random.randn(N, 2) * 0.01
            current = np.clip(current, -0.5 + 0.005, 0.5 - 0.005)
        
        optimised, radius = _optimize_with_flips(current, max_iters=800, step_size=0.004)
        
        if radius > best_radius:
            best_radius = radius
            best_centres = optimised.copy()
    
    return best_centres
# EVOLVE-BLOCK-END
