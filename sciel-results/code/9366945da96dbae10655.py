import time
import numpy as np
from scipy.spatial.distance import cdist
from itertools import combinations

N = 14
D = 3
DEADLINE = 140.0
_t0 = None

def _true_score(pts):
    """
    Calculate the squared ratio of max to min pairwise distance.
    """
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2

def _generate_candidates(num_fib=2000, num_icosa=100):
    """
    Generate a diverse set of candidate points on the unit sphere.
    1. Fibonacci sphere for uniform coverage.
    2. Icosahedral symmetry points for structural candidates.
    """
    candidates = []
    
    # 1. Fibonacci Sphere
    i = np.arange(num_fib)
    phi_angle = np.arccos(1 - 2 * (i + 0.5) / num_fib)
    theta_angle = np.pi * (1 + 5 ** 0.5) * i
    
    x = np.sin(phi_angle) * np.cos(theta_angle)
    y = np.sin(phi_angle) * np.sin(theta_angle)
    z = np.cos(phi_angle)
    candidates.append(np.column_stack([x, y, z]))
    
    # 2. Icosahedron Vertices and Edge Midpoints (High Symmetry)
    phi = (1 + 5 ** 0.5) / 2
    icosa_verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, phi, 1], [0, phi, -1], [0, -phi, 1], [0, -phi, -1],
        [1, 0, phi], [1, 0, -phi], [-1, 0, phi], [-1, 0, -phi],
        [phi, 1, 0], [phi, -1, 0], [-phi, 1, 0], [-phi, -1, 0]
    ], dtype=float)
    icosa_verts = icosa_verts / np.linalg.norm(icosa_verts, axis=1, keepdims=True)
    candidates.append(icosa_verts)
    
    # Add midpoints of edges of the icosahedron to allow for "flattened" or "expanded" structures
    # This helps if the optimal shape isn't strictly vertices.
    # We approximate by taking linear combinations of nearby vertices.
    # For simplicity, we just rely on the dense Fibonacci grid for the "continuous" part 
    # and Icosahedron for the "discrete" part.
    
    all_cands = np.vstack(candidates)
    # Normalize
    all_cands = all_cands / np.linalg.norm(all_cands, axis=1, keepdims=True)
    return all_cands

def _greedy_select(candidates, k, rng):
    """
    Greedy selection of k points to maximize the minimum distance (packing).
    This provides a good initial structure for the ratio minimization.
    """
    selected = []
    remaining = np.arange(len(candidates))
    
    # Start with a random point
    first_idx = rng.choice(len(remaining))
    selected.append(remaining[first_idx])
    remaining = np.delete(remaining, first_idx)
    
    for _ in range(k - 1):
        # Compute min distance from each remaining candidate to selected set
        sel_pts = candidates[selected]
        # dists shape: (len(remaining), k)
        dists = cdist(candidates[remaining], sel_pts)
        min_dists = dists.min(axis=1)
        
        # Choose the point with the maximum minimum distance
        best_local_idx = np.argmax(min_dists)
        best_global_idx = remaining[best_local_idx]
        
        selected.append(best_global_idx)
        remaining = np.delete(remaining, best_local_idx)
        
    return np.array(selected)

def _local_search_swap(candidates, selected_indices, time_limit=5.0):
    """
    Local search: try swapping one selected point with one unselected point.
    Accept if the score (max/min)^2 improves.
    """
    start_time = time.time()
    current_score = _true_score(candidates[selected_indices])
    best_score = current_score
    best_indices = selected_indices.copy()
    
    unselected = np.setdiff1d(np.arange(len(candidates)), best_indices)
    
    # Precompute distances for efficiency? 
    # cdist on 2000x2000 is fast enough.
    
    while time.time() - start_time < time_limit:
        improved = False
        # Randomize the order of swaps to escape local minima
        rng = np.random.default_rng()
        swap_order_i = rng.permutation(len(best_indices))
        swap_order_j = rng.permutation(len(unselected))
        
        for i_idx in swap_order_i:
            if time.time() - start_time > time_limit:
                break
            old_point = best_indices[i_idx]
            
            for j_idx in swap_order_j:
                if time.time() - start_time > time_limit:
                    break
                new_point = unselected[j_idx]
                
                if old_point == new_point:
                    continue
                
                # Test swap
                test_indices = best_indices.copy()
                test_indices[i_idx] = new_point
                test_pts = candidates[test_indices]
                
                new_score = _true_score(test_pts)
                
                if new_score < best_score:
                    best_score = new_score
                    best_indices = test_indices
                    # Update unselected set
                    unselected = np.setdiff1d(np.arange(len(candidates)), best_indices)
                    improved = True
                    break # Restart inner loop with new state
            
            if improved:
                break # Restart outer loop with new state
        
        if not improved:
            # If no improvement in a full pass, we might be stuck.
            # Try a random perturbation of the whole set? 
            # For now, just break to save time for other restarts.
            break
            
    return best_indices, best_score

def construct():
    global _t0
    if _t0 is None:
        _t0 = time.time()
    
    rng = np.random.default_rng(42)
    
    # 1. Generate Candidates
    candidates = _generate_candidates(num_fib=3000, num_icosa=20)
    
    best_pts = None
    best_score = np.inf
    
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() - _t0 > DEADLINE - 10:
            break
            
        # 2. Greedy Initialization
        # Different random seeds for the initial point in greedy
        init_indices = _greedy_select(candidates, N, rng)
        
        # 3. Local Search (Swaps)
        # Allocate time based on remaining budget
        time_left = DEADLINE - (time.time() - _t0)
        # Reserve 5s for final polish
        search_time = max(1.0, min(5.0, (time_left - 5) / max(1, num_restarts - r)))
        
        refined_indices, score = _local_search_swap(candidates, init_indices, time_limit=search_time)
        
        if score < best_score:
            best_score = score
            best_pts = candidates[refined_indices].copy()
            
        # Early stopping if we find a very good solution
        if best_score < 1.2: # Theoretical lower bound for N=14 is likely around 1.1-1.3
             # We can try to do a more intensive search or break early
             # For now, just continue to see if we can beat it slightly
             pass

    # 4. Final Polish: 
    # The discrete search might have found a good set, but the points are constrained to the candidate grid.
    # We can try to relax the constraints by doing a small continuous optimization on the best discrete set.
    if best_pts is not None:
        # Simple gradient descent to "relax" the points slightly off the grid
        # Objective: minimize (max/min)^2
        # We use a soft version: minimize sum((d - target)^2) where target is current mean?
        # Or just minimize variance of distances?
        # Let's try a simple repulsion/attraction balance for a few steps.
        
        pts = best_pts.copy()
        for _ in range(50):
            d = cdist(pts, pts)
            np.fill_diagonal(d, np.inf)
            mn = d.min()
            mx = d.max()
            if mn < 1e-6: break
            
            # Gradient of (mx/mn)^2 is complex.
            # Approximation: 
            # Force on i from j:
            # If d_ij is close to mn, push apart.
            # If d_ij is close to mx, pull together.
            # Weight by how much they contribute to the extremum.
            
            forces = np.zeros_like(pts)
            for i in range(N):
                for j in range(N):
                    if i == j: continue
                    diff = pts[i] - pts[j]
                    dist = d[i, j]
                    
                    # Soft weights
                    w_min = 1.0 / (1.0 + (dist - mn)**2 * 100) # Strong if close to min
                    w_max = 1.0 / (1.0 + (mx - dist)**2 * 100) # Strong if close to max
                    
                    # Push apart if w_min is high
                    f_push = diff / dist * w_min
                    # Pull together if w_max is high
                    f_pull = -diff / dist * w_max
                    
                    forces[i] += f_push + f_pull
            
            # Normalize forces
            norm_f = np.linalg.norm(forces, axis=1, keepdims=True)
            norm_f[norm_f < 1e-9] = 1.0
            forces = forces / norm_f
            
            # Small step
            pts += 0.01 * forces
            # Re-center
            pts = pts - pts.mean(axis=0)
            # Re-normalize to sphere (approximation, as optimal is likely on sphere)
            pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
            
            new_score = _true_score(pts)
            if new_score < best_score:
                best_score = new_score
                best_pts = pts.copy()
            else:
                # If it got worse, revert
                pts = best_pts.copy()
                break

    # Fallback
    if best_pts is None:
        # Return a simple regular grid
        i = np.arange(N)
        phi = np.arccos(1 - 2*(i+0.5)/N)
        theta = np.pi*(1+5**0.5)*i
        x = np.sin(phi)*np.cos(theta)
        y = np.sin(phi)*np.sin(theta)
        z = np.cos(phi)
        best_pts = np.column_stack([x, y, z])

    return best_pts
