import numpy as np
import time
import math

N = 11
TIME_LIMIT = 140.0

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2
AREA_EQUI = 0.5 * H

def bary_to_cart(bary):
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def cart_to_bary(cart):
    # Inverse of bary_to_cart
    x = cart[:, 0]
    y = cart[:, 1]
    v = x - 0.5 * (1 - y/H) # w = 1 - y/H
    v = x - 0.5 * (y/H)     # w = y/H
    # Let's derive properly:
    # x = v + 0.5(1-u-v) = 0.5 + 0.5v - 0.5u
    # y = (1-u-v)H => u+v = 1 - y/H
    # x = 0.5 + 0.5v - 0.5u
    # 2x - 1 = v - u
    # u + v = 1 - y/H
    # 2v = (2x - 1) + (1 - y/H) => v = x - 0.5 - 0.5y/H
    # 2u = (1 - y/H) - (2x - 1) => u = 1 - y/H - x + 0.5 = 1.5 - x - y/H
    u = 1.5 - x - y/H
    v = x - 0.5 - 0.5*y/H
    return np.column_stack([u, v])

def clamp_bary(bary, eps=1e-12):
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def calc_areas_bary(bary):
    """Calculate areas of all triangles formed by N points in barycentric coords."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Pre-compute indices for all combinations of 3 points
    # We will compute this on the fly or cache it. Caching is better.
    # Since N is fixed at 11, we can cache the indices.
    if not hasattr(calc_areas_bary, 'tri_idx'):
        indices = list(zip(*np.triu_indices(N, k=1)))
        # Actually combinations is simpler
        from itertools import combinations
        tri_indices = list(combinations(range(N), 3))
        calc_areas_bary.tri_idx = np.array(tri_indices)
        
    tri_idx = calc_areas_bary.tri_idx
    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]
    
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]
    
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]
    
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def get_min_area(bary):
    areas = calc_areas_bary(bary)
    return np.min(areas)

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Lattice parameters
    # We use a triangular lattice.
    # Step size in barycentric coordinates.
    # A grid of M x M x M effectively.
    # Let's define the lattice by spacing in Cartesian, then convert?
    # Or just spacing in barycentric.
    # Let's use a spacing of 1/50 in barycentric u and v.
    # This creates a grid of points.
    step = 1.0 / 50.0
    
    # Generate valid lattice points
    # u from 0 to 1, v from 0 to 1-u
    u_vals = np.arange(0, 1.0 + step/2, step)
    lattice_points = []
    for u in u_vals:
        v_max = 1.0 - u
        v_vals = np.arange(0, v_max + step/2, step)
        for v in v_vals:
            lattice_points.append([u, v])
            
    lattice_points = np.array(lattice_points)
    # Clamp to ensure strict interior if needed, but for lattice search
    # we can allow boundary? The problem says "on or inside".
    # But area 0 if on boundary with other points?
    # Usually "on or inside" implies we can use boundary.
    # However, if two points are on the boundary, area might be small.
    # Let's keep them strictly inside for the optimization to be safe,
    # or just clamp.
    lattice_points = clamp_bary(lattice_points, eps=1e-9)
    
    num_points = len(lattice_points)
    
    # Map Cartesian to nearest lattice point index?
    # Or just store lattice points and use indices.
    # Using indices allows O(1) lookup for moves.
    
    # Initial configuration
    # Start with a hexagonal packing approximation mapped to lattice
    rng = np.random.default_rng(42)
    
    # Generate a random initial assignment of N points to N distinct lattice points
    # To avoid duplicates, we sample without replacement.
    # But N=11 is small compared to lattice size (~2500 points).
    
    # A better init: pick 11 points roughly evenly spaced.
    # Let's just pick 11 random distinct points.
    init_indices = rng.choice(num_points, size=N, replace=False)
    
    current_indices = init_indices.copy()
    current_bary = lattice_points[current_indices]
    current_min_area = get_min_area(current_bary)
    
    best_indices = current_indices.copy()
    best_bary = current_bary.copy()
    best_min_area = current_min_area
    
    # Simulated Annealing
    # Temperature schedule
    T_start = 0.01
    T_end = 1e-6
    # We don't know the number of iterations, so we use time-based cooling.
    # Or we can estimate iterations.
    # Let's use a fixed number of iterations if time allows, or time-based.
    # Time-based is safer.
    
    # Pre-compute areas for all pairs? No, we need min area of all triples.
    # Updating min area efficiently:
    # When one point moves, only triangles involving that point change.
    # There are C(10, 2) = 45 such triangles.
    # We can maintain an array of areas and update only the relevant ones.
    
    # Let's implement a fast update for min area.
    # areas array of size 165.
    # When point i moves, update areas for all triples (i, j, k) with j<k, j!=i, k!=i.
    
    # To do this efficiently, we need to map (i, j, k) to index in areas array.
    # This is complex.
    # Alternative: Recompute all areas?
    # 165 triangles. Calculation is fast.
    # get_min_area takes ~0.5ms?
    # If we can do 1000 iterations/sec, we have 100k iterations in 100s.
    # That's plenty.
    
    # Let's benchmark get_min_area roughly.
    # It involves 165 determinants. Vectorized.
    # Should be very fast.
    
    # Strategy:
    # 1. Pick a random point index in the N points.
    # 2. Pick a random neighbor in the lattice.
    # 3. Check if the neighbor is occupied by another point.
    # 4. If not, move the point.
    # 5. Calculate new min area.
    # 6. Accept based on SA.
    
    # To check occupancy, maintain a set of used lattice indices.
    used_set = set(current_indices.tolist())
    
    # Neighbor list for each lattice point
    # Lattice is triangular.
    # Neighbors of (u, v) are (u+1, v), (u-1, v), (u, v+1), (u, v-1), (u+1, v-1), (u-1, v+1)
    # In index space, this is hard.
    # Let's precompute neighbors.
    
    # Map (u, v) to index
    # u = i * step, v = j * step
    # i, j are integers.
    # index = ?
    # We can create a dictionary (i, j) -> index
    # u_vals are multiples of step.
    # Let's reconstruct i, j from u, v.
    # i = round(u / step)
    # j = round(v / step)
    
    # Let's create a grid array for fast neighbor lookup.
    # Grid size: max_i = 1/step = 50.
    # grid[i][j] = index in lattice_points, or -1 if invalid.
    
    max_i = int(1.0 / step)
    grid = np.full((max_i + 1, max_i + 1), -1, dtype=int)
    
    for idx, pt in enumerate(lattice_points):
        i = int(round(pt[0] / step))
        j = int(round(pt[1] / step))
        if 0 <= i <= max_i and 0 <= j <= max_i:
            grid[i][j] = idx
            
    # Precompute neighbors for each lattice point
    neighbors = [[] for _ in range(num_points)]
    for idx in range(num_points):
        pt = lattice_points[idx]
        i = int(round(pt[0] / step))
        j = int(round(pt[1] / step))
        
        # 6 neighbors
        di_dj = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
        for di, dj in di_dj:
            ni = i + di
            nj = j + dj
            if 0 <= ni <= max_i and 0 <= nj <= max_i:
                nidx = grid[ni][nj]
                if nidx != -1:
                    neighbors[idx].append(nidx)
    
    # Main Loop
    iterations = 0
    while time.time() < deadline - 5.0:
        # Cooling
        # Approximate iterations per second?
        # Let's just use a simple cooling based on iterations if we can estimate.
        # Or use a fixed T for a while, then decrease.
        # Let's use iterations based cooling.
        # Assume 1000 iter/sec.
        # Total iterations approx 100 * 1000 = 100000.
        # T = T_start * (T_end/T_start)^(iter/total_iter)
        
        # We don't know total_iter.
        # Let's use time-based cooling.
        elapsed = time.time() - start_time
        progress = elapsed / (TIME_LIMIT - 5.0)
        T = T_start * (T_end / T_start) ** progress
        
        # Pick a point to move
        p_idx = rng.integers(0, N)
        curr_latt_idx = current_indices[p_idx]
        
        # Pick a neighbor
        nbrs = neighbors[curr_latt_idx]
        if not nbrs:
            continue
            
        new_latt_idx = nbrs[rng.integers(0, len(nbrs))]
        
        # Check occupancy
        if new_latt_idx in used_set:
            # Skip if occupied
            # Could try to swap? No, just skip.
            continue
            
        # Move
        # Update used_set
        used_set.remove(curr_latt_idx)
        used_set.add(new_latt_idx)
        
        old_idx_val = current_indices[p_idx]
        current_indices[p_idx] = new_latt_idx
        
        # Calculate new min area
        new_bary = lattice_points[current_indices]
        new_min_area = get_min_area(new_bary)
        
        # Acceptance
        delta = new_min_area - current_min_area
        if delta > 0 or rng.random() < math.exp(delta / T):
            current_min_area = new_min_area
            if new_min_area > best_min_area:
                best_min_area = new_min_area
                best_indices = current_indices.copy()
                best_bary = new_bary.copy()
        else:
            # Revert
            current_indices[p_idx] = old_idx_val
            used_set.remove(new_latt_idx)
            used_set.add(curr_latt_idx)
            
        iterations += 1
        
    # Final refinement?
    # Maybe a few steps of local search on the best solution?
    # The lattice search itself is a form of local search with global moves.
    # We can try to refine the best solution by trying all neighbors for all points
    # and accepting improvements (hill climbing).
    
    # Hill climbing on the best solution
    curr_indices = best_indices.copy()
    curr_bary = lattice_points[curr_indices]
    curr_min_area = best_min_area
    used_set = set(curr_indices.tolist())
    
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for p_idx in range(N):
            curr_latt_idx = curr_indices[p_idx]
            nbrs = neighbors[curr_latt_idx]
            for new_latt_idx in nbrs:
                if new_latt_idx in used_set:
                    continue
                
                used_set.remove(curr_latt_idx)
                used_set.add(new_latt_idx)
                curr_indices[p_idx] = new_latt_idx
                
                new_bary = lattice_points[curr_indices]
                new_min_area = get_min_area(new_bary)
                
                if new_min_area > curr_min_area:
                    curr_min_area = new_min_area
                    improved = True
                    # Keep the move
                    if curr_min_area > best_min_area:
                        best_min_area = curr_min_area
                        best_indices = curr_indices.copy()
                        best_bary = new_bary.copy()
                else:
                    # Revert
                    curr_indices[p_idx] = curr_latt_idx
                    used_set.remove(new_latt_idx)
                    used_set.add(curr_latt_idx)
                    
    return bary_to_cart(best_bary)

if __name__ == "__main__":
    pts = construct()
    print(pts)
