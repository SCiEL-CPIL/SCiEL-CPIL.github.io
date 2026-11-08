"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull

DEADLINE = 145.0  # Safety margin before 150s kill

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _hard_score(pts):
    """
    Compute min triangle area / hull area.
    Vectorized for speed.
    """
    n = len(pts)
    if n < 3:
        return 0.0
    
    # Vectorized triangle area calculation
    # We need min over all (i, j, k) with i < j < k
    # Area = 0.5 * | (x_j - x_i)*(y_k - y_i) - (x_k - x_i)*(y_j - y_i) |
    
    # To save memory and time, we can iterate i, but vectorize j, k
    min_area = float('inf')
    x = pts[:, 0]
    y = pts[:, 1]
    
    for i in range(n - 2):
        # Points j > i and k > j
        # Let's just get all pairs (j, k) where j > i and k > j
        # This is effectively all triangles with vertex i as the first one in sorted order
        # But we need to be careful not to double count or miss any.
        # Actually, iterating i and taking all pairs (j, k) from the remaining points 
        # covers all unique triples exactly once if we are careful with indices.
        
        # Indices for j: i+1 to n-1
        # Indices for k: j+1 to n-1
        # This is a bit tricky to fully vectorize in a single matrix operation without 
        # creating a huge array. 
        # However, for N=14, the total number of triangles is C(14,3) = 364.
        # This is small. We can just compute all 364 areas quickly.
        
        # Let's use a simpler vectorized approach for the inner loop
        # For a fixed i, we have m = n - i - 1 remaining points.
        # We need min area of triangles formed by (i, j, k) where j, k are from remaining.
        
        rem_idx = np.arange(i + 1, n)
        m = len(rem_idx)
        if m < 2: continue
        
        # Get coordinates of remaining points
        x_rem = x[rem_idx]
        y_rem = y[rem_idx]
        
        # We need areas of triangles (i, p, q) where p, q are distinct indices in rem_idx.
        # Area = 0.5 * | cross( P_p - P_i, P_q - P_i ) |
        
        # Vectorize over all pairs (p, q) in rem_idx
        # Create meshgrid of indices
        p_idx, q_idx = np.meshgrid(np.arange(m), np.arange(m), indexing='ij')
        # Only keep p < q to avoid duplicates and zero areas
        mask = p_idx < q_idx
        p_flat = p_idx[mask]
        q_flat = q_idx[mask]
        
        if len(p_flat) == 0: continue
        
        # Coordinates
        px = x_rem[p_flat]
        py = y_rem[p_flat]
        qx = x_rem[q_flat]
        qy = y_rem[q_flat]
        
        # Vectors from i
        v1x = px - x[i]
        v1y = py - y[i]
        v2x = qx - x[i]
        v2y = qy - y[i]
        
        # Cross product
        cross = v1x * v2y - v1y * v2x
        areas = 0.5 * np.abs(cross)
        
        local_min = np.min(areas)
        if local_min < min_area:
            min_area = local_min
            if min_area < 1e-12:
                return 0.0 # Degenerate or nearly so
                
    if min_area == float('inf'):
        return 0.0
        
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
        
    return min_area / hull_a

def _generalized_pattern_search(pts, max_time=None, start_step=0.1):
    """
    Direct optimization of the hard score using Generalized Pattern Search.
    """
    start_t = time.time()
    current_pts = pts.copy()
    current_score = _hard_score(current_pts)
    
    step = start_step
    n_dims = current_pts.size
    n_coords = current_pts.shape[0]
    
    # Directions: ±e_i for each coordinate
    # We can update them in batches to reduce Python loop overhead slightly, 
    # but for N=14 (28 coords), individual steps are fine if vectorized score is fast.
    
    while True:
        if max_time and (time.time() - start_t) > max_time:
            break
        
        improved = False
        # Shrink step if no improvement in previous full cycle? 
        # GPS typically shrinks step if no improvement in a full cycle of directions.
        # Here we check if we improved in this iteration. If not, we might shrink.
        # But standard GPS: try all directions. If any improve, accept. 
        # If none improve in a full cycle, shrink step.
        
        found_improvement_in_cycle = False
        
        for d in range(n_dims):
            # Try + direction
            trial_pts = current_pts.copy()
            # Determine which point and coord
            coord_idx = d
            point_idx = coord_idx // 2
            axis = coord_idx % 2
            
            trial_pts[point_idx, axis] += step
            
            trial_score = _hard_score(trial_pts)
            
            if trial_score > current_score:
                current_pts = trial_pts
                current_score = trial_score
                improved = True
                found_improvement_in_cycle = True
                # Optional: restart direction search? No, just continue to next direction
                # or restart from same direction? 
                # Standard GPS: accept and continue to next direction.
                
            # Try - direction
            trial_pts = current_pts.copy()
            trial_pts[point_idx, axis] -= step
            
            trial_score = _hard_score(trial_pts)
            
            if trial_score > current_score:
                current_pts = trial_pts
                current_score = trial_score
                improved = True
                found_improvement_in_cycle = True
                
        if not found_improvement_in_cycle:
            # No improvement in full cycle, shrink step
            step *= 0.5
            if step < 1e-8:
                break
        else:
            # If we improved, we can keep the step or even grow it slightly?
            # Standard GPS usually keeps step if improved, or grows if allowed.
            # Let's keep it stable to be safe, or grow slowly.
            # step *= 1.1 # Aggressive, might oscillate
            pass
            
    return current_pts, current_score

def construct():
    start_time = time.time()
    
    # 1. Generate diverse initial configurations
    inits = []
    
    # Regular polygon
    angles = 2 * np.pi * np.arange(N) / N
    pts_reg = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(pts_reg)
    
    # Hexagonal-like
    pts_hex = []
    pts_hex.append([0,0])
    for i in range(6):
        a = 2*np.pi*i/6
        pts_hex.append([np.cos(a), np.sin(a)])
    for i in range(7):
        a = 2*np.pi*i/7 + np.pi/7
        pts_hex.append([1.7*np.cos(a), 1.7*np.sin(a)])
    inits.append(np.array(pts_hex))
    
    # Random
    rng = np.random.RandomState(42)
    for _ in range(5):
        pts_rand = rng.uniform(-1, 1, size=(N, 2))
        inits.append(pts_rand)
        
    # Normalize all inits to unit hull area to start on comparable footing
    normalized_inits = []
    for pts in inits:
        try:
            hull = ConvexHull(pts)
            area = hull.volume
            if area > 1e-6:
                pts_norm = pts / np.sqrt(area)
                normalized_inits.append(pts_norm)
            else:
                # Fallback to unit circle
                angles = 2 * np.pi * np.arange(N) / N
                pts_norm = np.column_stack([np.cos(angles), np.sin(angles)])
                normalized_inits.append(pts_norm)
        except:
            angles = 2 * np.pi * np.arange(N) / N
            pts_norm = np.column_stack([np.cos(angles), np.sin(angles)])
            normalized_inits.append(pts_norm)
            
    best_pts = normalized_inits[0]
    best_score = _hard_score(best_pts)
    
    # 2. Optimize each initial configuration using GPS
    # We allocate time dynamically. 
    # GPS is fast. Let's give each init a chance.
    # We have ~140 seconds. 8 inits. ~17s each? 
    # GPS with N=14 is very fast. We can do many iterations.
    # Let's run GPS until time runs out for the best candidate, 
    # but we need to explore all inits first.
    
    # Strategy: 
    # Phase 1: Quick GPS on all inits to find a good basin.
    # Phase 2: Deep GPS on the best basin found.
    
    candidates = []
    for i, pts in enumerate(normalized_inits):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        # Give each init 2-3 seconds for a quick search
        time_limit_per_init = 2.5
        opt_pts, opt_score = _generalized_pattern_search(pts, max_time=time_limit_per_init, start_step=0.05)
        candidates.append((opt_score, opt_pts))
        
        if opt_score > best_score:
            best_score = opt_score
            best_pts = opt_pts
            
    # Sort candidates by score to prioritize polishing the best ones
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 2: Polish the top candidates with remaining time
    # We will keep polishing the best one until time is nearly up.
    # But we might find that a second-best one overtakes it with more polish?
    # Unlikely if GPS is converging. Let's focus on the current best.
    
    # However, it's safer to alternate or just stick to the best.
    # Let's stick to the best one found so far.
    
    if best_pts is not None:
        # Deep polish
        # Start with a slightly larger step to escape local minima if any, then shrink
        # GPS handles this via shrinking.
        # Let's run until deadline - 2s
        remaining_time = DEADLINE - 2 - (time.time() - start_time)
        if remaining_time > 0:
            final_pts, final_score = _generalized_pattern_search(best_pts, max_time=remaining_time, start_step=0.01)
            if final_score > best_score:
                best_pts = final_pts
                best_score = final_score
            else:
                # If deep polish didn't improve, maybe try a different start step?
                # Or just keep the previous best.
                pass
                
    # Final safety check
    if best_pts is None:
        best_pts = normalized_inits[0]
        
    return best_pts
# EVOLVE-BLOCK-END
