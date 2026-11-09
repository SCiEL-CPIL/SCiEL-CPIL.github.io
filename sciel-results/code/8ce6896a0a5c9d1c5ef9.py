"""
Strategy: Discrete Combinatorial Optimization
Instead of continuous Simulated Annealing, we restrict the search space to a dense
triangular lattice of candidate positions and a discrete set of rotations.
We use a greedy set-packing approach to select non-overlapping hexagons that minimize
the enclosing hexagon size. This exploits the fact that optimal hexagonal packings
often align with the lattice.
"""

import math
import time
import random
import numpy as np

N = 11
SQRT3 = math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # 1. Define Candidate Positions and Orientations
    # We search in a grid around the origin. 
    # The outer hexagon side length is roughly sqrt(N) * 1. For N=11, S ~ 3.3.
    # The hexagons must be within radius ~ S.
    # We generate candidates on a triangular lattice with spacing 1.0 (touching) and 0.9 (overlapping allowed in search, filtered later).
    # Actually, for non-overlapping unit hexagons on a triangular lattice, spacing must be >= 1.0.
    # Let's use spacing 1.0 for the lattice points.
    
    # Orientations: 0, 10, 20, 30, 40, 50 degrees (30 is aligned with lattice normals)
    # Let's use 0, 15, 30, 45, 60? No, 0-60 is symmetry. 
    # Let's use 0, 5, 10, 15, 20, 25, 30 degrees.
    orientations = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    
    # Grid range: need to cover a hexagon of side ~4. 
    # Radius of hexagon side 4 is 4.
    # Let's use a square grid from -5 to 5 with step 1.0.
    # To allow some flexibility, we can also use a finer grid or shift the grid.
    # Let's stick to integer lattice points (q, r) in axial coordinates.
    # Axial to Cartesian:
    # x = SQRT3 * (q + r/2)
    # y = 1.5 * r
    
    candidates = []
    # We want candidates that can form a cluster of size 11.
    # The cluster diameter is roughly 4-5 units.
    # So centers should be within radius ~3.
    for q in range(-4, 5):
        for r in range(-4, 5):
            x = SQRT3 * (q + r / 2.0)
            y = 1.5 * r
            # Filter by distance to origin to keep list manageable
            if x*x + y*y < 36.0: # radius 6
                for angle in orientations:
                    candidates.append((x, y, angle))
    
    # 2. Precompute Vertices for all candidates
    # This is expensive if we do it for every pair. 
    # Better: Precompute vertices for each candidate.
    # But SAT requires axes too.
    
    def get_hex_vertices(cx, cy, angle_deg):
        verts = []
        for i in range(6):
            theta = math.radians(angle_deg + i * 60)
            x = cx + math.cos(theta)
            y = cy + math.sin(theta)
            verts.append((x, y))
        return verts

    def get_hex_axes(angle_deg):
        axes = []
        for i in range(6):
            normal_angle = math.radians(angle_deg + i * 60 + 30)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            axes.append((nx, ny))
        return axes

    # Precompute vertices and axes for candidates
    # This might use a lot of memory if candidates are many.
    # 9*9*7 = 567 candidates. 567 * 6 vertices is fine.
    
    cand_data = []
    for c in candidates:
        x, y, a = c
        verts = get_hex_vertices(x, y, a)
        axes = get_hex_axes(a)
        cand_data.append((verts, axes, x, y, a))

    # 3. Precompute Overlap Matrix
    # M[i][j] = True if candidate i and j overlap
    # 567 candidates -> 567x567 matrix. ~320k checks.
    # Each check involves 12 axes * 6 vertices * 2 = 144 projections.
    # Total operations ~ 46M. This is fast in Python if optimized, or slow if not.
    # Let's use a faster disjoint check.
    
    def is_disjoint_fast(verts1, verts2, axes1, axes2):
        # SAT
        # Combine axes
        # We can optimize by checking bounding boxes first?
        # For now, standard SAT.
        all_axes = axes1 + axes2
        for axis in all_axes:
            nx, ny = axis
            min1 = min(v[0]*nx + v[1]*ny for v in verts1)
            max1 = max(v[0]*nx + v[1]*ny for v in verts1)
            min2 = min(v[0]*nx + v[1]*ny for v in verts2)
            max2 = max(v[0]*nx + v[1]*ny for v in verts2)
            if max1 < min2 or max2 < min1:
                return True
        return False

    num_cands = len(cand_data)
    # Build adjacency matrix (1 if disjoint)
    # Using a list of lists of indices that are disjoint
    disjoint_with = [[] for _ in range(num_cands)]
    
    # Optimization: Only check pairs if their centers are not too far?
    # No, we need to know if they overlap.
    
    # To speed up, we can use NumPy for the SAT projections?
    # Or just loop. 567^2 / 2 = 160k pairs.
    # 160k * 12 axes = 2M axis checks.
    # 2M * 12 projections (6 per poly) = 24M projections.
    # Python can do ~10M simple ops/sec. This might take 2-5 seconds. Acceptable.
    
    print(f"Precomputing overlap matrix for {num_cands} candidates...")
    overlap_start = time.time()
    
    for i in range(num_cands):
        verts_i, axes_i, x_i, y_i, a_i = cand_data[i]
        for j in range(i + 1, num_cands):
            verts_j, axes_j, x_j, y_j, a_j = cand_data[j]
            
            # Quick reject: if center distance > 2.0, they might still overlap if rotated?
            # Max distance between centers for non-overlap is 2.0 (touching).
            # If distance > 2.0, they are disjoint.
            dx = x_i - x_j
            dy = y_i - y_j
            if dx*dx + dy*dy > 4.0:
                disjoint_with[i].append(j)
                disjoint_with[j].append(i)
                continue
            
            # Check SAT
            if is_disjoint_fast(verts_i, verts_j, axes_i, axes_j):
                disjoint_with[i].append(j)
                disjoint_with[j].append(i)
                
    print(f"Overlap matrix done in {time.time() - overlap_start:.2f}s")
    
    # 4. Greedy Set Packing with Local Search
    # We want to select N=11 candidates such that:
    # 1. All are pairwise disjoint
    # 2. The enclosing hexagon size is minimized
    
    # Heuristic:
    # 1. Sort candidates by some "centrality" or just iterate.
    # 2. Greedily pick a candidate.
    # 3. Remove all overlapping candidates.
    # 4. Repeat until N picked or no candidates left.
    # 5. If N picked, calculate score.
    # 6. Use local search (swap) to improve.
    
    # Better: Randomized Greedy with Restarts
    # Since the space is discrete, we can do many fast restarts.
    
    def calc_score(selected_indices):
        # Calculate min enclosing hexagon side length
        # Outer hexagon centered at (0,0) (we will center the cluster later)
        # First, center the cluster
        cx = sum(cand_data[i][2] for i in selected_indices) / N
        cy = sum(cand_data[i][3] for i in selected_indices) / N
        
        # Shifted centers
        shifted = [(cand_data[i][2] - cx, cand_data[i][3] - cy) for i in selected_indices]
        
        # Vertices of shifted hexagons
        # Note: Rotation is absolute, so shifting center doesn't change vertices relative to center, 
        # but we need to recompute or just shift the precomputed vertices.
        # Precomputed vertices are absolute. 
        # v_new = v_old - (cx, cy)
        
        max_proj = 0.0
        # Outer hexagon axes (angle 0 for now, optimize later)
        # Axes for outer hexagon at angle 0:
        # Normals at 30, 90, 150, 210, 270, 330
        outer_axes = [
            (math.cos(math.radians(30)), math.sin(math.radians(30))),
            (math.cos(math.radians(90)), math.sin(math.radians(90))),
            (math.cos(math.radians(150)), math.sin(math.radians(150))),
            (math.cos(math.radians(210)), math.sin(math.radians(210))),
            (math.cos(math.radians(270)), math.sin(math.radians(270))),
            (math.cos(math.radians(330)), math.sin(math.radians(330)))
        ]
        
        for i in selected_indices:
            verts, _, _, _, _ = cand_data[i]
            for vx, vy in verts:
                svx = vx - cx
                svy = vy - cy
                for nx, ny in outer_axes:
                    proj = abs(svx * nx + svy * ny)
                    if proj > max_proj:
                        max_proj = proj
        
        side = 2.0 * max_proj / SQRT3
        return side

    best_score = float('inf')
    best_selection = None
    
    # Local Search / Greedy
    # Strategy: 
    # 1. Start with a random valid set (or empty).
    # 2. Try to add a candidate that minimizes score increase.
    # 3. Try to swap a candidate with another to decrease score.
    
    # Since N=11 is small, we can use a simple backtracking or greedy.
    # Let's use a randomized greedy that tries to build a set.
    
    def greedy_build():
        selected = []
        available = set(range(num_cands))
        
        # Sort candidates randomly to break symmetry
        cand_list = list(available)
        random.shuffle(cand_list)
        
        for i in cand_list:
            if len(selected) >= N:
                break
            if i not in available:
                continue
            
            # Check if i is disjoint from all selected
            valid = True
            for s in selected:
                if s not in disjoint_with[i]:
                    valid = False
                    break
            
            if valid:
                selected.append(i)
                # Remove overlapping candidates from available
                for j in available:
                    if j not in disjoint_with[i]:
                        available.remove(j)
        
        if len(selected) == N:
            score = calc_score(selected)
            return selected, score
        return None, float('inf')
    
    # Run multiple greedy builds
    num_restarts = 500
    for r in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
            
        sel, score = greedy_build()
        if sel and score < best_score:
            best_score = score
            best_selection = sel.copy()
            
        # Local Search on best_selection
        if best_selection:
            improved = True
            while improved and time.time() < deadline - 10.0:
                improved = False
                # Try swapping one hexagon in best_selection with one not in it
                for idx_in in range(N):
                    current_idx = best_selection[idx_in]
                    # Try replacing with a random candidate
                    for _ in range(10): # Limited swaps per iteration
                        new_idx = random.randint(0, num_cands - 1)
                        if new_idx in best_selection:
                            continue
                        
                        # Check if new_idx is disjoint from all other selected
                        valid = True
                        for other_idx in best_selection:
                            if other_idx == current_idx:
                                continue
                            if other_idx not in disjoint_with[new_idx]:
                                valid = False
                                break
                        
                        if valid:
                            new_selection = best_selection.copy()
                            new_selection[idx_in] = new_idx
                            new_score = calc_score(new_selection)
                            if new_score < best_score:
                                best_score = new_score
                                best_selection = new_selection
                                improved = True
                                break
                    if improved:
                        break
                        
    if not best_selection:
        # Fallback: Just return a valid lattice packing
        # Construct a standard hexagonal cluster of 11
        # Center + 6 ring + 4 more?
        # 1, 6, 12... 
        # 1 center, 6 around. Total 7.
        # Need 4 more.
        # Add 4 to the ring? No, ring is full.
        # Add a second layer partial.
        # Coords:
        # (0,0)
        # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        # (2,0), (2,-1), (2,1), (1,1) -> 11 hexagons
        coords = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,-1), (2,1), (1,1)]
        sel = []
        for q, r in coords:
            x = SQRT3 * (q + r / 2.0)
            y = 1.5 * r
            # Find candidate index
            # This is approximate. We should just use the first 11 candidates that are disjoint.
            # Or better, just return a hardcoded valid config.
            pass
        
        # Hardcoded fallback
        inner_list = []
        for q, r in coords:
            x = SQRT3 * (q + r / 2.0)
            y = 1.5 * r
            inner_list.append([x, y, 0.0])
        
        cx = sum(p[0] for p in inner_list) / N
        cy = sum(p[1] for p in inner_list) / N
        inner_list = [[x - cx, y - cy, a] for x, y, a in inner_list]
        
        # Calculate side
        side = calc_score_fallback(inner_list)
        return (inner_list, (0.0, 0.0), side, 0.0)

    # Optimize Outer Angle for best_selection
    # We need to re-verify the selection is valid (it should be)
    # And optimize the outer rotation.
    
    cx = sum(cand_data[i][2] for i in best_selection) / N
    cy = sum(cand_data[i][3] for i in best_selection) / N
    
    # Build final inner list
    inner_list = []
    for i in best_selection:
        x, y, a = cand_data[i][2], cand_data[i][3], cand_data[i][4]
        inner_list.append([x - cx, y - cy, a])
        
    # Precompute shifted vertices for angle optimization
    shifted_verts = []
    for i in best_selection:
        verts, _, x, y, a = cand_data[i]
        sv = [(vx - cx, vy - cy) for vx, vy in verts]
        shifted_verts.append(sv)
        
    best_side = float('inf')
    best_oa = 0.0
    # Search 0 to 60 degrees
    for oa_deg in range(0, 600, 1): # 1 degree steps
        oa = oa_deg * 0.1
        outer_axes = []
        for k in range(6):
            ang = math.radians(oa + 30 + k * 60)
            outer_axes.append((math.cos(ang), math.sin(ang)))
            
        max_proj = 0.0
        for sv in shifted_verts:
            for vx, vy in sv:
                for nx, ny in outer_axes:
                    proj = abs(vx * nx + vy * ny)
                    if proj > max_proj:
                        max_proj = proj
        
        side = 2.0 * max_proj / SQRT3
        if side < best_side:
            best_side = side
            best_oa = oa
            
    return (inner_list, (0.0, 0.0), best_side, best_oa)

def calc_score_fallback(inner_list):
    # Simple calc for fallback
    max_proj = 0.0
    outer_axes = [
        (math.cos(math.radians(30)), math.sin(math.radians(30))),
        (math.cos(math.radians(90)), math.sin(math.radians(90))),
        (math.cos(math.radians(150)), math.sin(math.radians(150))),
        (math.cos(math.radians(210)), math.sin(math.radians(210))),
        (math.cos(math.radians(270)), math.sin(math.radians(270))),
        (math.cos(math.radians(330)), math.sin(math.radians(330)))
    ]
    for x, y, a in inner_list:
        verts = get_hex_vertices(x, y, a)
        for vx, vy in verts:
            for nx, ny in outer_axes:
                proj = abs(vx * nx + vy * ny)
                if proj > max_proj:
                    max_proj = proj
    return 2.0 * max_proj / SQRT3

# Helper used in fallback, defined outside construct for scope
def get_hex_vertices(cx, cy, angle_deg):
    verts = []
    for i in range(6):
        theta = math.radians(angle_deg + i * 60)
        x = cx + math.cos(theta)
        y = cy + math.sin(theta)
        verts.append((x, y))
    return verts
