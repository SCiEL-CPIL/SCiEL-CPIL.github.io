# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd

def _normalize_line(A, B, C):
    """Normalize line (A,B,C) so it's canonical."""
    if A == 0 and B == 0:
        return None
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g > 0:
        A, B, C = A // g, B // g, C // g
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _build_candidate_lines(nx, ny, max_slope_range=12):
    """Build a diverse pool of candidate lines for an nx x ny grid."""
    candidates = []
    seen = set()
    
    def add_line(A, B, C):
        norm = _normalize_line(A, B, C)
        if norm is not None and norm not in seen:
            seen.add(norm)
            candidates.append(norm)
    
    # Vertical lines: x = c
    for c in range(nx):
        add_line(1, 0, -c)
    
    # Horizontal lines: y = c
    for c in range(ny):
        add_line(0, 1, -c)
    
    # Diagonal lines with various slopes
    # Line through points: dy*x - dx*y + C = 0, gcd(dx,dy)=1
    for dx in range(1, max(nx, ny) + 1):
        for dy in range(-max_slope_range, max_slope_range + 1):
            if dx == 0:
                continue
            if gcd(dx, abs(dy)) != 1:
                continue
            # Line: dy*x - dx*y + C = 0
            # For point (i,j): C = dx*j - dy*i
            # Find all C values that give at least 2 points
            c_set = set()
            for i in range(nx):
                for j in range(ny):
                    C = dx * j - dy * i
                    c_set.add(C)
            
            # For each C, count points on the line
            for C in c_set:
                count = 0
                for i in range(nx):
                    for j in range(ny):
                        if dx * j - dy * i == C:
                            count += 1
                if count >= 2:
                    add_line(dy, -dx, C)
    
    return candidates

def _count_incidence(line, points):
    """Count how many points lie on the line."""
    A, B, C = line
    vals = A * points[:, 0] + B * points[:, 1] + C
    return int(np.sum(vals == 0))

def _solve_bipartite_matching(candidate_lines, points, n_lines=27, n_points=54):
    """
    Solve the bipartite b-matching problem to maximize total incidences.
    
    We model this as: select n_lines lines and n_points points to maximize
    the number of incidences between them.
    
    This is equivalent to finding a maximum weight subgraph in a bipartite graph
    where one side has line nodes and the other has point nodes, with edges
    weighted by 1 if the point lies on the line.
    
    We use a greedy + local search approach since exact min-cost flow on
    this large graph would be too slow.
    """
    n_cand = len(candidate_lines)
    n_pts = len(points)
    
    # Compute incidence matrix: inc[i][j] = 1 if point j is on line i
    # This is n_cand x n_pts
    inc = np.zeros((n_cand, n_pts), dtype=np.int8)
    for i, line in enumerate(candidate_lines):
        A, B, C = line
        vals = A * points[:, 0] + B * points[:, 1] + C
        inc[i] = (vals == 0).astype(np.int8)
    
    # Line scores: number of points on each line
    line_scores = inc.sum(axis=1)
    # Point scores: number of lines through each point
    point_scores = inc.sum(axis=0)
    
    # Greedy initialization: pick top n_lines lines by score, then top n_points points
    line_order = np.argsort(-line_scores)
    selected_lines = line_order[:n_lines]
    
    # For these lines, pick top n_points points
    sub_inc = inc[selected_lines, :]
    sub_point_scores = sub_inc.sum(axis=0)
    point_order = np.argsort(-sub_point_scores)
    selected_points = point_order[:n_points]
    
    current_score = int(inc[selected_lines][:, selected_points].sum())
    
    # Local search: try swapping lines and points
    improved = True
    max_iterations = 5000
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Try swapping one line with an unselected line
        best_delta = 0
        best_swap = None
        
        sel_line_set = set(selected_lines.tolist())
        unselected_lines = [i for i in range(n_cand) if i not in sel_line_set]
        
        # For each selected line, try replacing with each unselected line
        for sel_idx in range(len(selected_lines)):
            old_line = selected_lines[sel_idx]
            # Score contribution of old line to selected points
            old_contrib = int(inc[old_line, selected_points].sum())
            
            for new_line in unselected_lines:
                new_contrib = int(inc[new_line, selected_points].sum())
                delta = new_contrib - old_contrib
                if delta > best_delta:
                    best_delta = delta
                    best_swap = (sel_idx, old_line, new_line)
        
        if best_swap is not None and best_delta > 0:
            sel_idx, old_line, new_line = best_swap
            selected_lines[sel_idx] = new_line
            current_score += best_delta
            improved = True
        
        # Try swapping one point with an unselected point
        best_delta = 0
        best_swap_pt = None
        
        sel_pt_set = set(selected_points.tolist())
        unselected_points = [j for j in range(n_pts) if j not in sel_pt_set]
        
        for sel_idx in range(len(selected_points)):
            old_pt = selected_points[sel_idx]
            old_contrib = int(inc[selected_lines, old_pt].sum())
            
            for new_pt in unselected_points:
                new_contrib = int(inc[selected_lines, new_pt].sum())
                delta = new_contrib - old_contrib
                if delta > best_delta:
                    best_delta = delta
                    best_swap_pt = (sel_idx, old_pt, new_pt)
        
        if best_swap_pt is not None and best_delta > 0:
            sel_idx, old_pt, new_pt = best_swap_pt
            selected_points[sel_idx] = new_pt
            current_score += best_delta
            improved = True
    
    return selected_lines, selected_points, current_score

def _local_refine(candidate_lines, points, selected_lines, selected_points, n_lines=27, n_points=54):
    """Further refine the selection using more aggressive local search."""
    n_cand = len(candidate_lines)
    n_pts = len(points)
    
    inc = np.zeros((n_cand, n_pts), dtype=np.int8)
    for i, line in enumerate(candidate_lines):
        A, B, C = line
        vals = A * points[:, 0] + B * points[:, 1] + C
        inc[i] = (vals == 0).astype(np.int8)
    
    selected_lines = list(selected_lines)
    selected_points = list(selected_points)
    current_score = int(inc[selected_lines][:, selected_points].sum())
    
    max_iterations = 10000
    for iteration in range(max_iterations):
        improved = False
        
        # Try all single swaps of lines
        sel_line_set = set(selected_lines)
        for sel_idx, old_line in enumerate(selected_lines):
            old_contrib = int(inc[old_line, selected_points].sum())
            for new_line in range(n_cand):
                if new_line in sel_line_set:
                    continue
                new_contrib = int(inc[new_line, selected_points].sum())
                delta = new_contrib - old_contrib
                if delta > 0:
                    selected_lines[sel_idx] = new_line
                    sel_line_set.add(new_line)
                    sel_line_set.discard(old_line)
                    current_score += delta
                    improved = True
                    break
        
        # Try all single swaps of points
        sel_pt_set = set(selected_points)
        for sel_idx, old_pt in enumerate(selected_points):
            old_contrib = int(inc[selected_lines, old_pt].sum())
            for new_pt in range(n_pts):
                if new_pt in sel_pt_set:
                    continue
                new_contrib = int(inc[selected_lines, new_pt].sum())
                delta = new_contrib - old_contrib
                if delta > 0:
                    selected_points[sel_idx] = new_pt
                    sel_pt_set.add(new_pt)
                    sel_pt_set.discard(old_pt)
                    current_score += delta
                    improved = True
                    break
        
        if not improved:
            break
    
    return selected_lines, selected_points, current_score

def _try_grid_configuration(nx, ny, candidate_lines, deadline):
    """Try a specific grid configuration and return (points, lines, score)."""
    if time.time() > deadline - 10:
        return None
    
    xs = np.arange(nx)
    ys = np.arange(ny)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    if len(points) != 54:
        return None
    
    # Filter candidate lines to those with at least 1 point on the grid
    relevant_lines = []
    for line in candidate_lines:
        A, B, C = line
        vals = A * points[:, 0] + B * points[:, 1] + C
        if np.sum(vals == 0) >= 2:
            relevant_lines.append(line)
    
    if len(relevant_lines) < 27:
        return None
    
    # Solve the matching problem
    sel_lines, sel_points, score = _solve_bipartite_matching(relevant_lines, points, 27, 54)
    
    if time.time() > deadline - 20:
        return None
    
    # Local refinement
    sel_lines, sel_points, score = _local_refine(relevant_lines, points, sel_lines, sel_points, 27, 54)
    
    # Construct final points array (54 points from the grid)
    final_points = points[sel_points]
    
    # Construct final lines array (27 lines)
    final_lines = np.array([relevant_lines[i] for i in sel_lines], dtype=np.int64)
    
    return final_points, final_lines, score

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 620.0  # Leave 10s margin
    
    best_points = None
    best_lines = None
    best_score = 0
    
    # Try different grid configurations
    grid_configs = [
        (6, 9),
        (9, 6),
        (3, 18),
        (18, 3),
        (2, 27),
        (27, 2),
    ]
    
    # Build a large candidate pool
    all_candidates = []
    seen = set()
    
    for nx, ny in grid_configs:
        cands = _build_candidate_lines(nx, ny, max_slope_range=15)
        for c in cands:
            if c not in seen:
                seen.add(c)
                all_candidates.append(c)
    
    # Also add some lines for larger ranges
    for nx, ny in [(7, 8), (8, 7)]:
        cands = _build_candidate_lines(nx, ny, max_slope_range=15)
        for c in cands:
            if c not in seen:
                seen.add(c)
                all_candidates.append(c)
    
    print(f"Built {len(all_candidates)} candidate lines", flush=True)
    
    # Try each grid configuration
    for nx, ny in grid_configs:
        if time.time() > deadline - 30:
            break
        
        result = _try_grid_configuration(nx, ny, all_candidates, deadline)
        if result is not None:
            pts, lns, score = result
            if score > best_score:
                best_score = score
                best_points = pts
                best_lines = lns
                print(f"Grid {nx}x{ny}: score={score}", flush=True)
    
    # Fallback: if no valid result, use the 6x9 grid with top 27 lines
    if best_points is None:
        nx, ny = 6, 9
        xs = np.arange(nx)
        ys = np.arange(ny)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        line_scores = []
        for line in all_candidates:
            A, B, C = line
            vals = A * points[:, 0] + B * points[:, 1] + C
            count = int(np.sum(vals == 0))
            line_scores.append((count, line))
        
        line_scores.sort(key=lambda x: -x[0])
        selected = [l for s, l in line_scores[:27]]
        lines = np.array(selected, dtype=np.int64)
        best_points = points
        best_lines = lines
        best_score = sum(s for s, _ in line_scores[:27])
    
    # Final validation
    # Check no duplicate points
    pt_tuples = [tuple(p) for p in best_points]
    assert len(set(pt_tuples)) == 54, "Duplicate points found"
    
    # Check no duplicate lines
    line_norms = [_normalize_line(*l) for l in best_lines]
    assert len(set(line_norms)) == 27, "Duplicate lines found"
    
    # Check no degenerate lines
    for A, B, C in best_lines:
        assert not (A == 0 and B == 0), "Degenerate line found"
    
    return best_points, best_lines

# EVOLVE-BLOCK-END
