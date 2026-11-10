import numpy as np
import itertools
import time
from math import gcd

def get_gcd(a, b, c):
    return gcd(gcd(abs(a), abs(b)), abs(c))

def normalize_line(A, B, C):
    g = get_gcd(A, B, C)
    if g == 0:
        return (0, 0, 0)
    A //= g
    B //= g
    C //= g
    # Make first nonzero entry positive
    if A < 0:
        A = -A; B = -B; C = -C
    elif A == 0 and B < 0:
        B = -B; C = -C
    return (A, B, C)

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 550 # Safety margin
    
    # Grid size
    N = 9
    xs = np.arange(N)
    ys = np.arange(N)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    grid_points = np.stack([px.ravel(), py.ravel()], axis=1)
    num_grid_points = len(grid_points)
    
    # Generate all lines passing through at least 2 grid points
    # A line is defined by 2 distinct points (x1, y1) and (x2, y2)
    # Equation: (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0
    # A = y2-y1, B = x1-x2, C = x2*y1 - x1*y2
    
    line_set = set()
    for i in range(num_grid_points):
        x1, y1 = grid_points[i]
        for j in range(i+1, num_grid_points):
            x2, y2 = grid_points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            norm_line = normalize_line(A, B, C)
            line_set.add(norm_line)
            
    lines_list = list(line_set)
    num_lines = len(lines_list)
    
    # Precompute incidence matrix: incidence[p, l] = 1 if point p is on line l
    # This is 81 x ~1000-2000 lines.
    # Let's estimate number of lines. In N x N grid, number of lines is O(N^4).
    # For N=9, it's manageable.
    
    incidence = np.zeros((num_grid_points, num_lines), dtype=np.int8)
    for l_idx, (A, B, C) in enumerate(lines_list):
        # A*x + B*y + C = 0
        vals = A * grid_points[:, 0] + B * grid_points[:, 1] + C
        incidence[vals == 0, l_idx] = 1
        
    # Score of a line is the number of points it hits in the full grid
    line_scores = incidence.sum(axis=0)
    
    # Initial selection: Top 27 lines by score
    m = 27
    n = 54
    
    # Sort lines by score
    line_indices = np.argsort(line_scores)[::-1]
    selected_lines = line_indices[:m]
    
    # Calculate point scores based on selected lines
    # point_score[p] = number of selected lines passing through p
    point_scores = incidence[:, selected_lines].sum(axis=1)
    
    # Select top 54 points
    point_indices = np.argsort(point_scores)[::-1]
    selected_points = point_indices[:n]
    
    current_incidence = incidence[selected_points][:, selected_lines].sum()
    
    # Local Search
    # 1. Try swapping lines
    # 2. Try swapping points
    
    unselected_lines = np.setdiff1d(np.arange(num_lines), selected_lines)
    unselected_points = np.setdiff1d(np.arange(num_grid_points), selected_points)
    
    improved = True
    while improved:
        improved = False
        
        # Line Swap Search
        # For each selected line, try swapping with an unselected line
        # To speed up, only try top unselected lines
        best_line_swap_gain = 0
        best_line_swap_i = -1
        best_line_swap_j = -1
        
        # Precompute contribution of each selected line to each point
        # contrib[p, i] = 1 if p is on selected_lines[i]
        
        for i_idx in range(m):
            l_old = selected_lines[i_idx]
            # Gain from removing l_old: - (number of selected points on l_old)
            loss = incidence[selected_points, l_old].sum()
            
            # Try adding l_new
            # We want to maximize (number of selected points on l_new) - loss
            # So we want max number of selected points on l_new.
            
            # Iterate over unselected lines, pick the one with max incidence with current points
            # Optimization: We can precompute for all lines the count of selected points.
            # But selected points change if we swap points.
            # Here we are only swapping lines, so points are fixed.
            
            # Let's just find the best unselected line for this specific i?
            # Actually, the loss is constant for a given i.
            # We want to find l_new that maximizes incidence[selected_points, l_new].
            # This is the same for all i!
            # So we just need the best unselected line overall.
            
            pass # Logic simplified below
            
        # Find best unselected line to add
        # Score of a line l is incidence[selected_points, l]
        # We want to swap a selected line with an unselected line if:
        # score(l_new) > score(l_old)
        
        # Calculate scores of all lines with respect to current selected points
        line_point_counts = incidence[selected_points].sum(axis=0)
        
        # Current scores of selected lines
        current_line_scores = line_point_counts[selected_lines]
        
        # Find min selected line score
        min_sel_idx = np.argmin(current_line_scores)
        min_sel_score = current_line_scores[min_sel_idx]
        l_to_remove = selected_lines[min_sel_idx]
        
        # Find max unselected line score
        # Mask selected lines
        unselected_mask = np.ones(num_lines, dtype=bool)
        unselected_mask[selected_lines] = False
        unselected_scores = line_point_counts.copy()
        unselected_scores[selected_lines] = -1
        
        max_unsel_idx = np.argmax(unselected_scores)
        max_unsel_score = unselected_scores[max_unsel_idx]
        
        if max_unsel_score > min_sel_score:
            selected_lines[min_sel_idx] = max_unsel_idx
            # Update selected points?
            # Point scores change because the set of lines changed.
            point_scores = incidence[:, selected_lines].sum(axis=1)
            point_indices = np.argsort(point_scores)[::-1]
            new_selected_points = point_indices[:n]
            
            new_incidence = incidence[new_selected_points][:, selected_lines].sum()
            if new_incidence > current_incidence:
                selected_points = new_selected_points
                current_incidence = new_incidence
                improved = True
                unselected_points = np.setdiff1d(np.arange(num_grid_points), selected_points)
                unselected_lines = np.setdiff1d(np.arange(num_lines), selected_lines)
            else:
                # Revert
                selected_lines[min_sel_idx] = l_to_remove
                point_scores = incidence[:, selected_lines].sum(axis=1)
                point_indices = np.argsort(point_scores)[::-1]
                selected_points = point_indices[:n]
                current_incidence = incidence[selected_points][:, selected_lines].sum()
                unselected_points = np.setdiff1d(np.arange(num_grid_points), selected_points)
                unselected_lines = np.setdiff1d(np.arange(num_lines), selected_lines)
                
        # Point Swap Search
        # Similar logic.
        # Calculate scores of all points with respect to current selected lines
        point_line_counts = incidence[:, selected_lines].sum(axis=1)
        
        # Current scores of selected points
        current_point_scores = point_line_counts[selected_points]
        
        # Find min selected point score
        min_sel_p_idx = np.argmin(current_point_scores)
        min_sel_p_score = current_point_scores[min_sel_p_idx]
        p_to_remove = selected_points[min_sel_p_idx]
        
        # Find max unselected point score
        unselected_p_mask = np.ones(num_grid_points, dtype=bool)
        unselected_p_mask[selected_points] = False
        unselected_p_scores = point_line_counts.copy()
        unselected_p_scores[selected_points] = -1
        
        max_unsel_p_idx = np.argmax(unselected_p_scores)
        max_unsel_p_score = unselected_p_scores[max_unsel_p_idx]
        
        if max_unsel_p_score > min_sel_p_score:
            selected_points[min_sel_p_idx] = max_unsel_p_idx
            # Update selected lines?
            # Line scores change because the set of points changed.
            line_scores_new = incidence[selected_points].sum(axis=0)
            line_indices_new = np.argsort(line_scores_new)[::-1]
            new_selected_lines = line_indices_new[:m]
            
            new_incidence = incidence[selected_points][:, new_selected_lines].sum()
            if new_incidence > current_incidence:
                selected_lines = new_selected_lines
                current_incidence = new_incidence
                improved = True
                unselected_points = np.setdiff1d(np.arange(num_grid_points), selected_points)
                unselected_lines = np.setdiff1d(np.arange(num_lines), selected_lines)
            else:
                # Revert
                selected_points[min_sel_p_idx] = p_to_remove
                line_scores_new = incidence[selected_points].sum(axis=0)
                line_indices_new = np.argsort(line_scores_new)[::-1]
                selected_lines = line_indices_new[:m]
                current_incidence = incidence[selected_points][:, selected_lines].sum()
                unselected_points = np.setdiff1d(np.arange(num_grid_points), selected_points)
                unselected_lines = np.setdiff1d(np.arange(num_lines), selected_lines)
                
        if time.time() > deadline:
            break
            
    # Final result
    points = grid_points[selected_points].astype(np.int64)
    lines = np.array([lines_list[l] for l in selected_lines], dtype=np.int64)
    
    return points, lines
