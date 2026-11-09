<<<<<<< SEARCH
def _compute_top27(points):
    """
    Compute the top 27 lines by incidence count for a set of points.
    Returns (score, lines_array) where score is the sum of incidences of the top 27 lines.
    """
    n = len(points)
    if n < 2:
        return 0, np.zeros((27, 3), dtype=np.int64)
    
    # Step 1: Find all unique lines defined by pairs of points
    line_set = set()
    pts = points
    
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            # Line through (x1,y1) and (x2,y2)
            A = y2 - y1
            B = -(x2 - x1)
            C = x2 * y1 - x1 * y2
            
            key = _normalize_line(A, B, C)
            if key is not None:
                line_set.add(key)
                
    # Step 2: Count incidences for each line
    counts = []
    for (A, B, C) in line_set:
        count = 0
        for p in pts:
            x, y = p
            if A * x + B * y + C == 0:
                count += 1
        counts.append((count, (A, B, C)))
        
    # Step 3: Sort by count descending, then by line tuple for determinism
    counts.sort(key=lambda x: (-x[0], x[1]))
    
    # Step 4: Select top 27
    if len(counts) >= 27:
        selected = counts[:27]
    else:
        selected = counts
        # Pad with dummy lines if necessary (should not happen for 54 points)
        while len(selected) < 27:
            selected.append((0, (0, 0, 1)))
            
    score = sum(c for c, _ in selected)
    lines = np.array([l for _, l in selected], dtype=np.int64)
    
    return score, lines
=======
def _compute_top27(points):
    """
    Compute the top 27 lines by incidence count for a set of points.
    Returns (score, lines_array) where score is the sum of incidences of the top 27 lines.
    Uses vectorized incidence counting for speed.
    """
    n = len(points)
    if n < 2:
        return 0, np.zeros((27, 3), dtype=np.int64)
    
    pts = points
    pts_f = pts.astype(np.float64)
    
    # Step 1: Find all unique lines defined by pairs of points
    line_set = set()
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            A = y2 - y1
            B = -(x2 - x1)
            C = x2 * y1 - x1 * y2
            key = _normalize_line(A, B, C)
            if key is not None:
                line_set.add(key)
    
    if not line_set:
        return 0, np.zeros((27, 3), dtype=np.int64)
    
    # Step 2: Vectorized incidence counting
    lines_arr = np.array(list(line_set), dtype=np.int64)  # (L, 3)
    L = len(lines_arr)
    
    # Compute A*x + B*y + C for all (point, line) pairs
    # pts_f: (n, 2), lines_arr: (L, 3)
    # vals[p, l] = A_l * x_p + B_l * y_p + C_l
    vals = (pts_f[:, 0:1] * lines_arr[0:1, 0].reshape(1, L) +
            pts_f[:, 1:2] * lines_arr[1:2, 0].reshape(1, L) +
            lines_arr[2:3, 0].reshape(1, L))
    # vals shape: (n, L)
    
    # Count incidences: for each line, how many points satisfy A*x+B*y+C == 0
    incidence_matrix = (vals == 0).astype(np.int64)  # (n, L)
    counts = incidence_matrix.sum(axis=0)  # (L,)
    
    # Step 3: Sort by count descending, then by line tuple for determinism
    # Create list of (count, line_tuple)
    counts_list = [(int(counts[i]), tuple(lines_arr[i])) for i in range(L)]
    counts_list.sort(key=lambda x: (-x[0], x[1]))
    
    # Step 4: Select top 27
    if len(counts_list) >= 27:
        selected = counts_list[:27]
    else:
        selected = counts_list
        while len(selected) < 27:
            selected.append((0, (0, 0, 1)))
            
    score = sum(c for c, _ in selected)
    lines = np.array([l for _, l in selected], dtype=np.int64)
    
    return score, lines
>>>>>>> REPLACE

