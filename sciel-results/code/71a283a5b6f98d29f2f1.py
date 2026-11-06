<<<<<<< SEARCH
    for n in [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0]:
        for a in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
            for b in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
                param_list.append((n, a, b))
    
    # Shuffle to get diversity early
    import random
    random.seed(42)
    random.shuffle(param_list)
    
    for n, a, b in param_list:
        if time.time() > deadline:
            break
        
        # Create the 16x16 top-left quadrant
        half_params = [[0]*half for _ in range(half)]
        
        for i in range(half):
            for j in range(half):
                # Pixel center coordinates
                px = i + 0.5
                py = j + 0.5
                
                # Distance from cell center (15.5, 15.5)
                dx = abs(px - cx)
                dy = abs(py - cy)
                
                # Super-ellipse test: (dx/a)^n + (dy/b)^n <= 1 means inside air region
                if a > 0 and b > 0:
                    val = (dx / a) ** n + (dy / b) ** n
                    if val <= 1.0:
                        half_params[i][j] = 0  # air
                    else:
                        half_params[i][j] = 1  # dielectric
                else:
                    half_params[i][j] = 1
        
        cell = make_cell(half_params)
        
        # We don't have the evaluator here, so we need a proxy metric
        # For now, just use the first valid design or a heuristic
        # Actually, we can't compute the band gap here without the full PWE solver
        # So let's just pick a reasonable design based on known good patterns
        
        if best_cell is None:
            best_cell = cell
=======
    # To ensure the TL quadrant is symmetric (half_params[i][j] == half_params[j][i]),
    # we must use a symmetric metric. Using a=b in the super-ellipse ensures this
    # because dx and dy are symmetric with respect to i and j in the TL quadrant
    # (dx = 15-i, dy = 15-j).
    
    param_list = []
    for n in [2.0, 4.0, 6.0, 8.0, 12.0, 20.0]:
        for a in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]:
            # b is forced to a to maintain quadrant symmetry
            param_list.append((n, a, a))
            
    # Shuffle to get diversity early
    import random
    random.seed(42)
    random.shuffle(param_list)
    
    for n, a, b in param_list:
        if time.time() > deadline:
            break
        
        # Create the 16x16 top-left quadrant
        half_params = [[0]*half for _ in range(half)]
        
        for i in range(half):
            for j in range(half):
                # Pixel center coordinates
                px = i + 0.5
                py = j + 0.5
                
                # Distance from cell center (15.5, 15.5)
                dx = abs(px - cx)
                dy = abs(py - cy)
                
                # Super-ellipse test: (dx/a)^n + (dy/b)^n <= 1 means inside air region
                # With a=b, this is symmetric in i,j
                val = (dx / a) ** n + (dy / b) ** n
                if val <= 1.0:
                    half_params[i][j] = 0  # air
                else:
                    half_params[i][j] = 1  # dielectric
        
        cell = make_cell(half_params)
        
        if best_cell is None:
            best_cell = cell
>>>>>>> REPLACE
