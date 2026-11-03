<<<<<<< SEARCH
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_inner_angle = 0.0
    
    num_layouts = 6
    layouts = [_build_positions(i) for i in range(num_layouts)]
    
    for layout_idx, positions in enumerate(layouts):
=======
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_inner_angle = 0.0
    
    num_layouts = 6
    layouts = [_build_positions(i) for i in range(num_layouts)]
    
    # Add combinatorial search results
    if time.time() < _DEADLINE - 30.0:
        comb_result = _combinatorial_search(None, N, _DEADLINE)
        if comb_result:
            sqrt3 = _SQRT3
            scale = 1.0000001
            u = (sqrt3*scale, 0.0)
            v = (sqrt3/2*scale, 1.5*scale)
            comb_positions = []
            for ti, tj in comb_result:
                x = ti*u[0] + tj*v[0]
                y = ti*u[1] + tj*v[1]
                comb_positions.append((x,y))
            layouts.append(comb_positions)
    
    for layout_idx, positions in enumerate(layouts):
>>>>>>> REPLACE

