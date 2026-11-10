<<<<<<< SEARCH
    # Fallback if optimization failed or no valid result found
    if best_inner is None:
        # Use the raw lattice packing, scaled slightly to ensure gap
        # We need to find a scale that gives gap > 1e-6
        # Approximation: gap scales linearly with (scale - 1)
        # For touching hexagons, the "distance" is 0.
        # We just pick a slightly larger scale.
        scale = 1.001 
        centers = _lattice_to_positions(scale)
        inner = np.zeros((11, 3))
        for i in range(11):
            inner[i, 0] = centers[i, 0]
            inner[i, 1] = centers[i, 1]
            inner[i, 2] = 0.0
        
        all_verts = np.vstack([_hex_vertices(h[0], h[1], 1.0, h[2]) for h in inner])
        side = _outer_side_for(all_verts, oc0, 0.0)
        
        # If this fallback is invalid (unlikely with scale 1.001), try even larger
        if not _validity(inner, oc0, 0.0, side):
             scale = 1.01
             centers = _lattice_to_positions(scale)
             for i in range(11):
                 inner[i, 0] = centers[i, 0]
                 inner[i, 1] = centers[i, 1]
             all_verts = np.vstack([_hex_vertices(h[0], h[1], 1.0, h[2]) for h in inner])
             side = _outer_side_for(all_verts, oc0, 0.0)
        
        best_inner = inner
        best_side = side
        best_ang = 0.0
=======
    # Fallback if optimization failed or no valid result found
    if best_inner is None:
        # Use the raw lattice packing, scaled slightly to ensure gap
        scale = 1.001 
        centers = _lattice_to_positions(scale)
        inner = np.zeros((11, 3))
        for i in range(11):
            inner[i, 0] = centers[i, 0]
            inner[i, 1] = centers[i, 1]
            inner[i, 2] = 0.0
        
        all_verts = np.vstack([_hex_vertices(h[0], h[1], 1.0, h[2]) for h in inner])
        side = _outer_side_for(all_verts, oc0, 0.0)
        
        # If this fallback is invalid (unlikely with scale 1.001), try even larger
        if not _validity(inner, oc0, 0.0, side):
             scale = 1.01
             centers = _lattice_to_positions(scale)
             for i in range(11):
                 inner[i, 0] = centers[i, 0]
                 inner[i, 1] = centers[i, 1]
             all_verts = np.vstack([_hex_vertices(h[0], h[1], 1.0, h[2]) for h in inner])
             side = _outer_side_for(all_verts, oc0, 0.0)
        
        best_inner = inner
        best_side = side
        best_ang = 0.0
>>>>>>> REPLACE
