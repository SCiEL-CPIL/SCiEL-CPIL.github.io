def build_lattice_config():
    """Build the hexagonal lattice configuration for 11 unit hexagons."""
    # Axial coordinates for a compact hexagonal cluster
    # We need exactly N=11 hexagons.
    # Center (1) + Ring 1 (6) + 4 from Ring 2 = 11
    axial_coords = [
        (0, 0),       # center
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),  # ring 1
        (2, 0), (1, 1), (0, 2), (-1, 2)  # 4 more to make 11
    ]
    
    # Ensure we don't exceed N if N changes, though it is fixed at 11
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
        
    epsilon = 1e-5
    scale = 1.0 + epsilon
    
    local_centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        local_centers.append((x, y))
    
    return local_centers
