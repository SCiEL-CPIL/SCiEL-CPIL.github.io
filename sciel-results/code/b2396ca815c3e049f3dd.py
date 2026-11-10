<<<<<<< SEARCH
    sqrt3 = np.sqrt(3.0)
    
    # The 6 neighbor vectors at distance sqrt(3)
    # Angles: 30, 90, 150, 210, 270, 330 degrees
    neighbor_angles = np.pi/6 + np.arange(6) * np.pi/3
    neighbors = np.array([[sqrt3 * np.cos(a), sqrt3 * np.sin(a)] for a in neighbor_angles])
    
    # Two basis vectors that span the lattice
    # u1 = neighbor at 30 deg = (1.5, sqrt(3)/2)
    # u2 = neighbor at 150 deg = (-1.5, sqrt(3)/2)
    u1 = np.array([1.5, sqrt3 / 2.0])
    u2 = np.array([-1.5, sqrt3 / 2.0])
=======
    sqrt3 = np.sqrt(3.0)
    
    # Use a slightly smaller lattice distance to ensure strict separation > 1e-6
    # The touching distance is sqrt(3). Reducing it creates a gap.
    # Gap = sqrt(3) - d. We need gap > 1e-6.
    # Let d = sqrt(3) - 1e-5.
    d_lattice = sqrt3 - 1e-5
    
    # The 6 neighbor vectors at distance d_lattice
    # Angles: 30, 90, 150, 210, 270, 330 degrees
    neighbor_angles = np.pi/6 + np.arange(6) * np.pi/3
    neighbors = np.array([[d_lattice * np.cos(a), d_lattice * np.sin(a)] for a in neighbor_angles])
    
    # Two basis vectors that span the lattice
    # u1 = neighbor at 30 deg = (d_lattice * cos(30), d_lattice * sin(30)) = (d_lattice * sqrt(3)/2, d_lattice * 0.5)
    # u2 = neighbor at 150 deg = (d_lattice * cos(150), d_lattice * sin(150)) = (-d_lattice * sqrt(3)/2, d_lattice * 0.5)
    u1 = np.array([d_lattice * sqrt3 / 2.0, d_lattice / 2.0])
    u2 = np.array([-d_lattice * sqrt3 / 2.0, d_lattice / 2.0])
>>>>>>> REPLACE

