<<<<<<< SEARCH
    # Define hex lattice basis
    sqrt3 = np.sqrt(3.0)
    # Lattice vectors for honeycomb of unit hexagons
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])
=======
    # Define hex lattice basis
    sqrt3 = np.sqrt(3.0)
    # Lattice vectors for honeycomb of unit hexagons
    # Add a small padding factor to ensure strict non-overlap (disjoint)
    # Distance between centers must be > 2 * apothem = sqrt(3)
    # Using 1.01 * sqrt(3) ensures a small gap.
    pad = 1.01
    a1 = np.array([sqrt3 * pad, 0.0])
    a2 = np.array([sqrt3/2.0 * pad, 1.5 * pad])
>>>>>>> REPLACE
