<<<<<<< SEARCH
    # All inner hexagons are aligned with the lattice (angle 0.0)
    angles_rad = np.zeros(N)
=======
    # Orient hexagons to match the honeycomb lattice packing.
    # The lattice vectors have length sqrt(3).
    # For unit hexagons to pack without overlap at this distance,
    # their width in the direction of the lattice vectors must be sqrt(3).
    # This corresponds to a 'flat-top' orientation (vertices at 30, 90, ...).
    # The default angle 0 puts a vertex at 0 (pointy-top), width 2.
    # Rotate by 30 degrees (pi/6) to get width sqrt(3) in x-direction.
    angles_rad = np.full(N, np.pi / 6.0)
>>>>>>> REPLACE
