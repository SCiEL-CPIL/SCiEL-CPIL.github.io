<<<<<<< SEARCH
    # Generate honeycomb positions with a small gap to ensure disjointness
    # Scale factor 1.001 ensures slight separation for numerical safety
    # Flat-top hexagons (30 deg) have width sqrt(3) in x, so honeycomb spacing sqrt(3) is valid
    scale = 1.001
    
    def honeycomb_positions(count, pattern):
        """Generate positions for 'count' hexagons in honeycomb pattern."""
        positions = []
        for j, row_len in enumerate(pattern):
            for i in range(row_len):
                x = (i + 0.5 * (j % 2)) * sqrt3 * scale
                y = j * 1.5 * scale
                positions.append((x, y, 0.0))
        return positions[:count]
=======
    # Generate honeycomb positions with a small gap to ensure disjointness
    # Scale factor 1.001 ensures slight separation for numerical safety
    # Flat-top hexagons (30 deg) have width sqrt(3) in x, so honeycomb spacing sqrt(3) is valid
    scale = 1.001
    
    def honeycomb_positions(count, pattern):
        """Generate positions for 'count' hexagons in honeycomb pattern."""
        positions = []
        total = sum(pattern)
        if total != count:
            return positions
        for j, row_len in enumerate(pattern):
            for i in range(row_len):
                x = (i + 0.5 * (j % 2)) * sqrt3 * scale
                y = j * 1.5 * scale
                positions.append((x, y, 0.0))
        return positions[:count]
>>>>>>> REPLACE

