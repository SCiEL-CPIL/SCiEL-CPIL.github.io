<<<<<<< SEARCH
    # Generate compact hexagonal cluster of 11 unit hexagons
    # Center + 6 first-ring + 4 second-ring
    centers = []
    centers.append((0.0, 0.0))
    
    # First ring: 6 neighbors at distance sqrt3
    for i in range(6):
        angle = np.radians(60 * i)
        x = sqrt3 * np.cos(angle)
        y = sqrt3 * np.sin(angle)
        centers.append((x, y))
    
    # Second ring: 12 positions at distance 2*sqrt3
    # Pick 4 that give most compact shape
    second_ring_angles = [np.radians(30 * i) for i in range(12)]
    # Choose 4 evenly spaced or optimal 4
    # Let's pick angles 0, 90, 180, 270 or 30, 120, 210, 300
    # Try: 0, 120, 240 and one more? No, need 4.
    # Best is probably 4 that are symmetric. Let's use 0, 90, 180, 270
    for angle in [np.radians(0), np.radians(90), np.radians(180), np.radians(270)]:
        x = 2 * sqrt3 * np.cos(angle)
        y = 2 * sqrt3 * np.sin(angle)
        centers.append((x, y))
    
    centers = np.array(centers)
    inner = [[c[0], c[1], 0.0] for c in centers]
=======
    # Generate compact hexagonal cluster of 11 unit hexagons
    # Center + 6 first-ring + 4 second-ring
    # Using distance 2.0 for ring 1 and 4.0 for ring 2 to ensure no overlap
    # for aligned (angle 0) hexagons.
    centers = []
    centers.append((0.0, 0.0))
    
    # First ring: 6 neighbors at distance 2.0
    for i in range(6):
        angle = np.radians(60 * i)
        x = 2.0 * np.cos(angle)
        y = 2.0 * np.sin(angle)
        centers.append((x, y))
    
    # Second ring: 4 positions at distance 4.0
    # Placed at 0, 90, 180, 270 degrees to avoid overlap with ring 1
    for angle in [np.radians(0), np.radians(90), np.radians(180), np.radians(270)]:
        x = 4.0 * np.cos(angle)
        y = 4.0 * np.sin(angle)
        centers.append((x, y))
    
    centers = np.array(centers)
    inner = [[c[0], c[1], 0.0] for c in centers]
>>>>>>> REPLACE
