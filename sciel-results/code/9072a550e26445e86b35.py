<<<<<<< SEARCH
    # Compact hexagonal cluster: center + 6 around + 5 more
    # Let's use a systematic approach: try all subsets of a hexagonal grid
    
    # Generate hex grid positions
    # Axial coordinates to Cartesian: x = q + r/2, y = r * sqrt(3)/2
    # For unit hexagons, center-to-center distance = 1.0
    
    hex_positions = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            x = q + r * 0.5
            y = r * np.sqrt(3) / 2
            hex_positions.append((x, y))
    
    # Try different compact subsets of size 12
    # Strategy: try arrangements where hexagons are close together
    
    # Arrangement A: 3x4 rectangular-ish
    arr_A = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0.5, np.sqrt(3)/2), (1.5, np.sqrt(3)/2), (2.5, np.sqrt(3)/2), (3.5, np.sqrt(3)/2),
        (1, np.sqrt(3)), (2, np.sqrt(3)), (3, np.sqrt(3)), (4, np.sqrt(3)),
    ]
    arrangements.append(arr_A)
    
    # Arrangement B: 4-3-4 hexagonal
    arr_B = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0.5, np.sqrt(3)/2), (1.5, np.sqrt(3)/2), (2.5, np.sqrt(3)/2),
        (1, np.sqrt(3)), (2, np.sqrt(3)), (3, np.sqrt(3)), (4, np.sqrt(3)),
        (1.5, 3*np.sqrt(3)/2), (2.5, 3*np.sqrt(3)/2),
    ]
    arrangements.append(arr_B)
    
    # Arrangement C: 2-4-4-2
    arr_C = [
        (1, 0), (2, 0),
        (0.5, np.sqrt(3)/2), (1.5, np.sqrt(3)/2), (2.5, np.sqrt(3)/2), (3.5, np.sqrt(3)/2),
        (0, np.sqrt(3)), (1, np.sqrt(3)), (2, np.sqrt(3)), (3, np.sqrt(3)),
        (0.5, 3*np.sqrt(3)/2), (1.5, 3*np.sqrt(3)/2),
    ]
    arrangements.append(arr_C)
    
    # Arrangement D: compact 4x3
    arr_D = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0.5, np.sqrt(3)/2), (1.5, np.sqrt(3)/2), (2.5, np.sqrt(3)/2), (3.5, np.sqrt(3)/2),
        (1, np.sqrt(3)), (2, np.sqrt(3)), (3, np.sqrt(3)), (4, np.sqrt(3)),
    ]
    arrangements.append(arr_D)
    
    # Arrangement E: more compact - centered hexagonal
    arr_E = [
        (0, 0),
        (1, 0), (0.5, np.sqrt(3)/2), (-0.5, np.sqrt(3)/2),
        (-1, 0), (-0.5, -np.sqrt(3)/2), (0.5, -np.sqrt(3)/2),
        (1.5, np.sqrt(3)/2), (0.5, np.sqrt(3)), (-0.5, np.sqrt(3)),
        (1.5, -np.sqrt(3)/2), (0.5, -np.sqrt(3)), (-0.5, -np.sqrt(3)),
    ]
    arrangements.append(arr_E)
=======
    # Use a scaling factor to ensure strict disjointness (distance > 1.0)
    S = 1.01
    
    # Arrangement A: 3x4 rectangular-ish
    arr_A = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0.5, np.sqrt(3)/2), (1.5, np.sqrt(3)/2), (2.5, np.sqrt(3)/2), (3.5, np.sqrt(3)/2),
        (1, np.sqrt(3)), (2, np.sqrt(3)), (3, np.sqrt(3)), (4, np.sqrt(3)),
    ]
    arr_A = [(x*S, y*S) for x, y in arr_A]
    arrangements.append(arr_A)
    
    # Arrangement B: 4-3-4 hexagonal
    arr_B = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0.5, np.sqrt(3)/2), (1.5, np.sqrt(3)/2), (2.5, np.sqrt(3)/2),
        (1, np.sqrt(3)), (2, np.sqrt(3)), (3, np.sqrt(3)), (4, np.sqrt(3)),
        (1.5, 3*np.sqrt(3)/2), (2.5, 3*np.sqrt(3)/2),
    ]
    arr_B = [(x*S, y*S) for x, y in arr_B]
    arrangements.append(arr_B)
    
    # Arrangement C: 2-4-4-2
    arr_C = [
        (1, 0), (2, 0),
        (0.5, np.sqrt(3)/2), (1.5, np.sqrt(3)/2), (2.5, np.sqrt(3)/2), (3.5, np.sqrt(3)/2),
        (0, np.sqrt(3)), (1, np.sqrt(3)), (2, np.sqrt(3)), (3, np.sqrt(3)),
        (0.5, 3*np.sqrt(3)/2), (1.5, 3*np.sqrt(3)/2),
    ]
    arr_C = [(x*S, y*S) for x, y in arr_C]
    arrangements.append(arr_C)
    
    # Arrangement D: compact 4x3
    arr_D = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0.5, np.sqrt(3)/2), (1.5, np.sqrt(3)/2), (2.5, np.sqrt(3)/2), (3.5, np.sqrt(3)/2),
        (1, np.sqrt(3)), (2, np.sqrt(3)), (3, np.sqrt(3)), (4, np.sqrt(3)),
    ]
    arr_D = [(x*S, y*S) for x, y in arr_D]
    arrangements.append(arr_D)
    
    # Arrangement E: more compact - centered hexagonal
    arr_E = [
        (0, 0),
        (1, 0), (0.5, np.sqrt(3)/2), (-0.5, np.sqrt(3)/2),
        (-1, 0), (-0.5, -np.sqrt(3)/2), (0.5, -np.sqrt(3)/2),
        (1.5, np.sqrt(3)/2), (0.5, np.sqrt(3)), (-0.5, np.sqrt(3)),
        (1.5, -np.sqrt(3)/2), (0.5, -np.sqrt(3)), (-0.5, -np.sqrt(3)),
    ]
    arr_E = [(x*S, y*S) for x, y in arr_E]
    arrangements.append(arr_E)
>>>>>>> REPLACE

We also need to update the overlap check in `_try_arrangement` to be consistent with the scaled positions. Since we scaled by 1.01, the minimum distance is 1.01. The check `dist < 1.01` might fail due to floating point if `dist` is exactly `1.01`. It's better to use a slightly smaller threshold for the check, or just rely on the scaling.
Actually, if we scale by 1.01, the distances are `1.01`.
If we check `dist < 1.0`, it will pass.
If we check `dist < 1.01`, it might fail if `dist` is `1.01 - epsilon`.
Let's change the overlap check to `1.0` to be safe, as the scaling ensures they are at least `1.01` apart.

