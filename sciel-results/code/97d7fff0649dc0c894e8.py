<<<<<<< SEARCH
    vertices = []
    for cx, cy in centers:
        for k in range(6):
            angle = math.radians(60 * k) # Pointy top vertices at 0, 60, ...
            vx = cx + 1.0 * math.cos(angle)
            vy = cy + 1.0 * math.sin(angle)
            vertices.append((vx, vy))
=======
    vertices = []
    inner_angle_offset = math.radians(30) # Rotate 30 deg so flat edges face grid directions
    for cx, cy in centers:
        for k in range(6):
            angle = math.radians(60 * k) + inner_angle_offset
            vx = cx + 1.0 * math.cos(angle)
            vy = cy + 1.0 * math.sin(angle)
            vertices.append((vx, vy))
>>>>>>> REPLACE
<<<<<<< SEARCH
    # Construct the return values
    inner = [[cx, cy, 0.0] for cx, cy in centers]
    return inner, best_center, best_side, best_angle
=======
    # Construct the return values
    inner = [[cx, cy, 30.0] for cx, cy in centers]
    return inner, best_center, best_side, best_angle
>>>>>>> REPLACE
