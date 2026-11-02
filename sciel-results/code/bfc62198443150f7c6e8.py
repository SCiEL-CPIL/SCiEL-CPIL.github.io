<<<<<<< SEARCH
    # Also try optimizing with non-zero inner hexagon rotations for the best configuration
    if best_positions is not None and time.time() < deadline - 30:
        # Try rotating all inner hexagons by the same angle
        for rot_angle in [0.0, 15.0, 30.0, 45.0, 60.0]:
            if time.time() > deadline - 10:
                break
            rotated_hexes = [(x, y, 1.0, rot_angle) for x, y in best_positions]
            cx_mean = sum(x for x, y, s, a in rotated_hexes) / len(rotated_hexes)
            cy_mean = sum(y for x, y, s, a in rotated_hexes) / len(rotated_hexes)
            
            for oang in [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]:
                if time.time() > deadline - 10:
                    break
                opt_cx, opt_cy, opt_angle, opt_side = _optimize_outer_hex(
                    rotated_hexes, cx_mean, cy_mean, oang, max_iters=50
                )
                if opt_side < best_side - 1e-10:
                    best_side = opt_side
                    best_center = (opt_cx, opt_cy)
                    best_angle = opt_angle
                    # Store the rotated inner hexagons
                    best_positions = [(x, y) for x, y, s, a in rotated_hexes]
                    best_inner_angle = rot_angle
                    break
=======
    # Also try optimizing with non-zero inner hexagon rotations for the best configuration
    if best_positions is not None and time.time() < deadline - 20:
        # Try rotating all inner hexagons by the same angle
        for rot_angle in [0.0, 15.0, 30.0]:
            if time.time() > deadline - 10:
                break
            rotated_hexes = [(x, y, 1.0, rot_angle) for x, y in best_positions]
            cx_mean = sum(x for x, y, s, a in rotated_hexes) / len(rotated_hexes)
            cy_mean = sum(y for x, y, s, a in rotated_hexes) / len(rotated_hexes)
            
            for oang in [0.0, 15.0, 30.0, 45.0]:
                if time.time() > deadline - 10:
                    break
                opt_cx, opt_cy, opt_angle, opt_side = _optimize_outer_hex(
                    rotated_hexes, cx_mean, cy_mean, oang, max_iters=40
                )
                if opt_side < best_side - 1e-10:
                    best_side = opt_side
                    best_center = (opt_cx, opt_cy)
                    best_angle = opt_angle
                    best_inner_angle = rot_angle
                    break
>>>>>>> REPLACE

