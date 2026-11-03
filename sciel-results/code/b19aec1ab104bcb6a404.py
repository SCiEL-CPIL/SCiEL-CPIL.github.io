<<<<<<< SEARCH
        # Try outer angles in steps of 1.0
        for outer_angle_deg in np.arange(0, 30.0, 1.0):
            if time.time() > _DEADLINE - 20.0:
                break
            res = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
            if res:
                cx, cy, side = res
                if side < best_local_side:
                    best_local_side = side
                    best_local_center = (cx, cy)
                    best_local_angle = outer_angle_deg
=======
        # Try outer angles in steps of 0.5 for finer resolution
        for outer_angle_deg in np.arange(0, 30.0, 0.5):
            if time.time() > _DEADLINE - 20.0:
                break
            res = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
            if res:
                cx, cy, side = res
                if side < best_local_side:
                    best_local_side = side
                    best_local_center = (cx, cy)
                    best_local_angle = outer_angle_deg
>>>>>>> REPLACE

