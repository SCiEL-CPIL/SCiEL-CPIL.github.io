            if steps_in_direction >= next_turn:
                # Turn 90 degrees (rotate direction)
                d = direction
                # Rotate in xy plane: (dx, dy, dz) -> (-dy, dx, dz)
                direction = (-d[1], d[0], d[2])
