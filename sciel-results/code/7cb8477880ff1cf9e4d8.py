        # Start at (ring, 0)
        q, r = ring, 0
        for _ in range(6):
            for _ in range(ring):
                points.append((q, r))
                # Move in the direction of the next side
                # Directions: (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1), (1, 0)
                # This standard loop might be tricky. Let's use a simpler generator.
                pass
