<<<<<<< SEARCH
    # Define candidate point sets (axial coordinates)
    candidates = []

    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    candidates.append(axial_1)
=======
    # Define candidate point sets (axial coordinates)
    candidates = []

    def is_disjoint(axial_list):
        """Check if no two hexagons are neighbors (share an edge)."""
        n = len(axial_list)
        for i in range(n):
            q1, r1 = axial_list[i]
            for j in range(i + 1, n):
                q2, r2 = axial_list[j]
                dq = q1 - q2
                dr = r1 - r2
                # Axial distance
                dist = (abs(dq) + abs(dr) + abs(dq + dr)) / 2
                if dist < 2: # Neighbors have dist 1. dist < 2 means dist 0 or 1.
                    # dist 0 is duplicate, dist 1 is neighbor
                    return False
        return True

    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    if is_disjoint(axial_1):
        candidates.append(axial_1)
>>>>>>> REPLACE

