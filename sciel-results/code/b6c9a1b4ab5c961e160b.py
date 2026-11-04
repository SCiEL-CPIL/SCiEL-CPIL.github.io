<<<<<<< SEARCH
    def generate_hexagonal_cluster():
        d = math.sqrt(3) + 0.01
        u = np.array([d, 0.0])
        v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
        centers = [np.array([0.0, 0.0])]
        ring1_vectors = [u, v, v - u, -u, -v, u - v]
        for vec in ring1_vectors:
            centers.append(vec.copy())
        ring2_candidates = []
        basis = [u, v, v - u, -u, -v, u - v]
        for i in range(6):
            p1 = ring1_vectors[i]
            for b in basis:
                p2 = p1 + b
                is_r0 = np.allclose(p2, np.array([0.0, 0.0]))
                is_r1 = False
                for p1j in ring1_vectors:
                    if np.allclose(p2, p1j):
                        is_r1 = True
                        break
                if not is_r0 and not is_r1:
                    duplicate = False
                    for pc in ring2_candidates:
                        if np.allclose(p2, pc):
                            duplicate = True
                            break
                    if not duplicate:
                        ring2_candidates.append(p2)
        ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
        for i in range(5):
            centers.append(ring2_candidates[i].copy())
        angles = [30.0] * 12
        return centers, angles

    def generate_rectangular_grid():
        dx = math.sqrt(3) + 0.01
        dy = 3.0 + 0.01
        centers = []
        angles = []
        for row in range(3):
            for col in range(4):
                cx = (col - 1.5) * dx
                cy = (row - 1.0) * dy
                centers.append([cx, cy])
                angles.append(30.0)
        return centers, angles

    def generate_circular_cluster():
        # Place hexagons in a more compact circular arrangement
        centers = []
        angles = []
        # Center
        centers.append([0.0, 0.0])
        angles.append(30.0)
        # Ring 1: 6 hexagons
        d1 = math.sqrt(3) + 0.01
        for i in range(6):
            angle = i * math.pi / 3
            cx = d1 * math.cos(angle)
            cy = d1 * math.sin(angle)
            centers.append([cx, cy])
            angles.append(30.0)
        # Ring 2: 5 more hexagons (partial ring)
        d2 = 2 * (math.sqrt(3) + 0.01)
        for i in range(5):
            angle = i * 2 * math.pi / 5 + math.pi / 10
            cx = d2 * math.cos(angle)
            cy = d2 * math.sin(angle)
            centers.append([cx, cy])
            angles.append(30.0)
        return centers, angles
=======
    def generate_hexagonal_cluster():
        """12 hexagons in a hexagonal lattice arrangement (center + 6 ring + 5 more)"""
        d = math.sqrt(3) + 0.01
        u = np.array([d, 0.0])
        v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
        centers = [np.array([0.0, 0.0])]
        ring1_vectors = [u, v, v - u, -u, -v, u - v]
        for vec in ring1_vectors:
            centers.append(vec.copy())
        ring2_candidates = []
        basis = [u, v, v - u, -u, -v, u - v]
        for i in range(6):
            p1 = ring1_vectors[i]
            for b in basis:
                p2 = p1 + b
                is_r0 = np.allclose(p2, np.array([0.0, 0.0]))
                is_r1 = False
                for p1j in ring1_vectors:
                    if np.allclose(p2, p1j):
                        is_r1 = True
                        break
                if not is_r0 and not is_r1:
                    duplicate = False
                    for pc in ring2_candidates:
                        if np.allclose(p2, pc):
                            duplicate = True
                            break
                    if not duplicate:
                        ring2_candidates.append(p2)
        ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
        for i in range(5):
            centers.append(ring2_candidates[i].copy())
        angles = [30.0] * 12
        return centers, angles

    def generate_rectangular_grid():
        """3x4 grid of hexagons"""
        dx = math.sqrt(3) + 0.01
        dy = 3.0 + 0.01
        centers = []
        angles = []
        for row in range(3):
            for col in range(4):
                cx = (col - 1.5) * dx
                cy = (row - 1.0) * dy
                centers.append([cx, cy])
                angles.append(30.0)
        return centers, angles

    def generate_circular_cluster():
        """Circular arrangement: center + 6 ring + 5 outer"""
        centers = []
        angles = []
        centers.append([0.0, 0.0])
        angles.append(30.0)
        d1 = math.sqrt(3) + 0.01
        for i in range(6):
            angle = i * math.pi / 3
            cx = d1 * math.cos(angle)
            cy = d1 * math.sin(angle)
            centers.append([cx, cy])
            angles.append(30.0)
        d2 = 2 * (math.sqrt(3) + 0.01)
        for i in range(5):
            angle = i * 2 * math.pi / 5 + math.pi / 10
            cx = d2 * math.cos(angle)
            cy = d2 * math.sin(angle)
            centers.append([cx, cy])
            angles.append(30.0)
        return centers, angles

    def generate_staggered_grid():
        """Staggered grid: 2 rows of 6, offset by half spacing"""
        dx = math.sqrt(3) + 0.01
        dy = 2.598 + 0.01  # sqrt(3)*1.5
        centers = []
        angles = []
        for row in range(2):
            for col in range(6):
                cx = (col - 2.5) * dx
                if row == 1:
                    cx += dx / 2.0
                cy = (row - 0.5) * dy
                centers.append([cx, cy])
                angles.append(30.0)
        return centers, angles

    def generate_compact_cluster():
        """More compact: try to fit 12 in a tighter arrangement"""
        centers = []
        angles = []
        # Center
        centers.append([0.0, 0.0])
        angles.append(30.0)
        # Ring 1: 6 hexagons at distance sqrt(3)
        d1 = math.sqrt(3) + 0.005
        for i in range(6):
            angle = i * math.pi / 3
            cx = d1 * math.cos(angle)
            cy = d1 * math.sin(angle)
            centers.append([cx, cy])
            angles.append(30.0)
        # Ring 2: 5 more hexagons at distance ~2*sqrt(3), but compressed
        d2 = 1.9 * math.sqrt(3) + 0.005
        for i in range(5):
            angle = i * 2 * math.pi / 5 + math.pi / 10
            cx = d2 * math.cos(angle)
            cy = d2 * math.sin(angle)
            centers.append([cx, cy])
            angles.append(30.0)
        return centers, angles
>>>>>>> REPLACE

