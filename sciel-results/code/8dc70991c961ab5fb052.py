<<<<<<< SEARCH
    def hexagons_separated(v1, v2):
        # v1, v2: (6,2). True if separated by > 1e-6 on at least one of 12 normals.
        for i in range(6):
            e1 = v1[(i + 1) % 6] - v1[i]
            n1 = np.array([-e1[1], e1[0]])
            norm = np.linalg.norm(n1)
            if norm > 1e-15:
                n1 = n1 / norm
            if v1 @ n1 > v2 @ n1:
                p1, p2 = v1 @ n1, v2 @ n1
            else:
                p1, p2 = v2 @ n1, v1 @ n1
            if p1.max() < p2.min() - 1e-6:
                return True
        for i in range(6):
            e2 = v2[(i + 1) % 6] - v2[i]
            n2 = np.array([-e2[1], e2[0]])
            norm = np.linalg.norm(n2)
            if norm > 1e-15:
                n2 = n2 / norm
            if v1 @ n2 > v2 @ n2:
                p1, p2 = v1 @ n2, v2 @ n2
            else:
                p1, p2 = v2 @ n2, v1 @ n2
            if p1.max() < p2.min() - 1e-6:
                return True
        return False
=======
    def hexagons_separated(v1, v2):
        # v1, v2: (6,2). True if separated by > 1e-6 on at least one of 12 normals.
        for i in range(6):
            e1 = v1[(i + 1) % 6] - v1[i]
            n1 = np.array([-e1[1], e1[0]])
            norm = np.linalg.norm(n1)
            if norm > 1e-15:
                n1 = n1 / norm
            p1 = v1 @ n1
            p2 = v2 @ n1
            if p1.max() < p2.min() - 1e-6 or p2.max() < p1.min() - 1e-6:
                return True
        for i in range(6):
            e2 = v2[(i + 1) % 6] - v2[i]
            n2 = np.array([-e2[1], e2[0]])
            norm = np.linalg.norm(n2)
            if norm > 1e-15:
                n2 = n2 / norm
            p1 = v1 @ n2
            p2 = v2 @ n2
            if p1.max() < p2.min() - 1e-6 or p2.max() < p1.min() - 1e-6:
                return True
        return False
>>>>>>> REPLACE
