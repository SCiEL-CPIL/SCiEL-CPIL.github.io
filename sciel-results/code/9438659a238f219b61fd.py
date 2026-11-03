    def generate_greedy_polyhex(n):
        poly = [(0, 0)]
        dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        
        for _ in range(n - 1):
            best_add = None
            best_score = float('inf')
            existing = set(poly)
            for q, r in poly:
                for dq, dr in dirs:
                    nq, nr = q + dq, r + dr
                    if (nq, nr) in existing:
                        continue
                    new_poly = poly + [(nq, nr)]
                    # Calculate bounding box area in axial coords (approx)
                    qs = [p[0] for p in new_poly]
                    rs = [p[1] for p in new_poly]
                    w = max(qs) - min(qs)
                    h = max(rs) - min(rs)
                    score = w * h
                    if score < best_score:
                        best_score = score
                        best_add = (nq, nr)
            if best_add:
                poly.append(best_add)
        return poly
