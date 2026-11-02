<<<<<<< SEARCH
    def _nullspace_int(C, tol=1e-7):
        """C: (rank, d). Return list of integer null vectors (length rank)."""
        try:
            s, vh = np.linalg.svd(C, full_matrices=True)
        except np.linalg.LinAlgError:
            return []
        t = 1e-9 * (s[0] if s.size else 1.0)
        nsv = int(np.sum(s > t))
        if nsv >= C.shape[0]:
            return []
        null = vh[nsv:]  # (rank-nsv, rank)
        out = []
        for row in null:
            xi = np.rint(row)
            if np.all(np.abs(xi - row) < 1e-5):
                # normalize to smallest integer form
                g = 1
                for v in xi:
                    if v != 0:
                        g = gcd(g, int(abs(v)))
                xi = xi // g if g else xi
                # verify it's actually in the null space
                if np.linalg.norm(C @ xi) < tol * max(1.0, np.linalg.norm(C, axis=1).max()):
                    out.append(xi)
        return out
=======
    def _nullspace_int(C, tol=1e-7):
        """C: (rank, d). Return list of integer null vectors (length rank).
        Uses exact rational arithmetic when possible."""
        rank = C.shape[0]
        # Try exact rational nullspace first
        try:
            C_frac = np.array([[Fraction(int(round(C[i,j]))) for j in range(C.shape[1])] for i in range(rank)])
            # Find nullspace via row reduction
            rows = [row[:] for row in C_frac]
            m, k = rank, C.shape[1]
            pivot_cols = []
            pivot_rows = []
            for col in range(k):
                # Find pivot
                pivot_row = None
                for r in range(len(rows)):
                    if rows[r][col] != 0:
                        pivot_row = r
                        break
                if pivot_row is None:
                    continue
                # Swap to front if not already
                if pivot_row != len(pivot_cols):
                    rows[len(pivot_cols)], rows[pivot_row] = rows[pivot_row], rows[len(pivot_cols)]
                # Normalize
                pv = rows[len(pivot_cols)][col]
                rows[len(pivot_cols)] = [x / pv for x in rows[len(pivot_cols)]]
                pivot_cols.append(col)
                pivot_rows.append(len(pivot_cols) - 1)
                # Eliminate from other rows
                for r in range(m):
                    if r != len(pivot_cols) - 1 and rows[r][col] != 0:
                        factor = rows[r][col]
                        rows[r] = [rows[r][c] - factor * rows[len(pivot_cols)-1][c] for c in range(k)]
            
            # Free variables correspond to columns not in pivot_cols
            free_cols = [c for c in range(k) if c not in pivot_cols]
            if not free_cols:
                return []
            
            # Build nullspace basis: one vector per free variable
            out = []
            for fc in free_cols:
                vec = [Fraction(0)] * rank
                vec[fc] = Fraction(1)
                for i, pc in enumerate(pivot_cols):
                    vec[pc] = -rows[i][fc]
                # Convert to integers
                # Find LCM of denominators
                from math import lcm
                denoms = [abs(v.denominator) for v in vec if v != 0]
                if not denoms:
                    continue
                lcm_den = denoms[0]
                for d in denoms[1:]:
                    lcm_den = lcm(lcm_den, d)
                int_vec = [int(v * lcm_den) for v in vec]
                # Normalize by GCD
                g = 0
                for v in int_vec:
                    if v != 0:
                        g = gcd(g, abs(v))
                if g > 1:
                    int_vec = [v // g for v in int_vec]
                # Verify
                C_int = np.array([[int(round(C[i,j])) for j in range(C.shape[1])] for i in range(rank)])
                if np.allclose(C_int @ np.array(int_vec), 0, atol=1e-9):
                    out.append(np.array(int_vec))
            return out
        except Exception:
            pass
        
        # Fallback: numerical SVD
        try:
            s, vh = np.linalg.svd(C, full_matrices=True)
        except np.linalg.LinAlgError:
            return []
        t = 1e-9 * (s[0] if s.size else 1.0)
        nsv = int(np.sum(s > t))
        if nsv >= C.shape[0]:
            return []
        null = vh[nsv:]  # (rank-nsv, rank)
        out = []
        for row in null:
            xi = np.rint(row)
            if np.all(np.abs(xi - row) < 1e-5):
                g = 1
                for v in xi:
                    if v != 0:
                        g = gcd(g, int(abs(v)))
                xi = xi // g if g else xi
                if np.linalg.norm(C @ xi) < tol * max(1.0, np.linalg.norm(C, axis=1).max()):
                    out.append(xi)
        return out
>>>>>>> REPLACE

