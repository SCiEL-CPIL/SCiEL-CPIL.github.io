        def spectral_radius(sgn):
            A = np.zeros((n, n), dtype=np.float64)
            A[u_pts, v_pts] = sgn
            A[v_pts, u_pts] = sgn
            w = np.linalg.eigvalsh(A)
            return float(np.max(np.abs(w)))
        