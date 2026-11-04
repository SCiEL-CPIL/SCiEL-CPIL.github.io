    def is_in_span(target, vectors):
        # vectors: list of indices
        if not vectors:
            return np.allclose(target, 0)
        
        V_mat = V_flat[vectors].T # (dim, num)
        # Solve V_mat @ x = target
        try:
            x, residuals, rank, s = np.linalg.lstsq(V_mat, target, rcond=None)
            # residuals is sum of squared residuals
            # If rank < num, it's underdetermined, but lstsq finds min norm solution.
            # We need to check if target is IN the span.
            # If target is in the span, residual should be 0.
            # Due to floating point, check if residual is small.
            if len(residuals) > 0:
                return residuals[0] < 1e-10
            else:
                # If rank == dim, and num >= dim, residuals might be empty?
                # No, lstsq always returns residuals if M > N.
                # Here M=6144, N<=48. So M > N.
                # So residuals will be present.
                return np.linalg.norm(V_mat @ x - target) < 1e-10
        except:
            return False
