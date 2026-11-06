    def objective(x):
        a_raw = x[:stages]
        b_raw = x[stages:]
        
        # Adjust to sum to 1
        a = a_raw + (1.0 - np.sum(a_raw)) / stages
        b = b_raw + (1.0 - np.sum(b_raw)) / stages
        
        errors = []
        for t in [0.25, 0.5, 1.0]:
            # U_exact is precomputed? No, depends on t.
            # But H_total is constant.
            # U_exact = expm(-i t H_total)
            # We can precompute eig of H_total too.
            
            U_exact = vH @ np.diag(np.exp(-1j * t * wH)) @ vH.conj().T
            
            U_approx = np.eye(dim, dtype=complex)
            for k in range(stages):
                # E_b = vB @ diag(exp(-i b_k t wB)) @ vB^H
                # E_a = vA @ diag(exp(-i a_k t wA)) @ vA^H
                Eb = vB @ np.diag(np.exp(-1j * b[k] * t * wB)) @ vB.conj().T
                Ea = vA @ np.diag(np.exp(-1j * a[k] * t * wA)) @ vA.conj().T
                U_approx = U_approx @ Eb @ Ea
            
            err = np.linalg.norm(U_approx - U_exact, 2)
            errors.append(err)
        return np.mean(errors)
