    def evaluate(signs, update_best=True):
        nonlocal best_signs, best_radius
        if not fast_connected(signs):
            return None
        
        # Construct A_signed
        A_s = np.zeros((n, n), dtype=float)
        for i in range(num_edges):
            u, v = eu[i], ev[i]
            sgn = float(signs[i])
            A_s[u, v] = sgn
            A_s[v, u] = sgn
            
        eigvals = np.linalg.eigvalsh(A_s)
        rho_s = np.max(np.abs(eigvals))
        
        total_radius = max(rho_base, rho_s)
        
        if update_best:
            if total_radius < best_radius:
                best_radius = total_radius
                best_signs = signs.copy()
            if len(candidates) < 10 or total_radius < candidates[-1][0]:
                candidates.append((total_radius, signs.copy()))
                candidates.sort(key=lambda x: x[0])
                if len(candidates) > 10:
                    candidates.pop()
        return total_radius
