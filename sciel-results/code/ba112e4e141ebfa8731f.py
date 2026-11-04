    def local_search(p, Q, cost):
        improved = True
        b_pj_pj = diag_b[p]
        while improved:
            improved = False
            best_delta = 0
            best_i, best_j = -1, -1
            
            for i in range(n):
                u = p[i]
                a_ii = diag_a[i]
                b_u_u = diag_b[u]
                
                term1 = (a_ii - diag_a) * (b_pj_pj - b_u_u)
                term2 = 2 * (Q[i, p] - Q[i, u])
                
                deltas = term1 + term2
                deltas[i] = 0
                
                min_idx = np.argmin(deltas)
                min_val = deltas[min_idx]
                
                if min_val < best_delta:
                    best_delta = min_val
                    best_i = i
                    best_j = min_idx
            
            if best_delta < 0:
                i, j = best_i, best_j
                u, v = p[i], p[j]
                
                cost += best_delta
                
                p[i], p[j] = v, u
                b_pj_pj[i] = diag_b[v]
                b_pj_pj[j] = diag_b[u]
                
                diff_a = a[:, i] - a[:, j]
                diff_b = b[:, v] - b[:, u]
                Q += np.outer(diff_a, diff_b)
                
                B_perm_new = b[:, p]
                Q[i, :] = a[i, :] @ B_perm_new - diag_a[i] * B_perm_new[:, i]
                Q[j, :] = a[j, :] @ B_perm_new - diag_a[j] * B_perm_new[:, j]
                
                improved = True
        return p, cost
