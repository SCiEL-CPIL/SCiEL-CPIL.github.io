    def local_search_best(p):
        p = p.copy()
        improved = True
        while improved:
            improved = False
            best_delta = -1e-10
            best_i, best_j = -1, -1
            
            for i in range(n):
                for j in range(i + 1, n):
                    u = p[i]
                    v = p[j]
                    
                    w = a[i] - a[j]
                    d = b[u] - b[v]
                    S = np.dot(w, d)
                    
                    term_i = (a[i, i] - a[i, j]) * (b[u, u] - b[u, v])
                    term_j = (a[i, j] - a[j, j]) * (b[u, v] - b[v, v])
                    
                    delta = (a[i, i] + a[j, j]) * (b[v, v] - b[u, u]) + 2.0 * (S - term_i - term_j)
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
                        
            if best_i != -1:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                improved = True
        return p
