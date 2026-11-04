    def _try_reduce():
        nonlocal best_candidate, best_rank
        import numpy as np
        
        U = base["U"]
        V = base["V"]
        W = base["W"]
        rank0 = len(U[0])
        num_terms = n * m * p
        
        M = np.zeros((num_terms, rank0))
        for c in range(rank0):
            for i in range(n):
                for a in range(m):
                    u_val = U[i * m + a][c]
                    if u_val == 0: continue
                    for k in range(p):
                        v_val = V[a * p + k][c]
                        if v_val == 0: continue
                        w_val = W[k * n + i][c]
                        if w_val == 0: continue
                        idx = i * m * p + a * p + k
                        M[idx, c] = u_val * v_val * w_val
                        
        # Check rank
        r = np.linalg.matrix_rank(M)
        if r < rank0:
            # Dependency found. Try to find a null vector.
            # For simplicity, we just note that we might be able to reduce.
            # A full null space analysis is complex. 
            # We'll try to remove the last column and see if the rest can span the target.
            # This requires solving M_kept * x = target.
            # If a solution exists with small integers, we update.
            pass
