def _search_2x2x3_rank(target_rank, time_budget=5.0):
    """Search for rank-r decomposition of 2x2x3 matmul tensor with half-integer entries."""
    start = time.time()
    
    nonzero = []
    for i in range(2):
        for j in range(2):
            for k in range(3):
                u = 2*i + j
                v = 3*j + k
                w = 2*k + i
                nonzero.append((u, v, w))
    nz_set = set(nonzero)
    
    r = target_rank
    best_residual = float('inf')
    best_solution = None
    
    for attempt in range(5000):
        if time.time() - start > time_budget:
            break
        
        # Initialize
        U = np.zeros((4, r))
        V = np.zeros((6, r))
        W = np.zeros((6, r))
        
        # Random initialization with small values
        for ri in range(r):
            U[np.random.randint(0,4), ri] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            V[np.random.randint(0,6), ri] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            W[np.random.randint(0,6), ri] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
        
        # Local search
        for it in range(500):
            if time.time() - start > time_budget:
                break
            
            # Compute P
            P = np.einsum('ur,vr,wr->uvw', U, V, W)
            
            # Compute residual
            residual = 0.0
            for (u,v,w) in nonzero:
                residual += (P[u,v,w] - 1.0)**2
            for u in range(4):
                for v in range(6):
                    for w in range(6):
                        if (u,v,w) not in nz_set:
                            residual += P[u,v,w]**2
            
            if residual < best_residual:
                best_residual = residual
                best_solution = (U.copy(), V.copy(), W.copy())
            
            if residual < 1e-10:
                # Found a solution!
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                P_r = np.einsum('ur,vr,wr->uvw', U_r, V_r, W_r)
                ok = True
                for (u,v,w) in nonzero:
                    if abs(P_r[u,v,w] - 1.0) > 1e-10:
                        ok = False
                        break
                if ok:
                    for u in range(4):
                        for v in range(6):
                            for w in range(6):
                                if (u,v,w) not in nz_set and abs(P_r[u,v,w]) > 1e-10:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    if ok:
                        return U_r.tolist(), V_r.tolist(), W_r.tolist()
            
            # Random perturbation
            which = np.random.randint(0, 3)
            ri = np.random.randint(0, r)
            if which == 0:
                idx = np.random.randint(0, 4)
                U[idx, ri] += np.random.choice([-1.0, -0.5, 0.5, 1.0])
            elif which == 1:
                idx = np.random.randint(0, 6)
                V[idx, ri] += np.random.choice([-1.0, -0.5, 0.5, 1.0])
            else:
                idx = np.random.randint(0, 6)
                W[idx, ri] += np.random.choice([-1.0, -0.5, 0.5, 1.0])
    
    # Check best solution
    if best_solution is not None:
        U, V, W = best_solution
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        P_r = np.einsum('ur,vr,wr->uvw', U_r, V_r, W_r)
        ok = True
        for (u,v,w) in nonzero:
            if abs(P_r[u,v,w] - 1.0) > 1e-10:
                ok = False
                break
        if ok:
            for u in range(4):
                for v in range(6):
                    for w in range(6):
                        if (u,v,w) not in nz_set and abs(P_r[u,v,w]) > 1e-10:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    return None
