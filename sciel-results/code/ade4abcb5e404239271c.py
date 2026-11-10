    x_sol = None
    y_sol = None
    p_sol = None
    
    max_iters = 10
    for _ in range(max_iters):
        if _elapsed() > T_LIMIT - 5.0:
            break
            
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                          options={
                              'presolve': True,
                              'dual_feasibility_tolerance': 1e-12,
                              'primal_feasibility_tolerance': 1e-12,
                              'time_limit': T_LIMIT - _elapsed() - 5.0
                          })
        except Exception:
            break
            
        if not res.success or res.x is None:
            break
            
        x = np.clip(res.x[:N], 0.0, 1.0)
        y = np.clip(res.x[N:2*N], 0.0, 1.0)
        p = np.maximum(res.x[2*N:3*N], 0.0)
        
        # Check FULL IC
        violation = False
        viol_pair = None
        max_viol = 0.0
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                lhs = p[i] - p[j]
                rhs = V[i,0]*(x[i]-x[j]) + V[i,1]*(y[i]-y[j])
                viol = lhs - rhs
                if viol > max_viol:
                    max_viol = viol
                    viol_pair = (i, j)
                if viol > 1e-9:
                    violation = True
        
        if not violation:
            x_sol = x
            y_sol = y
            p_sol = p
            break
        else:
            if viol_pair and viol_pair not in ic_pairs:
                ic_pairs.append(viol_pair)
            
            A_ic_new, b_ic_new = _build_ic_matrix(ic_pairs)
            A_ub_list = [A_ir, A_ic_new]
            b_ub_list = [np.zeros(N), b_ic_new]
            A_ub = np.vstack(A_ub_list)
            b_ub = np.concatenate(b_ub_list)
            
            if len(ic_pairs) > 200:
                break

    if x_sol is None:
        return {'allocations': np.zeros((N, 2)), 'payments': np.zeros(N)}
        
    x = x_sol
    y = y_sol
    p = p_sol
