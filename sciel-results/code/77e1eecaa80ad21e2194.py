def _slsqp_constraints_vectorized(params):
    points = params[:2*N].reshape(N, 2)
    t = params[2*N]
    
    # Boundary constraints
    x = points[:, 0]
    y = points[:, 1]
    
    cons_boundary = np.concatenate([
        y,
        np.sqrt(3)*x - y,
        np.sqrt(3)*(1-x) - y
    ])
    
    # Area constraints
    i_idx = np.array([tr[0] for tr in TRIPLES])
    j_idx = np.array([tr[1] for tr in TRIPLES])
    k_idx = np.array([tr[2] for tr in TRIPLES])
    
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    v1 = pj - pi
    v2 = pk - pi
    dets = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(dets)
    
    cons_area = areas - t
    
    return np.concatenate([cons_boundary, cons_area])
