def construct_auction(distribution_spec, support, constraints):
    import numpy as np
    
    # Parse distribution
    if isinstance(distribution_spec, np.ndarray):
        P = distribution_spec.copy()
    elif isinstance(distribution_spec, dict):
        P = np.zeros((4,4))
        for key, p in distribution_spec.items():
            if isinstance(key, tuple) and len(key) == 2:
                v1, v2 = key
                P[v1, v2] = p
            else:
                # Try to parse as (v1, v2)
                v1, v2 = int(key[0]), int(key[1])
                P[v1, v2] = p
    else:
        P = np.array(distribution_spec, dtype=float)
    
    # Ensure shape is (4,4)
    if P.shape != (4,4):
        P = P.reshape((4,4))
    
    # Normalize
    total = P.sum()
    if total > 0:
        P = P / total
    
    # Compute marginals
    P1 = P.sum(axis=1)  # P(v1) for v1 in 0..3
    P2 = P.sum(axis=0)  # P(v2) for v2 in 0..3
    
    # Determine optimal allocation for each good
    # For good 1: q1(v) = 1 if v >= threshold, else 0
    if P1[3] >= P1[2]:
        q1 = np.array([0.0, 0.0, 0.0, 1.0])
    else:
        q1 = np.array([0.0, 0.0, 1.0, 1.0])
    
    if P2[3] >= P2[2]:
        q2 = np.array([0.0, 0.0, 0.0, 1.0])
    else:
        q2 = np.array([0.0, 0.0, 1.0, 1.0])
    
    # Compute payments using payment identity
    def compute_payment(q):
        p = np.zeros(4)
        p[0] = 0.0
        p[1] = 0.0
        p[2] = q[2] - q[1]
        p[3] = 2.0*q[3] - q[2] - q[1]
        return p
    
    p1 = compute_payment(q1)
    p2 = compute_payment(q2)
    
    # Construct joint allocation and payment
    allocations = np.zeros((16, 2))
    payments = np.zeros(16)
    
    idx = 0
    for v1 in range(4):
        for v2 in range(4):
            allocations[idx, 0] = q1[v1]
            allocations[idx, 1] = q2[v2]
            payments[idx] = p1[v1] + p2[v2]
            idx += 1
    
    # Verify constraints
    # Check nonnegative payments
    assert np.all(payments >= -1e-12), "Negative payment detected"
    
    # Check IR: utility >= 0 for all types
    for v1 in range(4):
        for v2 in range(4):
            idx = v1 * 4 + v2
            util = v1 * allocations[idx, 0] + v2 * allocations[idx, 1] - payments[idx]
            assert util >= -1e-12, f"IR violated at ({v1},{v2}): util={util}"
    
    # Check IC: for all types (v1,v2) and reports (v1',v2'), utility of truth >= utility of report
    for v1 in range(4):
        for v2 in range(4):
            for v1p in range(4):
                for v2p in range(4):
                    idx_true = v1 * 4 + v2
                    idx_rep = v1p * 4 + v2p
                    util_true = v1 * allocations[idx_true, 0] + v2 * allocations[idx_true, 1] - payments[idx_true]
                    util_rep = v1 * allocations[idx_rep, 0] + v2 * allocations[idx_rep, 1] - payments[idx_rep]
                    regret = util_rep - util_true
                    assert regret <= 1e-8, f"IC violated: type ({v1},{v2}) reporting ({v1p},{v2p}) gives regret {regret}"
    
    return {'allocations': allocations, 'payments': payments}
