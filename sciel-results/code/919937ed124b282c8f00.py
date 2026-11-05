"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def coefficients(stages):
    """4th-order symmetric 12-stage product formula via composition of 3 second-order 4-stage splittings."""
    if stages != 12:
        # Fallback for other stage counts: use simple Strang
        substeps = max(1, stages // 2)
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b
    
    # 4th-order 12-stage symmetric splitting
    # w = 1/(2*(2 - 2^(1/3)))
    w = 1.0 / (2.0 * (2.0 - 2.0 ** (1.0 / 3.0)))
    
    # Each substep has 4 stages, 3 substeps total
    # a_sub = [w, 0.5-w, 0.5-w, w] scaled by 1/3
    # b_sub = [0.5-w, w, w, 0.5-w] scaled by 1/3
    a_sub = [w / 3.0, (0.5 - w) / 3.0, (0.5 - w) / 3.0, w / 3.0]
    b_sub = [(0.5 - w) / 3.0, w / 3.0, w / 3.0, (0.5 - w) / 3.0]
    
    a = []
    b = []
    for _ in range(3):
        a.extend(a_sub)
        b.extend(b_sub)
    
    return a, b
# EVOLVE-BLOCK-END
