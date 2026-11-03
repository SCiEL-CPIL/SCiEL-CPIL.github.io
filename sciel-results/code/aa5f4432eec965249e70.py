    # Actually, let's do it more carefully
    n1, m1, p1 = U1.shape[0] // m1, U1.shape[1], V1.shape[0] // m1
    # This is getting complex. Let me use a different approach.
    
    # For 4x4x5, decompose as (2,2,2) x (2,2,1) x (1,1,5)? No.
    # Better: 4x4x5 can be seen as Kronecker of (2,2,2) and (2,2,5)
    # Or more simply, use (2,2,2) Strassen (rank 7) for the 2x2 blocks
