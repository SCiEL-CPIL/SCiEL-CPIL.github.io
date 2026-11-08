def construct():
    n, m, p = 2, 3, 9
    # Rank 14 decomposition for 2x3x3
    # U_block: 6x14
    # V_block: 9x14
    # W_block: 9x14 (indices k*n+i)
    
    # ... define U_block, V_block, W_block ...
    
    # Assemble for 2x3x9
    U = []
    V = []
    W = []
    for block in range(3):
        # Append U_block
        # Append V_block with shifted k indices
        # Append W_block with shifted k indices
        pass
    return {"U": U, "V": V, "W": W}
