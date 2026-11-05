def add_laderman_4x4(B_col_offset):
    """Add 19 terms for 4x4 * 4x4 using Laderman's algorithm."""
    # Terms defined as (u_dict, v_dict, w_dict)
    # u_dict: {(i,j): coeff}
    # v_dict: {(j,k): coeff}
    # w_dict: {(row_C, col_C): coeff}
    
    # M1
    add_term(
        {(0,0):1, (1,1):1},
        {(0,0):1, (1,1):1},
        {(0,0):1, (1,1):1}
    )
    # M2
    add_term(
        {(0,0):1, (1,1):1},
        {(2,2):1, (3,3):1},
        {(0,2):1, (1,3):1}
    )
    # M3
    add_term(
        {(2,2):1, (3,3):1},
        {(0,0):1, (1,1):1},
        {(2,0):1, (3,1):1}
    )
    # M4
    add_term(
        {(2,2):1, (3,3):1},
        {(2,2):1, (3,3):1},
        {(2,2):1, (3,3):1}
    )
    # M5
    add_term(
        {(0,2):1, (1,3):1},
        {(0,2):1, (1,3):1},
        {(0,0):1, (1,1):1}
    )
    # M6
    add_term(
        {(0,3):1, (1,2):1},
        {(0,3):1, (1,2):1},
        {(0,0):1, (1,1):1} # Wait, C_00, C_11? No, M6 contributes to C_03, C_12?
        # Check reconstruction:
        # C_03 = M6 - M15 - M19
        # C_12 = M5 - M14 - M17
        # C_11 = M1 - M6 + M10 - M17
        # So M6 contributes to C_03 (coeff 1), C_11 (coeff -1).
        # My w_dict for M6 should be {(0,3):1, (1,1):-1}
    )
