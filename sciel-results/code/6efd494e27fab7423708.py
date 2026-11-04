import numpy as np

def construct():
    n, m, p = 6, 6, 6
    
    # Strassen 2x2 decomposition (rank 7)
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M0 = (a+d)(e+h)
    U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[3,0]=1; W2[0,0]=1; W2[3,0]=1
    # M1 = (c+d)e
    U2[2,1]=1; U2[3,1]=1; V2[0,1]=1; W2[2,1]=1; W2[3,1]=-1
    # M2 = a(f-h)
    U2[0,2]=1; V2[1,2]=1; V2[3,2]=-1; W2[1,2]=1; W2[3,2]=1
    # M3 = d(g-e)
    U2[3,3]=1; V2[2,3]=1; V2[0,3]=-1; W2[0,3]=1; W2[2,3]=1
    # M4 = (a+b)h
    U2[0,4]=1; U2[1,4]=1; V2[3,4]=1; W2[0,4]=-1; W2[1,4]=1
    # M5 = (c-a)(e+f)
    U2[2,5]=1; U2[0,5]=-1; V2[0,5]=1; V2[1,5]=1; W2[3,5]=1
    # M6 = (b-d)(g+h)
    U2[1,6]=1; U2[3,6]=-1; V2[2,6]=1; V2[3,6]=1; W2[0,6]=1
    
    # Liu 3x3 decomposition (rank 19)
    U3 = np.zeros((9, 19))
    V3 = np.zeros((9, 19))
    W3 = np.zeros((9, 19))
    
    # Define the 19 multipliers
    # Using standard Liu formulation
    multipliers = [
        # (U_indices, U_values, V_indices, V_values)
        ([0,4,8], [1,1,1], [0,4,8], [1,1,1]),       # M0
        ([0,4,8], [1,-1,1], [0,4,8], [1,-1,1]),      # M1
        ([0,4,8], [1,1,-1], [0,4,8], [1,1,-1]),      # M2
        ([1,5,6], [1,1,1], [3,7,2], [1,1,1]),        # M3
        ([1,5,6], [1,-1,1], [3,7,2], [1,-1,1]),      # M4
        ([1,5,6], [1,1,-1], [3,7,2], [1,1,-1]),      # M5
        ([1,5,6], [1,-1,-1], [3,7,2], [1,-1,-1]),    # M6
        ([3,7,2], [1,1,1], [6,1,5], [1,1,1]),        # M7
        ([3,7,2], [1,-1,1], [6,1,5], [1,-1,1]),      # M8
        ([3,7,2], [1,1,-1], [6,1,5], [1,1,-1]),      # M9
        ([3,7,2], [1,-1,-1], [6,1,5], [1,-1,-1]),    # M10
        ([0,1], [1,1], [4,3], [1,1]),                # M11
        ([0,2], [1,1], [7,6], [1,1]),                # M12
        ([3,4], [1,1], [1,0], [1,1]),                # M13
        ([3,5], [1,1], [7,6], [1,1]),                # M14
        ([6,7], [1,1], [1,0], [1,1]),                # M15
        ([6,8], [1,1], [4,3], [1,1]),                # M16
        ([2,5], [1,1], [6,3], [1,1]),                # M17
        ([1,3], [1,1], [8,7], [1,1]),                # M18
    ]
    
    for c, (ui, uv, vi, vv) in enumerate(multipliers):
        for idx, val in zip(ui, uv):
            U3[idx, c] = val
        for idx, val in zip(vi, vv):
            V3[idx, c] = val
    
    # Solve for W3
    # For each (i,k): sum_c W3[i*3+k, c] * U3[i*3+j, c] * V3[j*3+k, c] = delta_{j,j'}
    # This gives a 9x9 system for each (i,k)
    
    for i in range(3):
        for k in range(3):
            # Build the 9x9 system
            # Unknowns: W3[i*3+k, 0..18]
            # Equations: for each (j,j'): sum_c W3[i*3+k,c]*U3[i*3+j,c]*V3[j'*3+k,c] = delta_{j,j'}
            
            A_sys = np.zeros((9, 19))
            b_sys = np.zeros(9)
            
            eq_idx = 0
            for j in range(3):
                for jp in range(3):
                    for c in range(19):
                        A_sys[eq_idx, c] = U3[i*3+j, c] * V3[jp*3+k, c]
                    b_sys[eq_idx] = 1.0 if j == jp else 0.0
                    eq_idx += 1
            
            # Solve least squares (should be exact if decomposition is valid)
            W3[i*3+k, :], residuals, rank, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
    
    # Now construct the 6x6 decomposition via Kronecker product
    rank = 19 * 7
    U6 = np.zeros((36, rank))
    V6 = np.zeros((36, rank))
    W6 = np.zeros((36, rank))
    
    for i3 in range(3):
        for i2 in range(2):
            i = i3 * 2 + i2
            for j3 in range(3):
                for j2 in range(2):
                    j = j3 * 2 + j2
                    for c3 in range(19):
                        for c2 in range(7):
                            c = c3 * 7 + c2
                            U6[i*6+j, c] = U3[i3*3+j3, c3] * U2[i2*2+j2, c2]
    
    for j3 in range(3):
        for j2 in range(2):
            j = j3 * 2 + j2
            for k3 in range(3):
                for k2 in range(2):
                    k = k3 * 2 + k2
                    for c3 in range(19):
                        for c2 in range(7):
                            c = c3 * 7 + c2
                            V6[j*6+k, c] = V3[j3*3+k3, c3] * V2[j2*2+k2, c2]
    
    for k3 in range(3):
        for k2 in range(2):
            k = k3 * 2 + k2
            for i3 in range(3):
                for i2 in range(2):
                    i = i3 * 2 + i2
                    for c3 in range(19):
                        for c2 in range(7):
                            c = c3 * 7 + c2
                            W6[k*6+i, c] = W3[i3*3+k3, c3] * W2[i2*2+k2, c2]
    
    # Convert to lists
    return {
        "U": U6.tolist(),
        "V": V6.tolist(),
        "W": W6.tolist()
    }
