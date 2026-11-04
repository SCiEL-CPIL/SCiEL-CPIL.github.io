import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=8
    rank = 98
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen steps
    # 7 steps for B_L, 7 steps for B_R
    # B_L is cols 0-3, B_R is cols 4-7
    
    # Define the 7 Strassen products for 2x2 matrices
    # X, Y are 2x2
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    
    # Reconstruction coeffs for Z00, Z01, Z10, Z11
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # We will iterate through the 14 Strassen steps
    # For each step, we define the linear forms for X and Y
    # and the reconstruction coeffs for the 4 output blocks
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec, col_idx):
        for i in range(n*m):
            U[i][col_idx] += u_vec[i]
        for i in range(m*p):
            V[i][col_idx] += v_vec[i]
        for i in range(p*n):
            W[i][col_idx] += w_vec[i]
            
    col = 0
    for step in range(14):
        is_left = (step < 7)
        t_idx = step % 7
        
        # Determine X and Y linear forms
        # X is from A blocks
        # Y is from B blocks (L or R)
        
        # A blocks indices:
        # A00: i in 0,1; j in 0,1
        # A01: i in 0,1; j in 2,3
        # A10: i in 2,3; j in 0,1
        # A11: i in 2,3; j in 2,3
        
        # B blocks indices (for L):
        # B00: j in 0,1; k in 0,1
        # B01: j in 0,1; k in 2,3
        # B10: j in 2,3; k in 0,1
        # B11: j in 2,3; k in 2,3
        
        # B blocks indices (for R):
        # B00: j in 0,1; k in 4,5
        # B01: j in 0,1; k in 6,7
        # B10: j in 2,3; k in 4,5
        # B11: j in 2,3; k in 6,7
        
        # We need to construct u_vec, v_vec for each of the 7 M_t
        
        # Let's define the coefficients for X and Y for each M_t
        # X coeffs: x00, x01, x10, x11
        # Y coeffs: y00, y01, y10, y11
        
        # M1: X=(1,0,0,1), Y=(1,0,0,1)
        # M2: X=(0,0,1,1), Y=(1,0,0,0)
        # M3: X=(1,0,0,0), Y=(0,1,0,-1)
        # M4: X=(0,0,0,1), Y=(0,0,1,-1)
        # M5: X=(1,1,0,0), Y=(0,0,0,1)
        # M6: X=(0,0,-1,0)?? No. X10-X00. X=( -1, 0, 1, 0). Y=(1,1,0,0)
        # M7: X=(0,1,0,-1). Y=(0,0,1,1)
        
        # Note: X indices are (00, 01, 10, 11) corresponding to (i,j) blocks
        # X00 is block A00. X01 is block A01. X10 is block A10. X11 is block A11.
        
        x_coeffs = [
            [1, 0, 0, 1], # M1
            [0, 0, 1, 1], # M2
            [1, 0, 0, 0], # M3
            [0, 0, 0, 1], # M4
            [1, 1, 0, 0], # M5
            [-1, 0, 1, 0],# M6
            [0, 1, 0, -1] # M7
        ]
        
        y_coeffs = [
            [1, 0, 0, 1], # M1
            [1, 0, 0, 0], # M2
            [0, 1, 0, -1],# M3
            [0, 0, 1, -1],# M4
            [0, 0, 0, 1], # M5
            [1, 1, 0, 0], # M6
            [0, 0, 1, 1]  # M7
        ]
        
        # Reconstruction coeffs for Z blocks (00, 01, 10, 11)
        # Z00 = M1 + M4 - M5 + M7
        # Z01 = M3 + M5
        # Z10 = M2 + M4
        # Z11 = M1 - M2 + M3 + M6
        
        # For a specific M_t, the coeffs are:
        # If t=0 (M1): Z00+=1, Z01+=0, Z10+=0, Z11+=1
        # If t=1 (M2): Z00+=0, Z01+=0, Z10+=1, Z11+=-1
        # If t=2 (M3): Z00+=0, Z01+=1, Z10+=0, Z11+=1
        # If t=3 (M4): Z00+=1, Z01+=0, Z10+=1, Z11+=0
        # If t=4 (M5): Z00+=-1, Z01+=1, Z10+=0, Z11+=0
        # If t=5 (M6): Z00+=0, Z01+=0, Z10+=0, Z11+=1
        # If t=6 (M7): Z00+=1, Z01+=0, Z10+=0, Z11+=0
        
        w_coeffs = [
            [1, 0, 0, 1], # M1
            [0, 0, 1, -1],# M2
            [0, 1, 0, 1], # M3
            [1, 0, 1, 0], # M4
            [-1, 1, 0, 0],# M5
            [0, 0, 0, 1], # M6
            [1, 0, 0, 0]  # M7
        ]
        
        xc = x_coeffs[t_idx]
        yc = y_coeffs[t_idx]
        wc = w_coeffs[t_idx]
        
        # Construct u_vec (size 16)
        u_vec = [0] * 16
        # xc[0] is coeff for A00 (i 0,1; j 0,1)
        for i in [0, 1]:
            for j in [0, 1]:
                u_vec[i*4+j] += xc[0]
        # xc[1] is coeff for A01 (i 0,1; j 2,3)
        for i in [0, 1]:
            for j in [2, 3]:
                u_vec[i*4+j] += xc[1]
        # xc[2] is coeff for A10 (i 2,3; j 0,1)
        for i in [2, 3]:
            for j in [0, 1]:
                u_vec[i*4+j] += xc[2]
        # xc[3] is coeff for A11 (i 2,3; j 2,3)
        for i in [2, 3]:
            for j in [2, 3]:
                u_vec[i*4+j] += xc[3]
        
        # Construct v_vec (size 32)
        v_vec = [0] * 32
        if is_left:
            # B_L blocks
            # yc[0] for B00 (j 0,1; k 0,1)
            for j in [0, 1]:
                for k in [0, 1]:
                    v_vec[j*8+k] += yc[0]
            # yc[1] for B01 (j 0,1; k 2,3)
            for j in [0, 1]:
                for k in [2, 3]:
                    v_vec[j*8+k] += yc[1]
            # yc[2] for B10 (j 2,3; k 0,1)
            for j in [2, 3]:
                for k in [0, 1]:
                    v_vec[j*8+k] += yc[2]
            # yc[3] for B11 (j 2,3; k 2,3)
            for j in [2, 3]:
                for k in [2, 3]:
                    v_vec[j*8+k] += yc[3]
        else:
            # B_R blocks
            # yc[0] for B00 (j 0,1; k 4,5)
            for j in [0, 1]:
                for k in [4, 5]:
                    v_vec[j*8+k] += yc[0]
            # yc[1] for B01 (j 0,1; k 6,7)
            for j in [0, 1]:
                for k in [6, 7]:
                    v_vec[j*8+k] += yc[1]
            # yc[2] for B10 (j 2,3; k 4,5)
            for j in [2, 3]:
                for k in [4, 5]:
                    v_vec[j*8+k] += yc[2]
            # yc[3] for B11 (j 2,3; k 6,7)
            for j in [2, 3]:
                for k in [6, 7]:
                    v_vec[j*8+k] += yc[3]
                    
        # Construct w_vec (size 32)
        # Output C is 4x8.
        # C blocks for this step:
        # If is_left: C_L (cols 0-3)
        # If is_right: C_R (cols 4-7)
        
        w_vec = [0] * 32
        
        # wc[0] for Z00 (i 0,1; k 0,1 relative to block)
        # wc[1] for Z01 (i 0,1; k 2,3 relative to block)
        # wc[2] for Z10 (i 2,3; k 0,1 relative to block)
        # wc[3] for Z11 (i 2,3; k 2,3 relative to block)
        
        if is_left:
            # C_L
            for i in [0, 1]:
                for k in [0, 1]:
                    w_vec[k*4+i] += wc[0]
            for i in [0, 1]:
                for k in [2, 3]:
                    w_vec[k*4+i] += wc[1]
            for i in [2, 3]:
                for k in [0, 1]:
                    w_vec[k*4+i] += wc[2]
            for i in [2, 3]:
                for k in [2, 3]:
                    w_vec[k*4+i] += wc[3]
        else:
            # C_R
            for i in [0, 1]:
                for k in [4, 5]:
                    w_vec[k*4+i] += wc[0]
            for i in [0, 1]:
                for k in [6, 7]:
                    w_vec[k*4+i] += wc[1]
            for i in [2, 3]:
                for k in [4, 5]:
                    w_vec[k*4+i] += wc[2]
            for i in [2, 3]:
                for k in [6, 7]:
                    w_vec[k*4+i] += wc[3]
                    
        add_term(u_vec, v_vec, w_vec, col)
        col += 1
        
    return {"U": U, "V": V, "W": W}
