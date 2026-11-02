import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = 2, 3, 6
    
    # Try to find a lower rank decomposition for 2x3x2
    # If found, use it 3 times for 2x3x6
    # Rank of 2x3x2 is suspected to be 8
    
    def get_2x3x2_decomp():
        # Returns U, V, W for 2x3x2 with minimal rank found
        # U: (2*3, R), V: (3*2, R), W: (2*2, R)
        # Target: sum_r U[i*3+j, r] * V[j*2+k, r] * W[k*2+i, r] == 1
        
        # Try Rank 8
        # We can try to construct it explicitly using Strassen + 1
        # Strassen for A[0:2, 0:2] * B[0:2, :]
        # Plus A[0:2, 2] * B[2, :]
        
        # Strassen Mults:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B11+B01)
        
        # M8 = A02*B20 ? No, A[0:2, 2] is a vector. B[2, 0:2] is a vector.
        # The term is A_i2 * B_2k.
        # This is rank 1.
        # U part: A_i2. So U[i*3+2, r] = 1 for the term.
        # V part: B_2k. So V[2*2+k, r] = 1.
        # W part: C_ik. So W[k*2+i, r] = 1.
        
        # Let's build the matrices for R=8
        
        R = 8
        U = np.zeros((6, R))
        V = np.zeros((6, R))
        W = np.zeros((4, R))
        
        # Strassen terms (indices 0-6)
        # M1
        # U: A00, A11. V: B00, B11. W: C00, C01, C10, C11 (with signs)
        # This is getting complex to hardcode correctly without errors.
        # Let's use a numerical solver to find a solution for R=8.
        
        # Variables: U, V, W entries.
        # Equations: 12.
        # We can fix some variables to integers/half-integers and solve?
        # Or just random search?
        
        # Given the risk, let's stick to Schoolbook if we can't guarantee the decomposition.
        # But 24 is a huge improvement.
        
        # Let's try a simple search for R=8.
        # We can use scipy.optimize.least_squares to find real solution, then round?
        # No, exact integer/half-integer required.
        
        # Alternative: Just return Schoolbook.
        # The prompt says "score is the rank".
        # If I return 36, score is 36.
        # If I return 24, score is 24.
        
        # I will attempt to implement the explicit Strassen+1 construction.
        
        # Strassen for 2x2:
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(g-h)
        # M4 = d(f-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+g)
        # M7 = (b-d)(h+f)
        
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M7
        # C10 = M2 + M4
        # C11 = M1 - M2 + M6 - M3
        
        # Mapping to U, V, W for 2x3x2:
        # i in 0..1, j in 0..2, k in 0..1
        # U index: i*3 + j
        # V index: j*2 + k
        # W index: k*2 + i
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        # This corresponds to 4 products: A00B00, A00B11, A11B00, A11B11
        # In the decomposition, we sum these.
        # U[0,0]=1 (A00), U[3,0]=1 (A11)
        # V[0,0]=1 (B00), V[2,0]=1 (B11) -- Wait, B index is j*2+k.
        # B00 is j=0,k=0 -> idx 0.
        # B11 is j=1,k=1 -> idx 1*2+1=3.
        # So V[0,0]=1, V[3,0]=1.
        # W contributions:
        # C00: +1
        # C01: 0
        # C10: 0
        # C11: +1
        # W index: k*2+i.
        # C00: k=0,i=0 -> idx 0.
        # C11: k=1,i=1 -> idx 3.
        # So W[0,0]=1, W[3,0]=1.
        
        # We need to check if this rank-1 term contributes correctly to the sum.
        # The sum over r of U[i,j]_r V[j,k]_r W[k,i]_r must equal 1 for each i,j,k.
        
        # Let's define the 8 terms explicitly.
        
        terms = []
        
        # Helper to add a term defined by lists of (idx, val) for U, V, W
        def add_term(u_vals, v_vals, w_vals):
            terms.append((u_vals, v_vals, w_vals))
            
        # M1: (A00+A11)(B00+B11)
        # This is not a single multiplication in the tensor sense, it's a sum of 4.
        # But Strassen combines them into 1 multiplication of sums.
        # The "multiplication" is of the linear forms.
        # So U has non-zero entries for A00 and A11.
        # V has non-zero entries for B00 and B11.
        # W has non-zero entries for C00 and C11.
        
        # Term 0 (M1)
        add_term([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)])
        
        # Term 1 (M2): (A10+A11)B00
        # U: A10 (idx 1*3+0=3? No, i=1,j=0 -> 3), A11 (i=1,j=1 -> 4)
        # V: B00 (j=0,k=0 -> 0)
        # W: C10 (k=0,i=1 -> 2), C11 (k=1,i=1 -> 3)
        # C10 = M2 + ...
        # C11 = ... - M2
        # So W[2,1]=1, W[3,1]=-1
        add_term([(3,1), (4,1)], [(0,1)], [(2,1), (3,-1)])
        
        # Term 2 (M3): A00(B10-B11)
        # U: A00 (0)
        # V: B10 (j=1,k=0 -> 2), B11 (j=1,k=1 -> 3)
        # W: C01 (k=1,i=0 -> 1), C11 (k=1,i=1 -> 3)
        # C01 = M3 + ...
        # C11 = ... - M3
        # W[1,2]=1, W[3,2]=-1
        add_term([(0,1)], [(2,1), (3,-1)], [(1,1), (3,-1)])
        
        # Term 3 (M4): A11(B01-B00)
        # U: A11 (4)
        # V: B01 (j=0,k=1 -> 1), B00 (j=0,k=0 -> 0)
        # W: C00 (0), C10 (2)
        # C00 = ... - M4
        # C10 = ... + M4
        # W[0,3]=-1, W[2,3]=1
        add_term([(4,1)], [(1,1), (0,-1)], [(0,-1), (2,1)])
        
        # Term 4 (M5): (A00+A01)B11
        # U: A00 (0), A01 (1)
        # V: B11 (3)
        # W: C00 (0), C01 (1)
        # C00 = ... + M5
        # C01 = ... + M5?
        # Strassen: C00 = M1 - M4 + M5 - M7. C01 = M3 + M7.
        # So M5 only in C00?
        # Wait, Strassen:
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M7
        # C10 = M2 + M4
        # C11 = M1 - M2 + M6 - M3
        # So M5 is only in C00.
        # W[0,4]=1
        add_term([(0,1), (1,1)], [(3,1)], [(0,1)])
        
        # Term 5 (M6): (A10-A00)(B00+B10)
        # U: A10 (3), A00 (0)
        # V: B00 (0), B10 (2)
        # W: C11 (3)
        # C11 = ... + M6
        # W[3,5]=1
        add_term([(3,1), (0,-1)], [(0,1), (2,1)], [(3,1)])
        
        # Term 6 (M7): (A01-A11)(B11+B01)
        # U: A01 (1), A11 (4)
        # V: B11 (3), B01 (1)
        # W: C00 (0), C01 (1)
        # C00 = ... - M7
        # C01 = ... + M7
        # W[0,6]=-1, W[1,6]=1
        add_term([(1,1), (4,-1)], [(3,1), (1,1)], [(0,-1), (1,1)])
        
        # Term 7 (M8): A_i2 B_2k
        # U: A02 (2), A12 (5)
        # V: B20 (j=2,k=0 -> 4), B21 (j=2,k=1 -> 5)
        # W: C00 (0), C01 (1), C10 (2), C11 (3)
        # All +1
        add_term([(2,1), (5,1)], [(4,1), (5,1)], [(0,1), (1,1), (2,1), (3,1)])
        
        # Construct U, V, W
        U = np.zeros((6, 8))
        V = np.zeros((6, 8))
        W = np.zeros((4, 8))
        
        for r, (u_vals, v_vals, w_vals) in enumerate(terms):
            for idx, val in u_vals:
                U[idx, r] = val
            for idx, val in v_vals:
                V[idx, r] = val
            for idx, val in w_vals:
                W[idx, r] = val
                
        # Verify
        # Check sum_r U[i*3+j, r] V[j*2+k, r] W[k*2+i, r] == 1
        valid = True
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    s = 0
                    for r in range(8):
                        s += U[i*3+j, r] * V[j*2+k, r] * W[k*2+i, r]
                    if s != 1:
                        valid = False
                        break
                if not valid: break
            if not valid: break
            
        if valid:
            return U, V, W
        else:
            return None

    U232, V232, W232 = get_2x3x2_decomp()
    
    if U232 is not None:
        # Combine 3 copies for 2x3x6
        # n=2, m=3, p=6
        # U: (2*3, 24)
        # V: (3*6, 24)
        # W: (6*2, 24)
        
        U = np.zeros((6, 24))
        V = np.zeros((18, 24))
        W = np.zeros((12, 24))
        
        for t in range(3):
            # U is same for all blocks (A is 2x3)
            U[:, t*8:(t+1)*8] = U232
            
            # V: B is 3x6. Block t is B[:, t*2:(t+1)*2]
            # V index for block t: j*6 + (t*2 + k)
            # U232 V index: j*2 + k
            for j in range(3):
                for k in range(2):
                    idx_v_232 = j*2 + k
                    idx_v_236 = j*6 + (t*2 + k)
                    V[idx_v_236, t*8:(t+1)*8] = V232[idx_v_232, :]
                    
            # W: C is 2x6. Block t is C[:, t*2:(t+1)*2]
            # W index for block t: (t*2 + k)*2 + i
            # U232 W index: k*2 + i
            for k in range(2):
                for i in range(2):
                    idx_w_232 = k*2 + i
                    idx_w_236 = (t*2 + k)*2 + i
                    W[idx_w_236, t*8:(t+1)*8] = W232[idx_w_232, :]
                    
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank = 36
        U = np.zeros((6, rank))
        V = np.zeros((18, rank))
        W = np.zeros((12, rank))
        
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(6):
                    U[i*3+j, col] = 1
                    V[j*6+k, col] = 1
                    W[k*2+i, col] = 1
                    col += 1
                    
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
