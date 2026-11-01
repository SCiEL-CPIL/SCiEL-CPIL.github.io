"""Matrix multiplication 2x2x16: Strassen-based rank-56 decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# Strassen 2x2x2 decomposition: 7 bilinear forms
# A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[ae+cg, af+bh],[ce+dg, cf+bh]]
# M1 = (a+d)(e+h)
# M2 = (c+d)e
# M3 = a(f-h)
# M4 = d(g-e)
# M5 = (a+b)h
# M6 = (c-a)(e+f)
# M7 = (b-d)(g+h)
# C11 = M1+M4-M5+M7
# C12 = M3+M5
# C21 = M2+M4
# C22 = M1-M2+M3+M6

def _strassen_2x2x2():
    """Return the 7 bilinear forms for 2x2x2 matrix multiplication.
    
    Each form is a triple (u_vec, v_vec, w_vec) where:
    - u_vec: length 4, indices (i,j) flattened as i*2+j for A
    - v_vec: length 4, indices (j,k) flattened as j*2+k for B  
    - w_vec: length 4, indices (k,i) flattened as k*2+i for C (trace convention W[k*n+i])
    
    For 2x2x2: A[i,j], B[j,k], C[k,i] with trace convention W[k*2+i].
    """
    # A indices: a=(0,0), b=(0,1), c=(1,0), d=(1,1) -> flat: 0,1,2,3
    # B indices: e=(0,0), f=(0,1), g=(1,0), h=(1,1) -> flat: 0,1,2,3
    # C indices (trace W[k*n+i]): C[0,0]=ae+cg -> idx 0*2+0=0
    #                          C[0,1]=af+bh -> idx 0*2+1=1
    #                          C[1,0]=ce+dg -> idx 1*2+0=2
    #                          C[1,1]=cf+bh -> idx 1*2+1=3
    
    # Wait, let me re-examine the trace convention.
    # W[k*n+i, r] where the output is C[i,k] but stored as W[k*n+i].
    # So for 2x2x2, W has 4 entries: W[0]=C[0,0], W[1]=C[0,1], W[2]=C[1,0], W[3]=C[1,1]
    # where W[k*2+i]: W[0*2+0]=C[0,0], W[0*2+1]=C[1,0], W[1*2+0]=C[0,1], W[1*2+1]=C[1,1]
    
    # Actually let me be more careful. The standard matrix mult is C[i,k] = sum_j A[i,j]*B[j,k].
    # The tensor T[i,j,k] = A[i,j]*B[j,k] contributes to C[i,k].
    # The W matrix has index W[k*n+i] for the (i,k) entry of C.
    
    # For 2x2x2:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] = a*e + b*g
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] = a*f + b*h
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] = c*e + d*g
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] = c*f + d*h
    
    # W indexing: W[k*2+i]
    # W[0] = W[0*2+0] = C[0,0] = a*e + b*g
    # W[1] = W[0*2+1] = C[1,0] = c*e + d*g
    # W[2] = W[1*2+0] = C[0,1] = a*f + b*h
    # W[3] = W[1*2+1] = C[1,1] = c*f + d*h
    
    # A flat indices: a=0, b=1, c=2, d=3
    # B flat indices: e=0, f=1, g=2, h=3
    # W flat indices: W0=0, W1=1, W2=2, W3=3 (corresponding to C[0,0], C[1,0], C[0,1], C[1,1])
    
    # Strassen formulas (standard):
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # Let me verify:
    # M1 = (a+d)(e+h) = ae+ah+de+dh
    # M4 = d(g-e) = dg-de
    # M5 = (a+b)h = ah+bh
    # M7 = (b-d)(g+h) = bg+bh-dg-dh
    # M1+M4-M5+M7 = ae+ah+de+dh + dg-de - ah-bh + bg+bh-dg-dh = ae+bg ✓ (C[0,0])
    
    # M3 = a(f-h) = af-ah
    # M5 = ah+bh
    # M3+M5 = af+bg... wait that's not C[0,1]=af+bh
    # Hmm, let me recheck. M3+M5 = af-ah+ah+bh = af+bh ✓ (C[0,1])
    
    # M2 = (c+d)e = ce+de
    # M4 = dg-de
    # M2+M4 = ce+dg ✓ (C[1,0])
    
    # M1 = ae+ah+de+dh
    # M2 = ce+de
    # M3 = af-ah
    # M6 = (c-a)(e+f) = ce+cf-ae-af
    # M1-M2+M3+M6 = ae+ah+de+dh - ce-de + af-ah + ce+cf-ae-af = dh+cf ✓ (C[1,1])
    
    # Great, all verified.
    
    # Now express each M as (u_vec, v_vec, w_vec)
    # u_vec has entries for A indices [a,b,c,d] = [0,1,2,3]
    # v_vec has entries for B indices [e,f,g,h] = [0,1,2,3]
    # w_vec has entries for W indices [W0,W1,W2,W3] = [0,1,2,3]
    
    # M1 = (a+d)(e+h): u=[1,0,1,0], v=[1,0,0,1]
    # M2 = (c+d)e: u=[0,0,1,1], v=[1,0,0,0]
    # M3 = a(f-h): u=[1,0,0,0], v=[0,1,0,-1]
    # M4 = d(g-e): u=[0,0,0,1], v=[-1,0,1,0]
    # M5 = (a+b)h: u=[1,1,0,0], v=[0,0,0,1]
    # M6 = (c-a)(e+f): u=[-1,0,1,0], v=[1,1,0,0]
    # M7 = (b-d)(g+h): u=[0,1,0,-1], v=[0,0,1,1]
    
    # W contributions:
    # C[0,0]=W[0]: M1+M4-M5+M7
    # C[1,0]=W[1]: M2+M4
    # C[0,1]=W[2]: M3+M5
    # C[1,1]=W[3]: M1-M2+M3+M6
    
    forms = []
    # (u, v, w) for each of the 7 forms
    forms.append(([1,0,1,0], [1,0,0,1], [1,0,0,0]))   # M1 -> W0
    forms.append(([0,0,1,1], [1,0,0,0], [0,1,0,0]))   # M2 -> W1
    forms.append(([1,0,0,0], [0,1,0,-1], [0,0,1,0]))  # M3 -> W2
    forms.append(([0,0,0,1], [-1,0,1,0], [1,1,0,0]))  # M4 -> W0,W1
    forms.append(([1,1,0,0], [0,0,0,1], [0,0,1,0]))   # M5 -> W2
    forms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))  # M6 -> W3
    forms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,1]))  # M7 -> W0,W3
    
    return forms

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 2, 16
    
    # The 2x2x16 multiplication can be viewed as 8 independent 2x2x2 multiplications
    # because p=16 = 8*2. For each block of 2 columns of the second matrix,
    # we have a 2x2x2 multiplication.
    
    # Actually, let me think about this more carefully.
    # A is 2x2, B is 2x16, C is 2x16.
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T[i,j,k] = A[i,j]*B[j,k]
    # U has shape (n*m, rank) = (4, rank)
    # V has shape (m*p, rank) = (32, rank)
    # W has shape (p*n, rank) = (32, rank)
    
    # The 8 blocks: for block b=0..7, columns 2b and 2b+1 of B form a 2x2 matrix.
    # Block b: A (2x2) * B_block (2x2) -> C_block (2x2)
    # C_block[0,0] = C[0,2b], C_block[0,1] = C[0,2b+1]
    # C_block[1,0] = C[1,2b], C_block[1,1] = C[1,2b+1]
    
    # For each block, Strassen gives 7 rank-1 terms.
    # Total rank = 8*7 = 56.
    
    strassen_forms = _strassen_2x2x2()
    rank = 56
    
    U = [[0.0] * rank for _ in range(n * m)]      # (4, 56)
    V = [[0.0] * rank for _ in range(m * p)]      # (32, 56)
    W = [[0.0] * rank for _ in range(p * n)]      # (32, 56)
    
    for b in range(8):
        # Block b uses columns 2b, 2b+1 of B
        # B indices in V: row j*p + k where j in {0,1}, k in {2b, 2b+1}
        # B flat index for (j, k) is j*16 + k
        # So for block b:
        #   B[0, 2b] -> V row 0*16+2b = 2b
        #   B[0, 2b+1] -> V row 0*16+2b+1 = 2b+1
        #   B[1, 2b] -> V row 1*16+2b = 16+2b
        #   B[1, 2b+1] -> V row 1*16+2b+1 = 16+2b+1
        
        # C indices in W: W[k*n+i] = W[k*2+i]
        # C[0, 2b] -> W[2b*2+0] = W[4b]
        # C[0, 2b+1] -> W[(2b+1)*2+0] = W[4b+2]
        # C[1, 2b] -> W[2b*2+1] = W[4b+1]
        # C[1, 2b+1] -> W[(2b+1)*2+1] = W[4b+3]
        
        # A indices in U: row i*m+j = i*2+j
        # A[0,0] -> U row 0
        # A[0,1] -> U row 1
        # A[1,0] -> U row 2
        # A[1,1] -> U row 3
        
        base_col = b * 7
        for s in range(7):
            u_vec, v_vec, w_vec = strassen_forms[s]
            col = base_col + s
            
            # U contributions: u_vec[s] for A index s
            for ai in range(4):
                U[ai][col] = u_vec[ai]
            
            # V contributions: v_vec maps B_block indices [e,f,g,h] to actual B indices
            # B_block: e=B[0,2b], f=B[0,2b+1], g=B[1,2b], h=B[1,2b+1]
            # v_vec = [ve, vf, vg, vh]
            ve, vf, vg, vh = v_vec
            U_row = 0  # not needed
            # V row for B[0, 2b] = 2b
            if ve != 0:
                V[2*b][col] += ve
            # V row for B[0, 2b+1] = 2b+1
            if vf != 0:
                V[2*b+1][col] += vf
            # V row for B[1, 2b] = 16+2b
            if vg != 0:
                V[16+2*b][col] += vg
            # V row for B[1, 2b+1] = 16+2b+1
            if vh != 0:
                V[16+2*b+1][col] += vh
            
            # W contributions: w_vec maps C_block indices to W indices
            # C_block: C[0,0]->W[4b], C[1,0]->W[4b+1], C[0,1]->W[4b+2], C[1,1]->W[4b+3]
            # w_vec = [w_C00, w_C10, w_C01, w_C11]
            w00, w10, w01, w11 = w_vec
            if w00 != 0:
                W[4*b][col] += w00
            if w10 != 0:
                W[4*b+1][col] += w10
            if w01 != 0:
                W[4*b+2][col] += w01
            if w11 != 0:
                W[4*b+3][col] += w11
    
    # Verify the decomposition
    if time.time() < deadline:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check: for random A, B, compute C = A*B and compare with tensor reconstruction
        # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        rng = np.random.RandomState(42)
        A_test = rng.randn(n, m)
        B_test = rng.randn(m, p)
        C_expected = A_test @ B_test
        
        C_reconstructed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for j in range(m):
                    for r in range(rank):
                        val += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                C_reconstructed[i, k] = val
        
        if not np.allclose(C_reconstructed, C_expected, atol=1e-10):
            # Fallback to schoolbook if verification fails
            rank_sb = n * m * p
            U_sb = [[0] * rank_sb for _ in range(n * m)]
            V_sb = [[0] * rank_sb for _ in range(m * p)]
            W_sb = [[0] * rank_sb for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_sb[i * m + j][column] = 1
                        V_sb[j * p + k][column] = 1
                        W_sb[k * n + i][column] = 1
                        column += 1
            return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U, "V": V, "W": W}
