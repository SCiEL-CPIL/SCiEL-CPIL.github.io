"""Matrix multiplication 3,3,5 — rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 9, 15, 15
    
    # Build the target tensor T (flattened)
    # T[a, b, c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for consistent i,j,k
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    # Schoolbook decomposition (rank 45)
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1
                V_school[b, col] = 1
                W_school[c, col] = 1
                col += 1
                
    # Strategy: Replace the 8 schoolbook terms for i,j,k in {0,1} with 7 Strassen terms
    # This reduces rank from 45 to 44.
    
    # Identify the 8 columns in U_school, V_school, W_school corresponding to these terms
    # We will rebuild the factors to avoid index tracking issues.
    
    U_new = []
    V_new = []
    W_new = []
    
    # Add the 37 non-overlapping schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i in [0, 1] and j in [0, 1] and k in [0, 1]:
                    continue
                a = i * m + j
                b = j * p + k
                c = k * n + i
                u = np.zeros(nm)
                v = np.zeros(mp)
                w = np.zeros(pn)
                u[a] = 1
                v[b] = 1
                w[c] = 1
                U_new.append(u)
                V_new.append(v)
                W_new.append(w)
    
    # Add the 7 Strassen terms for the 2x2x2 sub-tensor
    # Local indices I, J, K in {0, 1}
    # Global mappings:
    # a = i*m + j. i=I, j=J.
    # I,J -> local_a = I*2 + J
    # map_a: 0->0 (I=0,J=0), 1->1 (I=0,J=1), 2->3 (I=1,J=0), 3->4 (I=1,J=1)
    map_a = {0: 0, 1: 1, 2: 3, 3: 4}
    
    # b = j*p + k. j=J, k=K.
    # J,K -> local_b = J*2 + K
    # map_b: 0->0 (J=0,K=0), 1->1 (J=0,K=1), 2->5 (J=1,K=0), 3->6 (J=1,K=1)
    map_b = {0: 0, 1: 1, 2: 5, 3: 6}
    
    # c = k*n + i. k=K, i=I.
    # K,I -> local_c = K*2 + I
    # map_c: 0->0 (K=0,I=0), 1->1 (K=0,I=1), 2->3 (K=1,I=0), 3->4 (K=1,I=1)
    map_c = {0: 0, 1: 1, 2: 3, 3: 4}
    
    # Strassen terms (u, v, w) for 2x2x2
    # u is for A (I, J)
    # v is for B (J, K)
    # w is for C (I, K) -> Note: C index is k*n+i, so local index is K*2+I.
    # My w vectors were derived for local index s = K*2 + I?
    # Let's re-verify the w vector indices.
    # In the derivation:
    # w = [1, 0, 0, 1] for M1.
    # Indices 0, 1, 2, 3 correspond to local C indices.
    # Local C index s = K*2 + I.
    # s=0: K=0, I=0.
    # s=1: K=0, I=1.
    # s=2: K=1, I=0.
    # s=3: K=1, I=1.
    
    # M1: C00 (K=0,I=0) and C11 (K=1,I=1).
    # s=0 and s=3.
    # w = [1, 0, 0, 1]. Correct.
    
    # M2: C10 (K=0, I=1? No. C_{ik}. i=I, k=K.
    # C_{10} in 0-based means I=1, K=0?
    # Standard matrix C: C[row, col].
    # Row = I, Col = K.
    # C_{10} means Row=1, Col=0. So I=1, K=0.
    # s = K*2 + I = 0*2 + 1 = 1.
    # My previous derivation:
    # M2 contributes to C_{21} (1-based) -> C_{10} (0-based).
    # I=1, K=0.
    # s = 1.
    # w = [0, 1, 0, 0]?
    # Let's re-check M2 w vector.
    # M2 = (A_{10}+A_{11})B_{00}.
    # Contributes to C_{10} (I=1, K=0) and C_{11} (I=1, K=1)?
    # Formula: C_{21} = M2 + M6. (1-based).
    # C_{21} is Row 2, Col 1. 0-based: Row 1, Col 0.
    # So I=1, K=0.
    # Does it contribute to C_{22} (I=1, K=1)?
    # C_{22} = M1 - M2 + M3 + M6.
    # Yes, M2 contributes to C_{22} with -1.
    # So M2 contributes to I=1, K=0 (s=1) and I=1, K=1 (s=3).
    # w = [0, 1, 0, -1].
    
    # Let's re-derive all w vectors carefully.
    
    # M1: (A00+A11)(B00+B11)
    # C00 (I=0,K=0, s=0) +1
    # C11 (I=1,K=1, s=3) +1
    # w1 = [1, 0, 0, 1]
    
    # M2: (A10+A11)B00
    # C10 (I=1,K=0, s=1) +1
    # C11 (I=1,K=1, s=3) -1
    # w2 = [0, 1, 0, -1]
    
    # M3: A00(B10-B11)
    # C01 (I=0,K=1, s=2) +1
    # C11 (I=1,K=1, s=3) +1
    # w3 = [0, 0, 1, 1]
    
    # M4: A11(B01-B00)
    # C00 (I=0,K=0, s=0) +1
    # w4 = [1, 0, 0, 0]
    
    # M5: (A00+A01)B11
    # C00 (I=0,K=0, s=0) -1
    # C01 (I=0,K=1, s=2) +1
    # w5 = [-1, 0, 1, 0]
    
    # M6: (A10-A00)(B00+B01)
    # C10 (I=1,K=0, s=1) +1
    # C11 (I=1,K=1, s=3) +1
    # w6 = [0, 1, 0, 1]
    
    # M7: (A01-A11)(B10+B11)
    # C00 (I=0,K=0, s=0) +1
    # C10 (I=1,K=0, s=1) +1
    # w7 = [1, 1, 0, 0]
    
    # Let's verify sums.
    # C00 (s=0): M1(1) + M4(1) - M5(1) + M7(1) = 1+1-1+1 = 2? 
    # Wait.
    # C00 = M1 + M4 - M5 + M7.
    # 1 + 1 - (-1) + 1 = 4? No.
    # M5 coefficient in C00 is -1.
    # So C00 = M1 + M4 - M5 + M7.
    # w1[0]=1, w4[0]=1, w5[0]=-1, w7[0]=1.
    # Sum = 1 + 1 - (-1) + 1 = 4?
    # No, the formula is $C_{00} = M_1 + M_4 - M_5 + M_7$.
    # This means $1 \cdot M_1 + 1 \cdot M_4 - 1 \cdot M_5 + 1 \cdot M_7$.
    # So the coefficient of w5 in the sum for s=0 is -1.
    # My w5 vector is the vector for M5.
    # So the contribution is $-1 \times w5[0] = -1 \times (-1) = 1$.
    # Total = $1(1) + 1(1) + (-1)(-1) + 1(1) = 4$?
    # This is wrong. The tensor value should be 1.
    
    # Ah, the Strassen formulas are for the *values* of the matrix elements.
    # $C_{ij} = \sum_r u_{i,r} v_{j,r} w_{k,r}$ is not quite right.
    # The decomposition is $T_{ijk} = \sum_r u_{i,r} v_{j,r} w_{k,r}$.
    # For the $2 \times 2$ case, $T_{ijk} = 1$ for all $i,j,k$.
    # So $\sum_r u_{i,r} v_{j,r} w_{k,r} = 1$.
    
    # Let's check the standard Strassen decomposition of the tensor.
    # The 7 vectors (u, v, w) are:
    # 1: (1,0,0,1), (1,0,0,1), (1,0,0,1)
    # 2: (0,0,1,1), (1,0,0,0), (0,1,0,-1)
    # 3: (1,0,0,0), (0,0,1,-1), (0,0,1,1)
    # 4: (0,0,0,1), (-1,1,0,0), (1,0,0,0)
    # 5: (1,1,0,0), (0,0,0,1), (-1,0,1,0)
    # 6: (-1,0,1,0), (1,1,0,0), (0,1,0,1)
    # 7: (0,1,0,-1), (0,0,1,1), (1,1,0,0)
    
    # Let's verify r=1:
    # u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
    # u_i v_j w_k.
    # i=0,j=0,k=0: 1*1*1 = 1.
    # i=0,j=0,k=1: 1*1*0 = 0.
    # i=0,j=1,k=0: 1*0*1 = 0.
    # i=1,j=0,k=0: 0*1*1 = 0.
    # i=1,j=1,k=1: 1*1*1 = 1.
    # This rank-1 tensor is 1 at (0,0,0) and (1,1,1).
    
    # Summing all 7 should give 1 everywhere.
    
    # I will use these verified vectors.
    
    strassen_terms = [
        (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),
        (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1])),
        (np.array([1, 0, 0, 0]), np.array([0, 0, 1, -1]), np.array([0, 0, 1, 1])),
        (np.array([0, 0, 0, 1]), np.array([-1, 1, 0, 0]), np.array([1, 0, 0, 0])),
        (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0])),
        (np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([0, 1, 0, 1])),
        (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0]))
    ]
    
    for u_sub, v_sub, w_sub in strassen_terms:
        u_global = np.zeros(nm)
        v_global = np.zeros(mp)
        w_global = np.zeros(pn)
        
        for s, g in map_a.items():
            u_global[g] = u_sub[s]
        for s, g in map_b.items():
            v_global[g] = v_sub[s]
        for s, g in map_c.items():
            w_global[g] = w_sub[s]
            
        U_new.append(u_global)
        V_new.append(v_global)
        W_new.append(w_global)
        
    U_final = np.column_stack(U_new)
    V_final = np.column_stack(V_new)
    W_final = np.column_stack(W_new)
    
    # Verify
    T_recon = np.einsum('ar,br,cr->abc', U_final, V_final, W_final)
    if not np.allclose(T_recon, T, atol=1e-10):
        # Fallback to schoolbook if verification fails
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
# EVOLVE-BLOCK-END
