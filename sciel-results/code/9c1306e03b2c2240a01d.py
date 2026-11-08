import numpy as np
import time

def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    """
    start_time = time.time()
    DEADLINE = 145
    
    n, m, p = 4, 4, 4
    rank = 49
    
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    # Flat indices for A, B elements
    # A[i,j] at index i*4+j
    # B[k,j] at index k*4+j  
    # C[i,j] at index j*4+i (trace convention W[k*n+i] with k=j_col, i=row)
    
    def lc(entries):
        d = {}
        for idx, coeff in entries:
            d[idx] = d.get(idx, 0) + coeff
        return d
    
    # A element indices
    A00,A01,A02,A03 = 0,1,2,3
    A10,A11,A12,A13 = 4,5,6,7
    A20,A21,A22,A23 = 8,9,10,11
    A30,A31,A32,A33 = 12,13,14,15
    
    # B element indices (same layout)
    B00,B01,B02,B03 = 0,1,2,3
    B10,B11,B12,B13 = 4,5,6,7
    B20,B21,B22,B23 = 8,9,10,11
    B30,B31,B32,B33 = 12,13,14,15
    
    # W element indices: C[i,j] -> W[j*4+i]
    Widx = lambda i,j: j*4+i
    
    # Block indices for 2x2 blocks
    # Block (br,bc) of C: C[2*br:r, 2*bc:c] for r,c in 0,1
    # C11 (block 0,0): C[0,0],C[0,1],C[1,0],C[1,1] -> W[0],W[1],W[4],W[5]
    # C12 (block 0,1): C[0,2],C[0,3],C[1,2],C[1,3] -> W[8],W[9],W[12],W[13]
    # C21 (block 1,0): C[2,0],C[2,1],C[3,0],C[3,1] -> W[2],W[3],W[6],W[7]
    # C22 (block 1,1): C[2,2],C[2,3],C[3,2],C[3,3] -> W[10],W[11],W[14],W[15]
    
    C11 = [Widx(0,0), Widx(0,1), Widx(1,0), Widx(1,1)]
    C12 = [Widx(0,2), Widx(0,3), Widx(1,2), Widx(1,3)]
    C21 = [Widx(2,0), Widx(2,1), Widx(3,0), Widx(3,1)]
    C22 = [Widx(2,2), Widx(2,3), Widx(3,2), Widx(3,3)]
    
    # 7 block-level Strassen terms
    # Each term: (L_matrix, R_matrix, contrib_dict)
    # L_matrix[r][c] = linear combo of A elements (flat index -> coeff)
    # R_matrix[r][c] = linear combo of B elements
    # contrib_dict[(zr,zc)] = list of W indices (for the 2x2 block result)
    
    terms = []
    
    # P1 = (A11+A22)(B11+B22)
    # A11 = [[A00,A01],[A10,A11]], A22 = [[A20,A21],[A30,A31]]
    # B11 = [[B00,B01],[B10,B11]], B22 = [[B20,B21],[B30,B31]]
    L0 = [[lc([(A00,1),(A20,1)]), lc([(A01,1),(A21,1)])],
          [lc([(A10,1),(A30,1)]), lc([(A11,1),(A31,1)])]]
    R0 = [[lc([(B00,1),(B20,1)]), lc([(B01,1),(B21,1)])],
          [lc([(B10,1),(B30,1)]), lc([(B11,1),(B31,1)])]]
    terms.append((L0, R0, {(0,0): C11, (1,1): C22}))
    
    # P2 = (A21+A22)B11
    # A21 = [[A02,A03],[A12,A13]], A22 = [[A20,A21],[A30,A31]]
    L1 = [[lc([(A02,1),(A20,1)]), lc([(A03,1),(A21,1)])],
          [lc([(A12,1),(A30,1)]), lc([(A13,1),(A31,1)])]]
    R1 = [[lc([(B00,1)]), lc([(B01,1)])],
          [lc([(B10,1)]), lc([(B11,1)])]]
    terms.append((L1, R1, {(1,0): C21, (1,1): C22}))
    
    # P3 = A11(B12-B22)
    L2 = [[lc([(A00,1)]), lc([(A01,1)])],
          [lc([(A10,1)]), lc([(A11,1)])]]
    R2 = [[lc([(B20,1),(B00,-1)]), lc([(B21,1),(B01,-1)])],
          [lc([(B30,1),(B10,-1)]), lc([(B31,1),(B11,-1)])]]
    terms.append((L2, R2, {(0,1): C12, (1,1): C22}))
    
    # P4 = A22(B21-B11)
    L3 = [[lc([(A20,1)]), lc([(A21,1)])],
          [lc([(A30,1)]), lc([(A31,1)])]]
    R3 = [[lc([(B20,1),(B00,-1)]), lc([(B21,1),(B01,-1)])],
          [lc([(B30,1),(B10,-1)]), lc([(B31,1),(B11,-1)])]]
    terms.append((L3, R3, {(0,0): C11, (1,0): C21}))
    
    # P5 = (A11+A12)B22
    # A11 = [[A00,A01],[A10,A11]], A12 = [[A02,A03],[A12,A13]]
    L4 = [[lc([(A00,1),(A02,1)]), lc([(A01,1),(A03,1)])],
          [lc([(A10,1),(A12,1)]), lc([(A11,1),(A13,1)])]]
    R4 = [[lc([(B20,1)]), lc([(B21,1)])],
          [lc([(B30,1)]), lc([(B31,1)])]]
    terms.append((L4, R4, {(0,0): C11, (0,1): C12}))
    
    # P6 = (A21-A11)(B11+B12)
    L5 = [[lc([(A02,1),(A00,-1)]), lc([(A03,1),(A01,-1)])],
          [lc([(A12,1),(A10,-1)]), lc([(A13,1),(A11,-1)])]]
    R5 = [[lc([(B00,1),(B20,1)]), lc([(B01,1),(B21,1)])],
          [lc([(B10,1),(B30,1)]), lc([(B11,1),(B31,1)])]]
    terms.append((L5, R5, {(1,1): C22}))
    
    # P7 = (A12-A22)(B21+B22)
    L6 = [[lc([(A02,1),(A20,-1)]), lc([(A03,1),(A21,-1)])],
          [lc([(A12,1),(A30,-1)]), lc([(A13,1),(A31,-1)])]]
    R6 = [[lc([(B20,1),(B00,1)]), lc([(B21,1),(B01,1)])],
          [lc([(B30,1),(B10,1)]), lc([(B31,1),(B11,1)])]]
    terms.append((L6, R6, {(0,0): C11}))
    
    # Inner Strassen for 2x2: Z = X*Y
    # m0=(x00+x11)(y00+y11), m1=(x10+x11)y00, m2=x00(y01-y11)
    # m3=x11(y10-y00), m4=(x00+x01)y11, m5=(x10-x00)(y00+y01), m6=(x01-x11)(y10+y11)
    # Z00=m0+m3-m4+m6, Z01=m2+m4, Z10=m1+m3, Z11=m0-m1+m2+m5
    
    inner_contrib = [
        [(0,0,1),(1,1,1)],
        [(1,0,1),(1,1,-1)],
        [(0,1,1),(1,1,1)],
        [(0,0,1),(1,0,1)],
        [(0,0,-1),(0,1,1)],
        [(1,1,1)],
        [(0,0,1)]
    ]
    
    for t in range(7):
        Lt, Rt, ct = terms[t]
        x00, x01, x10, x11 = Lt[0][0], Lt[0][1], Lt[1][0], Lt[1][1]
        y00, y01, y10, y11 = Rt[0][0], Rt[0][1], Rt[1][0], Rt[1][1]
        
        subs = [
            ( {**{k:x00.get(k,0)+x11.get(k,0) for k in set(x00)|set(x11) if x00.get(k,0)+x11.get(k,0)!=0},
              **{}}, 
              {**{k:y00.get(k,0)+y11.get(k,0) for k in set(y00)|set(y11) if y00.get(k,0)+y11.get(k,0)!=0},
              **{}} ),
            ( {**{k:x10.get(k,0)+x11.get(k,0) for k in set(x10)|set(x11) if x10.get(k,0)+x11.get(k,0)!=0}}, 
              dict(y00) ),
            ( dict(x00), 
              {**{k:y01.get(k,0)-y11.get(k,0) for k in set(y01)|set(y11) if y01.get(k,0)-y11.get(k,0)!=0}} ),
            ( dict(x11), 
              {**{k:y10.get(k,0)-y00.get(k,0) for k in set(y10)|set(y00) if y10.get(k,0)-y00.get(k,0)!=0}} ),
            ( {**{k:x00.get(k,0)+x01.get(k,0) for k in set(x00)|set(x01) if x00.get(k,0)+x01.get(k,0)!=0}}, 
              dict(y11) ),
            ( {**{k:x10.get(k,0)-x00.get(k,0) for k in set(x10)|set(x00) if x10.get(k,0)-x00.get(k,0)!=0}}, 
              {**{k:y00.get(k,0)+y01.get(k,0) for k in set(y00)|set(y01) if y00.get(k,0)+y01.get(k,0)!=0}} ),
            ( {**{k:x01.get(k,0)-x11.get(k,0) for k in set(x01)|set(x11) if x01.get(k,0)-x11.get(k,0)!=0}}, 
              {**{k:y10.get(k,0)+y11.get(k,0) for k in set(y10)|set(y11) if y10.get(k,0)+y11.get(k,0)!=0}} ),
        ]
        
        for k in range(7):
            xk, yk = subs[k]
            ridx = t*7 + k
            for idx, coeff in xk.items():
                U[idx][ridx] = coeff
            for idx, coeff in yk.items():
                V[idx][ridx] = coeff
            for (zr, zc, sc) in inner_contrib[k]:
                if (zr, zc) in ct:
                    for w_idx in ct[(zr, zc)]:
                        W[w_idx][ridx] += sc
    
    return {"U": U, "V": V, "W": W}
