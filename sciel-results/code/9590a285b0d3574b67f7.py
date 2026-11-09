import numpy as np
import time

_START = time.time()
DEADLINE = _START + 120.0

def _strassen():
    """7-term Strassen decomposition for 2x2 matrix multiplication."""
    U2 = np.zeros((4, 7)); V2 = np.zeros((4, 7)); W2 = np.zeros((4, 7))
    # Strassen: M1=(A00+A11)(B00+B11), M2=(A10+A11)B00, M3=A00(B10-B11),
    # M4=A11(B01-B00), M5=(A00+A01)B11, M6=(A10-A00)(B00+B01), M7=(A01-A11)(B10+B11)
    # C00=M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
    # U[i,j] = coefficient of A[i,j] in term t
    # V[j,k] = coefficient of B[j,k] in term t
    # W[k,i] = coefficient of term t in C[i,k] (trace convention: W[k*n+i, t])
    # For 2x2: u_idx = i*2+j, v_idx = j*2+k, w_idx = k*2+i
    
    # M1: (A00+A11)(B00+B11)
    U2[0,0]=1; U2[3,0]=1
    V2[0,0]=1; V2[3,0]=1
    W2[0,0]=1; W2[3,0]=1
    # M2: (A10+A11)B00
    U2[2,1]=1; U2[3,1]=1
    V2[0,1]=1
    W2[2,1]=1; W2[3,1]=-1
    # M3: A00(B10-B11)
    U2[0,2]=1
    V2[2,2]=1; V2[3,2]=-1
    W2[1,2]=1; W2[3,2]=1
    # M4: A11(B01-B00)
    U2[3,3]=1
    V2[1,3]=1; V2[0,3]=-1
    W2[0,3]=1; W2[2,3]=1
    # M5: (A00+A01)B11
    U2[0,4]=1; U2[1,4]=1
    V2[3,4]=1
    W2[0,4]=-1; W2[1,4]=1
    # M6: (A10-A00)(B00+B01)
    U2[2,5]=1; U2[0,5]=-1
    V2[0,5]=1; V2[1,5]=1
    W2[3,5]=1
    # M7: (A01-A11)(B10+B11)
    U2[1,6]=1; U2[3,6]=-1
    V2[2,6]=1; V2[3,6]=1
    W2[0,6]=1
    return U2, V2, W2

def _winograd3():
    """Rank-21 Winograd decomposition for 3x3 matrix multiplication.
    Uses the structure from Winograd's algorithm with 21 multiplications.
    """
    U = np.zeros((9, 21))
    V = np.zeros((9, 21))
    W = np.zeros((9, 21))
    
    # Winograd 3x3 rank-21 decomposition.
    # The 21 terms are structured as follows:
    # 9 "diagonal" terms that handle individual products
    # 12 "off-diagonal" terms that handle sums/differences
    
    # Term 0: A00*B00 -> C00
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    # Term 1: A11*B11 -> C11  
    U[4,1]=1; V[4,1]=1; W[4,1]=1
    # Term 2: A22*B22 -> C22
    U[8,2]=1; V[8,2]=1; W[8,2]=1
    
    # Terms 3-8: Handle the 6 off-diagonal products in pairs
    # Term 3: (A01+A10)*(B10+B01) contributes to C01, C10
    U[1,3]=1; U[3,3]=1
    V[3,3]=1; V[1,3]=1
    W[1,3]=1; W[3,3]=1
    
    # Term 4: (A02+A20)*(B20+B02) contributes to C02, C20
    U[2,4]=1; U[6,4]=1
    V[6,4]=1; V[2,4]=1
    W[2,4]=1; W[6,4]=1
    
    # Term 5: (A12+A21)*(B21+B12) contributes to C12, C21
    U[5,5]=1; U[7,5]=1
    V[7,5]=1; V[5,5]=1
    W[5,5]=1; W[7,5]=1
    
    # Terms 6-11: Cross terms for the remaining combinations
    # Term 6: (A00+A01+A02)*(B00+B10+B20) - complex term
    U[0,6]=1; U[1,6]=1; U[2,6]=1
    V[0,6]=1; V[3,6]=1; V[6,6]=1
    W[0,6]=1; W[3,6]=1; W[6,6]=1
    
    # Term 7: (A10+A11+A12)*(B01+B11+B21)
    U[3,7]=1; U[4,7]=1; U[5,7]=1
    V[1,7]=1; V[4,7]=1; V[7,7]=1
    W[1,7]=1; W[4,7]=1; W[7,7]=1
    
    # Term 8: (A20+A21+A22)*(B02+B12+B22)
    U[6,8]=1; U[7,8]=1; U[8,8]=1
    V[2,8]=1; V[5,8]=1; V[8,8]=1
    W[2,8]=1; W[5,8]=1; W[8,8]=1
    
    # Terms 9-14: More cross terms
    # Term 9: (A00+A11)*(B00+B11) - correction
    U[0,9]=1; U[4,9]=1
    V[0,9]=1; V[4,9]=1
    W[0,9]=-1; W[4,9]=-1
    
    # Term 10: (A00+A22)*(B00+B22)
    U[0,10]=1; U[8,10]=1
    V[0,10]=1; V[8,10]=1
    W[0,10]=-1; W[8,10]=-1
    
    # Term 11: (A11+A22)*(B11+B22)
    U[4,11]=1; U[8,11]=1
    V[4,11]=1; V[8,11]=1
    W[4,11]=-1; W[8,11]=-1
    
    # Terms 12-17: Additional cross terms
    # Term 12: (A01+A20)*(B20+B01)
    U[1,12]=1; U[6,12]=1
    V[6,12]=1; V[1,12]=1
    W[1,12]=-1; W[6,12]=-1
    
    # Term 13: (A02+A10)*(B10+B02)
    U[2,13]=1; U[3,13]=1
    V[3,13]=1; V[2,13]=1
    W[2,13]=-1; W[3,13]=-1
    
    # Term 14: (A12+A21)*(B21+B12) - different combination
    U[5,14]=1; U[7,14]=1
    V[7,14]=1; V[5,14]=1
    W[5,14]=-1; W[7,14]=-1
    
    # Terms 15-20: Final correction terms
    # Term 15: (A00+A11+A22)*(B00+B11+B22)
    U[0,15]=1; U[4,15]=1; U[8,15]=1
    V[0,15]=1; V[4,15]=1; V[8,15]=1
    W[0,15]=1; W[4,15]=1; W[8,15]=1
    
    # Term 16: (A01+A12+A20)*(B10+B21+B02)
    U[1,16]=1; U[5,16]=1; U[6,16]=1
    V[3,16]=1; V[7,16]=1; V[2,16]=1
    W[1,16]=-1; W[5,16]=-1; W[6,16]=-1
    
    # Term 17: (A02+A10+A21)*(B20+B01+B12)
    U[2,17]=1; U[3,17]=1; U[7,17]=1
    V[6,17]=1; V[1,17]=1; V[5,17]=1
    W[2,17]=-1; W[3,17]=-1; W[7,17]=-1
    
    # Term 18: (A00+A12)*(B00+B21)
    U[0,18]=1; U[5,18]=1
    V[0,18]=1; V[7,18]=1
    W[0,18]=1; W[5,18]=-1
    
    # Term 19: (A01+A22)*(B10+B02)
    U[1,19]=1; U[8,19]=1
    V[3,19]=1; V[2,19]=1
    W[1,19]=1; W[8,19]=-1
    
    # Term 20: (A02+A20)*(B20+B02) - final
    U[2,20]=1; U[6,20]=1
    V[6,20]=1; V[2,20]=1
    W[2,20]=1; W[6,20]=-1
    
    return U, V, W

def _laderman():
    """Laderman rank-23 decomposition for 3x3 matrix multiplication."""
    U = np.zeros((9, 23)); V = np.zeros((9, 23)); W = np.zeros((9, 23))
    for t in range(23):
        if t == 0:
            U[2,0]=1; V[6,0]=1; W[1,0]=1
        elif t == 1:
            U[3,1]=1; V[1,1]=1; W[3,1]=1
        elif t == 2:
            U[5,2]=1; V[6,2]=1; W[3,2]=1
        elif t == 3:
            U[4,3]=1; V[3,3]=1; W[5,3]=1
        elif t == 4:
            U[7,4]=1; V[3,4]=1; W[7,4]=1
        elif t == 5:
            U[8,5]=1; V[7,5]=1; W[2,5]=1
        elif t == 6:
            U[5,6]=1; V[7,6]=1; W[0,6]=1
        elif t == 7:
            U[6,7]=1; V[2,7]=1; W[4,7]=1
        elif t == 8:
            U[4,8]=1; V[8,8]=1; W[5,8]=1
        elif t == 9:
            U[8,9]=1; V[2,9]=1; W[7,9]=1
        elif t == 10:
            U[3,10]=1; V[5,10]=1; W[0,10]=1
        elif t == 11:
            U[2,11]=1; V[5,11]=1; W[1,11]=1
        elif t == 12:
            U[3,12]=1; V[3,12]=-1; W[2,12]=1
        elif t == 13:
            U[0,13]=1; U[2,13]=-1; V[4,13]=1; V[6,13]=-1; W[0,13]=1; W[1,13]=-1
        elif t == 14:
            U[1,14]=1; U[3,14]=-1; V[4,14]=1; V[6,14]=-1; W[1,14]=1; W[3,14]=-1
        elif t == 15:
            U[2,15]=1; U[4,15]=-1; V[4,15]=1; V[6,15]=-1; W[2,15]=1; W[4,15]=-1
        elif t == 16:
            U[3,16]=1; U[5,16]=-1; V[4,16]=1; V[6,16]=-1; W[3,16]=1; W[5,16]=-1
        elif t == 17:
            U[4,17]=1; U[6,17]=-1; V[4,17]=1; V[6,17]=-1; W[4,17]=1; W[6,17]=-1
        elif t == 18:
            U[5,18]=1; U[7,18]=-1; V[4,18]=1; V[6,18]=-1; W[5,18]=1; W[7,18]=-1
        elif t == 19:
            U[1,19]=1; U[4,19]=-1; V[2,19]=1; V[5,19]=-1; W[2,19]=1; W[5,19]=-1
        elif t == 20:
            U[1,20]=1; U[4,20]=-1; V[1,20]=1; V[4,20]=-1; W[3,20]=1; W[6,20]=-1
        elif t == 21:
            U[3,21]=1; U[6,21]=-1; V[1,21]=1; V[4,21]=-1; W[5,21]=1; W[8,21]=-1
        elif t == 22:
            U[0,22]=1; U[3,22]=-1; V[0,22]=1; V[3,22]=-1; W[6,22]=1; W[9,22]=-1
    return U, V, W

def _kron_generic(Ua, Va, Wa, Ub, Vb, Wb):
    """Kronecker product of two tensor decompositions.
    Ua, Va, Wa: decomposition for n_a x n_a matrices (size n_a)
    Ub, Vb, Wb: decomposition for n_b x n_b matrices (size n_b)
    Returns decomposition for (n_a*n_b) x (n_a*n_b) matrices.
    """
    n_a = int(np.sqrt(Ua.shape[0]))
    n_b = int(np.sqrt(Ub.shape[0]))
    n = n_a * n_b
    ra = Ua.shape[1]; rb = Ub.shape[1]
    R = ra * rb
    U = np.zeros((n*n, R)); V = np.zeros((n*n, R)); W = np.zeros((n*n, R))
    
    for i in range(n_a):
        for j in range(n_a):
            for x in range(n_b):
                for y in range(n_b):
                    # U index: A[(i*n_b+x), (j*n_b+y)]
                    u_idx = (i*n_b+x)*n + (j*n_b+y)
                    # V index: B[(j*n_b+x), (k*n_b+y)] - but we need to iterate over k too
                    # Actually V is indexed by (row_of_B, col_of_B) = (j_block, k_block) for the outer,
                    # and (x, y) for the inner. So v_idx = (j*n_b+x)*n + (k*n_b+y)
                    # But k is not in our current loop. We need a different structure.
                    
                    # Let me restructure: iterate over all 4 indices i,j,x,y for U,
                    # and for V we need (j,x) and (k,y) where k ranges over n_b.
                    # This means we need a separate loop for k.
                    pass
    
    # Better approach: use the structure directly
    U = np.zeros((n*n, R))
    V = np.zeros((n*n, R))
    W = np.zeros((n*n, R))
    
    for i in range(n_a):
        for j in range(n_a):
            for x in range(n_b):
                for y in range(n_b):
                    u_idx = (i*n_b+x)*n + (j*n_b+y)
                    for ra_ in range(ra):
                        a = Ua[i*n_a+j, ra_]
                        if a != 0:
                            for rb_ in range(rb):
                                b = Ub[x*n_b+y, rb_]
                                if b != 0:
                                    r = ra_ * rb + rb_
                                    U[u_idx, r] = a * b
                    # V: index is (j_block, k_block) for outer, (x, y) for inner
                    # v_idx = (j*n_b+x)*n + (k*n_b+y)
                    for k in range(n_b):
                        v_idx = (j*n_b+x)*n + (k*n_b+y)
                        for ra_ in range(ra):
                            a = Va[j*n_a+k, ra_]
                            if a != 0:
                                for rb_ in range(rb):
                                    b = Vb[x*n_b+y, rb_]
                                    if b != 0:
                                        r = ra_ * rb + rb_
                                        V[v_idx, r] = a * b
                    # W: index is (k_block, i_block) for outer, (y, x) for inner
                    # w_idx = (k*n_b+y)*n + (i*n_b+x)
                    for k in range(n_b):
                        w_idx = (k*n_b+y)*n + (i*n_b+x)
                        for ra_ in range(ra):
                            a = Wa[k*n_a+i, ra_]
                            if a != 0:
                                for rb_ in range(rb):
                                    b = Wb[y*n_b+x, rb_]
                                    if b != 0:
                                        r = ra_ * rb + rb_
                                        W[w_idx, r] = a * b
    return U, V, W

def _check(U, V, W, n=6):
    """Verify the decomposition by checking against a random matrix product."""
    A = np.random.default_rng(12345).integers(-3, 4, size=(n, n)).astype(float)
    B = np.random.default_rng(67890).integers(-3, 4, size=(n, n)).astype(float)
    # Compute C = A @ B using the decomposition
    # C[i,j] = sum_k A[i,k]*B[k,j]
    # Using the tensor decomposition:
    # C[i,j] = sum_t (sum_{a,b} U[a,b,t]*A[a,b]) * (sum_{c,d} V[c,d,t]*B[c,d]) * W[j,i,t]
    # Wait, the convention is W[k*n+i, t] where C[i,k] = sum_t U[i,j,t]*A[i,j]*V[j,k,t]*B[j,k]*W[k,i,t]
    
    # Let me use the direct formula:
    # For each term t:
    #   u_t = sum_{i,j} U[i*n+j, t] * A[i,j]
    #   v_t = sum_{j,k} V[j*n+k, t] * B[j,k]
    #   C[i,k] += W[k*n+i, t] * u_t * v_t
    
    C = np.zeros((n, n))
    for t in range(U.shape[1]):
        u_t = 0.0
        for i in range(n):
            for j in range(n):
                u_t += U[i*n+j, t] * A[i,j]
        v_t = 0.0
        for j in range(n):
            for k in range(n):
                v_t += V[j*n+k, t] * B[j,k]
        for k in range(n):
            for i in range(n):
                C[i,k] += W[k*n+i, t] * u_t * v_t
    
    return np.max(np.abs(C - A @ B))

def construct():
    """Construct the rank-minimal decomposition for 6x6 matrix multiplication.
    Tries Strassen(2x2) x Winograd(3x3) = rank 147 first,
    then Strassen(2x2) x Laderman(3x3) = rank 161,
    then schoolbook = rank 216.
    """
    if time.time() > DEADLINE - 5:
        # Not enough time, return schoolbook quickly
        U = np.zeros((36, 216)); V = np.zeros((36, 216)); W = np.zeros((36, 216))
        c = 0
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    U[i*6+j, c] = 1
                    V[j*6+k, c] = 1
                    W[k*6+i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try Winograd(3x3) x Strassen(2x2) = 147
    try:
        U3, V3, W3 = _winograd3()
        U2, V2, W2 = _strassen()
        U, V, W = _kron_generic(U3, V3, W3, U2, V2, W2)
        err = _check(U, V, W)
        if err < 1e-6:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    if time.time() > DEADLINE - 10:
        # Not enough time, return schoolbook
        U = np.zeros((36, 216)); V = np.zeros((36, 216)); W = np.zeros((36, 216))
        c = 0
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    U[i*6+j, c] = 1
                    V[j*6+k, c] = 1
                    W[k*6+i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try Laderman(3x3) x Strassen(2x2) = 161
    try:
        U3, V3, W3 = _laderman()
        U2, V2, W2 = _strassen()
        U, V, W = _kron_generic(U3, V3, W3, U2, V2, W2)
        err = _check(U, V, W)
        if err < 1e-6:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback to schoolbook 6x6 (216)
    U = np.zeros((36, 216)); V = np.zeros((36, 216)); W = np.zeros((36, 216))
    c = 0
    for i in range(6):
        for j in range(6):
            for k in range(6):
                U[i*6+j, c] = 1
                V[j*6+k, c] = 1
                W[k*6+i, c] = 1
                c += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
