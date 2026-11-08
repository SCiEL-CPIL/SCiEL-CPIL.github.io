import time
import numpy as np
import itertools

def _strassen_2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    U shape: (4, 7)  # indices i*2+j
    V shape: (4, 7)  # indices j*2+k
    W shape: (4, 7)  # indices k*2+i
    """
    # Verified Strassen 7 products
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # p1 = (a - d)(e + f)
    # p2 = (c + d)e
    # p3 = a(h - f)
    # p4 = d(g - e)
    # p5 = (a + b)h
    # p6 = (c - a)(e + g)
    # p7 = (b - d)(f + h)
    
    # C00 = p1 + p4 - p5 + p7
    # C01 = p3 + p5
    # C10 = p2 + p4
    # C11 = p1 - p2 + p3 + p6
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # r=0: p1 = (a - d)(e + f)
    # a=0, d=3. e=0, f=1.
    U[0][0] = 1; U[3][0] = -1
    V[0][0] = 1; V[1][0] = 1
    # C00 += 1, C11 -= 1, C01 += 1, C10 += 0? 
    # p1 = ae + af - de - df
    # C00 (ae) +1. C01 (af) +1. C10 (0). C11 (-df) -1.
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 0; W[3][0] = -1
    
    # r=1: p2 = (c + d)e
    # c=2, d=3. e=0.
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # p2 = ce + de
    # C10 (ce) +1. C11 (0).
    W[2][1] = 1; W[3][1] = 0; W[0][1] = 0; W[1][1] = 0
    
    # r=2: p3 = a(h - f)
    # a=0. h=3, f=1.
    U[0][2] = 1
    V[3][2] = 1; V[1][2] = -1
    # p3 = ah - af
    # C01 (-af) -1. C11 (ah) +1.
    W[1][2] = -1; W[3][2] = 1; W[0][2] = 0; W[2][2] = 0
    
    # r=3: p4 = d(g - e)
    # d=3. g=2, e=0.
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    # p4 = dg - de
    # C10 (dg) +1. C11 (-de) -1.
    W[2][3] = 1; W[3][3] = -1; W[0][3] = 0; W[1][3] = 0
    
    # r=4: p5 = (a + b)h
    # a=0, b=1. h=3.
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # p5 = ah + bh
    # C01 (bh) +1. C11 (ah) +1.
    W[1][4] = 1; W[3][4] = 1; W[0][4] = 0; W[2][4] = 0
    
    # r=5: p6 = (c - a)(e + g)
    # c=2, a=0. e=0, g=2.
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    # p6 = ce + cg - ae - ag
    # C10 (ce) +1. C00 (-ae) -1. C00 (-ag) -1? No, ag is A00 B10. C00 = A00 B00 + A01 B10. So ag is in C00.
    # Wait, C00 = ae + bg.
    # p6 terms: ce (C10), cg (C10? No, cg is A10 B10. C10 = A10 B00 + A11 B10. So cg is NOT in C10. cg is in C11? No.
    # Let's re-verify indices.
    # A = [[a, b], [c, d]]. B = [[e, f], [g, h]].
    # C00 = ae + bg.
    # C01 = af + bh.
    # C10 = ce + dg.
    # C11 = cf + dh.
    
    # p6 = (c-a)(e+g) = ce + cg - ae - ag.
    # ce: C10.
    # cg: A10 B10. Not in C10 (which is A10 B00 + A11 B10). Not in C11 (A10 B01 + A11 B11).
    # Is cg in C00? No. C01? No.
    # This implies p6 is not a standard Strassen product for this indexing?
    # Standard Strassen p6 is (c-a)(e+g).
    # Let's check Wikipedia.
    # p6 = (c - a)(e + g).
    # C10 = p2 + p4. p2 = (c+d)e = ce + de. p4 = d(g-e) = dg - de.
    # p2+p4 = ce + dg. Correct.
    # C11 = p1 - p2 + p3 + p6.
    # p1 = ae + af - de - df.
    # -p2 = -ce - de.
    # p3 = ah - af.
    # p6 = ce + cg - ae - ag.
    # Sum:
    # ae: 1 - 1 = 0.
    # af: 1 - 1 = 0.
    # -de: -1 - 1 = -2.
    # -df: -1.
    # -ce: -1 + 1 = 0.
    # cg: 1.
    # ah: 1.
    # -ag: -1.
    # Result: -2de - df + cg + ah - ag.
    # C11 should be cf + dh.
    # This does not match.
    
    # It seems my memory of Strassen formulas or their application to this specific index mapping is flawed.
    # However, the prompt says the current program (score 1.0) uses a specific _strassen_2x2.
    # I will use the _strassen_2x2 from the current program as it is verified to work.
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # From current program:
    # M1=(A11-A10)(B00+B01)
    U[3][0]=1; U[2][0]=-1; V[0][0]=1; V[1][0]=1; W[0][0]=1; W[2][0]=-1; W[3][0]=1
    # M2=(A10+A11)(B10+B11)
    U[2][1]=1; U[3][1]=1; V[2][1]=1; V[3][1]=1; W[2][1]=1; W[3][1]=1
    # M3=(A00+A10)(B00)
    U[0][2]=1; U[2][2]=1; V[0][2]=1; W[2][2]=1; W[3][2]=1
    # M4=(A11+A01)(B11)
    U[3][3]=1; U[1][3]=1; V[3][3]=1; W[0][3]=1; W[2][3]=-1
    # M5=(A00-A10)(B01+B11)
    U[0][4]=1; U[2][4]=-1; V[1][4]=1; V[3][4]=1; W[0][4]=-1; W[2][4]=1
    # M6=(A01+A11)(B10-B00)
    U[1][5]=1; U[3][5]=1; V[2][5]=1; V[0][5]=-1; W[2][5]=-1; W[3][5]=-1
    # M7=(A00+A01)(B00-B10)
    U[0][6]=1; U[1][6]=1; V[0][6]=1; V[2][6]=-1; W[0][6]=1; W[2][6]=-1
    
    return U, V, W

def _schoolbook(n, m, p):
    """
    Returns U, V, W for n x m by m x p multiplication with rank n*m*p.
    """
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx, col] = 1
                V[b_idx, col] = 1
                W[c_idx, col] = 1
                col += 1
                
    return U, V, W

def _kronecker_product(U1, V1, W1, U2, V2, W2, dims1, dims2):
    """
    Computes the Kronecker product of two tensor decompositions.
    
    Tensor T is defined by indices (a, b, c) -> sum_r U[a,r] V[b,r] W[c,r].
    If T1 has dims (n1, m1, p1) and T2 has dims (n2, m2, p2),
    the Kronecker product T = T1 x T2 has dims (n1*n2, m1*m2, p1*p2).
    
    The index mapping is:
    a = a1 * n2 + a2
    b = b1 * m2 + b2
    c = c1 * p2 + c2
    
    U[a, r1*R2 + r2] = U1[a1, r1] * U2[a2, r2]
    V[b, r1*R2 + r2] = V1[b1, r1] * V2[b2, r2]
    W[c, r1*R2 + r2] = W1[c1, r1] * W2[c2, r2]
    
    Note: This assumes the standard Kronecker product structure where the
    "block" indices are the outer indices and "element" indices are inner.
    Specifically:
    A = A1 x A2. A[i1*n2+i2, j1*m2+j2] = A1[i1,j1] * A2[i2,j2].
    B = B1 x B2. B[j1*m2+j2, k1*p2+k2] = B1[j1,k1] * B2[j2,k2].
    C = C1 x C2. C[k1*p2+k2, i1*n2+i2] = C1[k1,i1] * C2[k2,i2].
    
    This corresponds to multiplying block matrices where blocks are themselves matrices.
    """
    n1, m1, p1 = dims1
    n2, m2, p2 = dims2
    
    R1 = U1.shape[1]
    R2 = U2.shape[1]
    R = R1 * R2
    
    U = np.zeros((n1 * n2, R))
    V = np.zeros((m1 * m2, R))
    W = np.zeros((p1 * p2, R))
    
    for r1 in range(R1):
        for r2 in range(R2):
            r = r1 * R2 + r2
            
            # U
            for a1 in range(n1):
                for a2 in range(n2):
                    a = a1 * n2 + a2
                    U[a, r] = U1[a1, r1] * U2[a2, r2]
                    
            # V
            for b1 in range(m1):
                for b2 in range(m2):
                    b = b1 * m2 + b2
                    V[b, r] = V1[b1, r1] * V2[b2, r2]
                    
            # W
            for c1 in range(p1):
                for c2 in range(p2):
                    c = c1 * p2 + c2
                    W[c, r] = W1[c1, r1] * W2[c2, r2]
                    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verifies if U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check diagonal (i, j, k) where a=i*m+j, b=j*p+k, c=k*n+i
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                actual = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                if abs(actual - 1.0) > 1e-9:
                    return False
                    
    # Check off-diagonal with random sampling
    np.random.seed(42)
    num_checks = 2000
    for _ in range(num_checks):
        a = np.random.randint(n * m)
        b = np.random.randint(m * p)
        c = np.random.randint(p * n)
        actual = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
        
        i = a // m
        j = a % m
        j2 = b // p
        k = b % p
        k2 = c // n
        i2 = c % n
        expected = 1.0 if (j == j2 and i == i2 and k == k2) else 0.0
        
        if abs(actual - expected) > 1e-9:
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 4, 4, 8
    
    best_rank = n * m * p
    best_U, best_V, best_W = _schoolbook(n, m, p)
    
    # Get base decompositions
    U2, V2, W2 = _strassen_2x2()
    U2_2 = np.array(U2)
    V2_2 = np.array(V2)
    W2_2 = np.array(W2)
    
    # 2x2x2 decomposition (dims 2,2,2)
    # 2x2x4 decomposition:
    # We can construct 2x2x4 as Kronecker product of 2x2x2 and 1x2x2? No.
    # 2x2x4 is multiplying 2x2 by 2x4.
    # This can be seen as 2x2 matrix of 1x2 blocks? No.
    # It is a 2x2 matrix A times a 2x4 matrix B.
    # B can be viewed as two 2x2 matrices side by side.
    # So it is two 2x2x2 multiplications.
    # Rank = 7 + 7 = 14.
    # Dims: (2, 2, 4).
    # We can build this using the Kronecker product logic if we define a 1x2x2 tensor?
    # Or simply concatenate.
    
    # Let's build 2x2x4 manually using the Kronecker product of 2x2x2 and 1x2x2 (identity-like)?
    # Actually, 2x2x4 is just the tensor for A (2x2) * B (2x4).
    # B has indices j (0,1) and k (0..3).
    # We can view B as B1 (2x2) and B2 (2x2).
    # C = A * B = [A*B1, A*B2].
    # So it's two independent 2x2x2 multiplications.
    # U_2x2x4: (4, 14). V_2x2x4: (8, 14). W_2x2x4: (8, 14).
    # For r in 0..6 (first block):
    #   U[i, r] = U2[i, r]
    #   V[j, r] = V2[j, r] if k<2 else 0.
    #   W[k, r] = W2[k, r] if k<2 else 0.
    # For r in 7..13 (second block):
    #   U[i, r] = U2[i, r-7]
    #   V[j, r] = 0 if k<2 else V2[j, r-7].
    #   W[k, r] = 0 if k<2 else W2[k, r-7].
    
    U_2x2x4 = np.zeros((4, 14))
    V_2x2x4 = np.zeros((8, 14))
    W_2x2x4 = np.zeros((8, 14))
    
    for r in range(7):
        for i in range(4):
            U_2x2x4[i, r] = U2_2[i, r]
        # V indices: j*4+k.
        # Block 1 (k=0,1): j=0,1. Indices 0,1,4,5.
        V_2x2x4[0, r] = V2_2[0, r]
        V_2x2x4[1, r] = V2_2[1, r]
        V_2x2x4[4, r] = V2_2[2, r]
        V_2x2x4[5, r] = V2_2[3, r]
        # W indices: k*2+i.
        # Block 1 (k=0,1): indices 0,1,2,3.
        W_2x2x4[0, r] = W2_2[0, r]
        W_2x2x4[1, r] = W2_2[1, r]
        W_2x2x4[2, r] = W2_2[2, r]
        W_2x2x4[3, r] = W2_2[3, r]
        
    for r in range(7):
        rr = r + 7
        for i in range(4):
            U_2x2x4[i, rr] = U2_2[i, r]
        # Block 2 (k=2,3): j=0,1. Indices 2,3,6,7.
        V_2x2x4[2, rr] = V2_2[0, r]
        V_2x2x4[3, rr] = V2_2[1, r]
        V_2x2x4[6, rr] = V2_2[2, r]
        V_2x2x4[7, rr] = V2_2[3, r]
        # Block 2 (k=2,3): indices 4,5,6,7.
        W_2x2x4[4, rr] = W2_2[0, r]
        W_2x2x4[5, rr] = W2_2[1, r]
        W_2x2x4[6, rr] = W2_2[2, r]
        W_2x2x4[7, rr] = W2_2[3, r]
        
    # Now we have:
    # T1: 2x2x2 (dims 2,2,2), Rank 7
    # T2: 2x2x4 (dims 2,2,4), Rank 14
    
    # We want 4x4x8.
    # Option A: (2x2x2) x (2x2x4) -> Dims (4, 4, 8). Rank 7*14 = 98.
    # Option B: (2x2x4) x (2x2x2) -> Dims (4, 4, 8). Rank 14*7 = 98.
    
    # Let's try Option A: T1 (2,2,2) x T2 (2,2,4).
    # Dims: n=2*2=4, m=2*2=4, p=2*4=8.
    U_A, V_A, W_A = _kronecker_product(U2_2, V2_2, W2_2, U_2x2x4, V_2x2x4, W_2x2x4, (2,2,2), (2,2,4))
    
    if time.time() < deadline:
        if _verify_decomposition(U_A, V_A, W_A, 4, 4, 8):
            if U_A.shape[1] < best_rank:
                best_rank = U_A.shape[1]
                best_U, best_V, best_W = U_A, V_A, W_A
    
    # Option B: T2 (2,2,4) x T1 (2,2,2).
    # Dims: n=2*2=4, m=2*2=4, p=4*2=8.
    U_B, V_B, W_B = _kronecker_product(U_2x2x4, V_2x2x4, W_2x2x4, U2_2, V2_2, W2_2, (2,2,4), (2,2,2))
    
    if time.time() < deadline:
        if _verify_decomposition(U_B, V_B, W_B, 4, 4, 8):
            if U_B.shape[1] < best_rank:
                best_rank = U_B.shape[1]
                best_U, best_V, best_W = U_B, V_B, W_B

    # Also check 4x4x4 x 1x1x2? No, 4x4x4 is not directly available.
    # But we can make 4x4x4 from (2x2x2) x (2x2x2). Rank 49.
    # Then 4x4x8 = 4x4x4 x 1x1x2?
    # 1x1x2 tensor: A(1x1) * B(1x2) = C(1x2).
    # Rank 2.
    # Total Rank 49 * 2 = 98. Same.
    
    # What about 2x4x2 x 2x1x4?
    # 2x4x2: A(2x4) * B(4x2) = C(2x2).
    # This is equivalent to 2x2x2 if we permute indices?
    # No, dimensions are different.
    
    # Let's just return the best found.
    
    # Convert to lists for return
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
