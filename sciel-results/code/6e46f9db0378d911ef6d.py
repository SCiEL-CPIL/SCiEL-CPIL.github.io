"""Find rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    def schoolbook():
        """Standard rank-128 decomposition (one rank-1 tensor per (i,j,k) entry)."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, atol=1e-10):
        """Verify that the decomposition is exact: T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1"""
        rank = len(U[0]) if U else 0
        # Build tensor from decomposition
        T = np.zeros((n, m, p))
        for c in range(rank):
            uc = np.array(U[:, c]).reshape(n, m)
            vc = np.array(V[:, c]).reshape(m, p)
            wc = np.array(W[:, c]).reshape(p, n)
            # T[i,j,k] += uc[i,j] * vc[j,k] * wc[k,i]
            # Vectorized: for each (i,j), add uc[i,j] * vc[j,:] * wc[:,i]
            for i in range(n):
                for j in range(m):
                    if uc[i, j] != 0:
                        T[i, j, :] += uc[i, j] * vc[j, :] * wc[:, i]
        return np.allclose(T, np.ones((n, m, p)), atol=atol)
    
    def kronecker_decompose(U1, V1, W1, U2, V2, W2):
        """
        Given two decompositions:
          MM1: A1(n1 x m1) * B1(m1 x p1) -> C1(p1 x n1) with rank r1
          MM2: A2(n2 x m2) * B2(m2 x p2) -> C2(p2 x n2) with rank r2
        Construct decomposition for:
          A = A1 ⊗ A2  (n1*n2 x m1*m2)
          B = B1 ⊗ B2  (m1*m2 x p1*p2)
          C = C1 ⊗ C2  (p1*p2 x n1*n2)
        with rank r1*r2.
        
        The tensor entry: T[i1,i2, j1,j2, k1,k2] = sum_{c1,c2} U1[i1,j1,c1]*V1[j1,k1,c1]*W1[k1,i1,c1]
                                                           * U2[i2,j2,c2]*V2[j2,k2,c2]*W2[k2,i2,c2]
        = (sum_{c1} ...) * (sum_{c2} ...)
        
        For the combined: T_comb[i1*n2+i2, j1*m2+j2, k1*p2+k2]
        We need: sum_c U_comb[(i1*n2+i2)*m2+j1*m2+j2, c] * V_comb[(j1*m2+j2)*p2+k1*p2+k2, c] * W_comb[(k1*p2+k2)*n2+i1*n2+i2, c]
        
        Let c = c1*r2 + c2
        U_comb[i1*n2+i2, j1*m2+j2, c1*r2+c2] = U1[i1,j1,c1] * U2[i2,j2,c2]
        V_comb[j1*m2+j2, k1*p2+k2, c1*r2+c2] = V1[j1,k1,c1] * V2[j2,k2,c2]
        W_comb[k1*p2+k2, i1*n2+i2, c1*r2+c2] = W1[k1,i1,c1] * W2[k2,i2,c2]
        
        Then the product sums over all c1,c2:
        sum_{c1,c2} U1[i1,j1,c1]*U2[i2,j2,c2] * V1[j1,k1,c1]*V2[j2,k2,c2] * W1[k1,i1,c1]*W2[k2,i2,c2]
        = (sum_{c1} U1[i1,j1,c1]*V1[j1,k1,c1]*W1[k1,i1,c1]) * (sum_{c2} U2[i2,j2,c2]*V2[j2,k2,c2]*W2[k2,i2,c2])
        = T1[i1,j1,k1] * T2[i2,j2,k2]
        
        This equals 1 if both T1[i1,j1,k1]=1 and T2[i2,j2,k2]=1, i.e., for all valid entries.
        """
        r1 = len(U1[0]) if U1 else 0
        r2 = len(U2[0]) if U2 else 0
        r = r1 * r2
        
        # Get dimensions from the matrices
        n1, m1 = U1[0].__len__(), 0
        # Actually let's infer: U1 has shape (n1*m1, r1)
        n1m1 = len(U1)
        m1p1 = len(V1)
        p1n1 = len(W1)
        n2m2 = len(U2)
        m2p2 = len(V2)
        p2n2 = len(W2)
        
        U_comb = [[0] * r for _ in range(n1m1 * n2m2)]
        V_comb = [[0] * r for _ in range(m1p1 * m2p2)]
        W_comb = [[0] * r for _ in range(p1n1 * p2n2)]
        
        for c1 in range(r1):
            for c2 in range(r2):
                c = c1 * r2 + c2
                for idx1 in range(n1m1):
                    for idx2 in range(n2m2):
                        if U1[idx1][c1] != 0 and U2[idx2][c2] != 0:
                            U_comb[idx1 * n2m2 + idx2][c] = U1[idx1][c1] * U2[idx2][c2]
                for idx1 in range(m1p1):
                    for idx2 in range(m2p2):
                        if V1[idx1][c1] != 0 and V2[idx2][c2] != 0:
                            V_comb[idx1 * m2p2 + idx2][c] = V1[idx1][c1] * V2[idx2][c2]
                for idx1 in range(p1n1):
                    for idx2 in range(p2n2):
                        if W1[idx1][c1] != 0 and W2[idx2][c2] != 0:
                            W_comb[idx1 * p2n2 + idx2][c] = W1[idx1][c1] * W2[idx2][c2]
        
        return {"U": U_comb, "V": V_comb, "W": W_comb}
    
    # Step 1: Get a base decomposition for 2x2 * 2x2 (Strassen rank 7)
    # Strassen's 7 rank-1 tensors for 2x2 matrix multiplication
    def strassen_2x2():
        """
        Strassen decomposition for 2x2 * 2x2.
        A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[i, j], [k, l]]
        i = a*e + b*g - (c+a-e-d)*g  -- wait, let me use the standard form.
        
        Standard Strassen:
        M1 = (a+d)(e+h)
        M2 = (c+d)e
        M3 = a(f-h)
        M4 = d(g-e)
        M5 = (a+b)h
        M6 = (c-d)(f+h)
        M7 = (b-c)(e+h)
        
        i = M1 + M4 - M5 + M7
        j = M3 + M5
        k = M2 + M4
        l = M1 - M2 + M3 + M6
        """
        # For the tensor decomposition, we need U, V, W of shape (4, 7), (4, 7), (4, 7)
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] where n=m=p=2
        # So U is 4x7, V is 4x7, W is 4x7
        # Indices: U[0]=a, U[1]=b, U[2]=c, U[3]=d
        #           V[0]=e, V[1]=f, V[2]=g, V[3]=h
        #           W[0]=i, W[1]=j, W[2]=k, W[3]=l
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (a+d)(e+h): U: a=1,d=1; V: e=1,h=1; W: i=1,l=1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (c+d)e: U: c=1,d=1; V: e=1; W: k=1,l=-1
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # M3 = a(f-h): U: a=1; V: f=1,h=-1; W: j=1,l=1
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4 = d(g-e): U: d=1; V: g=1,e=-1; W: i=1,k=1
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5 = (a+b)h: U: a=1,b=1; V: h=1; W: i=-1,j=1
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6 = (c-d)(f+h): U: c=1,d=-1; V: f=1,h=1; W: l=1
        U[2][5] = 1; U[3][5] = -1
        V[1][5] = 1; V[3][5] = 1
        W[3][5] = 1
        
        # M7 = (b-c)(e+h): U: b=1,c=-1; V: e=1,h=1; W: i=1
        U[1][6] = 1; U[2][6] = -1
        V[0][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return {"U": U, "V": V, "W": W}
    
    # Verify Strassen
    strassen = strassen_2x2()
    # For verification, we need to adapt since verify() expects n=m=p dimensions
    # Let's do a quick manual check
    # Actually, let me just use it and verify the Kronecker result directly
    
    # Step 2: Use Kronecker to get 4x4 * 4x4 from two 2x2 * 2x2
    # A(4x4) = A1(2x2) ⊗ A2(2x2), B(4x4) = B1(2x2) ⊗ B2(2x2)
    # C(4x4) = C1(2x2) ⊗ C2(2x2)
    # Rank = 7 * 7 = 49
    # But we need 4x4 * 4x8, not 4x4 * 4x4
    
    # Step 3: For 4x4 * 4x8, we can think of it as:
    # A(4x4) * B(4x8) where B = [B1 | B2] with B1, B2 both 4x4
    # C = [A*B1 | A*B2] which is 4x8
    # 
    # Alternatively, we can use the Kronecker structure differently:
    # 4x4 * 4x8 = (2x2 ⊗ 2x2) * (2x2 ⊗ 2x4)
    # But 2x2 * 2x4 isn't standard.
    #
    # Better approach: 
    # 4x4 * 4x8 can be decomposed as the Kronecker product of:
    #   (4x4 * 4x4) [rank 49 via Strassen Kronecker] and (1x1 * 2x2) [rank 2]
    # No, that doesn't work dimensionally.
    #
    # Let's think more carefully:
    # n=4, m=4, p=8
    # We want U(16, r), V(32, r), W(32, r)
    #
    # Approach: Split p=8 into two blocks of 4: p = 4+4
    # C = [C1 | C2] where C1 = A*B1, C2 = A*B2, each 4x4
    # The tensor T[i,j,k] for k=0..7:
    #   k=0..3: C1[i,k] = sum_j A[i,j]*B1[j,k]
    #   k=4..7: C2[i,k-4] = sum_j A[i,j]*B2[j,k-4]
    #
    # This is like two separate 4x4*4x4 multiplications sharing the same A.
    #
    # If we have a rank-49 decomposition for 4x4*4x4:
    #   C1[i,k] = sum_c U1[i,k_row, c] * ... 
    # Actually the standard decomposition has:
    #   T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # For 4x4*4x4: n=m=p=4, rank=49
    # U: 16x49, V: 16x49, W: 16x49
    #
    # For 4x4*4x8, we can use two copies of the 4x4*4x4 decomposition:
    # For k in 0..3 (first block): use decomposition 1
    # For k in 4..7 (second block): use decomposition 2
    #
    # Total rank = 49 + 49 = 98
    #
    # U[i*m+j, c] for c=0..47: from first decomposition (same for both A's)
    # U[i*m+j, c] for c=49..97: from second decomposition (same A)
    #
    # Wait, A is shared! So:
    # C1[i,k] = sum_j A[i,j] B1[j,k] = sum_c sum_j U1[i*4+j, c] V1[j*4+k, c] W1[k*4+i, c]
    # C2[i,k] = sum_j A[i,j] B2[j,k] = sum_c sum_j U2[i*4+j, c] V2[j*4+k, c] W2[k*4+i, c]
    #
    # If we use the same A decomposition for both:
    # U[i*4+j, c] for c=0..47: U1[i*4+j, c]
    # U[i*4+j, c] for c=49..97: U1[i*4+j, c-49]  (same U for both)
    #
    # V[j*8+k, c] for c=0..47: if k<4, V1[j*4+k, c]; if k>=4, 0
    # V[j*8+k, c] for c=49..97: if k<4, 0; if k>=4, V1[j*4+(k-4), c-49]
    #
    # W[k*4+i, c] for c=0..47: if k<4, W1[k*4+i, c]; if k>=4, 0
    # W[k*4+i, c] for c=49..97: if k<4, 0; if k>=4, W1[(k-4)*4+i, c-49]
    #
    # This gives rank 98.
    
    # But can we do better? What if we interleave?
    # Actually, the above is valid. Let's also consider:
    # Using the Kronecker approach: 4x4*4x8 = (2x2*2x4) ⊗ (2x2*2x4)? No...
    #
    # Let me try: 4x4*4x8 via Kronecker of (2x2*2x4) and (2x2*2x2)?
    # n1*n2=4, m1*m2=4, p1*p2=8
    # Options: (n1,m1,p1)=(2,2,4), (n2,m2,p2)=(2,2,2)
    #   2x2*2x4: rank? Schoolbook is 16. Can we do better?
    #   2x2*2x2: rank 7 (Strassen)
    #   Total rank: 16*7 = 112. Worse than 98.
    #
    # Options: (n1,m1,p1)=(4,1,4), (n2,m2,p2)=(1,4,2)
    #   4x1*4x4: rank = 16 (schoolbook, since it's just 4 dot products each of length 4)
    #   Actually 4x1 * 4x4 = sum over one intermediate index. Rank is at most 4*4=16, and schoolbook achieves it.
    #   1x4 * 4x2: rank = 8 (schoolbook)
    #   Total: 16*8 = 128. Same as schoolbook.
    #
    # So the best structured approach so far is rank 98 (two copies of rank-49 for 4x4*4x4).
    #
    # But wait - can we share more? Since A is the same in both multiplications,
    # maybe we can find a joint decomposition with rank < 98.
    #
    # Actually, let's reconsider. The problem is:
    # T[i,j,k] for i in 0..3, j in 0..3, k in 0..7
    # This is a (4,4,8) tensor with all entries = 1.
    #
    # The rank-98 construction:
    # For each of the two 4x4*4x4 sub-problems, we use the rank-49 Strassen-Kronecker decomposition.
    #
    # Let's construct this properly.
    
    # First, get the 4x4*4x4 decomposition via Kronecker of two 2x2*2x2 Strassen decompositions
    def k4x4x4():
        """Kronecker product of two Strassen 2x2 decompositions to get 4x4*4x4, rank 49."""
        s = strassen_2x2()
        # U1, V1, W1 for first 2x2*2x2
        # U2, V2, W2 for second 2x2*2x2
        # Both are the same Strassen decomposition
        result = kronecker_decompose(s["U"], s["V"], s["W"], s["U"], s["V"], s["W"])
        return result
    
    k4 = k4x4x4()
    
    # Now build 4x4*4x8 from two copies of the 4x4*4x4 decomposition
    def build_4x4x8_from_2x4x4x4(k4):
        """
        Build 4x4*4x8 decomposition from two 4x4*4x4 decompositions.
        n=4, m=4, p=8
        U: 16 x 98, V: 32 x 98, W: 32 x 98
        
        First half (c=0..47): handles k=0..3 (C1 = A*B1)
        Second half (c=49..97): handles k=4..7 (C2 = A*B2)
        """
        r = 49
        rank = 2 * r  # 98
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(32)]
        W = [[0] * rank for _ in range(32)]
        
        # Copy first decomposition
        for i in range(16):
            for c in range(r):
                U[i][c] = k4["U"][i][c]
        
        for j in range(4):
            for k in range(4):
                for c in range(r):
                    V[j*8 + k][c] = k4["V"][j*4 + k][c]
        
        for k in range(4):
            for i in range(4):
                for c in range(r):
                    W[k*4 + i][c] = k4["W"][k*4 + i][c]
        
        # Copy second decomposition (offset by r)
        for i in range(16):
            for c in range(r):
                U[i][c + r] = k4["U"][i][c]
        
        for j in range(4):
            for k in range(4):
                for c in range(r):
                    V[j*8 + (k + 4)][c + r] = k4["V"][j*4 + k][c]
        
        for k in range(4):
            for i in range(4):
                for c in range(r):
                    W[(k + 4)*4 + i][c + r] = k4["W"][k*4 + i][c]
        
        return {"U": U, "V": V, "W": W}
    
    candidate = build_4x4x8_from_2x4x4x4(k4)
    
    # Verify the candidate
    if time.time() < deadline - 10:
        if verify(candidate["U"], candidate["V"], candidate["W"]):
            return candidate
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
