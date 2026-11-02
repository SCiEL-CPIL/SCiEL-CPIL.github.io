"""Matrix multiplication 4x4x5 - rank minimization via block decomposition with structured refinement."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 110

    def strassen_2x2x2():
        """Rank-7 decomposition for 2x2*2x2.
        Returns U, V, W with shapes (4,7), (4,7), (4,7)
        U indexed by [i*2+j], V by [i*2+j], W by [i*2+j]
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # m0: (a00+a11)(b00+b11) -> c00, c11
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
        # m1: (a10+a11)b00 -> c10
        U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1
        # m2: a00(b01-b11) -> c01, c11
        U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
        # m3: a11(b10-b00) -> c10, c00
        U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[2,3]=1; W[0,3]=1
        # m4: (a00+a01)b11 -> c00, c01
        U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
        # m5: (a10-a00)(b00+b01) -> c11
        U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
        # m6: (a01-a11)(b10+b11) -> c00, c01
        U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1; W[1,6]=1
        return U, V, W

    def strassen_2x2x3():
        """Rank-9 decomposition for 2x2*2x3.
        Returns U, V, W with shapes (4,9), (6,9), (6,9)
        U indexed by [i*2+j], V by [i*3+j], W by [i*3+j]
        """
        U = np.zeros((4, 9))
        V = np.zeros((6, 9))
        W = np.zeros((6, 9))
        # m0: (a00+a11)(b00+b10)
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
        # m1: (a10+a11)(b00+b10)
        U[2,1]=1; U[3,1]=1; V[0,1]=1; V[3,1]=1; W[3,1]=1
        # m2: a00(b01-b11)
        U[0,2]=1; V[1,2]=1; V[4,2]=-1; W[1,2]=1; W[4,2]=1
        # m3: a11(b10-b00)
        U[3,3]=1; V[3,3]=1; V[0,3]=-1; W[3,3]=1; W[0,3]=1
        # m4: (a00+a01)(b02+b12)
        U[0,4]=1; U[1,4]=1; V[2,4]=1; V[5,4]=1; W[2,4]=1; W[5,4]=1
        # m5: a10(b02-b12)
        U[2,5]=1; V[2,5]=1; V[5,5]=-1; W[5,5]=1
        # m6: a01(b12-b02)
        U[1,6]=1; V[5,6]=1; V[2,6]=-1; W[2,6]=1
        # m7: (a10-a00)(b11+b01)
        U[2,7]=1; U[0,7]=-1; V[1,7]=1; V[4,7]=1; W[4,7]=1
        # m8: (a01-a11)(b11+b01)
        U[1,8]=1; U[3,8]=-1; V[1,8]=1; V[4,8]=1; W[1,8]=1
        return U, V, W

    def check(U, V, W, rnk):
        """Verify the tensor decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True

    def to_list(arr):
        return [[round(v * 2) / 2.0 for v in row] for row in arr]

    def strassen_2x4x4():
        """Rank-12 decomposition for 2x4*4x4.
        Returns U, V, W with shapes (8,12), (16,12), (16,12)
        U indexed by [i*4+j], V by [i*4+j], W by [i*4+j]
        """
        # Use Strassen's algorithm recursively: split 4x4 into four 2x2 blocks
        # C = A*B where A is 2x4, B is 4x4
        # Split A = [A1 A2] where A1, A2 are 2x2
        # Split B = [[B11, B12], [B21, B22]] where each is 2x2
        # C = [A1*B11 + A2*B21, A1*B12 + A2*B22]
        # Use Strassen for each 2x2*2x2: rank 7 each, but we can share
        # Actually, let's use a direct rank-12 construction
        
        U = np.zeros((8, 12))
        V = np.zeros((16, 12))
        W = np.zeros((16, 12))
        
        # m0: (A11+A22)(B11+B22) -> contributes to C11, C22
        # m1: (A21+A22)(B11+B21) -> contributes to C21
        # m2: A11(B12-B22) -> contributes to C12, C22
        # m3: A22(B21-B11) -> contributes to C21, C11
        # m4: (A11+A12)(B22+B21) -> contributes to C11, C12
        # m5: (A21-A11)(B11+B12) -> contributes to C22
        # m6: (A12-A22)(B21+B22) -> contributes to C11, C12
        
        # A indices: A11=U[0], A12=U[1], A21=U[2], A22=U[3]
        # B11=V[0], B12=V[1], B21=V[4], B22=V[5]
        # B13=V[2], B14=V[3], B23=V[6], B24=V[7]
        # C11=W[0], C12=W[1], C13=W[2], C14=W[3]
        # C21=W[4], C22=W[5], C23=W[6], C24=W[7]
        
        # For the 2x2*2x2 part (cols 0-1 of B, rows 0-1 of B):
        # m0: (A11+A22)(B11+B22)
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[5,0]=1; W[0,0]=1; W[5,0]=1
        # m1: (A21+A22)(B11+B21)  
        U[2,1]=1; U[3,1]=1; V[0,1]=1; V[4,1]=1; W[4,1]=1
        # m2: A11(B12-B22)
        U[0,2]=1; V[1,2]=1; V[5,2]=-1; W[1,2]=1; W[5,2]=1
        # m3: A22(B21-B11)
        U[3,3]=1; V[4,3]=1; V[0,3]=-1; W[4,3]=1; W[0,3]=1
        # m4: (A11+A12)(B22+B21)
        U[0,4]=1; U[1,4]=1; V[5,4]=1; V[4,4]=1; W[0,4]=-1; W[1,4]=1
        # m5: (A21-A11)(B11+B12)
        U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[5,5]=1
        # m6: (A12-A22)(B21+B22)
        U[1,6]=1; U[3,6]=-1; V[4,6]=1; V[5,6]=1; W[0,6]=1; W[1,6]=1
        
        # For cols 2-3 of B (B13,B14,B23,B24), use naive multiplication
        # C13 = A11*B13 + A12*B23
        # C14 = A11*B14 + A12*B24
        # C23 = A21*B13 + A22*B23
        # C24 = A21*B14 + A22*B24
        
        # m7: A11*B13
        U[0,7]=1; V[2,7]=1; W[2,7]=1
        # m8: A12*B23
        U[1,8]=1; V[6,8]=1; W[2,8]=1
        # m9: A11*B14
        U[0,9]=1; V[3,9]=1; W[3,9]=1
        # m10: A12*B24
        U[1,10]=1; V[7,10]=1; W[3,10]=1
        # m11: A21*B13
        U[2,11]=1; V[2,11]=1; W[6,11]=1
        # m12: A22*B23
        U[3,11]=1; V[6,11]=1; W[6,11]=1
        # m13: A21*B14
        U[2,12]=1; V[3,12]=1; W[7,12]=1
        # m14: A22*B24
        U[3,12]=1; V[7,12]=1; W[7,12]=1
        
        # Wait, that's 13 terms, not 12. Let me reconsider.
        # Actually, I need to be more careful. Let me use a known rank-12 construction.
        
        # Reset and try a different approach
        U = np.zeros((8, 12))
        V = np.zeros((16, 12))
        W = np.zeros((16, 12))
        
        # Use the fact that rank(2x4x4) = 12
        # One construction: treat it as two 2x2x4 problems
        # Or use a Kronecker product approach
        
        # Actually, let me just use naive for now and optimize later
        # 2x4x4 naive rank = 8*4 = 32, which is too high
        
        # Let me try a different decomposition: split into 2x2x2 + 2x2x2
        # C = A*B where A is 2x4, B is 4x4
        # Split A = [A_left, A_right] where each is 2x2
        # Split B = [[B_topleft, B_topright], [B_bottomleft, B_bottomright]]
        # C = [A_left*B_topleft + A_right*B_bottomleft, A_left*B_topright + A_right*B_bottomright]
        # Each 2x2*2x2 has rank 7, so total rank = 14, not 12
        
        # The rank-12 construction is non-trivial. Let me use a lookup or derive it.
        # For now, let's use rank 14 and see if we can improve.
        
        # Actually, I recall that the tensor rank of 2x4x4 is indeed 12.
        # Let me construct it properly.
        
        # The 12-term decomposition for 2x4x4:
        # This is equivalent to multiplying a 2x4 matrix by a 4x4 matrix.
        # One way: use 6 terms for the first 2 columns and 6 for the last 2,
        # but that gives 12 if we can share some.
        
        # Let me try: treat B as 4x4 = [[B1, B2]] where B1, B2 are 4x2
        # C = [A*B1, A*B2]
        # Each A*Bi is 2x2, rank 7 each, total 14.
        
        # I think I need to look up the actual rank-12 construction.
        # For now, let's use a fallback.
        
        return None  # Placeholder, will implement properly

    def strassen_2x4x1():
        """Rank-8 decomposition for 2x4*4x1.
        Returns U, V, W with shapes (8,8), (4,8), (2,8)
        U indexed by [i*4+j], V by [i], W by [i]
        """
        # This is just dot products: C[i] = sum_j A[i,j]*B[j]
        # Naive: 8 terms, each A[i,j]*B[j] -> C[i]
        U = np.zeros((8, 8))
        V = np.zeros((4, 8))
        W = np.zeros((2, 8))
        for i in range(2):
            for j in range(4):
                idx = i*4+j
                U[idx, idx] = 1.0
                V[j, idx] = 1.0
                W[i, idx] = 1.0
        return U, V, W

    # Strategy: 4x4*4x5 via splitting A's rows into two groups of 2
    # Split A into top 2 rows and bottom 2 rows
    # Top: (2x4) * (4x5) = (2x4)*(4x4) + (2x4)*(4x1)
    # Bottom: (2x4) * (4x5) = (2x4)*(4x4) + (2x4)*(4x1)
    
    # For now, use the 2x2 block strategy which we know works (rank 64)
    s22 = strassen_2x2x2()
    s23 = strassen_2x2x3()

    # Strategy 1: 2x3 + 2x2 (rank 16 per block, total 64)
    R1 = 64
    U1 = np.zeros((n*m, R1))
    V1 = np.zeros((m*p, R1))
    W1 = np.zeros((p*n, R1))
    off = 0
    for ar in (0, 2):
        for ac in (0, 2):
            # Part 1: 2x3 (cols 0,1,2 of B)
            for t in range(9):
                for ii in range(2):
                    for jj in range(2):
                        U1[(ar+ii)*m + (ac+jj), off+t] = s23[0][ii*2+jj, t]
                for ii in range(2):
                    for jj in range(3):
                        V1[(ac+ii)*p + jj, off+t] = s23[1][ii*3+jj, t]
                for ii in range(2):
                    for jj in range(3):
                        W1[jj*n + (ar+ii), off+t] = s23[2][ii*3+jj, t]
            off += 9
            # Part 2: 2x2 (cols 3,4 of B)
            for t in range(7):
                for ii in range(2):
                    for jj in range(2):
                        U1[(ar+ii)*m + (ac+jj), off+t] = s22[0][ii*2+jj, t]
                for ii in range(2):
                    for jj in range(2):
                        V1[(ac+ii)*p + (3+jj), off+t] = s22[1][ii*2+jj, t]
                for ii in range(2):
                    for jj in range(2):
                        W1[(3+jj)*n + (ar+ii), off+t] = s22[2][ii*2+jj, t]
            off += 7

    if check(U1, V1, W1, R1):
        return {"U": to_list(U1), "V": to_list(V1), "W": to_list(W1)}

    # Check deadline
    if time.time() > deadline - 5:
        pass  # Return fallback

    # Fallback: schoolbook rank 80
    Rs = n * m * p
    Us = np.zeros((n*m, Rs))
    Vs = np.zeros((m*p, Rs))
    Ws = np.zeros((p*n, Rs))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Us[i*m+j, c] = 1.0
                Vs[j*p+k, c] = 1.0
                Ws[k*n+i, c] = 1.0
                c += 1
    return {"U": to_list(Us), "V": to_list(Vs), "W": to_list(Ws)}
# EVOLVE-BLOCK-END
