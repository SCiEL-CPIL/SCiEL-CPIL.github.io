"""Rewritten program for 3x4x5 matrix multiplication tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 12
    mp = m * p  # 20
    pn = p * n  # 15
    
    def verify(U, V, W, tol=1e-9):
        """Check if decomposition is exact (within tolerance)."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def build_schoolbook():
        r = n * m * p
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W, r
    
    def strassen_2x2x2():
        """Return Strassen decomposition for 2x2x2 matrix multiplication.
        Returns U(4x7), V(4x7), W(4x7) for the 2x2x2 tensor.
        Indices: U indexed by (i,j) with i in {0,1}, j in {0,1} -> i*2+j
                 V indexed by (j,k) with j in {0,1}, k in {0,1} -> j*2+k
                 W indexed by (k,i) with k in {0,1}, i in {0,1} -> k*2+i
        """
        # Strassen's 7 products for 2x2 matrix multiplication
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # C[0,0] = p1*p5 - p4*p8 + p7*p8 + p1*p6  ... let me use standard form
        # Standard Strassen:
        # p1 = (a+d)(e+h)
        # p2 = (c+d)(e)
        # p3 = (a-b)(f+h)
        # p4 = (c+d)(g+h)
        # p5 = (a+c)(e)
        # p6 = (b+d)(h)
        # p7 = (a-b)(f+g)
        # C[0,0] = p1 - p4 + p7 + p6
        # C[0,1] = p3 + p5
        # C[1,0] = p2 + p4
        # C[1,1] = p1 - p3 + p2 - p5
        
        # For each product pc, we need U, V, W vectors such that
        # sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = C[i,k]
        
        # Product 1: (a+d)(e+h)
        # a = A[0,0], d = A[1,1], e = B[0,0], h = B[1,1]
        # Contributes to C[0,0] with +1, C[1,1] with +1
        # U[0,0]=1, U[1,1]=1; V[0,0]=1, V[1,1]=1; W[0,0]=1, W[1,1]=1
        # But this gives C[0,0]+=1, C[0,1]+=0, C[1,0]+=0, C[1,1]+=1? 
        # Wait: U[i,j]*V[j,k]*W[k,i]
        # For (i=0,j=0,k=0): U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1 -> contributes to C[0,0]
        # For (i=0,j=1,k=0): U[0,1]*V[1,0]*W[0,0] = 0*0*1 = 0
        # For (i=0,j=0,k=1): U[0,0]*V[0,1]*W[1,0] = 1*0*0 = 0
        # For (i=0,j=1,k=1): U[0,1]*V[1,1]*W[1,1] = 0*1*1 = 0
        # For (i=1,j=0,k=0): U[1,0]*V[0,0]*W[0,1] = 0*1*0 = 0
        # For (i=1,j=1,k=0): U[1,1]*V[1,0]*W[0,1] = 1*0*0 = 0
        # For (i=1,j=0,k=1): U[1,0]*V[0,1]*W[1,1] = 0*0*1 = 0
        # For (i=1,j=1,k=1): U[1,1]*V[1,1]*W[1,1] = 1*1*1 = 1 -> contributes to C[1,1]
        # So this gives C[0,0]+=1, C[1,1]+=1. Good.
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # p1 = (a+d)(e+h): U has 1 at (0,0) and (1,1); V has 1 at (0,0) and (1,1); W has 1 at (0,0) and (1,1)
        # Index: U[i*2+j], V[j*2+k], W[k*2+i]
        U[0, 0] = 1; U[3, 0] = 1  # (i=0,j=0) and (i=1,j=1)
        V[0, 0] = 1; V[3, 0] = 1  # (j=0,k=0) and (j=1,k=1)
        W[0, 0] = 1; W[3, 0] = 1  # (k=0,i=0) and (k=1,i=1)
        
        # p2 = (c+d)(e): c=A[1,0], d=A[1,1], e=B[0,0]
        # U: (i=1,j=0) and (i=1,j=1) -> indices 2, 3
        # V: (j=0,k=0) -> index 0
        # W: (k=0,i=1) -> index 1
        # Contributes to C[1,0] with +1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1
        
        # p3 = (a-b)(f+h): a=A[0,0], b=A[0,1], f=B[0,1], h=B[1,1]
        # U: (i=0,j=0)=+1, (i=0,j=1)=-1 -> indices 0, 1
        # V: (j=0,k=1)=+1, (j=1,k=1)=+1 -> indices 1, 3
        # W: (k=1,i=0)=+1 -> index 2
        # Contributes to C[0,1] with +1
        U[0, 2] = 1; U[1, 2] = -1
        V[1, 2] = 1; V[3, 2] = 1
        W[2, 2] = 1
        
        # p4 = (c+d)(g+h): c=A[1,0], d=A[1,1], g=B[1,0], h=B[1,1]
        # U: (i=1,j=0)=+1, (i=1,j=1)=+1 -> indices 2, 3
        # V: (j=1,k=0)=+1, (j=1,k=1)=+1 -> indices 2, 3
        # W: (k=0,i=1)=+1 -> index 1
        # Contributes to C[1,0] with +1
        U[2, 3] = 1; U[3, 3] = 1
        V[2, 3] = 1; V[3, 3] = 1
        W[1, 3] = 1
        
        # p5 = (a+c)(e): a=A[0,0], c=A[1,0], e=B[0,0]
        # U: (i=0,j=0)=+1, (i=1,j=0)=+1 -> indices 0, 2
        # V: (j=0,k=0)=+1 -> index 0
        # W: (k=0,i=0)=+1 -> index 0
        # Contributes to C[0,0] with +1
        U[0, 4] = 1; U[2, 4] = 1
        V[0, 4] = 1
        W[0, 4] = 1
        
        # p6 = (b+d)(h): b=A[0,1], d=A[1,1], h=B[1,1]
        # U: (i=0,j=1)=+1, (i=1,j=1)=+1 -> indices 1, 3
        # V: (j=1,k=1)=+1 -> index 3
        # W: (k=1,i=0)=+1 -> index 2
        # Contributes to C[0,1] with +1
        U[1, 5] = 1; U[3, 5] = 1
        V[3, 5] = 1
        W[2, 5] = 1
        
        # p7 = (a-b)(f+g): a=A[0,0], b=A[0,1], f=B[0,1], g=B[1,0]
        # U: (i=0,j=0)=+1, (i=0,j=1)=-1 -> indices 0, 1
        # V: (j=0,k=1)=+1, (j=1,k=0)=+1 -> indices 1, 2
        # W: (k=0,i=0)=+1 -> index 0
        # Contributes to C[0,0] with +1
        U[0, 6] = 1; U[1, 6] = -1
        V[1, 6] = 1; V[2, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    def build_block_decomposition():
        """Build block decomposition for 3x4x5.
        Split: i={0,1},{2}; j={0,1},{2,3}; k={0,1},{2,3},{4}
        Blocks and their ranks:
        (i={0,1},j={0,1},k={0,1}): 2x2x2, rank 7 (Strassen)
        (i={0,1},j={0,1},k={2,3}): 2x2x2, rank 7 (Strassen)
        (i={0,1},j={0,1},k={4}):   2x2x1, rank 4 (schoolbook)
        (i={0,1},j={2,3},k={0,1}): 2x2x2, rank 7 (Strassen)
        (i={0,1},j={2,3},k={2,3}): 2x2x2, rank 7 (Strassen)
        (i={0,1},j={2,3},k={4}):   2x2x1, rank 4 (schoolbook)
        (i={2},j={0,1},k={0,1}):   1x2x2, rank 4 (schoolbook)
        (i={2},j={0,1},k={2,3}):   1x2x2, rank 4 (schoolbook)
        (i={2},j={0,1},k={4}):     1x2x1, rank 2 (schoolbook)
        (i={2},j={2,3},k={0,1}):   1x2x2, rank 4 (schoolbook)
        (i={2},j={2,3},k={2,3}):   1x2x2, rank 4 (schoolbook)
        (i={2},j={2,3},k={4}):     1x2x1, rank 2 (schoolbook)
        Total rank: 7+7+4+7+7+4+4+4+2+4+4+2 = 56
        """
        U = np.zeros((nm, 56))
        V = np.zeros((mp, 56))
        W = np.zeros((pn, 56))
        col = 0
        
        def add_block(i_indices, j_indices, k_indices, block_U, block_V, block_W):
            """Add a block decomposition. block_U, block_V, block_W are for the local indices.
            Local U index: local_i * len(j_indices) + local_j
            Local V index: local_j * len(k_indices) + local_k
            Local W index: local_k * len(i_indices) + local_i
            """
            r_block = block_U.shape[1]
            for c in range(r_block):
                for li, i in enumerate(i_indices):
                    for lj, j in enumerate(j_indices):
                        U[i*m + j, col + c] += block_U[li*len(j_indices) + lj, c]
                for lj, j in enumerate(j_indices):
                    for lk, k in enumerate(k_indices):
                        V[j*p + k, col + c] += block_V[lj*len(k_indices) + lk, c]
                for lk, k in enumerate(k_indices):
                    for li, i in enumerate(i_indices):
                        W[k*n + i, col + c] += block_W[lk*len(i_indices) + li, c]
            return r_block
        
        # Block 1: i={0,1}, j={0,1}, k={0,1} - 2x2x2 Strassen
        bU, bV, bW = strassen_2x2x2()
        col += add_block([0,1], [0,1], [0,1], bU, bV, bW)
        
        # Block 2: i={0,1}, j={0,1}, k={2,3} - 2x2x2 Strassen
        bU, bV, bW = strassen_2x2x2()
        col += add_block([0,1], [0,1], [2,3], bU, bV, bW)
        
        # Block 3: i={0,1}, j={0,1}, k={4} - 2x2x1 schoolbook (rank 4)
        # 2x2x1: C[0,0]=a*e+c*e, C[1,0]=b*e+d*e where A=[[a,b],[c,d]], B=[[e],[f]]... 
        # Wait, 2x2x1 means A is 2x2, B is 2x1, C is 2x1
        # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
        # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
        # 4 rank-1 terms
        bU = np.zeros((4, 4))  # 2x2 -> 4 entries
        bV = np.zeros((2, 4))  # 2x1 -> 2 entries
        bW = np.zeros((2, 4))  # 1x2 -> 2 entries
        # Term 1: A[0,0]*B[0] -> C[0]
        bU[0, 0] = 1; bV[0, 0] = 1; bW[0, 0] = 1  # W index: k*2+i, k=0, i=0 -> 0
        # Term 2: A[0,1]*B[1] -> C[0]
        bU[1, 1] = 1; bV[1, 1] = 1; bW[0, 1] = 1
        # Term 3: A[1,0]*B[0] -> C[1]
        bU[2, 2] = 1; bV[0, 2] = 1; bW[1, 2] = 1  # W index: k=0, i=1 -> 1
        # Term 4: A[1,1]*B[1] -> C[1]
        bU[3, 3] = 1; bV[1, 3] = 1; bW[1, 3] = 1
        col += add_block([0,1], [0,1], [4], bU, bV, bW)
        
        # Block 4: i={0,1}, j={2,3}, k={0,1} - 2x2x2 Strassen
        bU, bV, bW = strassen_2x2x2()
        col += add_block([0,1], [2,3], [0,1], bU, bV, bW)
        
        # Block 5: i={0,1}, j={2,3}, k={2,3} - 2x2x2 Strassen
        bU, bV, bW = strassen_2x2x2()
        col += add_block([0,1], [2,3], [2,3], bU, bV, bW)
        
        # Block 6: i={0,1}, j={2,3}, k={4} - 2x2x1 schoolbook (rank 4)
        bU = np.zeros((4, 4))
        bV = np.zeros((2, 4))
        bW = np.zeros((2, 4))
        bU[0, 0] = 1; bV[0, 0] = 1; bW[0, 0] = 1
        bU[1, 1] = 1; bV[1, 1] = 1; bW[0, 1] = 1
        bU[2, 2] = 1; bV[0, 2] = 1; bW[1, 2] = 1
        bU[3, 3] = 1; bV[1, 3] = 1; bW[1, 3] = 1
        col += add_block([0,1], [2,3], [4], bU, bV, bW)
        
        # Block 7: i={2}, j={0,1}, k={0,1} - 1x2x2 schoolbook (rank 4)
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        bU = np.zeros((2, 4))  # 1x2 -> 2 entries
        bV = np.zeros((4, 4))  # 2x2 -> 4 entries
        bW = np.zeros((2, 4))  # 2x1 -> 2 entries
        # Term 1: A[0,0]*B[0,0] -> C[0,0]
        bU[0, 0] = 1; bV[0, 0] = 1; bW[0, 0] = 1  # W: k*1+i, k=0,i=0 -> 0
        # Term 2: A[0,1]*B[1,0] -> C[0,0]
        bU[1, 1] = 1; bV[2, 1] = 1; bW[0, 1] = 1
        # Term 3: A[0,0]*B[0,1] -> C[0,1]
        bU[0, 2] = 1; bV[1, 2] = 1; bW[1, 2] = 1  # W: k=1,i=0 -> 1
        # Term 4: A[0,1]*B[1,1] -> C[0,1]
        bU[1, 3] = 1; bV[3, 3] = 1; bW[1, 3] = 1
        col += add_block([2], [0,1], [0,1], bU, bV, bW)
        
        # Block 8: i={2}, j={0,1}, k={2,3} - 1x2x2 schoolbook (rank 4)
        bU = np.zeros((2, 4))
        bV = np.zeros((4, 4))
        bW = np.zeros((2, 4))
        bU[0, 0] = 1; bV[0, 0] = 1; bW[0, 0] = 1
        bU[1, 1] = 1; bV[2, 1] = 1; bW[0, 1] = 1
        bU[0, 2] = 1; bV[1, 2] = 1; bW[1, 2] = 1
        bU[1, 3] = 1; bV[3, 3] = 1; bW[1, 3] = 1
        col += add_block([2], [0,1], [2,3], bU, bV, bW)
        
        # Block 9: i={2}, j={0,1}, k={4} - 1x2x1 schoolbook (rank 2)
        # C[0,0] = A[0,0]*B[0] + A[0,1]*B[1]
        bU = np.zeros((2, 2))
        bV = np.zeros((2, 2))
        bW = np.zeros((1, 2))
        bU[0, 0] = 1; bV[0, 0] = 1; bW[0, 0] = 1
        bU[1, 1] = 1; bV[1, 1] = 1; bW[0, 1] = 1
        col += add_block([2], [0,1], [4], bU, bV, bW)
        
        # Block 10: i={2}, j={2,3}, k={0,1} - 1x2x2 schoolbook (rank 4)
        bU = np.zeros((2, 4))
        bV = np.zeros((4, 4))
        bW = np.zeros((2, 4))
        bU[0, 0] = 1; bV[0, 0] = 1; bW[0, 0] = 1
        bU[1, 1] = 1; bV[2, 1] = 1; bW[0, 1] = 1
        bU[0, 2] = 1; bV[1, 2] = 1; bW[1, 2] = 1
        bU[1, 3] = 1; bV[3, 3] = 1; bW[1, 3] = 1
        col += add_block([2], [2,3], [0,1], bU, bV, bW)
        
        # Block 11: i={2}, j={2,3}, k={2,3} - 1x2x2 schoolbook (rank 4)
        bU = np.zeros((2, 4))
        bV = np.zeros((4, 4))
        bW = np.zeros((2, 4))
        bU[0, 0] = 1; bV[0, 0] = 1; bW[0, 0] = 1
        bU[1, 1] = 1; bV[2, 1] = 1; bW[0, 1] = 1
        bU[0, 2] = 1; bV[1, 2] = 1; bW[1, 2] = 1
        bU[1, 3] = 1; bV[3, 3] = 1; bW[1, 3] = 1
        col += add_block([2], [2,3], [2,3], bU, bV, bW)
        
        # Block 12: i={2}, j={2,3}, k={4} - 1x2x1 schoolbook (rank 2)
        bU = np.zeros((2, 2))
        bV = np.zeros((2, 2))
        bW = np.zeros((1, 2))
        bU[0, 0] = 1; bV[0, 0] = 1; bW[0, 0] = 1
        bU[1, 1] = 1; bV[1, 1] = 1; bW[0, 1] = 1
        col += add_block([2], [2,3], [4], bU, bV, bW)
        
        assert col == 56, f"Expected 56 columns, got {col}"
        return U, V, W, 56
    
    # Build block decomposition
    U_block, V_block, W_block, r_block = build_block_decomposition()
    
    # Verify
    if verify(U_block, V_block, W_block, tol=1e-9):
        best_U, best_V, best_W, best_rank = U_block, V_block, W_block, r_block
    else:
        # Fall back to schoolbook
        U_sb, V_sb, W_sb, r_sb = build_schoolbook()
        best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, r_sb
    
    # Convert to lists
    def to_list(mat):
        mat = np.round(mat * 2) / 2.0
        return mat.tolist()
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
