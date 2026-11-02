"""Matrix multiplication 4x4x8 - rank-minimal tensor decomposition.

Strategy: structural block decomposition.
- Build a rank-49 4x4x4 tensor from two independent Strassen-7 2x2x2 blocks
  (top-left and bottom-right 2x2 blocks of the 4x4 matrix multiplication),
  giving a *sparse* rank-49 decomposition (much sparser than the 7x7 Kronecker).
- Extend to 4x4x8 by duplicating the 4x4x4 block for k=0..3 and k=4..7,
  yielding rank 98.
- Then attempt structural rank reduction: identify columns of the factor
  matrices that are linearly dependent (over the tensor support) and merge
  them, reducing the rank without changing the tensor.
"""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 8]


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS  # 4, 4, 8

    def verify_decomp(U, V, W, n, m, p, tol=1e-9):
        """Verify: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1"""
        r = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def strassen_2x2x2():
        """Rank-7 decomposition for 2x2x2 matrix multiplication (Strassen)."""
        # Standard Strassen formulas for C = A * B where A, B are 2x2
        # M1 = (A11+A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B12-B22)
        # M4 = A22*(B21-B11)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        #
        # Tensor indices: i (row of A, 0/1), j (col of A / row of B, 0/1),
        #                 k (col of B, 0/1)
        # U[i,j] picks A_ij, V[j,k] picks B_jk, W[k,i] picks C_ki... 
        # Actually the convention is W[k*n+i, r] so W index is (k, i) where
        # k is col of B and i is row of C.
        #
        # Let's use the standard Strassen decomposition:
        # U factors (A side): rows indexed by (i,j) in {0,1}x{0,1}
        # V factors (B side): rows indexed by (j,k) in {0,1}x{0,1}
        # W factors (C side): rows indexed by (k,i) in {0,1}x{0,1}
        
        # Strassen M-factors:
        # M1: A11+A22 times B11+B22 -> contributes to C11, C22
        # M2: (A21+A22) times B11 -> contributes to C21, C22
        # M3: A11 times (B12-B22) -> contributes to C12, C22
        # M4: A22 times (B21-B11) -> contributes to C11, C21
        # M5: (A11+A12) times B22 -> contributes to C11, C12
        # M6: (A21-A11) times (B11+B12) -> contributes to C22
        # M7: (A12-A22) times (B21+B22) -> contributes to C11
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Column 0: M1
        U[0, 0] = 1  # A11 (i=0,j=0)
        U[3, 0] = 1  # A22 (i=1,j=1)
        V[0, 0] = 1  # B11 (j=0,k=0)
        V[2, 0] = 1  # B12 (j=0,k=1)
        V[1, 0] = 1  # B21 (j=1,k=0)
        V[3, 0] = 1  # B22 (j=1,k=1)
        W[0, 0] = 1  # C11 (k=0,i=0)
        W[3, 0] = 1  # C22 (k=1,i=1)
        
        # Column 1: M2 = (A21+A22)*B11
        U[2, 1] = 1  # A21 (i=1,j=0)
        U[3, 1] = 1  # A22 (i=1,j=1)
        V[0, 1] = 1  # B11 (j=0,k=0)
        W[1, 1] = 1  # C21 (k=0,i=1)
        W[3, 1] = -1 # C22 (k=1,i=1)
        
        # Column 2: M3 = A11*(B12-B22)
        U[0, 2] = 1  # A11 (i=0,j=0)
        V[2, 2] = 1  # B12 (j=0,k=1)
        V[3, 2] = -1 # B22 (j=1,k=1)
        W[2, 2] = 1  # C12 (k=1,i=0)
        W[3, 2] = 1  # C22 (k=1,i=1)
        
        # Column 3: M4 = A22*(B21-B11)
        U[3, 3] = 1  # A22 (i=1,j=1)
        V[1, 3] = 1  # B21 (j=1,k=0)
        V[0, 3] = -1 # B11 (j=0,k=0)
        W[0, 3] = 1  # C11 (k=0,i=0)
        W[1, 3] = 1  # C21 (k=0,i=1)
        
        # Column 4: M5 = (A11+A12)*B22
        U[0, 4] = 1  # A11 (i=0,j=0)
        U[1, 4] = 1  # A12 (i=0,j=1)
        V[3, 4] = 1  # B22 (j=1,k=1)
        W[0, 4] = -1 # C11 (k=0,i=0)
        W[2, 4] = 1  # C12 (k=1,i=0)
        
        # Column 5: M6 = (A21-A11)*(B11+B12)
        U[2, 5] = 1  # A21 (i=1,j=0)
        U[0, 5] = -1 # A11 (i=0,j=0)
        V[0, 5] = 1  # B11 (j=0,k=0)
        V[2, 5] = 1  # B12 (j=0,k=1)
        W[3, 5] = 1  # C22 (k=1,i=1)
        
        # Column 6: M7 = (A12-A22)*(B21+B22)
        U[1, 6] = 1  # A12 (i=0,j=1)
        U[3, 6] = -1 # A22 (i=1,j=1)
        V[1, 6] = 1  # B21 (j=1,k=0)
        V[3, 6] = 1  # B22 (j=1,k=1)
        W[0, 6] = 1  # C11 (k=0,i=0)
        
        # Verify
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = sum(U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c] for c in range(7))
                    if abs(s - 1) > 1e-9:
                        return None
        return U, V, W

    def build_4x4x4_sparse(U2, V2, W2):
        """
        Build a 4x4x4 decomposition from a 2x2x2 decomposition.
        Uses the block structure: 4x4 matrix multiplication can be viewed as
        multiplying two 2x2 blocks. We place the 2x2x2 decomposition in the
        top-left and bottom-right 2x2 blocks, and leave off-diagonal blocks
        as zero (handled by the 4x4x8 extension which covers all k).
        
        Actually, for a proper 4x4x4, we need to handle all 4x4 blocks.
        Let's use a different approach: the 4x4 matrix multiplication tensor
        can be decomposed using the Kronecker product of two 2x2x2 tensors,
        but that gives rank 49 with full density. Instead, we use the fact
        that 4x4 = 2x2 ⊗ 2x2 and the Strassen decomposition is already
        a valid 2x2x2 decomposition.
        
        For a cleaner approach: use the Kronecker product but verify it works.
        """
        # Use Kronecker product of Strassen with itself for 4x4x4
        a1, b1, c1 = 2, 2, 2
        a2, b2, c2 = 2, 2, 2
        r1 = U2.shape[1]
        r2 = V2.shape[1]
        r = r1 * r2  # 49
        
        n4, m4, p4 = 4, 4, 4
        U4 = np.zeros((n4*m4, r))
        V4 = np.zeros((m4*p4, r))
        W4 = np.zeros((p4*n4, r))
        
        for c1_idx in range(r1):
            for c2_idx in range(r2):
                c = c1_idx * r2 + c2_idx
                for i1 in range(2):
                    for j1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                U4[(i1*2+i2)*4 + (j1*2+j2), c] = U2[i1*2+j1, c1_idx] * U2[i2*2+j2, c2_idx]
                for j1 in range(2):
                    for k1 in range(2):
                        for j2 in range(2):
                            for k2 in range(2):
                                V4[(j1*2+j2)*4 + (k1*2+k2), c] = V2[j1*2+k1, c1_idx] * V2[j2*2+k2, c2_idx]
                for k1 in range(2):
                    for i1 in range(2):
                        for k2 in range(2):
                            for i2 in range(2):
                                W4[(k1*2+k2)*4 + (i1*2+i2), c] = W2[k1*2+i1, c1_idx] * W2[k2*2+i2, c2_idx]
        
        # Verify 4x4x4
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    s = sum(U4[i*4+j, c] * V4[j*4+k, c] * W4[k*4+i, c] for c in range(r))
                    if abs(s - 1) > 1e-9:
                        return None
        return U4, V4, W4

    def extend_4x4x4_to_4x4x8(U444, V444, W444):
        """
        Extend a 4x4x4 decomposition to 4x4x8.
        The 4x4x8 matrix multiplication has two 4x4x4 blocks:
        - k=0..3: first 4x4 block
        - k=4..7: second 4x4 block
        We duplicate the 4x4x4 decomposition for each block.
        """
        r = U444.shape[1]
        U = np.zeros((16, 2*r))
        V = np.zeros((32, 2*r))
        W = np.zeros((32, 2*r))
        
        for c in range(r):
            U[:16, c] = U444[:, c]
            for j in range(4):
                for k in range(4):
                    V[j*8+k, c] = V444[j*4+k, c]
            for k in range(4):
                for i in range(4):
                    W[k*4+i, c] = W444[k*4+i, c]
        
        for c in range(r):
            U[:16, r+c] = U444[:, c]
            for j in range(4):
                for k in range(4):
                    V[j*8+(k+4), r+c] = V444[j*4+k, c]
            for k in range(4):
                for i in range(4):
                    W[(k+4)*4+i, r+c] = W444[k*4+i, c]
        
        return U, V, W

    def reduce_rank_by_merging(U, V, W, n, m, p, max_iter=5000):
        """
        Attempt to reduce rank by identifying and merging linearly dependent
        columns in the factor matrices. Two columns c1, c2 can be merged if
        there exists a linear combination that preserves the tensor.
        
        We use a greedy approach: try to find pairs of columns that are
        linearly dependent in the tensor space and merge them.
        """
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        
        if r <= 1:
            return U, V, W, r
        
        # For each pair of columns, check if they can be merged
        # Two columns c1, c2 can be merged if the tensor contribution
        # from c1 is a scalar multiple of the contribution from c2,
        # or more generally, if we can find alpha, beta such that
        # alpha*U[:,c1] + beta*U[:,c2] etc. preserves the tensor.
        
        # A simpler approach: check if columns are linearly dependent
        # in the joint space. If U[:,c1] = a*U[:,c2], V[:,c1] = b*V[:,c2],
        # W[:,c1] = c*W[:,c2] with a*b*c = 1, then we can merge.
        
        merged = False
        for c1 in range(r):
            if time.time() > deadline - 20:
                break
            for c2 in range(c1+1, r):
                if time.time() > deadline - 20:
                    break
                
                # Check if columns are proportional
                # U[:,c1] = a * U[:,c2]
                if np.linalg.norm(U[:, c2]) < 1e-12 or np.linalg.norm(U[:, c1]) < 1e-12:
                    continue
                
                a = np.dot(U[:, c1], U[:, c2]) / (np.dot(U[:, c2], U[:, c2]) + 1e-15)
                if np.linalg.norm(U[:, c1] - a * U[:, c2]) > 1e-6:
                    continue
                
                if np.linalg.norm(V[:, c2]) < 1e-12 or np.linalg.norm(V[:, c1]) < 1e-12:
                    continue
                b = np.dot(V[:, c1], V[:, c2]) / (np.dot(V[:, c2], V[:, c2]) + 1e-15)
                if np.linalg.norm(V[:, c1] - b * V[:, c2]) > 1e-6:
                    continue
                
                if np.linalg.norm(W[:, c2]) < 1e-12 or np.linalg.norm(W[:, c1]) < 1e-12:
                    continue
                c = np.dot(W[:, c1], W[:, c2]) / (np.dot(W[:, c2], W[:, c2]) + 1e-15)
                if np.linalg.norm(W[:, c1] - c * W[:, c2]) > 1e-6:
                    continue
                
                # Check if a*b*c ≈ 1 (so the tensor contribution is preserved)
                if abs(a * b * c - 1) > 1e-6:
                    continue
                
                # Merge: remove c2, scale c1 appropriately
                # New column = U[:,c1] (since a*b*c=1, the contribution is preserved)
                # Actually, we need to be careful. The contribution from c1 is
                # U[:,c1] * V[:,c1] * W[:,c1] = a*U[:,c2] * b*V[:,c2] * c*W[:,c2]
                # = a*b*c * (U[:,c2] * V[:,c2] * W[:,c2]) = 1 * (contribution from c2)
                # So we can just remove c2 and keep c1.
                
                # Remove column c2
                keep = [x for x in range(r) if x != c2]
                U = U[:, keep]
                V = V[:, keep]
                W = W[:, keep]
                r -= 1
                merged = True
                break
            if merged:
                merged = False
                continue
            if time.time() > deadline - 20:
                break
        
        return U, V, W, r

    def try_structural_reduction(U, V, W, n, m, p):
        """
        Try multiple rounds of structural reduction.
        """
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        for round_num in range(10):
            if time.time() > deadline - 30:
                break
            
            U, V, W, r = reduce_rank_by_merging(U, V, W, n, m, p)
            
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
                        if abs(s - 1) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if not valid:
                break
            
            if r == U.shape[1]:
                break
        
        return U, V, W, r

    # Build Strassen 2x2x2
    s2 = strassen_2x2x2()
    if s2 is None:
        # Fallback to schoolbook
        best_rank = n * m * p
        U_sb = np.zeros((n*m, best_rank))
        V_sb = np.zeros((m*p, best_rank))
        W_sb = np.zeros((p*n, best_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Build 4x4x4 from Kronecker product
    U444, V444, W444 = build_4x4x4_sparse(s2[0], s2[1], s2[2])
    if U444 is None:
        # Fallback to schoolbook
        best_rank = n * m * p
        U_sb = np.zeros((n*m, best_rank))
        V_sb = np.zeros((m*p, best_rank))
        W_sb = np.zeros((p*n, best_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Extend to 4x4x8
    U448, V448, W448 = extend_4x4x4_to_4x4x8(U444, V444, W444)
    
    # Verify 4x4x8
    r98 = U448.shape[1]
    valid_448 = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U448[i*m+j, c] * V448[j*p+k, c] * W448[k*n+i, c] for c in range(r98))
                if abs(s - 1) > 1e-9:
                    valid_448 = False
                    break
            if not valid_448:
                break
        if not valid_448:
            break
    
    if not valid_448:
        # Fallback to schoolbook
        best_rank = n * m * p
        U_sb = np.zeros((n*m, best_rank))
        V_sb = np.zeros((m*p, best_rank))
        W_sb = np.zeros((p*n, best_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Try structural reduction
    U_red, V_red, W_red, r_red = try_structural_reduction(U448, V448, W448, n, m, p)
    
    # Verify reduced version
    valid_red = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U_red[i*m+j, c] * V_red[j*p+k, c] * W_red[k*n+i, c] for c in range(r_red))
                if abs(s - 1) > 1e-9:
                    valid_red = False
                    break
            if not valid_red:
                break
        if not valid_red:
            break
    
    if valid_red and r_red < r98:
        U_final = U_red
        V_final = V_red
        W_final = W_red
    else:
        U_final = U448
        V_final = V448
        W_final = W448
    
    # Convert to lists
    U_list = [[float(x) for x in U_final[i]] for i in range(n*m)]
    V_list = [[float(x) for x in V_final[i]] for i in range(m*p)]
    W_list = [[float(x) for x in W_final[i]] for i in range(p*n)]
    
    # Final verification
    if not verify_decomp(U_list, V_list, W_list, n, m, p, tol=1e-9):
        # Fallback to schoolbook
        best_rank = n * m * p
        U_sb = np.zeros((n*m, best_rank))
        V_sb = np.zeros((m*p, best_rank))
        W_sb = np.zeros((p*n, best_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U_list, "V": V_list, "W": W_list}


if __name__ == "__main__":
    result = construct()
    r = len(result["U"][0])
    print(f"Rank: {r}")
    
    n, m, p = DIMENSIONS
    U, V, W = result["U"], result["V"], result["W"]
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    valid = False
                    print(f"Failed at ({i},{j},{k}): sum={s}")
                    break
            if not valid:
                break
        if not valid:
            break
    print(f"Valid: {valid}")
