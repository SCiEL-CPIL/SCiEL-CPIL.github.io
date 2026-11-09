"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 5, 5, 5
    nm = n * m  # 25
    mp = m * p  # 25
    pn = p * n  # 25
    
    # Schoolbook decomposition as fallback
    def _schoolbook():
        rank = n * m * p  # 125
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Verify decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should be 1
    def _verify(U, V, W, tol=1e-9):
        rank = len(U[0]) if U else 0
        # Convert to numpy for speed
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    val = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Structural approach: Block decomposition with Strassen for 2x2 subblock
    # Split 5x5 as 2+3 blocks
    # A = [[A11(2x2), A12(2x3)], [A21(3x2), A22(3x3)]]
    # B = [[B11(2x2), B12(2x3)], [B21(3x2), B22(3x3)]]
    #
    # C11(2x2) = A11*B11 + A12*B21
    # C12(2x3) = A11*B12 + A12*B22
    # C21(3x2) = A21*B11 + A22*B21
    # C22(3x3) = A21*B12 + A22*B22
    #
    # Use Strassen (rank 7) for A11*B11
    # Use schoolbook for all other terms
    
    def _strassen_2x2():
        """Return U,V,W for 2x2x2 Strassen decomposition.
        U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
        Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        """
        # Strassen's 7 multiplications for 2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        #
        # Each M corresponds to a rank-1 term where:
        # M_c = (sum_a U[a,c]*A[a]) * (sum_b V[b,c]*B[b]) contributes to C via W
        #
        # For rank-1 term c: U_c is a 2x2 matrix, V_c is a 2x2 matrix, W_c is a 2x2 matrix
        # The contribution to C[i,j] is sum_c U_c[i,j] * V_c[i,j] * W_c[i,j] ... no
        # Actually: C[i,j] = sum_k sum_c U[i,k,c] * V[k,j,c] * W[i,j,c]
        # Wait, the tensor decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # For 2x2: T[i,j,k] = sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c]
        # This means for each rank-1 term c, we have matrices U_c, V_c, W_c (each 2x2)
        # such that the multiplication tensor is sum_c U_c ⊗ V_c ⊗ W_c (in appropriate indices)
        
        # Let me think differently. The standard way:
        # C = A * B where C[i,j] = sum_k A[i,k] * B[k,j]
        # Tensor: T[i,j,k] = 1 for all i,j,k (the multiplication tensor has all 1s)
        # Wait no. T[i,j,k] = δ_{i,k} * δ_{k,j}? No.
        # The multiplication tensor is T[i,j,k] where C[i,j] = sum_k A[i,k] B[k,j]
        # So T[i,j,k] = 1 for all (i,j,k) in the range? No!
        # T[i,j,k] is the coefficient: C[i,j] = sum_k T[i,j,k] * A[i,k] * B[k,j]
        # For standard matrix multiplication, T[i,j,k] = 1 for all i,j,k.
        # So the tensor is all-ones!
        
        # Rank-1 decomposition: T = sum_c u_c ⊗ v_c ⊗ w_c
        # where (u_c ⊗ v_c ⊗ w_c)[i,j,k] = u_c[i] * v_c[j] * w_c[k] ... no
        # In our indexing: T[i,j,k] = sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c]
        
        # For Strassen, the 7 rank-1 terms correspond to:
        # Term 1: (A11+A22)(B11+B22)
        #   U part: A11+A22 means U[i*2+j, 0] = 1 if (i,j) in {(0,0),(1,1)} else 0
        #   V part: B11+B22 means V[j*2+k, 0] = 1 if (j,k) in {(0,0),(1,1)} else 0  
        #   W part: contributes to C11 and C22
        #   C11 gets +1, C22 gets +1 from this term
        #   So W[0*2+0, 0] = 1 (C11), W[1*2+1, 0] = 1 (C22), others 0
        
        # Let me just directly construct the Strassen decomposition.
        # The 7 terms for 2x2 Strassen:
        
        terms = []
        
        # Term 1: M1 = (A11+A22)(B11+B22)
        # U: A11+A22 -> U[0,0]=1, U[3,0]=1 (i.e., U[0*2+0]=1, U[1*2+1]=1)
        # V: B11+B22 -> V[0,0]=1, V[3,0]=1 (i.e., V[0*2+0]=1, V[1*2+1]=1)
        # W: contributes to C11 and C22 -> W[0,0]=1, W[3,0]=1 (i.e., W[0*2+0]=1, W[1*2+1]=1)
        terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
        
        # Term 2: M2 = (A21+A22)B11
        # U: A21+A22 -> U[2,0]=1, U[3,0]=1 (i.e., U[1*2+0]=1, U[1*2+1]=1)
        # V: B11 -> V[0,0]=1 (i.e., V[0*2+0]=1)
        # W: contributes to C21 and C22 -> W[2,0]=1, W[3,0]=1 (i.e., W[1*2+0]=1, W[1*2+1]=1)
        terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,1]))
        
        # Term 3: M3 = A11(B12-B22)
        # U: A11 -> U[0,0]=1 (i.e., U[0*2+0]=1)
        # V: B12-B22 -> V[1,0]=1, V[3,0]=-1 (i.e., V[0*2+1]=1, V[1*2+1]=-1)
        # W: contributes to C12 and C22 -> W[1,0]=1, W[3,0]=1 (i.e., W[0*2+1]=1, W[1*2+1]=1)
        terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,1]))
        
        # Term 4: M4 = A22(B21-B11)
        # U: A22 -> U[3,0]=1 (i.e., U[1*2+1]=1)
        # V: B21-B11 -> V[2,0]=1, V[0,0]=-1 (i.e., V[1*2+0]=1, V[0*2+0]=-1)
        # W: contributes to C11 and C21 -> W[0,0]=1, W[2,0]=1 (i.e., W[0*2+0]=1, W[1*2+0]=1)
        terms.append(([0,0,0,1], [-1,0,1,0], [1,0,1,0]))
        
        # Term 5: M5 = (A11+A12)B22
        # U: A11+A12 -> U[0,0]=1, U[1,0]=1 (i.e., U[0*2+0]=1, U[0*2+1]=1)
        # V: B22 -> V[3,0]=1 (i.e., V[1*2+1]=1)
        # W: contributes to C11 and C12 -> W[0,0]=-1, W[1,0]=-1 (i.e., W[0*2+0]=-1, W[0*2+1]=-1)
        # Wait, let me recheck the Strassen formulas:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # So M5 contributes -1 to C11 and +1 to C12
        terms.append(([1,1,0,0], [0,0,0,1], [-1,-1,0,0]))
        
        # Term 6: M6 = (A21-A11)(B11+B12)
        # U: A21-A11 -> U[2,0]=1, U[0,0]=-1 (i.e., U[1*2+0]=1, U[0*2+0]=-1)
        # V: B11+B12 -> V[0,0]=1, V[1,0]=1 (i.e., V[0*2+0]=1, V[0*2+1]=1)
        # W: contributes to C22 -> W[3,0]=1 (i.e., W[1*2+1]=1)
        terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
        
        # Term 7: M7 = (A12-A22)(B21+B22)
        # U: A12-A22 -> U[1,0]=1, U[3,0]=-1 (i.e., U[0*2+1]=1, U[1*2+1]=-1)
        # V: B21+B22 -> V[2,0]=1, V[3,0]=1 (i.e., V[1*2+0]=1, V[1*2+1]=1)
        # W: contributes to C11 -> W[0,0]=1 (i.e., W[0*2+0]=1)
        terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        for c, (uc, vc, wc) in enumerate(terms):
            for idx in range(4):
                U[idx][c] = uc[idx]
                V[idx][c] = vc[idx]
                W[idx][c] = wc[idx]
        
        return U, V, W
    
    def _build_block_decomposition():
        """Build the 5x5x5 decomposition using 2+3 block split with Strassen for 2x2 part."""
        U_terms = []
        V_terms = []
        W_terms = []
        
        def add_term(u_dict, v_dict, w_dict):
            """Add a rank-1 term. u_dict maps (i,j) -> val for A indices,
            v_dict maps (j,k) -> val for B indices,
            w_dict maps (k,i) -> val for C indices."""
            U_row = [0.0] * nm
            V_row = [0.0] * mp
            W_row = [0.0] * pn
            for (ii, jj), val in u_dict.items():
                if val != 0:
                    U_row[ii * m + jj] = val
            for (jj, kk), val in v_dict.items():
                if val != 0:
                    V_row[jj * p + kk] = val
            for (kk, ii), val in w_dict.items():
                if val != 0:
                    W_row[kk * n + ii] = val
            U_terms.append(U_row)
            V_terms.append(V_row)
            W_terms.append(W_row)
        
        # Block indices for 2+3 split:
        # Rows 0-1: first block (size 2)
        # Rows 2-4: second block (size 3)
        # Same for columns
        
        # === Strassen for A11*B11 (2x2 * 2x2 -> 2x2) ===
        sU, sV, sW = _strassen_2x2()
        for c in range(7):
            u_dict = {}
            v_dict = {}
            w_dict = {}
            for i in range(2):
                for j in range(2):
                    if sU[i*2+j][c] != 0:
                        u_dict[(i, j)] = sU[i*2+j][c]
            for j in range(2):
                for k in range(2):
                    if sV[j*2+k][c] != 0:
                        v_dict[(j, k)] = sV[j*2+k][c]
            for k in range(2):
                for i in range(2):
                    if sW[k*2+i][c] != 0:
                        w_dict[(k, i)] = sW[k*2+i][c]
            add_term(u_dict, v_dict, w_dict)
        
        # === A12*B21 (2x3 * 3x2 -> 2x2): schoolbook, 2*3*2=12 terms ===
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    add_term(
                        {(i, 2+j): 1},
                        {(2+j, k): 1},
                        {(k, i): 1}
                    )
        
        # === A12*B22 (2x3 * 3x3 -> 2x3): schoolbook, 2*3*3=18 terms ===
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    add_term(
                        {(i, 2+j): 1},
                        {(2+j, 2+k): 1},
                        {(2+k, i): 1}
                    )
        
        # === A21*B11 (3x2 * 2x2 -> 3x2): schoolbook, 3*2*2=12 terms ===
        for i in range(3):
            for j in range(2):
                for k in range(2):
                    add_term(
                        {(2+i, j): 1},
                        {(j, k): 1},
                        {(k, 2+i): 1}
                    )
        
        # === A21*B12 (3x2 * 2x3 -> 3x3): schoolbook, 3*2*3=18 terms ===
        for i in range(3):
            for j in range(2):
                for k in range(3):
                    add_term(
                        {(2+i, j): 1},
                        {(j, 2+k): 1},
                        {(2+k, 2+i): 1}
                    )
        
        # === A22*B22 (3x3 * 3x3 -> 3x3): schoolbook, 3*3*3=27 terms ===
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    add_term(
                        {(2+i, 2+j): 1},
                        {(2+j, 2+k): 1},
                        {(2+k, 2+i): 1}
                    )
        
        rank = len(U_terms)
        U = U_terms
        V = V_terms
        W = W_terms
        
        return {"U": U, "V": V, "W": W}
    
    # Try the block decomposition
    result = _build_block_decomposition()
    
    # Verify
    if _verify(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
