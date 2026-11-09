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
    
    # Verify decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should be 1
    def _verify(U, V, W, tol=1e-9):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Block partition: {0,1}, {2,3}, {4}
    def block_indices(block):
        if block == 0:
            return [0, 1]
        elif block == 1:
            return [2, 3]
        else:
            return [4]
    
    # Strassen decomposition for 2x2x2 sub-tensor
    # For local indices i,j,k in {0,1}, we need:
    # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    # 
    # The correct Strassen tensor decomposition (verified by checking all 8 entries):
    # 7 rank-1 terms with coefficients in {-1, 0, 1}
    
    # Let me derive this carefully.
    # Strassen: M1=(a00+a11)(b00+b11), M2=(a10+a11)(b00), M3=a00(b10-b11),
    #           M4=a11(b01-b00), M5=(a00+a01)(b11), M6=(a10-a00)(b00+b10), M7=(a01-a11)(b01+b11)
    # c00 = M1+M4-M5+M7, c01=M3+M5, c10=M2+M4, c11=M1-M2+M3+M6
    #
    # In tensor form, each term m has U_m[i,j], V_m[j,k], W_m[k,i] such that
    # the contribution to T[i,j,k] from term m is U_m[i,j]*V_m[j,k]*W_m[k,i]
    #
    # For term m, U_m[i,j]*V_m[j,k] = M_m (the Strassen product), and W_m[k,i] 
    # determines how M_m contributes to each c_ik.
    #
    # M1 = (a00+a11)(b00+b11): U1[0,0]=1,U1[1,1]=1; V1[0,0]=1,V1[1,1]=1
    #   M1 contributes +1 to c00 and +1 to c11, 0 to c01,c10
    #   So W1[0,0]=1, W1[1,1]=1, W1[0,1]=0, W1[1,0]=0
    #   But wait: M1 = a00*b00 + a00*b11 + a11*b00 + a11*b11
    #   a00*b00 -> c00, a00*b11 -> c01, a11*b00 -> c10, a11*b11 -> c11
    #   In the tensor sum, T[0,0,0] gets U1[0,0]*V1[0,0]*W1[0,0] = 1*1*W1[0,0]
    #   T[0,0,1] gets U1[0,0]*V1[0,1]*W1[1,0] = 1*0*W1[1,0] = 0 (V1[0,1]=0)
    #   T[0,1,0] gets U1[0,1]*V1[1,0]*W1[0,0] = 0*0*W1[0,0] = 0
    #   T[0,1,1] gets U1[0,1]*V1[1,1]*W1[1,0] = 0*1*W1[1,0] = 0
    #   T[1,0,0] gets U1[1,0]*V1[0,0]*W1[0,1] = 0*1*W1[0,1] = 0
    #   T[1,0,1] gets U1[1,0]*V1[0,1]*W1[1,1] = 0*0*W1[1,1] = 0
    #   T[1,1,0] gets U1[1,1]*V1[1,0]*W1[0,1] = 1*0*W1[0,1] = 0
    #   T[1,1,1] gets U1[1,1]*V1[1,1]*W1[1,1] = 1*1*W1[1,1]
    #   So term 1 contributes: T[0,0,0]+=W1[0,0], T[1,1,1]+=W1[1,1]
    #   For this to match Strassen where M1 contributes to c00 and c11:
    #   W1[0,0]=1, W1[1,1]=1
    #
    # But we also need T[0,0,1]=1, T[0,1,0]=1, T[0,1,1]=1, T[1,0,0]=1, T[1,0,1]=1, T[1,1,0]=1
    # These must come from other terms.
    
    # Let me just enumerate all 7 Strassen terms properly.
    # Each term m defines U_m (2x2), V_m (2x2), W_m (2x2) as sparse matrices.
    # The key: U_m[i,j]*V_m[j,k] should equal the coefficient of a_ij*b_jk in M_m.
    
    # M1 = (a00+a11)(b00+b11) = a00*b00 + a00*b11 + a11*b00 + a11*b11
    #   U1: U1[0,0]=1, U1[1,1]=1 (all else 0)
    #   V1: V1[0,0]=1, V1[1,1]=1 (all else 0)
    #   Check: U1[i,j]*V1[j,k] is nonzero when i=j=0,k=0 (val=1), i=0,j=0,k=1? No, V1[0,1]=0.
    #   Actually U1[0,0]*V1[0,0]=1 (i=0,j=0,k=0), U1[0,0]*V1[0,1]=0, 
    #   U1[1,1]*V1[1,0]=0, U1[1,1]*V1[1,1]=1 (i=1,j=1,k=1)
    #   So M1 only produces a00*b00 and a11*b11, not the cross terms!
    #   This is the problem - a rank-1 tensor U⊗V⊗W only captures products where
    #   the same (i,j) and (j,k) pairs appear.
    
    # I see the issue now. The Strassen algorithm works because M1 = (a00+a11)(b00+b11)
    # is a SINGLE scalar multiplication, not a tensor product. In the tensor decomposition,
    # we need to represent the full 2x2x2 tensor (all 8 entries = 1) as a sum of 7
    # rank-1 tensors. The Strassen algorithm tells us HOW to compute the matrix product,
    # but the tensor decomposition is a different mathematical object.
    
    # Let me think about this differently. The 2x2x2 matrix multiplication tensor
    # T[i,j,k]=1 for all i,j,k in {0,1} needs to be decomposed into 7 rank-1 tensors.
    # This is a known result. Let me find the correct decomposition.
    
    # From the theory of tensor rank, the 2x2x2 matrix multiplication tensor has
    # rank 7. One valid decomposition can be found by solving the system of equations.
    
    # Let me use a known correct decomposition. After careful derivation:
    # The 7 terms are (U_m, V_m, W_m) where each is a 2x2 matrix (flattened to 4 entries).
    
    # I'll use the following verified decomposition:
    # Term 1: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]  (diagonal)
    # Term 2: U=[0,1,0,1], V=[1,0,0,0], W=[0,1,0,0]
    # Term 3: U=[1,0,0,0], V=[0,1,-1,0], W=[0,0,1,0]
    # Term 4: U=[0,0,0,1], V=[0,1,-1,0], W=[0,0,0,1]
    # Term 5: U=[1,1,0,0], V=[0,0,0,1], W=[1,0,0,0]
    # Term 6: U=[-1,0,1,0], V=[1,0,1,0], W=[0,0,0,1]
    # Term 7: U=[0,1,0,-1], V=[0,1,0,1], W=[1,0,0,0]
    
    # Let me verify T[0,0,0] (i=0,j=0,k=0):
    # Term 1: U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1
    # Term 2: U[0,0]*V[0,0]*W[1,0] = 0*1*0 = 0
    # Term 3: U[0,0]*V[0,0]*W[1,0] = 1*0*0 = 0
    # Term 4: U[0,0]*V[0,0]*W[0,1] = 0*0*0 = 0
    # Term 5: U[0,0]*V[0,0]*W[0,0] = 1*0*1 = 0
    # Term 6: U[0,0]*V[0,0]*W[0,1] = -1*1*0 = 0
    # Term 7: U[0,0]*V[0,0]*W[0,0] = 0*0*1 = 0
    # Sum = 1 ✓
    
    # T[0,0,1] (i=0,j=0,k=1):
    # Term 1: U[0,0]*V[0,1]*W[1,0] = 1*0*0 = 0
    # Term 2: U[0,0]*V[0,1]*W[1,0] = 0*0*0 = 0
    # Term 3: U[0,0]*V[0,1]*W[1,0] = 1*1*0 = 0
    # Term 4: U[0,0]*V[0,1]*W[0,1] = 0*1*0 = 0
    # Term 5: U[0,0]*V[0,1]*W[0,0] = 1*0*1 = 0
    # Term 6: U[0,0]*V[0,1]*W[0,1] = -1*0*0 = 0
    # Term 7: U[0,0]*V[0,1]*W[0,0] = 0*1*1 = 0
    # Sum = 0 ✗ Should be 1!
    
    # My decomposition is wrong. Let me try a completely different approach.
    # I'll directly construct the decomposition using the Strassen algorithm structure.
    
    # In Strassen, we compute:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)  
    # M3 = (a00)(b10-b11)
    # M4 = (a11)(b01-b00)
    # M5 = (a00+a01)(b11)
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    # c00 = M1+M4-M5+M7
    # c01 = M3+M5
    # c10 = M2+M4
    # c11 = M1-M2+M3+M6
    
    # The tensor T[i,j,k] = 1 for all i,j,k means:
    # c00 = a00*b00 + a01*b10 = 1 (as a bilinear form)
    # c01 = a00*b01 + a01*b11 = 1
    # c10 = a10*b00 + a11*b10 = 1
    # c11 = a10*b01 + a11*b11 = 1
    
    # In the tensor decomposition T[i,j,k] = sum_m U_m[i,j] * V_m[j,k] * W_m[k,i]:
    # The coefficient of a_ij*b_jk in c_ki is U_m[i,j]*V_m[j,k]*W_m[k,i] summed over m.
    # For T[i,j,k]=1, we need: for each (i,j,k), sum_m U_m[i,j]*V_m[j,k]*W_m[k,i] = 1.
    
    # This is equivalent to saying the 3-way tensor T has entries all 1, and we're
    # finding its rank-1 decomposition.
    
    # Let me use a systematic approach: set up the 7x(2*2*2) system and solve.
    # Actually, let me just hardcode a known correct Strassen decomposition.
    
    # After research, the correct Strassen decomposition of the 2x2x2 tensor is:
    # Using 0-indexed (i,j) for U, (j,k) for V, (k,i) for W:
    
    # I'll verify each term against all 8 tensor entries.
    # Let me define the 7 terms as tuples of (U_entries, V_entries, W_entries)
    # where entries are dicts mapping (row,col) -> value.
    
    strassen_terms = [
        # Term 1: (a00+a11)(b00+b11) -> contributes to c00 and c11
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        # Term 2: (a10+a11)(b00) -> contributes to c10
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(0,1):1}),
        # Term 3: a00*(b10-b11) -> contributes to c01
        ({(0,0):1}, {(1,0):1,(1,1):-1}, {(1,0):1}),
        # Term 4: a11*(b01-b00) -> contributes to c00 and c10
        ({(1,1):1}, {(0,1):1,(0,0):-1}, {(0,1):1,(1,0):1}),
        # Term 5: (a00+a01)(b11) -> contributes to c01 and c00
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(1,0):1,(0,0):-1}),
        # Term 6: (a10-a00)(b00+b10) -> contributes to c11
        ({(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}, {(1,1):1}),
        # Term 7: (a01-a11)(b01+b11) -> contributes to c00 and c11
        ({(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}, {(0,0):1,(1,1):1}),
    ]
    
    # Verify all 8 entries
    for i in range(2):
        for j in range(2):
            for k in range(2):
                total = 0
                for U_dict, V_dict, W_dict in strassen_terms:
                    u_val = U_dict.get((i,j), 0)
                    v_val = V_dict.get((j,k), 0)
                    w_val = W_dict.get((k,i), 0)
                    total += u_val * v_val * w_val
                if total != 1:
                    print(f"ERROR: T[{i},{j},{k}] = {total}, expected 1")
                    # Fall back to schoolbook
                    rank = n * m * p
                    U = [[0] * rank for _ in range(nm)]
                    V = [[0] * rank for _ in range(mp)]
                    W = [[0] * rank for _ in range(pn)]
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U[ii * m + jj][col] = 1
                                V[jj * p + kk][col] = 1
                                W[kk * n + ii][col] = 1
                                col += 1
                    return {"U": U, "V": V, "W": W}
    
    # Build the full 5x5x5 decomposition using 2+2+1 block partition
    # Blocks: 0={0,1}, 1={2,3}, 2={4}
    block_sizes = [2, 2, 1]
    
    U = [[0] * 117 for _ in range(nm)]
    V = [[0] * 117 for _ in range(mp)]
    W = [[0] * 117 for _ in range(pn)]
    col = 0
    
    for bi in range(3):  # A-row block
        for bj in range(3):  # B-col block (same as A-col)
            for bk in range(3):  # C-col block
                # Sub-tensor: A-indices in block bi, B-indices in block bj, C-indices in block bk
                # Size: block_sizes[bi] x block_sizes[bj] x block_sizes[bk]
                
                i_idx = block_indices(bi)
                j_idx = block_indices(bj)
                k_idx = block_indices(bk)
                
                if block_sizes[bi] == 2 and block_sizes[bj] == 2 and block_sizes[bk] == 2:
                    # Use Strassen (7 terms)
                    for U_dict, V_dict, W_dict in strassen_terms:
                        for (li, lj), u_val in U_dict.items():
                            gi = i_idx[li]
                            gj = j_idx[lj]
                            U[gi * m + gj][col] += u_val
                        for (lj, lk), v_val in V_dict.items():
                            gj = j_idx[lj]
                            gk = k_idx[lk]
                            V[gj * p + gk][col] += v_val
                        for (lk, li), w_val in W_dict.items():
                            gk = k_idx[lk]
                            gi = i_idx[li]
                            W[gk * n + gi][col] += w_val
                        col += 1
                else:
                    # Schoolbook for this sub-block
                    for li in range(block_sizes[bi]):
                        for lj in range(block_sizes[bj]):
                            for lk in range(block_sizes[bk]):
                                gi = i_idx[li]
                                gj = j_idx[lj]
                                gk = k_idx[lk]
                                U[gi * m + gj][col] = 1
                                V[gj * p + gk][col] = 1
                                W[gk * n + gi][col] = 1
                                col += 1
    
    # col should be 117
    if col != 117:
        print(f"Warning: expected 117 columns, got {col}")
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    if not _verify(result["U"], result["V"], result["W"]):
        print("Verification failed, falling back to schoolbook")
        rank = n * m * p
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
        result = {"U": U, "V": V, "W": W}
    
    return result
# EVOLVE-BLOCK-END
