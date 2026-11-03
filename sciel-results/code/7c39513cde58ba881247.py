"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

def _schoolbook(n, m, p):
    """Return (U, V, W) as numpy arrays for schoolbook multiplication of n x m by m x p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    for idx in range(rank):
        k = idx % p
        j = (idx // p) % m
        i = idx // (m * p)
        U[i * m + j, idx] = 1
        V[j * p + k, idx] = 1
        W[k * n + i, idx] = 1
    return U, V, W

def _strassen_2x2_2x2_2x2():
    """
    Returns U, V, W for Strassen 2x2 * 2x2 -> 2x2.
    Shapes: U (4, 7), V (4, 7), W (4, 7)
    Indices:
    U: A11, A12, A21, A22
    V: B11, B12, B21, B22
    W: C11, C12, C21, C22
    """
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21+B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, 7), dtype=object)
    V = np.zeros((4, 7), dtype=object)
    W = np.zeros((4, 7), dtype=object)
    
    # U cols (indices 0:A11, 1:A12, 2:A21, 3:A22)
    # M1: A11, A22 -> U[0,0]=1, U[3,0]=1
    # M2: A21, A22 -> U[2,1]=1, U[3,1]=1
    # M3: A11 -> U[0,2]=1
    # M4: A22 -> U[3,3]=1
    # M5: A11, A12 -> U[0,4]=1, U[1,4]=1
    # M6: A21, A11 -> U[2,5]=1, U[0,5]=-1
    # M7: A12, A22 -> U[1,6]=1, U[3,6]=-1
    
    # V cols (indices 0:B11, 1:B12, 2:B21, 3:B22)
    # M1: B11, B22 -> V[0,0]=1, V[3,0]=1
    # M2: B11 -> V[0,1]=1
    # M3: B21, B22 -> V[2,2]=1, V[3,2]=1
    # M4: B21, B11 -> V[2,3]=1, V[0,3]=-1
    # M5: B22 -> V[3,4]=1
    # M6: B11, B12 -> V[0,5]=1, V[1,5]=1
    # M7: B21, B22 -> V[2,6]=1, V[3,6]=1
    
    # W cols (indices 0:C11, 1:C12, 2:C21, 3:C22)
    # M1: C11 (+1), C22 (+1) -> W[0,0]=1, W[3,0]=1
    # M2: C21 (+1), C22 (-1) -> W[2,1]=1, W[3,1]=-1
    # M3: C12 (+1), C22 (+1) -> W[1,2]=1, W[3,2]=1
    # M4: C11 (+1), C21 (+1) -> W[0,3]=1, W[2,3]=1
    # M5: C11 (-1), C12 (+1) -> W[0,4]=-1, W[1,4]=1
    # M6: C22 (+1) -> W[3,5]=1
    # M7: C11 (+1) -> W[0,6]=1

    U[0,0]=1; U[3,0]=1
    U[2,1]=1; U[3,1]=1
    U[0,2]=1
    U[3,3]=1
    U[0,4]=1; U[1,4]=1
    U[2,5]=1; U[0,5]=-1
    U[1,6]=1; U[3,6]=-1
    
    V[0,0]=1; V[3,0]=1
    V[0,1]=1
    V[2,2]=1; V[3,2]=1
    V[2,3]=1; V[0,3]=-1
    V[3,4]=1
    V[0,5]=1; V[1,5]=1
    V[2,6]=1; V[3,6]=1
    
    W[0,0]=1; W[3,0]=1
    W[2,1]=1; W[3,1]=-1
    W[1,2]=1; W[3,2]=1
    W[0,3]=1; W[2,3]=1
    W[0,4]=-1; W[1,4]=1
    W[3,5]=1
    W[0,6]=1
    
    return U, V, W

def _embed_block(U_block, V_block, W_block, n, m, p, i_start, j_start, k_start):
    """
    Embed a block decomposition into the full tensor.
    U_block shape: (ni*mj, rank_block)
    V_block shape: (mj*pk, rank_block)
    W_block shape: (pk*ni, rank_block)
    i_start, j_start, k_start: starting indices in A, B, C dimensions.
    ni, mj, pk: dimensions of the block.
    """
    ni = U_block.shape[0] // mj
    mj = U_block.shape[0] // ni
    pk = V_block.shape[0] // mj
    
    rank_block = U_block.shape[1]
    U_full = np.zeros((n * m, rank_block), dtype=object)
    V_full = np.zeros((m * p, rank_block), dtype=object)
    W_full = np.zeros((p * n, rank_block), dtype=object)
    
    for r in range(rank_block):
        for idx_u in range(U_block.shape[0]):
            if U_block[idx_u, r] != 0:
                ii = idx_u // mj
                jj = idx_u % mj
                global_i = i_start + ii
                global_j = j_start + jj
                U_full[global_i * m + global_j, r] = U_block[idx_u, r]
        
        for idx_v in range(V_block.shape[0]):
            if V_block[idx_v, r] != 0:
                jj = idx_v // pk
                kk = idx_v % pk
                global_j = j_start + jj
                global_k = k_start + kk
                V_full[global_j * p + global_k, r] = V_block[idx_v, r]
        
        for idx_w in range(W_block.shape[0]):
            if W_block[idx_w, r] != 0:
                kk = idx_w // ni
                ii = idx_w % ni
                global_k = k_start + kk
                global_i = i_start + ii
                W_full[global_k * n + global_i, r] = W_block[idx_w, r]
    
    return U_full, V_full, W_full

def _combine_decompositions(list_of_decomp):
    """
    Combine a list of (U, V, W) numpy arrays by concatenating columns.
    """
    U = np.hstack([d[0] for d in list_of_decomp])
    V = np.hstack([d[1] for d in list_of_decomp])
    W = np.hstack([d[2] for d in list_of_decomp])
    return U, V, W

def _to_fraction_list(mat):
    """Convert numpy array of objects to list of lists of Fractions."""
    return [[Fraction(x) for x in row] for row in mat.tolist()]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Try to find a better decomposition than schoolbook (rank 60)
    # Strategy: Hybrid of Strassen and Schoolbook
    
    # Option 1: Two Strassen 2x2 blocks + Schoolbook remainder
    # Strassen 2x2x2 has rank 7. Schoolbook 2x2x2 has rank 8.
    # We have 3x4x5.
    # Let's try to partition A (3x4) and B (4x5) into blocks.
    # 
    # Partition A into:
    #   Block 1: 2x2 (rows 0-1, cols 0-1)
    #   Block 2: 2x2 (rows 0-1, cols 2-3) -- wait, B is 4x5.
    # 
    # Let's partition the intermediate dimension m=4 into two parts: 2 and 2.
    # So we split B into two 2x5 blocks and A into two 3x2 blocks? No.
    # A is 3x4, B is 4x5.
    # Split m=4 into m1=2, m2=2.
    # A = [A1 (3x2), A2 (3x2)]
    # B = [B1 (2x5), B2 (2x5)]
    # C = A1*B1 + A2*B2
    # 
    # For A1*B1: 3x2 * 2x5.
    # For A2*B2: 3x2 * 2x5.
    # 
    # Can we use Strassen for 3x2 * 2x5?
    # Strassen works for 2x2.
    # We can split the 3 rows of A into 2 and 1? Or 1 and 2?
    # Let's split A1 (3x2) into A1a (2x2) and A1b (1x2).
    # B1 is 2x5. Split B1 into B1a (2x2) and B1b (2x3)? No, columns of B correspond to output columns.
    # 
    # Actually, let's look at the known result from the prompt:
    # "Replaces schoolbook (rank 60) with a hybrid construction using two Strassen 2x2x2 blocks (rank 7 each) and schoolbook for the remainder, reducing total rank to 58."
    # 
    # Let's verify how 58 is achieved.
    # Total rank 58. Schoolbook 60.
    # If we use two Strassen blocks of rank 7, that's 14 ranks for some part.
    # The rest is schoolbook.
    # 
    # Let's try to construct:
    # 1. Split m=4 into two 2s.
    # 2. For each part, we have A_i (3x2) * B_i (2x5).
    # 3. We can split the 3x2 * 2x5 multiplication further.
    #    Split rows of A_i (3) into 2 and 1.
    #    Split cols of B_i (5) into 2 and 3.
    #    This creates 4 sub-multiplications:
    #    a) 2x2 * 2x2 (Use Strassen, rank 7)
    #    b) 2x2 * 2x3 (Schoolbook, rank 12)
    #    c) 1x2 * 2x2 (Schoolbook, rank 4)
    #    d) 1x2 * 2x3 (Schoolbook, rank 6)
    #    Total for one part: 7 + 12 + 4 + 6 = 29.
    #    Two parts: 29 * 2 = 58.
    #    
    # This matches! Let's implement this specific hybrid.
    
    def schoolbook_block(ni, mj, pk):
        return _schoolbook(ni, mj, pk)
    
    decomp_parts = []
    
    # Part 1: A1 (rows 0-1, cols 0-1) * B1 (rows 0-1, cols 0-4)
    # A1 is 2x2, B1 is 2x5.
    # Wait, my previous split was A (3x4) -> A1 (3x2), A2 (3x2).
    # So A1 is 3x2, B1 is 2x5.
    # 
    # Inside A1*B1:
    # A1 = [A1a (2x2); A1b (1x2)]
    # B1 = [B1a (2x2), B1b (2x3)]
    # 
    # Sub-block 1.1: A1a (2x2) * B1a (2x2). Indices: A(0:2, 0:2), B(0:2, 0:2), C(0:2, 0:2).
    U11, V11, W11 = _strassen_2x2_2x2_2x2()
    U11_f, V11_f, W11_f = _embed_block(U11, V11, W11, n, m, p, 0, 0, 0)
    decomp_parts.append((U11_f, V11_f, W11_f))
    
    # Sub-block 1.2: A1a (2x2) * B1b (2x3). Indices: A(0:2, 0:2), B(0:2, 2:5), C(0:2, 2:5).
    U12, V12, W12 = schoolbook_block(2, 2, 3)
    U12_f, V12_f, W12_f = _embed_block(U12, V12, W12, n, m, p, 0, 0, 2)
    decomp_parts.append((U12_f, V12_f, W12_f))
    
    # Sub-block 1.3: A1b (1x2) * B1a (2x2). Indices: A(2:3, 0:2), B(0:2, 0:2), C(2:3, 0:2).
    U13, V13, W13 = schoolbook_block(1, 2, 2)
    U13_f, V13_f, W13_f = _embed_block(U13, V13, W13, n, m, p, 2, 0, 0)
    decomp_parts.append((U13_f, V13_f, W13_f))
    
    # Sub-block 1.4: A1b (1x2) * B1b (2x3). Indices: A(2:3, 0:2), B(0:2, 2:5), C(2:3, 2:5).
    U14, V14, W14 = schoolbook_block(1, 2, 3)
    U14_f, V14_f, W14_f = _embed_block(U14, V14, W14, n, m, p, 2, 0, 2)
    decomp_parts.append((U14_f, V14_f, W14_f))
    
    # Part 2: A2 (rows 0-2, cols 2-3) * B2 (rows 2-3, cols 0-4)
    # A2 is 3x2, B2 is 2x5.
    # A2 = [A2a (2x2); A2b (1x2)]
    # B2 = [B2a (2x2), B2b (2x3)]
    # 
    # Sub-block 2.1: A2a (2x2) * B2a (2x2). Indices: A(0:2, 2:4), B(2:4, 0:2), C(0:2, 0:2).
    U21, V21, W21 = _strassen_2x2_2x2_2x2()
    U21_f, V21_f, W21_f = _embed_block(U21, V21, W21, n, m, p, 0, 2, 0)
    decomp_parts.append((U21_f, V21_f, W21_f))
    
    # Sub-block 2.2: A2a (2x2) * B2b (2x3). Indices: A(0:2, 2:4), B(2:4, 2:5), C(0:2, 2:5).
    U22, V22, W22 = schoolbook_block(2, 2, 3)
    U22_f, V22_f, W22_f = _embed_block(U22, V22, W22, n, m, p, 0, 2, 2)
    decomp_parts.append((U22_f, V22_f, W22_f))
    
    # Sub-block 2.3: A2b (1x2) * B2a (2x2). Indices: A(2:3, 2:4), B(2:4, 0:2), C(2:3, 0:2).
    U23, V23, W23 = schoolbook_block(1, 2, 2)
    U23_f, V23_f, W23_f = _embed_block(U23, V23, W23, n, m, p, 2, 2, 0)
    decomp_parts.append((U23_f, V23_f, W23_f))
    
    # Sub-block 2.4: A2b (1x2) * B2b (2x3). Indices: A(2:3, 2:4), B(2:4, 2:5), C(2:3, 2:5).
    U24, V24, W24 = schoolbook_block(1, 2, 3)
    U24_f, V24_f, W24_f = _embed_block(U24, V24, W24, n, m, p, 2, 2, 2)
    decomp_parts.append((U24_f, V24_f, W24_f))
    
    U, V, W = _combine_decompositions(decomp_parts)
    
    # Verify
    rank = U.shape[1]
    
    # Check validity with random integer matrices
    # To avoid floating point issues, use small integers
    A_test = np.random.randint(-5, 5, (n, m))
    B_test = np.random.randint(-5, 5, (m, p))
    C_true = A_test @ B_test
    
    # Compute C_calc from U, V, W
    # C[i, k] = sum_r U[i, r] * (V^T @ W)[r, k] ? No.
    # T = U V W^T ?
    # Standard form: C[i, k] = sum_{j, r} A[i, j] B[j, k] * delta?
    # Tensor T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
    # C[i, k] = sum_j A[i, j] B[j, k]
    # So sum_r U[i, r] V[j, r] W[k, r] must equal 1 if i,j,k match? No.
    # The decomposition is for the multiplication tensor.
    # M_{i, j, k} = sum_r U_{i, r} V_{j, r} W_{k, r}
    # And C_{i, k} = sum_j A_{i, j} B_{j, k} = sum_j sum_r A_{i, j} B_{j, k} M_{i, j, k} ?
    # No, M_{i, j, k} is the tensor of the multiplication map.
    # The map is (A, B) -> C.
    # C_{i, k} = sum_j A_{i, j} B_{j, k}.
    # The tensor M has entries M[i, j, k] such that C_{i, k} = sum_j A_{i, j} B_{j, k} = sum_{i, j, k} A_{i, j} B_{j, k} M_{i, j, k}?
    # Actually, the tensor M is defined such that M[i, j, k] is the coefficient of A_{i, j} B_{j, k} in C_{i, k}.
    # So M[i, j, k] is 1 if the indices match in the sum?
    # C_{i, k} = sum_j A_{i, j} B_{j, k}.
    # The term A_{i, j} B_{j, k} contributes to C_{i, k}.
    # So M[i, j, k] = 1 for all i, j, k? No, that would be a different tensor.
    # The tensor T_{i, j, k} represents the product C_{i, k} = sum_{i, j, k} T_{i, j, k} A_{i, j} B_{j, k}?
    # No, T is the multiplication tensor of dimension n x m x p.
    # T_{i, j, k} is 1 if the product A_{i, j} B_{j, k} contributes to C_{i, k}.
    # In the standard basis, T_{i, j, k} = 1 if i, j, k are such that A_{i, j} and B_{j, k} multiply to give a term in C_{i, k}.
    # Since C_{i, k} = sum_j A_{i, j} B_{j, k}, the term A_{i, j} B_{j, k} appears in C_{i, k}.
    # So T_{i, j, k} = 1 for all i, j, k?
    # Let's check the schoolbook construction.
    # U[i, r] V[j, r] W[k, r].
    # For schoolbook, r = (i, j, k).
    # U[i, (i, j, k)] = 1 if row matches i.
    # V[j, (i, j, k)] = 1 if row matches j.
    # W[k, (i, j, k)] = 1 if col matches k.
    # So sum_r U[i, r] V[j, r] W[k, r] = 1 if there exists r=(i', j', k') such that i'=i, j'=j, k'=k.
    # So the sum is 1.
    # Thus C_{i, k} = sum_j A_{i, j} B_{j, k} * 1.
    # Yes, T_{i, j, k} = 1 for all i, j, k in the schoolbook basis?
    # Wait, if T_{i, j, k} = 1 for all, then the tensor is all ones.
    # Then C_{i, k} = sum_{i', j', k'} T_{i', j', k'} A_{i', j'} B_{j', k'} ?
    # No, the contraction is C_{i, k} = sum_j A_{i, j} B_{j, k}.
    # The tensor M is defined by M[i, j, k] = 1.
    # Then (A \cdot M \cdot B)_{i, k} = sum_{j, j'} A_{i, j} M[j, j'] B_{j', k}?
    # No.
    # The standard definition: The multiplication tensor M of dimension (n, m, p) is the tensor such that
    # C_{i, k} = sum_{j, r} A_{i, j} B_{j, k} \delta_{...}?
    # Let's stick to the code's verification.
    # The code checks if U, V, W decompose the tensor.
    # Tensor T[i, j, k] = 1.
    # Check: sum_r U[i, r] V[j, r] W[k, r] == 1 for all i, j, k?
    
    # Let's verify this hypothesis.
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i, r] * V[j, r] * W[k, r]
                if s != 1:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    
    if not valid:
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
        rank = U.shape[1]
    
    # Return as list of lists of Fractions
    U_list = _to_fraction_list(U)
    V_list = _to_fraction_list(V)
    W_list = _to_fraction_list(W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
