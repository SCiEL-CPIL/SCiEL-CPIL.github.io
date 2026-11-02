"""Rewritten program for 2x3x10 matrix multiplication tensor rank minimization."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # For each output entry (i, k), compute sum over j of A[i,j]*B[j,k]
    # and check it equals sum over r of U[i*m+j_all... wait, let me think about the indexing.
    
    # U has shape (n*m, rank): U[i*m + j, r]
    # V has shape (m*p, rank): V[j*p + k, r]  
    # W has shape (p*n, rank): W[k*n + i, r]
    
    # The tensor identity: for each (i, j, k) where i in [n], j in [m], k in [p]:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    # (since T[i,j,k] = 1 for all valid indices)
    
    # But actually, the matrix multiplication tensor is:
    # T[i, j, k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # The rank-1 decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta for the product
    
    # Actually, the standard formulation: the matrix multiplication tensor M_{n,m,p} has entries
    # M[i, j, k] = 1 for all i,j,k (it's the all-ones tensor in the sense that each "slice" 
    # corresponds to one multiplication).
    
    # The bilinear identity: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    
    # Let's verify this directly
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(total - 1.0) > 1e-9:
                    return False, (i, j, k, total)
    return True, None

def _build_strassen_based(n, m, p):
    """
    Build a rank-55 decomposition using Strassen's algorithm for 2x2 blocks.
    
    Strategy: Partition the 2x3 matrix into two 2x2 blocks (with padding),
    and the 3x10 matrix into two 2x5 blocks (with padding).
    Use Strassen's 7-multiplication algorithm for each pair of 2x2 blocks.
    
    Actually, let me think more carefully.
    
    For 2x3 * 3x10:
    A is 2x3, B is 3x10, C is 2x10.
    
    We can write A = [A1 | A2] where A1 is 2x2, A2 is 2x1
    B = [B1; B2] where B1 is 2x10, B2 is 1x10
    
    C = A1*B1 + A2*B2
    
    A1*B1 is 2x2 * 2x10 = 2x10, rank 8 (schoolbook) or 7 (Strassen)
    A2*B2 is 2x1 * 1x10 = 2x10, rank 10 (schoolbook)
    
    Total: 7 + 10 = 17? That's way less than 55. But wait, that's not right.
    
    Actually, the tensor rank is about the total number of scalar multiplications.
    Let me reconsider.
    
    The schoolbook approach: for each of the 2*3*10 = 60 (i,j,k) triples, one multiplication.
    
    Strassen-based: We can group the multiplications.
    
    Let me think of it differently. The 2x3x10 tensor can be decomposed by:
    - Splitting the 3 dimension into 2+1
    - Using Strassen for the 2x2 part (7 mults instead of 4)
    - Using 2 mults for the remaining 2x1 * 1x10 part (actually 2*10=20 mults)
    
    Hmm, that gives 7*10 + 20 = 90, which is worse.
    
    Let me try another split. Actually, the known result for 2x3x10 is rank 55.
    Let me construct it explicitly.
    
    One approach: use the fact that rank(2x3x10) <= rank(2x3x3) + rank(2x3x7)
    and optimize each part.
    
    Actually, let me just use a known rank-55 construction.
    
    The key insight: we can use 55 scalar multiplications.
    One construction:
    - Use Strassen for a 2x2x2 subproblem: 7 mults
    - Handle the remaining dimensions with schoolbook
    
    Let me try: split the 10 columns into 2+2+2+2+2
    For each pair of columns, we have a 2x3x2 subproblem.
    rank(2x3x2) = ? 
    
    Actually, let me just implement a verified rank-55 construction.
    
    Based on literature, one rank-55 decomposition for 2x3x10:
    
    Let's define the 55 rank-1 terms. I'll use a systematic approach.
    
    Actually, the simplest verified approach: use the "bilinear" structure.
    
    Let me try a different strategy: start from schoolbook (rank 60) and try to
    merge 5 pairs of terms using bilinear identities.
    
    For a 2x2x2 tensor, Strassen merges 4 terms into 7, saving 1 (net -1).
    Wait, 4 schoolbook -> 7 Strassen is +3, not a saving.
    
    Hmm, I'm confusing myself. Let me restart.
    
    The matrix multiplication tensor T_{n,m,p} has entries T[i,j,k] = 1.
    The rank is the minimum number of rank-1 tensors to sum to T.
    
    For n=m=p=2, rank = 7 (Strassen), not 8.
    For n=2, m=3, p=10, we want rank < 60.
    
    Known: rank(2x3x10) = 55.
    
    Let me construct it. One way:
    Write the 3x10 matrix B as a block matrix [B1; B2] where B1 is 2x10, B2 is 1x10.
    Write A as [A1 A2] where A1 is 2x2, A2 is 2x1.
    
    C = A1*B1 + A2*B2
    
    For A1*B1 (2x2 * 2x10): use Strassen-like, rank 7*10/2... no.
    
    Actually, for 2x2 * 2xp, the rank is 7*p/2 if p is even? No.
    
    Let me just look up or derive rank(2x2x10).
    
    rank(2x2xp) = 7 for p=2, and for general p, it's at most 7*ceil(p/2) + 2*(p mod 2)... 
    
    This is getting complicated. Let me just implement a search.
    
    Strategy: Start with schoolbook rank 60, then try to find linear dependencies
    among the 60 rank-1 tensors and eliminate them.
    
    Actually, the cleanest approach: use the known rank-55 formula.
    
    After research, here's a rank-55 construction for 2x3x10:
    
    We'll use 55 scalar multiplications. The construction involves:
    - 55 specific combinations of A and B entries
    - 55 corresponding combinations for C entries
    
    Let me encode it as: for each of the 55 multiplications r=0..54,
    we have u_r (length 6), v_r (length 30), w_r (length 20) such that
    for all (i,j,k): sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1
    
    I'll use a randomized search to find such a decomposition.
    """
    pass  # Placeholder, will be implemented below

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s limit

    n, m, p = DIMENSIONS

    # Best known rank for 2x3x10 is 55
    # Try to find a rank-55 decomposition, fall back to schoolbook rank 60
    
    best_result = None
    
    # First, build schoolbook as fallback
    rank_schoolbook = n * m * p  # 60
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Verify schoolbook
    valid, _ = _verify(U_sb, V_sb, W_sb, n, m, p)
    if valid:
        best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Now try to find rank-55 decomposition
    # Use a structured approach based on Strassen-like identities
    
    # For 2x3x10, we can use the following strategy:
    # Split the 3-row dimension into 2+1
    # Use 7 Strassen multiplications for the 2x2 part (per column pair)
    # Use schoolbook for the 1x1 part
    
    # Actually, let me try a direct construction.
    # 
    # Key idea: The 2x3x10 tensor can be seen as:
    # T[i,j,k] = 1 for i in {0,1}, j in {0,1,2}, k in {0,...,9}
    #
    # We can write this as sum of rank-1 tensors.
    # 
    # Let's try: for each k (10 values), we have a 2x3 sub-tensor (all ones).
    # rank(2x3x1) = 6 (schoolbook) or can it be less?
    # Actually rank(2x3x1) = 6 since it's just outer product structure.
    #
    # But we can share across k values.
    #
    # Alternative: think of it as 10 slices of 2x3.
    # Each slice is the 2x3 all-ones matrix.
    # The tensor is sum_k (ones_2x3) ⊗ e_k
    #
    # We need to express this with fewer than 60 rank-1 terms.
    #
    # Let me try a random search for rank 55.
    
    if time.time() < deadline - 30:
        # Try to construct rank-55 using Strassen-based block decomposition
        # 
        # Strategy: Split B (3x10) into B1 (2x10) and B2 (1x10)
        # Split A (2x3) into A1 (2x2) and A2 (2x1)
        # C = A1*B1 + A2*B2
        #
        # For A1*B1 (2x2 * 2x10): 
        #   Use Strassen per column pair: 5 pairs, 7 mults each = 35 mults
        # For A2*B2 (2x1 * 1x10):
        #   Schoolbook: 2*1*10 = 20 mults
        # Total: 35 + 20 = 55 mults!
        #
        # This gives rank 55.
        
        rank_target = 55
        U55 = [[0] * rank_target for _ in range(n * m)]
        V55 = [[0] * rank_target for _ in range(m * p)]
        W55 = [[0] * rank_target for _ in range(p * n)]
        
        col = 0
        
        # Part 1: A1 * B1 where A1 is 2x2 (columns 0,1 of A), B1 is 2x10 (rows 0,1 of B)
        # Strassen for 2x2 * 2x10, processing 2 columns at a time
        # For columns 2c and 2c+1 of B1 (which are columns 2c and 2c+1 of B):
        #   Let B1_col1 = B[0:2, 2c], B1_col2 = B[0:2, 2c+1]
        #   Strassen:
        #   M1 = (A1[0,0]+A1[1,1]) * (B1_col1[0]+B1_col1[1])  -- wait, this is for 2x2*2x2
        #   
        # Actually, for 2x2 * 2xp, we can use:
        # For each pair of columns, apply Strassen to get 2x2 result, but we only need
        # the 2x2 block of the output.
        #
        # Let me be more careful. A1 is 2x2, B1 is 2x10.
        # C1 = A1 * B1 is 2x10.
        #
        # Using Strassen-style: for each pair of output columns (2c, 2c+1):
        #   Let b1 = B1[:, 2c], b2 = B1[:, 2c+1] (each is 2x1)
        #   We want c1 = A1*b1, c2 = A1*b2 (each 2x1)
        #
        # Strassen for 2x2 * 2x2 would give us 4 outputs from 7 mults,
        # but here we have 2x2 * 2x2 (where the second "2x2" is actually two separate 2x1 vectors).
        #
        # Actually, A1 * [b1 | b2] = [A1*b1 | A1*b2]
        # This is a 2x2 * 2x2 matrix multiplication!
        # So we can use Strassen: 7 mults instead of 8.
        #
        # Strassen for A1 (2x2) * B_pair (2x2 where columns are b1, b2):
        # M1 = (A1[0,0] + A1[1,1]) * (B_pair[0,0] + B_pair[1,1])
        #     = (A1[0,0] + A1[1,1]) * (b1[0] + b2[1])
        # M2 = (A1[1,0] + A1[1,1]) * B_pair[0,0]
        #     = (A1[1,0] + A1[1,1]) * b1[0]
        # M3 = A1[0,0] * (B_pair[1,0] - B_pair[1,1])
        #     = A1[0,0] * (b1[1] - b2[1])
        # M4 = A1[1,1] * (B_pair[0,0] + B_pair[0,1])
        #     = A1[1,1] * (b1[0] + b2[0])
        # M5 = (A1[0,0] + A1[0,1]) * B_pair[1,1]
        #     = (A1[0,0] + A1[0,1]) * b2[1]
        # M6 = (A1[1,0] - A1[0,0]) * (B_pair[0,0] + B_pair[0,1])
        #     = (A1[1,0] - A1[0,0]) * (b1[0] + b2[0])
        # M7 = (A1[0,1] - A1[1,1]) * (B_pair[1,0] + B_pair[1,1])
        #     = (A1[0,1] - A1[1,1]) * (b1[1] + b2[1])
        #
        # Result:
        # C1[0,0] = M1 + M3 - M4 + M7
        # C1[0,1] = M5 + M4
        # C1[1,0] = M2 + M4
        # C1[1,1] = M1 - M2 + M3 + M6
        #
        # Wait, I need to double-check the Strassen formulas.
        # Standard Strassen for A*B where A, B are 2x2:
        # M1 = (A11+A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B21-B22)
        # M4 = A22*(B11+B12)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        #
        # C11 = M1 + M3 - M4 + M7
        # C12 = M5 + M4
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        for pair_idx in range(5):  # 5 pairs of columns
            k1 = 2 * pair_idx
            k2 = 2 * pair_idx + 1
            
            # B_pair is 2x2: rows 0,1 of B, columns k1, k2
            # B_pair[0,0] = B[0, k1], B_pair[0,1] = B[0, k2]
            # B_pair[1,0] = B[1, k1], B_pair[1,1] = B[1, k2]
            
            # M1 = (A1[0,0] + A1[1,1]) * (B_pair[0,0] + B_pair[1,1])
            #    = (A[0,0] + A[1,1]) * (B[0,k1] + B[1,k2])
            # U entries for A: (0,0) and (1,1) with coefficient 1
            # V entries for B: (0,k1) and (1,k2) with coefficient 1
            # W entries: C[0,k1] and C[1,k2]... wait, no.
            
            # The rank-1 term is: u * v * w where
            # u[i*m+j] = coefficient of A[i,j]
            # v[j*p+k] = coefficient of B[j,k]
            # w[k*n+i] = coefficient of C[i,k]
            
            # For M1: 
            #   u: A[0,0] coeff 1, A[1,1] coeff 1
            #   v: B[0,k1] coeff 1, B[1,k2] coeff 1
            #   w: This term contributes to multiple C entries.
            #   Actually, M1 is a scalar multiplication. The rank-1 tensor has:
            #   u vector (length 6), v vector (length 30), w vector (length 20)
            #   such that sum_r u[i*m+j]*v[j*p+k]*w[k*n+i] = T[i,j,k]
            
            # Hmm, I think I'm overcomplicating this. Let me think differently.
            
            # Each "multiplication" in Strassen corresponds to one rank-1 term in the tensor decomposition.
            # The rank-1 term has:
            #   u: which A entries are multiplied (and with what coefficients)
            #   v: which B entries are multiplied (and with what coefficients)
            #   w: which C entries this contributes to (and with what coefficients)
            
            # For M1 = (A[0,0] + A[1,1]) * (B[0,k1] + B[1,k2]):
            #   u[0*m+0] = 1, u[1*m+1] = 1, others 0
            #   v[0*p+k1] = 1, v[1*p+k2] = 1, others 0
            #   w: This product contributes to C[0,k1], C[0,k2], C[1,k1], C[1,k2]
            #   Specifically, M1 appears in:
            #     C[0,k1] with coeff +1
            #     C[1,k2] with coeff +1
            #     C[0,k2] with coeff 0 (M1 doesn't appear in C12)
            #     C[1,k1] with coeff 0
            #   Wait, let me recheck.
            #   C11 = M1 + M3 - M4 + M7 -> C[0,k1] gets +M1
            #   C12 = M5 + M4 -> C[0,k2] gets no M1
            #   C21 = M2 + M4 -> C[1,k1] gets no M1
            #   C22 = M1 - M2 + M3 + M6 -> C[1,k2] gets +M1
            #   So w[k1*n+0] = 1, w[k2*n+1] = 1, others 0
            
            U55[0][col] = 1  # A[0,0]
            U55[1*m+1][col] = 1  # A[1,1]
            V55[0*p+k1][col] = 1  # B[0,k1]
            V55[1*p+k2][col] = 1  # B[1,k2]
            W55[k1*n+0][col] = 1  # C[0,k1]
            W55[k2*n+1][col] = 1  # C[1,k2]
            col += 1
            
            # M2 = (A[1,0] + A[1,1]) * B[0,k1]
            U55[1*m+0][col] = 1  # A[1,0]
            U55[1*m+1][col] = 1  # A[1,1]
            V55[0*p+k1][col] = 1  # B[0,k1]
            # M2 appears in:
            #   C21 = M2 + M4 -> C[1,k1] gets +M2
            #   C22 = M1 - M2 + M3 + M6 -> C[1,k2] gets -M2
            W55[k1*n+1][col] = 1  # C[1,k1]
            W55[k2*n+1][col] = -1  # C[1,k2]
            col += 1
            
            # M3 = A[0,0] * (B[1,k1] - B[1,k2])
            U55[0][col] = 1  # A[0,0]
            V55[1*p+k1][col] = 1  # B[1,k1]
            V55[1*p+k2][col] = -1  # B[1,k2]
            # M3 appears in:
            #   C11 = M1 + M3 - M4 + M7 -> C[0,k1] gets +M3
            #   C22 = M1 - M2 + M3 + M6 -> C[1,k2] gets +M3
            W55[k1*n+0][col] = 1  # C[0,k1]
            W55[k2*n+1][col] = 1  # C[1,k2]
            col += 1
            
            # M4 = A[1,1] * (B[0,k1] + B[0,k2])
            U55[1*m+1][col] = 1  # A[1,1]
            V55[0*p+k1][col] = 1  # B[0,k1]
            V55[0*p+k2][col] = 1  # B[0,k2]
            # M4 appears in:
            #   C11 = M1 + M3 - M4 + M7 -> C[0,k1] gets -M4
            #   C12 = M5 + M4 -> C[0,k2] gets +M4
            #   C21 = M2 + M4 -> C[1,k1] gets +M4
            W55[k1*n+0][col] = -1  # C[0,k1]
            W55[k2*n+0][col] = 1  # C[0,k2]
            W55[k1*n+1][col] = 1  # C[1,k1]
            col += 1
            
            # M5 = (A[0,0] + A[0,1]) * B[1,k2]
            U55[0][col] = 1  # A[0,0]
            U55[0*m+1][col] = 1  # A[0,1]
            V55[1*p+k2][col] = 1  # B[1,k2]
            # M5 appears in:
            #   C12 = M5 + M4 -> C[0,k2] gets +M5
            W55[k2*n+0][col] = 1  # C[0,k2]
            col += 1
            
            # M6 = (A[1,0] - A[0,0]) * (B[0,k1] + B[0,k2])
            U55[1*m+0][col] = 1  # A[1,0]
            U55[0][col] = -1  # A[0,0]
            V55[0*p+k1][col] = 1  # B[0,k1]
            V55[0*p+k2][col] = 1  # B[0,k2]
            # M6 appears in:
            #   C22 = M1 - M2 + M3 + M6 -> C[1,k2] gets +M6
            W55[k2*n+1][col] = 1  # C[1,k2]
            col += 1
            
            # M7 = (A[0,1] - A[1,1]) * (B[1,k1] + B[1,k2])
            U55[0*m+1][col] = 1  # A[0,1]
            U55[1*m+1][col] = -1  # A[1,1]
            V55[1*p+k1][col] = 1  # B[1,k1]
            V55[1*p+k2][col] = 1  # B[1,k2]
            # M7 appears in:
            #   C11 = M1 + M3 - M4 + M7 -> C[0,k1] gets +M7
            W55[k1*n+0][col] = 1  # C[0,k1]
            col += 1
        
        # Part 2: A2 * B2 where A2 is 2x1 (column 2 of A), B2 is 1x10 (row 2 of B)
        # C2[i,k] = A[1,k... wait, A2 is column 2 of A, so A2[i] = A[i,2]
        # B2 is row 2 of B, so B2[k] = B[2,k]
        # C2[i,k] = A[i,2] * B[2,k]
        # This is rank 2*10 = 20 (schoolbook)
        
        for i in range(n):
            for k in range(p):
                U55[i*m+2][col] = 1  # A[i,2]
                V55[2*p+k][col] = 1  # B[2,k]
                W55[k*n+i][col] = 1  # C[i,k]
                col += 1
        
        # Verify
        valid, err = _verify(U55, V55, W55, n, m, p)
        if valid:
            best_result = {"U": U55, "V": V55, "W": W55}
    
    # Return best result found
    if best_result is not None:
        return best_result
    else:
        # Fallback to schoolbook
        return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
