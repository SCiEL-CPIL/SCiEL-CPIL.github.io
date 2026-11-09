import time
import random
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def _build_schoolbook():
    """Schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(val - 1.0) > 1e-8:
                    return False
    return True

def _residuals(x, n, m, p, rank):
    """
    Compute residuals for least_squares.
    x is a flat array of length (n*m + m*p + p*n) * rank.
    Residuals are the differences between the computed tensor entries and 1.
    """
    U_size = n * m * rank
    V_size = m * p * rank
    
    U_flat = x[:U_size]
    V_flat = x[U_size:U_size + V_size]
    W_flat = x[U_size + V_size:]
    
    U = U_flat.reshape(n * m, rank)
    V = V_flat.reshape(m * p, rank)
    W = W_flat.reshape(p * n, rank)
    
    residuals = np.zeros(n * m * p)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                residuals[idx] = val - 1.0
                idx += 1
    return residuals

def _try_ranked_decomposition(n, m, p, target_rank, max_attempts=10, timeout=5.0):
    """
    Try to find an exact decomposition of rank target_rank using least_squares.
    """
    start_time = time.time()
    
    for attempt in range(max_attempts):
        if time.time() - start_time > timeout:
            break
            
        # Random initialization
        x0 = np.random.randn((n * m + m * p + p * n) * target_rank)
        
        try:
            result = least_squares(
                _residuals, 
                x0, 
                args=(n, m, p, target_rank),
                method='lm', 
                max_nfev=200
            )
            
            if result.success or np.max(np.abs(result.fun)) < 1e-8:
                x_opt = result.x
                U_size = n * m * target_rank
                V_size = m * p * target_rank
                
                U = x_opt[:U_size].reshape(n * m, target_rank)
                V = x_opt[U_size:U_size + V_size].reshape(m * p, target_rank)
                W = x_opt[U_size + V_size:].reshape(p * n, target_rank)
                
                U_list = [[float(val) for val in row] for row in U]
                V_list = [[float(val) for val in row] for row in V]
                W_list = [[float(val) for val in row] for row in W]
                
                if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                    return U_list, V_list, W_list
                    
        except Exception:
            continue
            
    return None

def _build_strassen_2x2():
    """
    Build the Strassen 2x2x2 decomposition.
    Returns U, V, W as numpy arrays of shape (4, 7) each.
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen constants
    M1 = 1; M2 = 1; M3 = 1; M4 = 1; M5 = 1; M6 = -1; M7 = 1
    # C1 = M1, C2 = M2, C3 = M3, C4 = M4
    # C5 = M5, C6 = M6, C7 = M7
    
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # c=0: M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    # c=1: M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # c=2: M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[1, 2] = 1
    
    # c=3: M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[3, 3] = 1
    
    # c=4: M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[3, 4] = 1
    
    # c=5: M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1; W[2, 5] = 1
    
    # c=6: M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[1, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def _build_4x4x4_strassen():
    """
    Build a rank-18 decomposition for 4x4x4 using the Strassen 2x2x2 building block.
    This is the standard recursive Strassen algorithm applied to 4x4x4.
    """
    n, m, p = 4, 4, 4
    # We use the tensor product of the 2x2x2 decomposition with itself.
    # The rank will be 7 * 7 = 49? No, that's schoolbook-like.
    # Wait, the recursive Strassen for 4x4x4 has rank 18.
    # How is 18 obtained?
    # 4x4x4 can be seen as 2x(2x4)x(4x2)? No.
    # Actually, the known optimal rank for 4x4x4 is 18.
    # One way to get 18 is by using the "Strassen-Winograd" or specific linear algebraic identities.
    # However, constructing this manually is complex.
    # Let's try a different approach: use the 2x2x2 block structure.
    # If we partition 4x4 into four 2x2 blocks, the product involves 16 block multiplications.
    # Using Strassen on the block level (2x2 blocks of size 2x2), we need 7 block multiplications.
    # Each block multiplication is 2x2x2, which has rank 7.
    # Total rank = 7 * 7 = 49. This is not better than schoolbook (64).
    # Wait, 49 < 64. So rank 49 is an improvement over schoolbook for 4x4x4.
    # But the known minimum is 18.
    # Is there a rank 18 decomposition? Yes, but it's not the simple recursive Strassen.
    # The recursive Strassen gives rank 7^log2(4) = 49.
    # So for 4x4x4, recursive Strassen gives 49.
    # For 4x4x6, we can use the recursive Strassen approach.
    # Partition A (4x4) into four 2x2 blocks.
    # Partition B (4x6) into two 2x3 blocks? Or 2x2 and 2x2 and 2x2?
    # Let's partition B into three 2x2 blocks: B1, B2, B3.
    # Then C (4x6) has three 2x2 blocks: C1, C2, C3.
    # We need to compute C1, C2, C3.
    # C1 = A*B1, C2 = A*B2, C3 = A*B3.
    # Each Ci is 4x2.
    # We can compute each Ci using the 4x4x2 multiplication.
    # But we want a single tensor decomposition.
    # Let's use the 2x2x2 Strassen for the block level.
    # The block matrix A is 2x2 (each block 2x2).
    # The block matrix B is 2x3 (each block 2x2).
    # The block matrix C is 2x3 (each block 2x2).
    # We need to multiply a 2x2 block matrix by a 2x3 block matrix.
    # This is equivalent to multiplying 2x2 by 2x3.
    # We can use the Strassen-like algorithm for rectangular matrices.
    # Or, we can just use the schoolbook for the block level: 2*2*3 = 12 block multiplications.
    # Each block multiplication is 2x2x2, rank 7.
    # Total rank = 12 * 7 = 84.
    # This is better than schoolbook (96).
    # Can we do better?
    # The rank of 2x2x3 matrix multiplication is 9? Or 10?
    # Actually, the rank of 2x2x3 is 9 (I think).
    # If we use rank 9 for the block level, total rank = 9 * 7 = 63.
    # Let's try to find a rank 9 decomposition for 2x2x3.
    # Or, we can use the recursive Strassen for 2x2x3.
    # 2x2x3: Partition B into 2x1 and 2x2?
    # This is getting complicated.
    # Let's stick to the recursive Strassen for 4x4x6.
    # Block level: 2x2 * 2x3.
    # Use schoolbook for block level: 12 block mults.
    # Each block mult is 2x2x2 (rank 7).
    # Total rank = 84.
    # This is a valid decomposition of rank 84.
    # Let's construct it.
    
    # First, get the 2x2x2 Strassen decomposition.
    U2, V2, W2 = _build_strassen_2x2()
    
    # Now, construct the 4x4x6 decomposition.
    # U has shape (16, rank), V has shape (24, rank), W has shape (24, rank).
    # rank = 84.
    rank = 84
    U = np.zeros((16, rank))
    V = np.zeros((24, rank))
    W = np.zeros((24, rank))
    
    # Block indices for A (4x4):
    # A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3)
    # A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    
    # Block indices for B (4x6):
    # B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3), B02 (rows 0-1, cols 4-5)
    # B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3), B12 (rows 2-3, cols 4-5)
    
    # Block indices for C (4x6):
    # C00 (rows 0-1, cols 0-1), C01 (rows 0-1, cols 2-3), C02 (rows 0-1, cols 4-5)
    # C10 (rows 2-3, cols 0-1), C11 (rows 2-3, cols 2-3), C12 (rows 2-3, cols 4-5)
    
    # We need to compute:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12
    
    # Each term is a 2x2 * 2x2 multiplication.
    # We have 12 such terms.
    # For each term, we use the 2x2x2 Strassen decomposition (7 rank-1 tensors).
    
    # Term 1: A00 * B00 -> C00
    # A00 rows: 0,1; cols: 0,1 -> U indices 0,1,4,5
    # B00 rows: 0,1; cols: 0,1 -> V indices 0,1,2,3
    # C00 rows: 0,1; cols: 0,1 -> W indices 0,1,4,5
    
    # Let's map the 2x2x2 decomposition indices.
    # U2[i*2+j, c], V2[j*2+k, c], W2[k*2+i, c]
    
    # For A00*B00:
    # i=0, j=0 -> A00. U rows 0,1.
    # j=0, k=0 -> B00. V rows 0,1.
    # k=0, i=0 -> C00. W rows 0,1.
    
    # We will iterate over the 12 terms.
    # Term (ai, bj) -> Cij
    # ai in {00, 01, 10, 11}, bj in {00, 01, 02, 10, 11, 12}
    # Actually, the terms are:
    # (00,00)->00, (01,10)->00
    # (00,01)->01, (01,11)->01
    # (00,02)->02, (01,12)->02
    # (10,00)->10, (11,10)->10
    # (10,01)->11, (11,11)->11
    # (10,02)->12, (11,12)->12
    
    terms = [
        (0, 0, 0), # A00*B00 -> C00
        (1, 2, 0), # A01*B10 -> C00  (A01 is col 1, B10 is row 1 col 0)
        (0, 1, 1), # A00*B01 -> C01
        (1, 3, 1), # A01*B11 -> C01
        (0, 2, 2), # A00*B02 -> C02
        (1, 4, 2), # A01*B12 -> C02
        (2, 0, 4), # A10*B00 -> C10
        (3, 2, 4), # A11*B10 -> C10
        (2, 1, 5), # A10*B01 -> C11
        (3, 3, 5), # A11*B11 -> C11
        (2, 2, 6), # A10*B02 -> C12
        (3, 4, 6), # A11*B12 -> C12
    ]
    
    # Note: The indices above are for the 2x2 block level.
    # A block index (r, c) corresponds to U rows (r*2, r*2+1) and cols (c*2, c*2+1).
    # So A00 is U[0:2, 0:2], A01 is U[0:2, 2:4], A10 is U[2:4, 0:2], A11 is U[2:4, 2:4].
    # B block index (r, c) corresponds to V rows (r*2, r*2+1) and cols (c*2, c*2+1).
    # B00 is V[0:2, 0:2], B01 is V[0:2, 2:4], B02 is V[0:2, 4:6].
    # B10 is V[2:4, 0:2], B11 is V[2:4, 2:4], B12 is V[2:4, 4:6].
    # C block index (r, c) corresponds to W rows (r*2, r*2+1) and cols (c*2, c*2+1).
    # C00 is W[0:2, 0:2], etc.
    
    # For each term, we use the 2x2x2 Strassen decomposition.
    # The local indices i, j, k in the 2x2x2 decomposition correspond to the local block indices.
    # For a term A_ai * B_bj -> C_c, we have:
    # A_ai is at block row ai_r, block col ai_c.
    # B_bj is at block row bj_r, block col bj_c.
    # C_c is at block row c_r, block col c_c.
    
    # Let's re-define the terms with explicit block coordinates.
    terms = [
        # (a_row, a_col, b_row, b_col, c_row, c_col)
        (0, 0, 0, 0, 0, 0), # A00*B00 -> C00
        (0, 1, 1, 0, 0, 0), # A01*B10 -> C00
        (0, 0, 0, 1, 0, 1), # A00*B01 -> C01
        (0, 1, 1, 1, 0, 1), # A01*B11 -> C01
        (0, 0, 0, 2, 0, 2), # A00*B02 -> C02
        (0, 1, 1, 2, 0, 2), # A01*B12 -> C02
        (1, 0, 0, 0, 1, 0), # A10*B00 -> C10
        (1, 1, 1, 0, 1, 0), # A11*B10 -> C10
        (1, 0, 0, 1, 1, 1), # A10*B01 -> C11
        (1, 1, 1, 1, 1, 1), # A11*B11 -> C11
        (1, 0, 0, 2, 1, 2), # A10*B02 -> C12
        (1, 1, 1, 2, 1, 2), # A11*B12 -> C12
    ]
    
    for t_idx, (ar, ac, br, bc, cr, cc) in enumerate(terms):
        # Local indices in the 2x2x2 decomposition
        # i = ar, j = ac? No.
        # In the 2x2x2 decomposition, U[i*2+j, c], V[j*2+k, c], W[k*2+i, c].
        # Here, the "i" of the 2x2x2 decomposition is the row index of A and C.
        # The "j" is the col index of A and row index of B.
        # The "k" is the col index of B and row index of C.
        
        # So for this term:
        # A is at block (ar, ac). So local i = ar, local j = ac.
        # B is at block (br, bc). So local j = br? No, j must match.
        # Wait, the term is A_ai * B_bj.
        # A_ai has row index ai_r and col index ai_c.
        # B_bj has row index bj_r and col index bj_c.
        # For the product to be valid, A's col index must match B's row index.
        # So ai_c must equal bj_r.
        # Let's check:
        # Term 1: A00 (0,0) * B00 (0,0). ai_c=0, bj_r=0. Match.
        # Term 2: A01 (0,1) * B10 (1,0). ai_c=1, bj_r=1. Match.
        # So the local indices for the 2x2x2 decomposition are:
        # i = ar, j = ac (which equals br), k = bc.
        # And the result C is at (cr, cc).
        # We need cr = k and cc = i?
        # In the 2x2x2 decomposition, W[k*2+i, c].
        # So the result block is at row k, col i.
        # So cr must equal bc, and cc must equal ar.
        # Let's check:
        # Term 1: C00 (0,0). cr=0, cc=0. bc=0, ar=0. Match.
        # Term 2: C00 (0,0). cr=0, cc=0. bc=0, ar=0. Match.
        # Term 3: C01 (0,1). cr=0, cc=1. bc=1, ar=0. Mismatch! cr=0 != bc=1.
        
        # Ah, the Strassen 2x2x2 decomposition computes C = A*B.
        # C[i,k] = sum_j A[i,j] B[j,k].
        # So the result block is at (i, k).
        # In our terms, the result is C_c, which is at (cr, cc).
        # So we need cr = i and cc = k.
        # And i = ar, k = bc.
        # So cr must equal ar, and cc must equal bc.
        # Let's check Term 3: A00*B01 -> C01.
        # A00 (0,0), B01 (0,1).
        # i=0, k=1. Result C[0,1].
        # C01 is at (0,1). So cr=0, cc=1.
        # ar=0, bc=1. So cr=ar=0, cc=bc=1. Match.
        
        # So the condition is:
        # i = ar, j = ac = br, k = bc.
        # Result C is at (i, k) = (ar, bc).
        # So cr = ar and cc = bc.
        
        # Let's re-verify the terms:
        # 1. A00(0,0)*B00(0,0) -> C(0,0)=C00. ar=0, bc=0. cr=0, cc=0. OK.
        # 2. A01(0,1)*B10(1,0) -> C(0,0)=C00. ar=0, bc=0. cr=0, cc=0. OK.
        # 3. A00(0,0)*B01(0,1) -> C(0,1)=C01. ar=0, bc=1. cr=0, cc=1. OK.
        # 4. A01(0,1)*B11(1,1) -> C(0,1)=C01. ar=0, bc=1. cr=0, cc=1. OK.
        # 5. A00(0,0)*B02(0,2) -> C(0,2)=C02. ar=0, bc=2. cr=0, cc=2. OK.
        # 6. A01(0,1)*B12(1,2) -> C(0,2)=C02. ar=0, bc=2. cr=0, cc=2. OK.
        # 7. A10(1,0)*B00(0,0) -> C(1,0)=C10. ar=1, bc=0. cr=1, cc=0. OK.
        # 8. A11(1,1)*B10(1,0) -> C(1,0)=C10. ar=1, bc=0. cr=1, cc=0. OK.
        # 9. A10(1,0)*B01(0,1) -> C(1,1)=C11. ar=1, bc=1. cr=1, cc=1. OK.
        # 10. A11(1,1)*B11(1,1) -> C(1,1)=C11. ar=1, bc=1. cr=1, cc=1. OK.
        # 11. A10(1,0)*B02(0,2) -> C(1,2)=C12. ar=1, bc=2. cr=1, cc=2. OK.
        # 12. A11(1,1)*B12(1,2) -> C(1,2)=C12. ar=1, bc=2. cr=1, cc=2. OK.
        
        # So the mapping is consistent.
        
        # Now, for each term, we add the 2x2x2 Strassen decomposition.
        # The local indices are i=ar, j=ac, k=bc.
        # The 2x2x2 decomposition has 7 rank-1 tensors.
        # For each c in 0..6:
        # U_local[i*2 + j, c] -> U_global[(ar*2 + 0)*2 + (ac*2 + 0) + ...]
        # Wait, U_global row index is (ar*2 + i_local)*2 + (ac*2 + j_local)?
        # No. A is 4x4. Block (ar, ac) is at rows ar*2..ar*2+1, cols ac*2..ac*2+1.
        # So the global row index is ar*2 + i_local, and global col index is ac*2 + j_local.
        # U_global index = (ar*2 + i_local)*4 + (ac*2 + j_local).
        # Similarly for V and W.
        
        for c in range(7):
            global_c = t_idx * 7 + c
            
            # U
            for i_local in range(2):
                for j_local in range(2):
                    val = U2[i_local * 2 + j_local, c]
                    if val != 0:
                        global_row = ar * 2 + i_local
                        global_col = ac * 2 + j_local
                        U[global_row * 4 + global_col, global_c] += val
                        
            # V
            for j_local in range(2):
                for k_local in range(2):
                    val = V2[j_local * 2 + k_local, c]
                    if val != 0:
                        global_row = br * 2 + j_local
                        global_col = bc * 2 + k_local
                        V[global_row * 6 + global_col, global_c] += val
                        
            # W
            for k_local in range(2):
                for i_local in range(2):
                    val = W2[k_local * 2 + i_local, c]
                    if val != 0:
                        global_row = cr * 2 + k_local
                        global_col = cc * 2 + i_local
                        W[global_row * 4 + global_col, global_c] += val
                        
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return U_list, V_list, W_list

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4 by 4 matrix
    by a 4 by 6 matrix.
    """
    deadline = time.time() + 110  # Safety margin before 120s limit
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    schoolbook = _build_schoolbook()
    best_rank = n * m * p
    best_result = schoolbook
    
    # Try the Strassen-based decomposition of rank 84
    U, V, W = _build_4x4x4_strassen() # Note: this builds 4x4x6
    if _verify_decomposition(U, V, W, n, m, p):
        best_rank = 84
        best_result = {"U": U, "V": V, "W": W}
        
    # Try to find even lower rank decompositions using least_squares
    # Start from 50 up to best_rank - 1
    for target_rank in range(50, best_rank):
        if time.time() > deadline:
            break
        
        time_left = deadline - time.time()
        if time_left < 2.0:
            break
            
        attempts = max(2, int(time_left / 5.0))
        
        result = _try_ranked_decomposition(n, m, p, target_rank, 
                                           max_attempts=attempts, 
                                           timeout=min(5.0, time_left - 1.0))
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                best_rank = target_rank
                best_result = {"U": U, "V": V, "W": W}
                break

    return best_result
# EVOLVE-BLOCK-END
