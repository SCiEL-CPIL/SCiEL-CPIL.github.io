import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def _strassen_2x2x2():
    """
    Return the Strassen decomposition for 2x2x2 matrix multiplication.
    Rank 7.
    U: 4x7, V: 4x7, W: 4x7
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    # Strassen's 7 products with correct coefficients:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # U (4x7): rows are (i,j) pairs: (0,0), (0,1), (1,0), (1,1)
    U = [
        [1,  0,  1,  0,  1, -1,  0],  # A00
        [0,  0,  0,  0,  1,  0,  1],  # A01
        [0,  1,  0,  0,  0,  1,  0],  # A10
        [1,  1,  0,  1,  0,  0, -1]   # A11
    ]
    
    # V (4x7): rows are (j,k) pairs: (0,0), (0,1), (1,0), (1,1)
    V = [
        [1,  1,  0, -1,  0,  1,  0],  # B00
        [0,  0,  1,  0,  0,  1,  0],  # B01
        [0,  0,  0,  1,  0,  0,  1],  # B10
        [1,  0, -1,  0,  1,  0,  1]   # B11
    ]
    
    # W (4x7): rows are (k,i) pairs: (0,0), (0,1), (1,0), (1,1)
    W = [
        [1,  0,  0,  0, -1, -1,  1],  # C00 (k=0,i=0)
        [0,  0,  0,  0,  0,  0,  0],  # C01 (k=0,i=1) - wait, this should be different
        [0,  1,  0,  1,  0,  1, -1],  # C10 (k=1,i=0)
        [1,  0,  1,  0,  1,  0,  1]   # C11 (k=1,i=1)
    ]
    
    # Let me re-derive W properly.
    # C[i,j] = sum_k A[i,k] B[k,j]
    # For 2x2: C[0,0] = A00*B00 + A01*B10
    #           C[0,1] = A00*B01 + A01*B11
    #           C[1,0] = A10*B00 + A11*B10
    #           C[1,1] = A10*B01 + A11*B11
    
    # The tensor T[i,j,k] = 1 for all i,j,k (it's the multiplication tensor)
    # We need: sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] = 1 for all i,j,k
    
    # Let me verify each Strassen term contributes correctly.
    # Term 0 (M1): U=[1,0,0,1], V=[1,0,0,1], W should make sum=1 for appropriate (i,j,k)
    # M1 = (A00+A11)(B00+B11)
    # This should contribute to C00, C01, C10, C11 with appropriate signs
    
    # Actually, let me just use the standard Strassen decomposition directly.
    # The standard form has:
    # U (4x7):
    # [1, 0, 1, 0, 1, 0, 0]
    # [0, 0, 0, 0, 0, 1, 1]
    # [0, 0, 0, 0, 0, -1, 1]
    # [1, 1, 0, 1, 0, 0, 0]
    # V (4x7):
    # [1, 1, 0, 1, 0, 1, 0]
    # [0, 0, 1, 0, 1, 0, 1]
    # [0, 0, -1, 0, 1, 0, 1]
    # [1, 1, 0, 1, 0, -1, 0]
    # W (4x7):
    # [1, 0, 0, 0, 0, 0, 0]
    # [0, 0, 0, 0, 1, 0, 0]
    # [0, 0, 0, 0, 0, 1, 0]
    # [1, 0, 0, 0, 0, 0, 1]
    
    # But I need to verify this is correct for the convention W[k*n+i].
    
    # Let me construct it from scratch using the Strassen formulas.
    # For each term t, I define U_t, V_t, W_t such that the sum reconstructs the multiplication.
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    # U: A00=1, A11=1 -> [1, 0, 0, 1]
    # V: B00=1, B11=1 -> [1, 0, 0, 1]
    # This product gives: A00*B00 + A00*B11 + A11*B00 + A11*B11
    # C00 needs A00*B00, C01 needs A00*B01 (not B11!), C10 needs A11*B10 (not B00!), C11 needs A11*B11
    # So M1 contributes: +1 to C00 (from A00*B00), +1 to C11 (from A11*B11)
    # But also +A00*B11 and +A11*B00 which don't belong in C00 or C11 directly.
    # These will be cancelled by other terms.
    
    # This is getting complex. Let me use a verified implementation.
    
    # Strassen decomposition derived from the 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # U[i,j] for i,j in {0,1}:
    # Term 0 (M1): A00=1, A11=1
    # Term 1 (M2): A10=1, A11=1
    # Term 2 (M3): A00=1
    # Term 3 (M4): A11=1
    # Term 4 (M5): A00=1, A01=1
    # Term 5 (M6): A10=1, A00=-1
    # Term 6 (M7): A01=1, A11=-1
    
    U = [
        [1,  0,  1,  0,  1, -1,  0],  # A00
        [0,  0,  0,  0,  1,  0,  1],  # A01
        [0,  1,  0,  0,  0,  1,  0],  # A10
        [1,  1,  0,  1,  0,  0, -1]   # A11
    ]
    
    # V[j,k] for j,k in {0,1}:
    # Term 0 (M1): B00=1, B11=1
    # Term 1 (M2): B00=1
    # Term 2 (M3): B01=1, B11=-1
    # Term 3 (M4): B10=1, B00=-1
    # Term 4 (M5): B11=1
    # Term 5 (M6): B00=1, B01=1
    # Term 6 (M7): B10=1, B11=1
    
    V = [
        [1,  1,  0, -1,  0,  1,  0],  # B00
        [0,  0,  1,  0,  0,  1,  0],  # B01
        [0,  0,  0,  1,  0,  0,  1],  # B10
        [1,  0, -1,  0,  1,  0,  1]   # B11
    ]
    
    # W[k,i] for k,i in {0,1} (note: W[k*n+i], so row index is k*n+i)
    # W[0] = W[0*2+0] = C[0,0]
    # W[1] = W[0*2+1] = C[1,0]
    # W[2] = W[1*2+0] = C[0,1]
    # W[3] = W[1*2+1] = C[1,1]
    
    # C00 = M1 + M4 - M5 - M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M7
    
    W = [
        [1,  0,  0,  1, -1,  0, -1],  # C00: M1+M4-M5-M7
        [0,  1,  0,  1,  0,  0,  0],  # C10: M2+M4
        [0,  0,  1,  0,  1,  0,  0],  # C01: M3+M5
        [1, -1, 1,  0,  0,  0,  1]    # C11: M1-M2+M3+M7
    ]
    
    return U, V, W

def _verify_strassen():
    """Verify the Strassen decomposition is correct."""
    U, V, W = _strassen_2x2x2()
    n = 2
    for i in range(n):
        for j in range(n):
            for k in range(n):
                s = 0
                for t in range(7):
                    s += U[i*n+j][t] * V[j*n+k][t] * W[k*n+i][t]
                if s != 1:
                    # Print debug info
                    print(f"Strassen verification failed at i={i}, j={j}, k={k}, s={s}")
                    return False
    return True

def _hybrid_decomposition(n, m, p):
    """
    Hybrid decomposition using Strassen for 2x2 blocks.
    For 2x3x7:
    - Split inner dim 3 into 2+1
    - For the 2x2x7 part: use Strassen on pairs of columns
    - 3 pairs (0,1), (2,3), (4,5) use Strassen: 3*7=21 rank
    - 1 leftover column (6) uses schoolbook: 4 rank
    - The 2x1x7 part (from the +1 in inner dim): 2*7=14 rank
    - Total: 21 + 4 + 14 = 39 rank
    """
    if n != 2 or m != 3 or p != 7:
        # Fallback to trivial
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
        return {"U": U, "V": V, "W": W}, rank
    
    # Get Strassen decomposition
    U_str, V_str, W_str = _strassen_2x2x2()
    
    # Total rank: 3 Strassen blocks (3*7=21) + 1 schoolbook column (4) + 14 for the extra row = 39
    total_rank = 3 * 7 + 4 + 14
    
    # Initialize matrices
    U = [[0] * total_rank for _ in range(n * m)]  # 6 x 39
    V = [[0] * total_rank for _ in range(m * p)]  # 21 x 39
    W = [[0] * total_rank for _ in range(p * n)]  # 14 x 39
    
    col = 0
    
    # Process 3 pairs of columns using Strassen: (0,1), (2,3), (4,5)
    for pair_idx in range(3):
        col1 = pair_idx * 2      # First column in pair
        col2 = pair_idx * 2 + 1  # Second column in pair
        
        for t in range(7):
            # U: For the 2x2 block, rows 0-3 correspond to A[0,0], A[0,1], A[1,0], A[1,1]
            # In our 2x3 matrix, A[0,0] is U[0], A[0,1] is U[1], A[1,0] is U[3], A[1,1] is U[4]
            # Wait, U[i*m+j] where m=3. So:
            # U[0] = A[0,0], U[1] = A[0,1], U[2] = A[0,2]
            # U[3] = A[1,0], U[4] = A[1,1], U[5] = A[1,2]
            
            # For Strassen on columns 0,1:
            # A00 = A[0,0] -> U[0]
            # A01 = A[0,1] -> U[1]
            # A10 = A[1,0] -> U[3]
            # A11 = A[1,1] -> U[4]
            
            U[0][col] = U_str[0][t]   # A00
            U[1][col] = U_str[1][t]   # A01
            U[3][col] = U_str[2][t]   # A10
            U[4][col] = U_str[3][t]   # A11
            
            # V: For the 2x2 block, rows 0-3 correspond to B[0,0], B[0,1], B[1,0], B[1,1]
            # In our 3x7 matrix, for columns col1, col2:
            # B[0,col1] = V[0*p + col1] = V[col1]
            # B[0,col2] = V[0*p + col2] = V[col2]
            # B[1,col1] = V[1*p + col1] = V[p + col1]
            # B[1,col2] = V[1*p + col2] = V[p + col2]
            
            V[col1][col] = V_str[0][t]   # B[0,col1]
            V[col2][col] = V_str[1][t]   # B[0,col2]
            V[p + col1][col] = V_str[2][t]   # B[1,col1]
            V[p + col2][col] = V_str[3][t]   # B[1,col2]
            
            # W: For the 2x2 block, rows 0-3 correspond to C[0,0], C[0,1], C[1,0], C[1,1]
            # W[k*n+i] where n=2.
            # C[0,0] = W[0*2+0] = W[0]
            # C[0,1] = W[0*2+1] = W[1]  -- wait, this is wrong
            # W[k*n+i]: k is the column index of C, i is the row index of C
            # C[i,k] where i is row, k is column
            # W[k*n + i]
            # C[0,0]: i=0, k=0 -> W[0*2+0] = W[0]
            # C[0,1]: i=0, k=1 -> W[1*2+0] = W[2]
            # C[1,0]: i=1, k=0 -> W[0*2+1] = W[1]
            # C[1,1]: i=1, k=1 -> W[1*2+1] = W[3]
            
            # But in our full problem, C is 2x7.
            # C[0,col1] = W[col1*2+0] = W[2*col1]
            # C[0,col2] = W[col2*2+0] = W[2*col2]
            # C[1,col1] = W[col1*2+1] = W[2*col1+1]
            # C[1,col2] = W[col2*2+1] = W[2*col2+1]
            
            W[2*col1][col] = W_str[0][t]   # C[0,col1]
            W[2*col2][col] = W_str[1][t]   # C[0,col2] -- wait, W_str[1] is for C[0,1] in the 2x2 block
            # Actually, W_str rows are indexed by (k,i) for the 2x2 block:
            # W_str[0] = C[0,0] in block
            # W_str[1] = C[0,1] in block (but in our mapping, this should be C[0,col2])
            # W_str[2] = C[1,0] in block (should be C[1,col1])
            # W_str[3] = C[1,1] in block (should be C[1,col2])
            
            # Let me re-check the Strassen W matrix indices.
            # In the 2x2x2 problem, W[k*n+i] where n=2.
            # W[0] = W[0*2+0] = C[0,0] (row 0, col 0)
            # W[1] = W[0*2+1] = C[1,0] (row 1, col 0)
            # W[2] = W[1*2+0] = C[0,1] (row 0, col 1)
            # W[3] = W[1*2+1] = C[1,1] (row 1, col 1)
            
            # So W_str[0] corresponds to C[0,0] in the block
            # W_str[1] corresponds to C[1,0] in the block
            # W_str[2] corresponds to C[0,1] in the block
            # W_str[3] corresponds to C[1,1] in the block
            
            # In our full problem:
            # Block C[0,0] -> Full C[0,col1] -> W[2*col1+0] = W[2*col1]
            # Block C[1,0] -> Full C[1,col1] -> W[2*col1+1]
            # Block C[0,1] -> Full C[0,col2] -> W[2*col2+0] = W[2*col2]
            # Block C[1,1] -> Full C[1,col2] -> W[2*col2+1]
            
            W[2*col1][col] = W_str[0][t]   # C[0,col1]
            W[2*col1+1][col] = W_str[1][t]   # C[1,col1]
            W[2*col2][col] = W_str[2][t]   # C[0,col2]
            W[2*col2+1][col] = W_str[3][t]   # C[1,col2]
            
            col += 1
    
    # Process leftover column 6 using schoolbook
    col6 = 6
    for i in range(n):
        for j in range(2):  # Only first 2 columns of A (the 2x2 part)
            U[i * m + j][col] = 1
            V[j * p + col6][col] = 1
            W[col6 * n + i][col] = 1
            col += 1
    
    # Process the extra row (row 2 of B, which is B[2,:])
    # A[0,2] * B[2,k] contributes to C[0,k]
    # A[1,2] * B[2,k] contributes to C[1,k]
    for k in range(p):
        U[0 * m + 2][col] = 1  # A[0,2]
        V[2 * p + k][col] = 1  # B[2,k]
        W[k * n + 0][col] = 1  # C[0,k]
        col += 1
        
        U[1 * m + 2][col] = 1  # A[1,2]
        V[2 * p + k][col] = 1  # B[2,k]
        W[k * n + 1][col] = 1  # C[1,k]
        col += 1
    
    return {"U": U, "V": V, "W": W}, total_rank

def _check_decomposition(U, V, W, n, m, p, rank):
    """Check if the decomposition is valid."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for t in range(rank):
                    s += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                if s != 1:
                    return False
    return True

def _trivial_decomposition(n, m, p):
    """Construct the trivial rank n*m*p decomposition."""
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
    return {"U": U, "V": V, "W": W}, rank

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Try hybrid decomposition first
    try:
        hybrid_result, hybrid_rank = _hybrid_decomposition(n, m, p)
        if _check_decomposition(hybrid_result['U'], hybrid_result['V'], hybrid_result['W'], n, m, p, hybrid_rank):
            return hybrid_result
    except Exception:
        pass
    
    # Fall back to trivial decomposition
    trivial_result, trivial_rank = _trivial_decomposition(n, m, p)
    if not _check_decomposition(trivial_result['U'], trivial_result['V'], trivial_result['W'], n, m, p, trivial_rank):
        raise Exception("Trivial decomposition failed verification")
    
    return trivial_result

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
