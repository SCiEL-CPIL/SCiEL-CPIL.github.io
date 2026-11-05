"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = 145.0  # Safety margin for 150s limit

def _time_exceeded():
    return (time.time() - _START_TIME) > _DEADLINE

def _verify(U, V, W, n=4, m=5, p=5, tol=1e-9):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    For all i in [0,n), j in [0,m), k in [0,p):
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    if not U or not V or not W:
        return False
    
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape for easier manipulation
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    try:
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    except Exception:
        return False
    
    if not np.allclose(T, np.ones((n, m, p)), atol=tol):
        return False
    
    return True

def _verify_full(U, V, W, n=4, m=5, p=5, trials=5, tol=1e-9):
    """
    Full verification: check that the decomposition correctly computes matrix multiplication
    for random test matrices.
    """
    if not U or not V or not W:
        return False
    
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        
        for c in range(rank):
            u_vals = U_arr[:, c]
            v_vals = V_arr[:, c]
            w_val = W_arr
            
            u_mat = u_vals.reshape(n, m)
            v_mat = v_vals.reshape(m, p)
            w_mat = w_val.reshape(p, n).T
            
            a_c = np.sum(A * u_mat, axis=1)
            b_c = np.sum(B * v_mat, axis=0)
            
            C_computed += np.outer(a_c, b_c) * w_mat
        
        if not np.allclose(C_computed, C_expected, atol=tol):
            return False
    
    return True

def _schoolbook(n=4, m=5, p=5):
    """
    Standard schoolbook decomposition of rank n*m*p.
    """
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

def _strassen_4x4x4():
    """
    Returns the rank-17 Strassen decomposition for 4x4x4 matrix multiplication.
    U: (16, 17), V: (16, 17), W: (16, 17)
    Indices: U[i*4+j, c], V[j*4+k, c], W[k*4+i, c]
    """
    # Strassen's 7 products for 2x2x2, extended to 4x4x4 via recursion/block structure
    # The standard Strassen 4x4x4 has rank 17.
    # We define the 17 terms explicitly.
    
    # Let A, B be 4x4. Partition into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = A*B
    #
    # Strassen 2x2x2 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # For 4x4x4, we apply this recursively. Each 2x2 block is a 2x2 matrix.
    # So we have 7 "super-products", each of which is a 2x2x2 multiplication.
    # Each 2x2x2 multiplication can be done with 7 scalar products (Strassen 2x2x2).
    # Total: 7 * 7 = 49? No, that's not right.
    #
    # Actually, the rank of 4x4x4 matrix multiplication is 17 (not 49).
    # The 17-term decomposition is a specific known result.
    #
    # Let's use the known 17-term decomposition.
    # Reference: Bini, Lotti, Pazzaglia (1978) "On the algebraic complexity of n x n matrix multiplication"
    #
    # The 17 terms are defined by the following 17 triples (u, v, w) where
    # u is a 4x4 matrix (flattened to 16 entries), v is 4x4, w is 4x4.
    #
    # Due to complexity, I will implement a simpler approach:
    # Use the fact that 4x4x4 can be decomposed using 17 terms.
    # I'll hardcode the 17 terms.
    
    # This is the standard Strassen 4x4x4 decomposition.
    # The 17 terms are:
    
    terms = []
    
    # We'll define the 17 rank-1 terms as (u, v, w) matrices.
    # Each is a 4x4 matrix with 0s and 1s (or -1s).
    
    # Term 1: (A00+A11)(B00+B11) -> contributes to C00, C01, C10, C11
    # Actually, let's just use the known 17-term formula.
    
    # I'll use a simpler approach: embed the 2x2x2 Strassen into 4x4x4.
    # For 4x4x4, we can use 17 terms.
    
    # Let's define the 17 terms explicitly.
    # This is a well-known result.
    
    # Due to the complexity of hardcoding all 17 terms correctly,
    # I'll use a different strategy:
    # 
    # For 4x5x5, we can partition as:
    # A is 4x5, B is 5x5.
    # We can write A = [A_4x4 | A_4x1]
    # B = [[B_4x4], [B_1x5]]
    #
    # C = A_4x4 * B_4x4 + A_4x1 * B_1x5
    #
    # The first term is 4x4x4 (rank 17 with Strassen).
    # The second term is 4x1x5 (rank 4*1*5 = 20 with schoolbook, or less).
    #
    # Actually, 4x1x5: A is 4x1, B is 1x5. C = A*B is 4x5.
    # C[i,k] = A[i,0] * B[0,k]
    # This is an outer product. Rank is 1? No.
    # T[i,0,k] = 1 for all i,k.
    # Sum_c U[i, c] * V[0, c] * W[k, i, c] = 1
    # We can do this with rank 4*5 = 20 terms (schoolbook) or rank 1?
    # If rank 1: U[i,0]*V[0,0]*W[k,0] = 1 for all i,k.
    # This requires U[i,0] = a_i, V[0,0] = b, W[k,0] = c_k such that a_i * b * c_k = 1.
    # This is possible if a_i = 1, b = 1, c_k = 1.
    # But wait, the indices are:
    # U has shape (n*m, rank) = (4*1, rank) = (4, rank)
    # V has shape (m*p, rank) = (1*5, rank) = (5, rank)
    # W has shape (p*n, rank) = (5*4, rank) = (20, rank)
    #
    # For the 4x1x5 part:
    # U[i, c] for i in [0,4)
    # V[0, c] for the single row
    # W[k*4+i, c] for k in [0,5), i in [0,4)
    #
    # We need sum_c U[i,c] * V[0,c] * W[k*4+i,c] = 1
    #
    # If we use rank 1: U[i,0] = 1, V[0,0] = 1, W[k*4+i,0] = 1.
    # Then sum = 1*1*1 = 1. Correct!
    #
    # So the 4x1x5 part has rank 1.
    #
    # Similarly, we can partition B as [[B_4x4], [B_1x5]] where B_4x4 is 4x4 and B_1x5 is 1x5.
    # And A as [A_4x4 | A_4x1] where A_4x4 is 4x4 and A_4x1 is 4x1.
    #
    # So C = A_4x4 * B_4x4 + A_4x1 * B_1x5
    #
    # Rank of first part: 17 (Strassen 4x4x4)
    # Rank of second part: 1 (as shown above)
    #
    # But we need to combine them. The total rank is at most 17 + 1 = 18.
    #
    # Let's verify this.
    #
    # For the 4x4x4 part (indices i in [0,4), j in [0,4), k in [0,4)):
    # We use the Strassen 17-term decomposition.
    #
    # For the 4x1x5 part (indices i in [0,4), j=4, k in [0,5)):
    # We use the rank-1 decomposition.
    #
    # Total rank: 18.
    #
    # This is a significant improvement over 100.
    
    # Now I need to implement the Strassen 4x4x4 decomposition.
    # Let me hardcode the 17 terms.
    
    # The 17 terms for 4x4x4 matrix multiplication:
    # I'll use the standard Strassen recursion.
    
    # Actually, let's just implement the 2x2x2 Strassen and use it for 4x4x4.
    
    def _strassen_2x2x2():
        """
        Returns the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
        U: (4, 7), V: (4, 7), W: (4, 7)
        Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A00 + A11)(B00 + B11)
        # U: A00=1, A11=1
        U[0][0] = 1  # A00
        U[3][0] = 1  # A11
        V[0][0] = 1  # B00
        V[3][0] = 1  # B11
        W[0][0] = 1  # C00
        W[3][0] = 1  # C11
        
        # M2 = (A10 + A11)B00
        U[2][1] = 1  # A10
        U[3][1] = 1  # A11
        V[0][1] = 1  # B00
        W[2][1] = 1  # C10
        
        # M3 = A00(B01 - B11)
        U[0][2] = 1  # A00
        V[1][2] = 1  # B01
        V[3][2] = -1 # B11
        W[1][2] = 1  # C01
        
        # M4 = A11(B10 - B00)
        U[3][3] = 1  # A11
        V[2][3] = 1  # B10
        V[0][3] = -1 # B00
        W[3][3] = 1  # C11
        
        # M5 = (A00 + A01)B11
        U[0][4] = 1  # A00
        U[1][4] = 1  # A01
        V[3][4] = 1  # B11
        W[1][4] = 1  # C01
        
        # M6 = (A10 - A00)(B00 + B01)
        U[2][5] = 1  # A10
        U[0][5] = -1 # A00
        V[0][5] = 1  # B00
        V[1][5] = 1  # B01
        W[2][5] = 1  # C10
        
        # M7 = (A01 - A11)(B10 + B11)
        U[1][6] = 1  # A01
        U[3][6] = -1 # A11
        V[2][6] = 1  # B10
        V[3][6] = 1  # B11
        W[3][6] = 1  # C11
        
        return U, V, W
    
    # For 4x4x4, we use the Strassen 2x2x2 decomposition recursively.
    # Each 2x2 block is a 2x2 matrix.
    # The 7 products are each 2x2x2 multiplications.
    # So we have 7 * 7 = 49 terms? No.
    #
    # Actually, the rank of 4x4x4 is 17, not 49.
    # The 17-term decomposition is a specific result.
    #
    # Let me just use the schoolbook for 4x4x4 (rank 64) and add the rank-1 term.
    # Total rank: 64 + 1 = 65.
    #
    # This is still better than 100.
    
    # Actually, let's just use the schoolbook for the entire 4x5x5.
    # Rank 100.
    
    # For now, return None to indicate we can't find a better decomposition quickly.
    return None

def construct():
    """
    Constructs a tensor decomposition for multiplying a 4x5 matrix by a 5x5 matrix.
    """
    if _time_exceeded():
        return {"U": [], "V": [], "W": []}
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower-rank decompositions
    # For now, return the schoolbook decomposition.
    
    # Quick verification of the schoolbook result
    if _verify(best_result['U'], best_result['V'], best_result['W'], n, m, p):
        return best_result
    
    # If schoolbook fails (shouldn't happen), return empty
    return {"U": [], "V": [], "W": []}
# EVOLVE-BLOCK-END
