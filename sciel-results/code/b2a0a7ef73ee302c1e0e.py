"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def construct():
    _start_time = time.time()
    _deadline = _start_time + 120.0  # 150s kill, leave margin
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    def _time_left():
        return _deadline - time.time()
    
    def _validate(U, V, W):
        """Validate that U, V, W form a valid matrix multiplication tensor decomposition.
        U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
        For each c: sum over c of U[i*m+j, c] * V[j*p+k, c] should give W[k*n+i, c] pattern
        Actually: the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1 if valid
        Wait, the decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        And T[i,j,k] = 1 for all i,j,k (the multiplication tensor has 1s everywhere)
        """
        rank = len(U[0])
        # Check shapes
        if len(U) != n * m:
            return False
        if len(V) != m * p:
            return False
        if len(W) != p * n:
            return False
        for c in range(rank):
            if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
                return False
        # Verify tensor: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    def _validate_fast(U, V, W):
        """Faster validation using numpy"""
        import numpy as np
        rank = len(U[0])
        U_arr = np.array(U, dtype=object)
        V_arr = np.array(V, dtype=object)
        W_arr = np.array(W, dtype=object)
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Reshape: U -> (n, m, rank), V -> (m, p, rank), W -> (p, n, rank)
        U3 = U_arr.reshape(n, m, rank)
        V3 = V_arr.reshape(m, p, rank)
        W3 = W_arr.reshape(p, n, rank)
        # Sum over c: T[i,j,k] = sum_c U3[i,j,c]*V3[j,k,c]*W3[k,i,c]
        # This is tricky with numpy. Let's do it differently.
        # For each c, we have outer products.
        # Actually let's just iterate but use the structure.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    def _schoolbook():
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
    
    def _strassen_2x2_block():
        """Use Strassen on 2x2 blocks. A is 4x4, B is 4x9.
        Split A into four 2x2 blocks, B into two 4x... hmm this is awkward.
        Let's think differently.
        
        A (4x4) * B (4x9) = C (4x9)
        
        Split A into 2x2 blocks of size 2x2:
        A = [[A11, A12], [A21, A22]] where each is 2x2
        
        Split B into 2 blocks of 2x9:
        B = [[B1], [B2]] where B1, B2 are 2x9
        
        C = [[A11*B1 + A12*B2], [A21*B1 + A22*B2]]
        
        For each 2x2 * 2x9 block multiplication, we can use a variant of Strassen.
        Actually, for 2x2 * 2x2, Strassen gives rank 7.
        For 2x2 * 2x9, we can think of B as three 2x3 blocks or nine 2x1 blocks.
        
        Alternative: treat the 4x4 * 4x9 as a sum of structured terms.
        """
        pass
    
    def _try_rank_r(r):
        """Try to find a rank-r decomposition via structured search."""
        if _time_left() < 5:
            return None
        
        # Try Strassen-inspired block decomposition
        # For 4x4 * 4x9, one approach:
        # Split into 2x2 blocks. A has 4 blocks of 2x2, B has 2 blocks of 2x9.
        # C has 2 blocks of 2x9.
        # C_top = A11*B_top + A12*B_bottom
        # C_bot = A21*B_top + A22*B_bottom
        
        # For each 2x2 * 2x9 multiplication, we need rank.
        # 2x2 * 2x9: U is 4x?, V is 18x?, W is 18x?
        # Schoolbook for 2x2*2x9: rank = 4*9 = 36
        # Can we do better? 
        # Treat B as 9 columns of 2x1. Then it's 9 independent 2x2*2x1 multiplications.
        # 2x2*2x1: U is 4x?, V is 2x?, W is 2x?
        # Schoolbook: rank = 4. Strassen doesn't directly apply to non-square.
        # Actually 2x2*2x1: C[i,k] = sum_j A[i,j]*B[j,k]
        # This is just matrix-vector. Rank 4 schoolbook.
        # Can we do rank 3? Unlikely for exact.
        
        # Let's try a different structured approach.
        # Use the fact that 4 = 2*2 and try Kronecker product structure.
        
        # Approach: decompose using 2x2 Strassen on the "row" dimension.
        # A(4x4) = [A_top(2x4); A_bot(2x4)]
        # B(4x9) = [B_top(2x9); B_bot(2x9)]
        # C(4x9) = [C_top(2x9); C_bot(2x9)]
        # C_top = A_top * B
        # C_bot = A_bot * B
        
        # A_top is 2x4, B is 4x9. This is 2x4 * 4x9.
        # Schoolbook rank = 2*4*9 = 72.
        # Can we do better? 2x4*4x9: treat as 2x2 * 2x9 done twice?
        
        # Let me try a simpler structured approach with rank = 4*9 = 36
        # by exploiting linearity in rows of A.
        # C[i,:] = sum_j A[i,j] * B[j,:]
        # For each i, C[i,:] is a linear combination of the 4 rows of B.
        # The tensor is: T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
        # Wait, the tensor for matrix mult A(4x4)*B(4x9) is T[i,j,k] = 1 for all i,j,k.
        # The rank of this tensor is what we want to minimize.
        
        # Known lower bounds: rank >= max(n*p, m*p, n*m) = max(36, 36, 16) = 36
        # So rank 36 is the lower bound. Let's see if we can achieve it.
        
        # For n=m=p, the rank of the tensor is the matrix multiplication rank.
        # For 4x4*4x4, the rank is known to be <= 49 (Strassen-based) and >= 36.
        # For 4x4*4x9, it's a rectangular case.
        
        # Let me try to construct rank 36.
        # If we can find U(16x36), V(36x36), W(36x36) such that the tensor equation holds.
        
        # Actually, let's just return schoolbook for now and try to improve.
        return None
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try to find better decompositions
    # For now, return schoolbook as it's guaranteed correct
    return best
# EVOLVE-BLOCK-END
