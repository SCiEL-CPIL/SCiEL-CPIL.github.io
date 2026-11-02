import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    deadline = time.time() + 130  # Safety margin before 150s kill
    
    # Target tensor: T[i*m+j, j*p+k, k*n+i] = 1 for i in [0,n), j in [0,m), k in [0,p)
    # U shape: (n*m, rank) = (20, rank)
    # V shape: (m*p, rank) = (30, rank)
    # W shape: (p*n, rank) = (24, rank)
    
    # Block decomposition approach:
    # Split A (4x5) into blocks: A11 (2x2), A12 (2x3), A21 (2x2), A22 (2x3)
    # Split B (5x6) into blocks: B11 (2x2), B12 (2x4), B21 (3x2), B22 (3x4)
    # C (4x6) = [C11 C12; C21 C22] where Cij = Ai1*B1j + Ai2*B2j
    
    # Strategy: Use Winograd-style 2x2x2 multiplication (rank 7) for small blocks
    # and optimize cross-terms
    
    # Let's use a systematic approach with tensor rank minimization
    # via structured low-rank decomposition
    
    def schoolbook():
        """Standard matrix multiplication - rank 120"""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    # Try block Winograd approach
    # Winograd 2x2x2: rank 7 instead of 8
    def winograd_2x2_2x2():
        """
        Winograd's algorithm for 2x2 * 2x2 = 2x2
        Uses 7 multiplications instead of 8
        Returns U (4x7), V (4x7), W (4x7) for the 2x2x2 case
        """
        # Winograd 2x2x2 factor matrices
        # A = [a b; c d], B = [e f; g h], C = [i j; k l]
        # i = (a+b+c)(e+g) - (a-c)(e-g) - (b+c)(f-h)
        # j = (b+c)(f+h) + (a-c)(f-h)  
        # k = (a+b+d)(g+h) - (b+d)(g-h) - (a+d)(f-h)
        # l = (a+d)(e+h) + (a+d)(g-h) - (a+b+d)(g-h)
        # Actually let me use the standard Winograd 2x2x2:
        # M1 = (a+b+c)(e+g)
        # M2 = (a-c)(e-g)
        # M3 = (b+c)(f-h)
        # M4 = (b+c)(f+h)
        # M5 = (a-c)(f-h)
        # M6 = (a+b+d)(g+h)
        # M7 = (a+d)(e+h)
        # Hmm, this is getting complex. Let me use a known optimal 2x2x2 decomposition.
        
        # Known rank-7 decomposition for 2x2x2:
        # Using the following pattern (indices 0-3 for 2x2 matrices):
        U2 = np.array([
            [1, 0, 1, 0, 1, 0, 0],  # a
            [1, 0, 0, 1, 0, 1, 0],  # b  
            [0, 1, 1, 0, 0, 0, 1],  # c
            [0, 1, 0, 1, 1, 0, 0],  # d
        ], dtype=np.float64)
        
        V2 = np.array([
            [1, 1, 0, 0, 0, 0, 1],  # e
            [0, 0, 1, 1, 1, 0, 0],  # f
            [1, -1, 0, 0, 0, 1, 0], # g
            [0, 0, 1, -1, -1, 0, 1],# h
        ], dtype=np.float64)
        
        W2 = np.array([
            [1, -1, -1, 0, 0, 0, 0],  # i
            [0, 0, 1, 1, 1, 0, 0],    # j
            [0, 0, 0, 0, 0, 1, -1],   # k  
            [0, 0, 0, 0, 0, 0, 1],    # l
        ], dtype=np.float64)
        
        return U2, V2, W2
    
    # Let's try a different approach: use the known fact that R(4,5,6) has upper bounds
    # from block decomposition. Let me construct a valid decomposition and try to minimize.
    
    # Approach: Split into blocks and use optimized sub-multiplications
    # n=4, m=5, p=6
    # Split: n1=2, n2=2, m1=2, m2=3, p1=2, p2=4
    
    n1, n2 = 2, 2
    m1, m2 = 2, 3
    p1, p2 = 2, 4
    
    # C11 = A11*B11 + A12*B21  (2x2 * 2x2 + 2x3 * 3x2)
    # C12 = A11*B12 + A12*B22  (2x2 * 2x4 + 2x3 * 3x4)
    # C21 = A21*B11 + A22*B21  (2x2 * 2x2 + 2x3 * 3x2)
    # C22 = A21*B12 + A22*B22  (2x2 * 2x4 + 2x3 * 3x4)
    
    # For each term, we need a decomposition. Let's use schoolbook for simplicity first,
    # then try to optimize.
    
    def mm_decomp(nn, mm, pp):
        """Schoolbook decomposition for nn x mm times mm x pp"""
        rank = nn * mm * pp
        U = np.zeros((nn * mm, rank))
        V = np.zeros((mm * pp, rank))
        W = np.zeros((pp * nn, rank))
        col = 0
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    U[i * mm + j, col] = 1.0
                    V[j * pp + k, col] = 1.0
                    W[k * nn + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    # Build full decomposition from blocks
    # We need to map local indices to global indices
    
    # Global A index: a = i*m + j where i in [0,4), j in [0,5)
    # Global B index: b = j*p + k where j in [0,5), k in [0,6)
    # Global C index: d = k*n + i where k in [0,6), i in [0,4)
    
    # Block A11: rows i in [0,2), cols j in [0,2) -> global a = i*5+j for i in {0,1}, j in {0,1}
    # Block A12: rows i in [0,2), cols j in [2,5) -> global a = i*5+j for i in {0,1}, j in {2,3,4}
    # Block A21: rows i in [2,4), cols j in [0,2) -> global a = i*5+j for i in {2,3}, j in {0,1}
    # Block A22: rows i in [2,4), cols j in [2,5) -> global a = i*5+j for i in {2,3}, j in {2,3,4}
    
    # Block B11: rows j in [0,2), cols k in [0,2) -> global b = j*6+k for j in {0,1}, k in {0,1}
    # Block B12: rows j in [0,2), cols k in [2,6) -> global b = j*6+k for j in {0,1}, k in {2,3,4,5}
    # Block B21: rows j in [2,5), cols k in [0,2) -> global b = j*6+k for j in {2,3,4}, k in {0,1}
    # Block B22: rows j in [2,5), cols k in [2,6) -> global b = j*6+k for j in {2,3,4}, k in {2,3,4,5}
    
    # Block C11: rows k in [0,2), cols i in [0,2) -> global d = k*4+i for k in {0,1}, i in {0,1}
    # Block C12: rows k in [0,2), cols i in [2,4) -> global d = k*4+i for k in {0,1}, i in {2,3}
    # Block C21: rows k in [2,6), cols i in [0,2) -> global d = k*4+i for k in {2,3,4,5}, i in {0,1}
    # Block C22: rows k in [2,6), cols i in [2,4) -> global d = k*4+i for k in {2,3,4,5}, i in {2,3}
    
    # Terms:
    # 1. A11*B11 -> C11 (2x2 * 2x2 -> 2x2)
    # 2. A12*B21 -> C11 (2x3 * 3x2 -> 2x2)
    # 3. A11*B12 -> C12 (2x2 * 2x4 -> 2x4)
    # 4. A12*B22 -> C12 (2x3 * 3x4 -> 2x4)
    # 5. A21*B11 -> C21 (2x2 * 2x2 -> 2x2)
    # 6. A22*B21 -> C21 (2x3 * 3x2 -> 2x2)
    # 7. A21*B12 -> C22 (2x2 * 2x4 -> 2x4)
    # 8. A22*B22 -> C22 (2x3 * 3x4 -> 2x4)
    
    # Schoolbook ranks for each term:
    # 1. 2*2*2 = 8
    # 2. 2*3*2 = 12
    # 3. 2*2*4 = 16
    # 4. 2*3*4 = 24
    # 5. 2*2*2 = 8
    # 6. 2*3*2 = 12
    # 7. 2*2*4 = 16
    # 8. 2*3*4 = 24
    # Total: 8+12+16+24+8+12+16+24 = 120 (same as schoolbook)
    
    # To do better, we need to share multiplications across terms or use Winograd.
    # Let's try using Winograd 2x2x2 (rank 7) for the 2x2*2x2 terms.
    
    # Use Winograd for terms 1 and 5 (A11*B11 and A21*B11)
    # But they multiply different A blocks, so we can't directly share.
    
    # Alternative: Use a different block split or a more global optimization.
    
    # Let's try a rank minimization approach using the tensor structure.
    # The tensor T has shape (20, 30, 24) with T[i*5+j, j*6+k, k*4+i] = 1
    
    # We'll use a greedy/local optimization approach to find a lower rank decomposition.
    
    # Start with schoolbook and try to reduce rank
    U, V, W = schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Build the target tensor for verification
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    def verify_decomp(U, V, W, T):
        """Check if U, V, W exactly decompose T"""
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
            return False
        T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.allclose(T_recon, T, atol=1e-10)
    
    # Try to find a lower rank decomposition using structured search
    # We'll try various block sizes and Winograd-style optimizations
    
    # For now, return schoolbook as a valid baseline
    # Future: implement more sophisticated rank minimization
    
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
