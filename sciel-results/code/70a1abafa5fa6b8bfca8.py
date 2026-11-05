import numpy as np
from fractions import Fraction
import time
import random

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact."""
    r = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) with integer entries."""
    # Strassen's 7 multiplications
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U[i,j] for i in {0,1}, j in {0,1}
    # V[j,k] for j in {0,1}, k in {0,1}
    # W[k,i] for k in {0,1}, i in {0,1}
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # A indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # B indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # C indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    
    # M1: (A00+A11)*(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # C00 += M1, C11 += M1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # C10 += M2, C11 -= M2
    W[2][1] = 1; W[3][1] = -1
    
    # M3: A00*(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    # C01 += M3, C11 += M3
    W[1][2] = 1; W[3][2] = 1
    
    # M4: A11*(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    # C00 += M4, C10 += M4
    W[0][3] = 1; W[2][3] = 1
    
    # M5: (A00+A01)*B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # C00 -= M5, C01 += M5
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (A10-A00)*(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # C11 += M6
    W[3][5] = 1
    
    # M7: (A01-A11)*(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    # C00 += M7
    W[0][6] = 1
    
    return U, V, W

def _block_decomposition_3x5x7():
    """
    Block decomposition: split the 5 columns of A into blocks of sizes [2, 2, 1].
    For each block of size b, we multiply a 3xb matrix by a bx7 matrix.
    
    Block 1: 3x2 * 2x7 -> use Strassen-like approach
    Block 2: 3x2 * 2x7 -> use Strassen-like approach  
    Block 3: 3x1 * 1x7 -> schoolbook (3*1*7 = 21)
    
    Actually, let me think more carefully.
    
    For 3x2 * 2x7:
    We can use a rank-? decomposition.
    Schoolbook: 3*2*7 = 42
    
    For 3x1 * 1x7:
    Schoolbook: 3*1*7 = 21
    
    Total schoolbook for blocks: 42 + 42 + 21 = 105 (same as before)
    
    Let me try a different approach. Use the known rank for specific small cases.
    
    Actually, the inspiration says rank 99 was achieved. Let me think about what structure gives 99.
    
    99 = 105 - 6. So we save 6 ranks.
    
    Let me try: split into blocks [3, 2] for the middle dimension.
    Block 1: 3x3 * 3x7
    Block 2: 3x2 * 2x7
    
    For 3x3*3x7: schoolbook = 63
    For 3x2*2x7: schoolbook = 42
    Total: 105
    
    Hmm, still 105.
    
    Let me try using Strassen for sub-blocks.
    
    Actually, let me reconsider. The key insight from inspiration 1 is that they used a specific block structure.
    
    Let me try: split the 5x7 part into a 2x7 and 3x7 part, and the 3x5 into 3x2 and 3x3.
    
    For 3x2 * 2x7:
    We can potentially use rank less than 42.
    
    Let me try a direct approach: for 3x2*2x7, the tensor has rank at most... 
    Actually, let me just try to find good decompositions computationally.
    
    Let me try a different strategy: use the known fact that we can sometimes reduce rank by using non-trivial decompositions for small blocks.
    
    For now, let me implement a search-based approach that tries to find low-rank decompositions for the full 3x5x7 problem, but with a smarter initialization.
    """
    n, m, p = 3, 5, 7
    
    # Let me try to construct a decomposition using block structure
    # Split middle dimension into [2, 3]
    # Block 1: 3x2 * 2x7
    # Block 2: 3x3 * 3x7
    
    # For 3x2*2x7, let's see if we can do better than 42
    # For 3x3*3x7, let's see if we can do better than 63
    
    # Total target: 99, so we need to save 6 from 105
    
    # Let me try a computational search for the full problem with better initialization
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # 5s margin for 150s limit
    
    # Schoolbook as fallback
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_fb
    
    # Try to find lower rank using a more structured approach
    # Strategy: use block decomposition with Strassen for 2x2x2 sub-blocks
    
    # First, let's try a direct search with better randomization
    # Use entries from {-1, 0, 1} for U and V
    
    for r in range(104, 49, -1):
        if time.time() > deadline:
            break
        
        # Try a few random attempts for this rank
        attempts = 30 if r > 85 else 15
        
        for attempt in range(attempts):
            if time.time() > deadline:
                break
            
            # Generate random U, V with small integer entries
            U_rand = np.random.randint(-1, 2, (n * m, r))
            V_rand = np.random.randint(-1, 2, (m * p, r))
            
            U_3d = U_rand.reshape(n, m, r)
            V_3d = V_rand.reshape(m, p, r)
            
            W_3d = np.zeros((p, n, r))
            success = True
            
            for k in range(p):
                for i in range(n):
                    # Construct A matrix of size m x r
                    A = np.zeros((m, r))
                    for j in range(m):
                        for c in range(r):
                            A[j, c] = U_3d[i, j, c] * V_3d[j, k, c]
                    
                    # Solve A @ w = 1
                    try:
                        w, residuals, rank_A, s = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        
                        # Check if it's an exact solution
                        if not np.allclose(A @ w, np.ones(m), atol=1e-9):
                            success = False
                            break
                        
                        # Check if w is close to half-integers
                        w_scaled = 2 * w
                        w_int = np.round(w_scaled)
                        
                        if not np.allclose(w_scaled, w_int, atol=1e-6):
                            success = False
                            break
                            
                        for c in range(r):
                            W_3d[k, i, c] = w_int[c] / 2.0
                            
                    except:
                        success = False
                        break
                if not success:
                    break
            
            if success:
                # Verify with Fractions
                U_frac = [[Fraction(int(U_rand[idx, c])) for c in range(r)] for idx in range(n * m)]
                V_frac = [[Fraction(int(V_rand[idx, c])) for c in range(r)] for idx in range(m * p)]
                W_frac = [[Fraction(int(W_3d[k, i, c] * 2)) / 2 for c in range(r)] for k in range(p) for i in range(n)]
                
                if _verify(U_frac, V_frac, W_frac, n, m, p):
                    best_result = {"U": U_frac, "V": V_frac, "W": W_frac}
                    best_rank = r
                    return best_result
    
    # Try a structured block decomposition approach
    if time.time() < deadline:
        # Split middle dimension into [2, 3]
        # Block 1: 3x2 * 2x7 (indices j=0,1)
        # Block 2: 3x3 * 3x7 (indices j=2,3,4)
        
        # For each block, try to find a low-rank decomposition
        # Block 1: 3x2*2x7, schoolbook rank = 42
        # Block 2: 3x3*3x7, schoolbook rank = 63
        
        # Let's try to find decompositions for each block separately
        # and combine them
        
        def find_block_decomp(n_b, m_b, p_b, max_rank, time_limit=5.0):
            """Find a rank-minimal decomposition for n_b x m_b * m_b x p_b."""
            start = time.time()
            rank_fb_b = n_b * m_b * p_b
            
            # Schoolbook
            U_b = [[0] * rank_fb_b for _ in range(n_b * m_b)]
            V_b = [[0] * rank_fb_b for _ in range(m_b * p_b)]
            W_b = [[0] * rank_fb_b for _ in range(p_b * n_b)]
            col = 0
            for i in range(n_b):
                for j in range(m_b):
                    for k in range(p_b):
                        U_b[i * m_b + j][col] = 1
                        V_b[j * p_b + k][col] = 1
                        W_b[k * n_b + i][col] = 1
                        col += 1
            
            best_r = rank_fb_b
            best_U, best_V, best_W = U_b, V_b, W_b
            
            for r in range(min(max_rank, rank_fb_b - 1), 10, -1):
                if time.time() - start > time_limit:
                    break
                
                for attempt in range(20):
                    if time.time() - start > time_limit:
                        break
                    
                    U_rand = np.random.randint(-1, 2, (n_b * m_b, r))
                    V_rand = np.random.randint(-1, 2, (m_b * p_b, r))
                    
                    U_3d = U_rand.reshape(n_b, m_b, r)
                    V_3d = V_rand.reshape(m_b, p_b, r)
                    
                    W_3d = np.zeros((p_b, n_b, r))
                    success = True
                    
                    for k in range(p_b):
                        for i in range(n_b):
                            A = np.zeros((m_b, r))
                            for j in range(m_b):
                                for c in range(r):
                                    A[j, c] = U_3d[i, j, c] * V_3d[j, k, c]
                            
                            try:
                                w, _, _, _ = np.linalg.lstsq(A, np.ones(m_b), rcond=None)
                                if not np.allclose(A @ w, np.ones(m_b), atol=1e-9):
                                    success = False
                                    break
                                w_scaled = 2 * w
                                w_int = np.round(w_scaled)
                                if not np.allclose(w_scaled, w_int, atol=1e-6):
                                    success = False
                                    break
                                for c in range(r):
                                    W_3d[k, i, c] = w_int[c] / 2.0
                            except:
                                success = False
                                break
                        if not success:
                            break
                    
                    if success:
                        U_frac = [[Fraction(int(U_rand[idx, c])) for c in range(r)] for idx in range(n_b * m_b)]
                        V_frac = [[Fraction(int(V_rand[idx, c])) for c in range(r)] for idx in range(m_b * p_b)]
                        W_frac = [[Fraction(int(W_3d[k, i, c] * 2)) / 2 for c in range(r)] for k in range(p_b) for i in range(n_b)]
                        
                        if _verify(U_frac, V_frac, W_frac, n_b, m_b, p_b):
                            best_r = r
                            best_U, best_V, best_W = U_frac, V_frac, W_frac
                            break  # Found minimal for this r, try lower
                # If we found a solution at rank r, we can stop since we're going down
                if best_r == r:
                    break
            
            return best_U, best_V, best_W, best_r
        
        # Try to find decompositions for blocks
        if time.time() < deadline - 10:
            # Block 1: 3x2 * 2x7
            U1, V1, W1, r1 = find_block_decomp(3, 2, 7, 42, time_limit=30.0)
            
            if time.time() < deadline - 10:
                # Block 2: 3x3 * 3x7
                U2, V2, W2, r2 = find_block_decomp(3, 3, 7, 63, time_limit=30.0)
                
                # Combine blocks
                total_r = r1 + r2
                if total_r < best_rank:
                    U_comb = [[0] * total_r for _ in range(3 * 5)]
                    V_comb = [[0] * total_r for _ in range(5 * 7)]
                    W_comb = [[0] * total_r for _ in range(7 * 3)]
                    
                    # Block 1: j indices 0,1
                    for i in range(3):
                        for j in range(2):
                            for c in range(r1):
                                U_comb[i * 5 + j][c] = U1[i * 2 + j][c]
                        for k in range(7):
                            for c in range(r1):
                                W_comb[k * 3 + i][c] = W1[k * 3 + i][c]
                    for j in range(2):
                        for k in range(7):
                            for c in range(r1):
                                V_comb[j * 7 + k][c] = V1[j * 7 + k][c]
                    
                    # Block 2: j indices 2,3,4
                    offset = r1
                    for i in range(3):
                        for j in range(3):
                            for c in range(r2):
                                U_comb[i * 5 + (j + 2)][offset + c] = U2[i * 3 + j][c]
                        for k in range(7):
                            for c in range(r2):
                                W_comb[k * 3 + i][offset + c] = W2[k * 3 + i][c]
                    for j in range(3):
                        for k in range(7):
                            for c in range(r2):
                                V_comb[(j + 2) * 7 + k][offset + c] = V2[j * 7 + k][c]
                    
                    # Verify combined decomposition
                    if _verify(U_comb, V_comb, W_comb, 3, 5, 7):
                        best_result = {"U": U_comb, "V": V_comb, "W": W_comb}
                        best_rank = total_r
                        return best_result
    
    return best_result
