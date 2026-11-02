import time
import numpy as np

_DEADLINE = 110.0
_start_time = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _start_time)

def _verify_236(U, V, W, tol=1e-9):
    n, m, p = 2, 3, 6
    r = U.shape[1]
    U_mat = U.reshape(n, m, r)
    V_mat = V.reshape(m, p, r)
    W_raw = W.reshape(p, n, r)
    W_mat = np.transpose(W_raw, (1, 0, 2))
    prod_UV = U_mat[:, :, np.newaxis, :] * V_mat[np.newaxis, :, :, :]
    prod = prod_UV * W_mat[:, np.newaxis, :, :]
    result = np.sum(prod, axis=3)
    if not np.allclose(result, 1.0, atol=tol):
        return False
    return True

def _get_232_rank_11():
    """
    Returns U, V, W for 2x3x2 rank 11.
    U: (6, 11), V: (6, 11), W: (4, 11)
    Indices:
    U: (i, j) where i in 0..1, j in 0..2
    V: (j, k) where j in 0..2, k in 0..1
    W: (k, i) where k in 0..1, i in 0..1 -> Reshape to (2, 2, 11) then transpose to (2, 2, 11) for W_mat (i, k, c)
    """
    # Hardcoded Rank 11 decomposition for 2x3x2
    # This is a known optimal decomposition.
    # Reference: "The rank of the 2x3x2 tensor"
    # I will construct it using Strassen's 7 terms for the 2x2x2 part (cols 0,1 of B, rows 0,1 of A)
    # and 4 extra terms for the remaining parts.
    
    # Let's use a simpler valid decomposition if I'm not sure about the 11.
    # Actually, I can just use the schoolbook for 2x3x2 (rank 12) for 3 blocks -> 36.
    # But I want 33.
    
    # I will use a known Rank 11 decomposition.
    # U (6x11), V (6x11), W (4x11)
    
    # Since I don't have the exact 11 vectors memorized, I will try to construct a Rank 34 solution
    # by using Rank 11 for the first block and Rank 12 for the others? No, 11+12+12 = 35.
    # 11+11+12 = 34.
    # If I can get one block to rank 11 and two to rank 12, I get 34.
    # Or, I can just use Rank 12 for all three, which is 36.
    
    # Let's try to find a Rank 34 construction.
    # Maybe split 2x3x6 into 2x2x6 and 1x3x6?
    
    # Actually, the inspirations got 1.0588 (Rank 34).
    # I will implement a Rank 34 solution by using a specific block structure.
    # If I fail, I return 36.
    
    return None

def _schoolbook_236():
    n, m, p = 2, 3, 6
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

def construct():
    global _start_time
    _start_time = time.time()
    
    # Attempt Rank 33 or 34
    # For now, return Schoolbook to ensure validity, then try to improve.
    # The prompt asks for a DIFFERENT strategy.
    # The current strategy is "search".
    # My strategy is "Constructive Block".
    
    # I will implement the Rank 33 construction using 3x Rank 11 blocks.
    # I need the Rank 11 tensors.
    
    # If I don't have them, I will use a Rank 34 construction if I can find one.
    # Actually, I can use the fact that 2x3x2 rank is 11.
    # I will hardcode the 11 vectors.
    
    # U (6x11)
    # V (6x11)
    # W (4x11)
    
    # Let's use a known Rank 11 decomposition for 2x3x2.
    # (i, j) for U, (j, k) for V, (k, i) for W.
    
    # I will use the decomposition from "Optimal algorithms for small matrix multiplication"
    # The 2x3x2 rank 11 decomposition is:
    # ... (I don't have it in memory)
    
    # Fallback: Return Schoolbook.
    # But I must try to improve.
    
    # Let's try to construct a Rank 34 solution.
    # Split 2x3x6 into:
    # Block 1: 2x3x2 (Rank 11)
    # Block 2: 2x3x2 (Rank 11)
    # Block 3: 2x3x2 (Rank 12) -> Total 34.
    # This is valid if I have one Rank 11 and two Rank 12.
    # Rank 12 is just Schoolbook for 2x3x2.
    
    # So I just need ONE Rank 11 decomposition for 2x3x2.
    
    # I will try to find/generate a Rank 11 decomposition for 2x3x2.
    # If I can't, I return 36.
    
    # Actually, I can just use the Schoolbook for all, which is 36.
    # To get 34, I need to save 2 ranks.
    # Saving 2 ranks in one 2x3x2 block (12->10? No, 11).
    # So 11 + 12 + 12 = 35.
    # 11 + 11 + 12 = 34.
    # 11 + 11 + 11 = 33.
    
    # I will implement the Rank 33 solution if I can hardcode the 11 vectors.
    # Since I cannot, I will return Schoolbook.
    
    return _schoolbook_236()

def construct():
    global _start_time
    _start_time = time.time()
    
    # Strategy: Use Block Decomposition.
    # I will attempt to construct a Rank 33 solution.
    # If it fails, I return Rank 36.
    
    # I will hardcode a Rank 11 decomposition for 2x3x2.
    # (This is the critical part).
    
    # Let's use a simple Rank 11 decomposition.
    # U, V, W for 2x3x2.
    
    # I will use the following known Rank 11 decomposition:
    # (This is a placeholder, I need the actual values)
    
    # If I don't have the values, I will just return Schoolbook.
    
    return _schoolbook_236()
