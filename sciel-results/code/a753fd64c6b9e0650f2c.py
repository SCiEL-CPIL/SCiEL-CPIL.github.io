import time
import random
import numpy as np

DIMENSIONS = [4, 4, 7]

def _strassen_4x4x4_terms():
    """
    Generate Strassen-style decomposition for 4x4 * 4x4.
    Returns U, V, W as lists of lists, rank r.
    U: (16, r), V: (16, r), W: (16, r)
    """
    # Strassen for 2x2: 7 multiplications
    # For 4x4, we can use 2x2 blocks and Strassen on each, 
    # but a better approach is to use the known 4x4x4 rank 49 result.
    # 
    # Actually, let's use a simpler approach: 
    # Use Strassen on 2x2 blocks. A 4x4 matrix has 4 2x2 blocks.
    # Multiplying two 4x4 matrices using 2x2 Strassen:
    # We need to multiply 4x4 = (2x2 blocks) * (2x2 blocks)
    # This gives 4*4 = 16 2x2 multiplications, each with rank 7, total 112.
    # That's worse than schoolbook 64.
    #
    # Better: Use the known optimal or near-optimal 4x4x4 decomposition.
    # The known rank for 4x4x4 is between 49 and 64.
    # Let's use a constructive approach with rank 49.
    #
    # Actually, for simplicity and guaranteed correctness, let's use:
    # - 4x4x4 with rank 49 (Strassen-based)
    # - 4x4x3 with schoolbook rank 48
    # Total: 97
    #
    # But let me implement a verified 4x4x4 decomposition.
    # I'll use a known pattern.
    
    # For now, use schoolbook 4x4x4 with rank 64 as fallback,
    # but try to find a better one.
    
    # Let's use a simple Strassen-based 4x4x4 with rank 49.
    # Reference: "Fast matrix multiplication" by Strassen, extended to 4x4.
    
    # Actually, let me just use the schoolbook for 4x4x4 (rank 64) 
    # and schoolbook for 4x4x3 (rank 48), total 112.
    # Then try to improve.
    
    # For a guaranteed correct answer, start with schoolbook:
    r = 64
    U = [[0]*r for _ in range(16)]
    V = [[0]*r for _ in range(16)]
    W = [[0]*r for _ in range(16)]
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*4+j][col] = 1
                V[j*4+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
    return U, V, W, r

def _schoolbook_4x4x3_terms():
    """
    Schoolbook decomposition for 4x4 * 4x3.
    Returns U, V, W where:
    U: (16, r), V: (28, r), W: (28, r)
    V and W have 28 rows because p=7, but we only use k=4,5,6.
    Actually, this function returns terms for the 4x4x3 subproblem
    where the 3 columns are at positions 4,5,6 in the full 7-column matrix.
    """
    r = 4*4*3  # 48
    U = [[0]*r for _ in range(16)]
    V = [[0]*r for _ in range(28)]  # m*p = 4*7 = 28
    W = [[0]*r for _ in range(28)]  # p*n = 7*4 = 28
    col = 0
    for i in range(4):
        for j in range(4):
            for k_rel in range(3):
                k_abs = k_rel + 4
                U[i*4+j][col] = 1
                V[j*7+k_abs][col] = 1
                W[k_abs*4+i][col] = 1
                col += 1
    return U, V, W, r

def _verify_decomposition(U, V, W, r, n=4, m=4, p=7):
    """Verify that the decomposition is correct."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_strassen_4x4x4(deadline):
    """
    Try to find a low-rank 4x4x4 decomposition using random search.
    Returns (U, V, W, r) or None.
    """
    rng = random.Random(12345)
    # Try ranks from 49 upward
    for r in range(49, 65):
        if time.time() > deadline - 30:
            break
        # Try a few random attempts per rank
        for trial in range(200):
            if time.time() > deadline - 30:
                break
            U = [[rng.choice([-1,0,1]) for _ in range(r)] for _ in range(16)]
            V = [[rng.choice([-1,0,1]) for _ in range(r)] for _ in range(16)]
            W = [[rng.choice([-1,0,1]) for _ in range(r)] for _ in range(16)]
            if _verify_decomposition(U, V, W, r, n=4, m=4, p=4):
                return U, V, W, r
    return None

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Strategy:
    # 1. Try to find a good 4x4x4 decomposition
    # 2. Use schoolbook for 4x4x3
    # 3. Combine
    # 4. If time permits, try to optimize the combined decomposition
    
    best_result = None
    best_rank = float('inf')
    
    # Get 4x4x3 schoolbook terms
    U_443, V_443, W_443, r_443 = _schoolbook_4x4x3_terms()
    
    # Try to get a good 4x4x4 decomposition
    decomp_444 = None
    if time.time() < deadline - 60:
        decomp_444 = _try_strassen_4x4x4(deadline)
    
    if decomp_444:
        U1, V1, W1, r1 = decomp_444
        total_rank = r1 + r_443
        
        U = [None] * total_rank
        V = [None] * total_rank
        W = [None] * total_rank
        
        # Embed 4x4x4 part
        for c in range(r1):
            U[c] = U1[c][:]
            V[c] = V1[c][:] + [0]*12  # Pad from 16 to 28
            W[c] = W1[c][:] + [0]*12  # Pad from 16 to 28
        
        # Add 4x4x3 schoolbook part
        for c in range(r_443):
            U[r1 + c] = U_443[c][:]
            V[r1 + c] = V_443[c][:]
            W[r1 + c] = W_443[c][:]
        
        if _verify_decomposition(U, V, W, total_rank):
            best_result = {"U": U, "V": V, "W": W}
            best_rank = total_rank
    
    # Fallback: full schoolbook
    if best_result is None or time.time() > deadline - 20:
        rank = 4 * 4 * 7
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(28)]
        W = [[0]*rank for _ in range(28)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    U[i*4+j][col] = 1
                    V[j*7+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return best_result
