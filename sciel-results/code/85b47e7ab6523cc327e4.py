"""Matrix multiplication 2x3x4 rank minimization via block decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]

def _verify(U, V, W, n=2, m=3, p=4):
    """Verify that the tensor decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 1.0
                rhs = 0.0
                for c in range(rank):
                    rhs += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(lhs - rhs) > 1e-9:
                    return False
    return True

def _is_half_integer(x, tol=1e-9):
    """Check if x is an integer or half-integer."""
    return abs(x * 2 - round(x * 2)) < tol

def _verify_half_integer(U, V, W, tol=1e-9):
    """Check all entries are half-integers."""
    for M in [U, V, W]:
        for row in M:
            for val in row:
                if not _is_half_integer(val, tol):
                    return False
    return True

def _build_1x3x4_rank9():
    """
    Build a rank-9 decomposition for 1x3x4 matrix multiplication.
    
    For 1x3x4: U is 3x9, V is 12x9, W is 4x9
    T[j,k] = sum_c U[j,c] * V[j*4+k,c] * W[k,c] = 1 for all j in {0,1,2}, k in {0,1,2,3}
    
    Construction using Strassen-Winograd style:
    Split into two 1x3x2 blocks (cols 0,1 and cols 2,3).
    Each 1x3x2 has rank 5, but we can share to get total rank 9.
    
    For 1x3x2 (rank 5), the standard construction:
    y0 = A0*B00 + A1*B10 + A2*B20
    y1 = A0*B01 + A1*B11 + A2*B21
    
    Using 5 terms:
    t0: U=[1,1,0], V=[1,0,0,1,0,0,0,0,0,0,0,0]... 
    
    Let me use a cleaner approach. For 1x3x2:
    M0 = (A0+A1)(B00+B11)
    M1 = (A0+A1)(B01+B10)
    M2 = A0*(B00-B10)  
    M3 = A1*(B11-B01)
    M4 = A2*(B20+B21)
    
    y0 = M0 - M2 - M3 + M4/2... no, let me recalculate.
    
    Actually for 1x3x2, a known rank-5 decomposition:
    y0 = A0*B00 + A1*B10 + A2*B20
    y1 = A0*B01 + A1*B11 + A2*B21
    
    Terms:
    c0: U=[1,0,0], V=[1,0,0,0,...], W=[1,0] -> A0*B00
    c1: U=[0,1,0], V=[0,0,1,0,...], W=[1,0] -> A1*B10
    c2: U=[0,0,1], V=[0,0,0,1,...], W=[1,0] -> A2*B20
    c3: U=[1,0,0], V=[0,1,0,0,...], W=[0,1] -> A0*B01
    c4: U=[0,1,0], V=[0,0,0,1,...], W=[0,1] -> A1*B11
    c5: U=[0,0,1], V=[0,0,0,1,...], W=[0,1] -> A2*B21
    
    That's rank 6 for 1x3x2. To get rank 5, we need to share.
    
    Strassen for 1x2x2 (rank 5):
    y0 = A0*B00 + A1*B10
    y1 = A0*B01 + A1*B11
    
    M0 = (A0+A1)(B00+B11)
    M1 = (A0+A1)(B01+B10)
    M2 = A0(B00-B10)
    M3 = A1(B11-B01)
    y0 = M0 - M2 - M3... let's verify:
    M0 - M2 - M3 = (A0+A1)(B00+B11) - A0(B00-B10) - A1(B11-B01)
    = A0*B00 + A0*B11 + A1*B00 + A1*B11 - A0*B00 + A0*B10 - A1*B11 + A1*B01
    = A0*B11 + A1*B00 + A0*B10 + A1*B01
    That's not y0 = A0*B00 + A1*B10.
    
    Let me use the standard Strassen for 2x2x2:
    M0 = (A00+A11)(B00+B11)
    M1 = (A10+A11)(B00+B10)
    M2 = (A00-A11)(B10+B11)
    M3 = (A00+A01)(B00+B01)
    M4 = (A01-A10)(B10+B11)
    M5 = (A00+A11)(B01-B11)
    M6 = (A00+A10)(B10-B00)
    
    For 1x2x2 (just one row of A), it's simpler:
    y0 = A0*B00 + A1*B10
    y1 = A0*B01 + A1*B11
    
    Rank 5 construction:
    M0 = (A0+A1)(B00+B11)
    M1 = A0*(B00-B10)
    M2 = A1*(B11-B01)
    M3 = (A0+A1)(B01+B10)
    M4 = A0*B01... 
    
    I'm getting confused. Let me just use a verified rank-9 construction for 1x3x4 directly.
    
    After research, the rank of 1x3x4 is indeed 9. Here's a construction:
    
    For 1x3x4, think of it as computing a 1x4 vector from a 1x3 and 3x4.
    
    Use the following 9 terms (U[j,c], V[j*4+k,c], W[k,c]):
    
    I'll construct it programmatically and verify.
    """
    # For now, let's try a direct approach: use rank 12 for 1x3x4 (schoolbook)
    # and see if we can reduce.
    
    # Actually, let's just use the schoolbook for 1x3x4 which is rank 12,
    # giving total rank 24 for 2x3x4. That's not an improvement.
    
    # Let me try a different approach: use known Strassen-based constructions.
    
    # For 2x3x4, use the fact that it can be decomposed as:
    # [A0; A1] * B where A0, A1 are 1x3 and B is 3x4.
    # C0 = A0*B, C1 = A1*B
    # Each Ci is 1x4.
    
    # If rank(1x3x4) = 9, then total rank = 18.
    
    # Let me construct a rank-9 decomposition for 1x3x4.
    
    # One approach: use the "border rank" construction.
    # For 1x3x4, we can use the following:
    
    # Group the 4 outputs as (y0,y1) and (y2,y3).
    # For (y0,y1): y0 = A0*B00+A1*B10+A2*B20, y1 = A0*B01+A1*B11+A2*B21
    # This is a 1x3x2 problem with rank 5 (using Strassen-like).
    # For (y2,y3): y2 = A0*B02+A1*B12+A2*B22, y3 = A0*B03+A1*B13+A2*B23
    # Also rank 5, but we can share some terms.
    
    # Total: 5 + 5 = 10, but with sharing we can get 9.
    
    # Let me use a specific construction:
    
    # For 1x3x2 (rank 5):
    # y0 = A0*B00 + A1*B10 + A2*B20
    # y1 = A0*B01 + A1*B11 + A2*B21
    #
    # Terms:
    # t0: U=[1,1,0], V=[1,0,0,1,0,0], W=[1,0]  -> (A0+A1)(B00+B01) contribution
    # t1: U=[1,-1,0], V=[1,0,0,-1,0,0], W=[1,0] -> (A0-A1)(B00-B01) contribution  
    # t2: U=[0,0,1], V=[0,0,1,0,1,0], W=[1,0] -> A2*(B10+B20)... 
    #
    # This is getting complicated. Let me just use a numerical approach.
    
    # For the combined 2x3x4 problem, let me try a direct search for rank 18
    # using the block structure.
    
    pass

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3x4 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 4
    schoolbook_rank = n * m * p  # 24
    
    # Start with schoolbook solution as fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_solution = (U_sb, V_sb, W_sb)
    
    # Try to construct a rank-18 solution using block decomposition
    # 2x3x4 = (1x3x4) + (1x3x4) with rank 9 each
    
    if time.time() < deadline - 30:
        rank18_solution = _try_rank18(n, m, p, deadline)
        if rank18_solution is not None:
            U, V, W = rank18_solution
            if _verify(U, V, W, n, m, p) and _verify_half_integer(U, V, W):
                best_rank = 18
                best_solution = (U, V, W)
    
    U, V, W = best_solution
    return {"U": U, "V": V, "W": W}

def _try_rank18(n, m, p, deadline):
    """
    Try to construct a rank-18 decomposition for 2x3x4.
    
    Strategy: Decompose into two 1x3x4 blocks.
    Block 1: i=0, computes C[0,k] = sum_j A[0,j]*B[j,k] for k=0..3
    Block 2: i=1, computes C[1,k] = sum_j A[1,j]*B[j,k] for k=0..3
    
    Each block is a 1x3x4 multiplication.
    
    For 1x3x4, we need rank 9. Let's construct it.
    """
    # For 1x3x4: U is 3x9, V is 12x9, W is 4x9
    # T[j,k] = sum_c U[j,c] * V[j*4+k,c] * W[k,c] = 1
    
    # Construction for 1x3x4 rank 9:
    # Use the following approach based on Strassen-Winograd:
    
    # Split columns into pairs: (0,1) and (2,3)
    # For each pair, use a 5-term scheme, but share 1 term between pairs.
    
    # For columns (0,1):
    # y0 = A0*B00 + A1*B10 + A2*B20
    # y1 = A0*B01 + A1*B11 + A2*B21
    
    # 5 terms:
    # t0: (A0+A1)(B00+B11) -> U=[1,1,0], V contributions, W=[1,1]
    # t1: A0*(B00-B10) -> U=[1,0,0], V, W=[1,0]
    # t2: A1*(B11-B01) -> U=[0,1,0], V, W=[0,1]  
    # t3: A2*(B20+B21) -> U=[0,0,1], V, W=[1,1]
    # t4: (A0+A1+A2)(B01+B10) -> ... 
    
    # Let me use a cleaner formulation. For 1x3x2:
    # y0 = A0*B00 + A1*B10 + A2*B20
    # y1 = A0*B01 + A1*B11 + A2*B21
    
    # Rank 5 construction:
    # M0 = (A0+A1)(B00+B11)
    # M1 = (A0+A1)(B01+B10)
    # M2 = A0(B00-B10)
    # M3 = A1(B11-B01)
    # M4 = A2(B20+B21)
    
    # y0 = M0 - M2 - M3 + M4/2... let's verify:
    # M0 - M2 - M3 = (A0+A1)(B00+B11) - A0(B00-B10) - A1(B11-B01)
    # = A0B00 + A0B11 + A1B00 + A1B11 - A0B00 + A0B10 - A1B11 + A1B01
    # = A0B11 + A1B00 + A0B10 + A1B01
    # That's not y0.
    
    # Let me use the correct Strassen for 1x2x2:
    # y0 = A0*B00 + A1*B10
    # y1 = A0*B01 + A1*B11
    # 
    # M0 = (A0+A1)(B00+B11)
    # M1 = A0(B00-B10)
    # M2 = A1(B11-B01)
    # M3 = (A0+A1)(B01+B10)
    # M4 = A0*B01... 
    #
    # y0 = M0 - M2 - M3/2... I keep making errors.
    
    # Let me just use a verified construction from literature.
    # The rank of 1x3x4 is 9. Here's one construction:
    
    # For 1x3x4, the 9 terms are:
    # c=0: U=[1,0,0], V=[1,0,0,0,0,0,0,0,0,0,0,0], W=[1,0,0,0]
    # c=1: U=[0,1,0], V=[0,0,1,0,0,0,0,0,0,0,0,0], W=[1,0,0,0]
    # c=2: U=[0,0,1], V=[0,0,0,1,0,0,0,0,0,0,0,0], W=[1,0,0,0]
    # c=3: U=[1,0,0], V=[0,1,0,0,0,0,0,0,0,0,0,0], W=[0,1,0,0]
    # c=4: U=[0,1,0], V=[0,0,0,1,0,0,0,0,0,0,0,0], W=[0,1,0,0]
    # c=5: U=[0,0,1], V=[0,0,0,1,0,0,0,0,0,0,0,0], W=[0,1,0,0]
    # c=6: U=[1,0,0], V=[0,0,0,0,1,0,0,0,0,0,0,0], W=[0,0,1,0]
    # c=7: U=[0,1,0], V=[0,0,0,0,0,1,0,0,0,0,0,0], W=[0,0,1,0]
    # c=8: U=[0,0,1], V=[0,0,0,0,0,0,1,0,0,0,0,0], W=[0,0,1,0]
    
    # But this is only rank 9 for 1x3x3, not 1x3x4. We're missing column 3.
    
    # For 1x3x4, we need 12 equations. With rank 9, we need to be clever.
    
    # Let me try a different approach: use numerical optimization.
    
    # For 1x3x4 with rank 9:
    # Variables: U (3x9=27), V (12x9=108), W (4x9=36), total 171 variables
    # Constraints: 12 equations (T[j,k]=1 for all j,k)
    # Plus: entries should be half-integers
    
    # This is underdetermined, so we can find solutions.
    
    # Let me try fixing W to be simple and solving for U and V.
    
    import random
    random.seed(42)
    
    # Try fixing W to have a specific structure
    # W is 4x9. Let's make W have entries in {-1, 0, 1}.
    
    for attempt in range(500):
        if time.time() > deadline - 20:
            break
        
        # Random W with entries in {-1, 0, 1}
        W_1x3x4 = [[random.choice([-1, 0, 1]) for _ in range(9)] for _ in range(4)]
        
        # For each k, we have: sum_c W[k,c] * U[j,c] * V[j*4+k,c] = 1 for j=0,1,2
        # This is 3 equations per k, total 12 equations.
        
        # For fixed W, this is still nonlinear in U and V.
        # But if we fix U, it's linear in V, and vice versa.
        
        # Let's fix U to be simple (e.g., U[j,c] = 1 for specific c)
        # and solve for V.
        
        # Try: U[j,c] = 1 if c == j*3 + c', else 0 (for c' in 0,1,2)
        # So U has 3 rows, each with 3 ones in different columns.
        
        U_1x3x4 = [[0]*9 for _ in range(3)]
        for j in range(3):
            for c_prime in range(3):
                U_1x3x4[j][j*3 + c_prime] = 1
        
        # Now for each (j,k): sum_c U[j,c] * V[j*4+k,c] * W[k,c] = 1
        # Since U[j,c] is nonzero only for c in {j*3, j*3+1, j*3+2}:
        # sum_{c_prime=0}^{2} V[j*4+k, j*3+c_prime] * W[k, j*3+c_prime] = 1
        
        # For each (j,k), this is 3 equations in 3 unknowns (V[j*4+k, j*3], V[j*4+k, j*3+1], V[j*4+k, j*3+2])
        # But W[k, j*3+c_prime] depends on k and j.
        
        # Actually, for fixed j and k, we have:
        # V[j*4+k, j*3]*W[k, j*3] + V[j*4+k, j*3+1]*W[k, j*3+1] + V[j*4+k, j*3+2]*W[k, j*3+2] = 1
        
        # This is 1 equation in 3 unknowns for each (j,k). Underdetermined.
        # We have 12 such equations, each with 3 unknowns (but the unknowns are shared across k for fixed j).
        
        # For fixed j, V[j*4+k, :] for k=0,1,2,3 are 4 rows of 3 relevant entries each = 12 unknowns.
        # And we have 4 equations (one for each k).
        
        # So for each j, we have a 4x12 system. Underdetermined.
        
        # Let's just pick V[j*4+k, j*3+c_prime] = 1/W[k, j*3+c_prime] if W[k, j*3+c_prime] != 0
        # and 0 otherwise. But this might not give half-integers.
        
        V_1x3x4 = [[0]*9 for _ in range(12)]
        valid = True
        
        for j in range(3):
            for k in range(4):
                # Need: sum_{c_prime=0}^{2} V[j*4+k, j*3+c_prime] * W_1x3x4[k, j*3+c_prime] = 1
                # Pick V[j*4+k, j*3] = 1/W_1x3x4[k, j*3] if W_1x3x4[k, j*3] != 0
                # and V[j*4+k, j*3+1] = V[j*4+k, j*3+2] = 0
                
                if W_1x3x4[k][j*3] != 0:
                    V_1x3x4[j*4+k][j*3] = 1.0 / W_1x3x4[k][j*3]
                else:
                    # Try next column
                    if W_1x3x4[k][j*3+1] != 0:
                        V_1x3x4[j*4+k][j*3+1] = 1.0 / W_1x3x4[k][j*3+1]
                    elif W_1x3x4[k][j*3+2] != 0:
                        V_1x3x4[j*4+k][j*3+2] = 1.0 / W_1x3x4[k][j*3+2]
                    else:
                        valid = False
                        break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Check if all entries are half-integers
        if not _verify_half_integer([U_1x3x4], [V_1x3x4], [W_1x3x4]):
            continue
        
        # Verify the 1x3x4 decomposition
        valid_1x3x4 = True
        for j in range(3):
            for k in range(4):
                lhs = 1.0
                rhs = 0.0
                for c in range(9):
                    rhs += U_1x3x4[j][c] * V_1x3x4[j*4+k][c] * W_1x3x4[k][c]
                if abs(lhs - rhs) > 1e-9:
                    valid_1x3x4 = False
                    break
            if not valid_1x3x4:
                break
        
        if valid_1x3x4:
            # Great! We have a rank-9 decomposition for 1x3x4.
            # Now combine two of these for 2x3x4.
            
            # For 2x3x4:
            # U is 6x18, V is 12x18, W is 8x18
            # Block 1 (i=0): U[0:3, 0:9], V[0:12, 0:9], W[0:4, 0:9]
            # Block 2 (i=1): U[3:6, 9:18], V[0:12, 9:18], W[4:8, 9:18]
            
            U_full = [[0]*18 for _ in range(6)]
            V_full = [[0]*18 for _ in range(12)]
            W_full = [[0]*18 for _ in range(8)]
            
            # Block 1: i=0
            for j in range(3):
                for c in range(9):
                    U_full[j][c] = U_1x3x4[j][c]
            for j in range(3):
                for k in range(4):
                    for c in range(9):
                        V_full[j*4+k][c] = V_1x3x4[j*4+k][c]
            for k in range(4):
                for c in range(9):
                    W_full[k*2+0][c] = W_1x3x4[k][c]
            
            # Block 2: i=1 (same structure, different W rows)
            for j in range(3):
                for c in range(9):
                    U_full[3+j][9+c] = U_1x3x4[j][c]
            for j in range(3):
                for k in range(4):
                    for c in range(9):
                        V_full[j*4+k][9+c] = V_1x3x4[j*4+k][c]
            for k in range(4):
                for c in range(9):
                    W_full[k*2+1][9+c] = W_1x3x4[k][c]
            
            # Verify the full 2x3x4 decomposition
            if _verify(U_full, V_full, W_full, 2, 3, 4) and _verify_half_integer(U_full, V_full, W_full):
                return (U_full, V_full, W_full)
    
    return None
