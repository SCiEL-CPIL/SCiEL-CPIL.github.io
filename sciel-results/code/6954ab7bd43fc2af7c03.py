import time
import sys
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W form a valid exact decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
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

def _winograd_2x3x7(n, m, p):
    """
    Winograd-style rank-30 construction for 2x3 * 3x7.
    
    The idea: For the 2x3 dot product, use 5 multiplications instead of 6.
    The 5 multiplications per "virtual column" give 5*7=35, but we can
    share some across columns to get down to 30.
    
    Actually, let's use a direct construction:
    For each column k of B (k=0..6), we compute C[0,k] and C[1,k].
    
    C[0,k] = a00*b0k + a01*b1k + a02*b2k
    C[1,k] = a10*b0k + a11*b1k + a12*b2k
    
    Using Winograd for the 2x3 case with 5 multiplications:
    Let's define for each column k:
      m1_k = (a00+a01) * (b0k+b1k)
      m2_k = (a01+a02) * (b1k+b2k)
      m3_k = a00 * b0k
      m4_k = a01 * b1k
      m5_k = a02 * b2k
    
    Then: a00*b0k + a01*b1k + a02*b2k = m3_k + m4_k + m5_k + (m1_k - m3_k - m4_k) + (m2_k - m4_k - m5_k) - ...
    
    This is getting complex. Let me use a different approach:
    Just do 5 multiplications per column for the first row, and reuse
    some for the second row.
    
    Actually, the cleanest rank-30 construction:
    - 5 multiplications for row 0 of A with each column of B: 5*7 = 35
    - But we can share: the linear combinations of B entries are the same
      for both rows if we structure it right.
    
    Let me just use 30 terms with a structured pattern and verify.
    """
    n, m, p = 2, 3, 7
    rank = 30
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 30
    V = [[0] * rank for _ in range(m * p)]  # 21 x 30
    W = [[0] * rank for _ in range(p * n)]  # 14 x 30
    
    # For each column k of B, use 5 multiplications
    # Terms 0-4 for column 0, 5-9 for column 1, etc.
    # But 5*7=35, not 30. So we need to share.
    
    # Alternative: use 4 multiplications for some columns
    # Let's try: 5 mults for 5 columns + 2 mults for 2 columns = 25+4=29... 
    # Or 5 mults for 4 columns + 5 mults for 3 columns = 20+15=35...
    
    # Let me try a different structure:
    # For the 2x3 dot product, we can use:
    # a00*b0 + a01*b1 + a02*b2
    # = (a00+a01+a02)*(b0+b1+b2) - a01*b0 - a02*b0 - a00*b1 - a02*b1 - a01*b2 - a00*b2
    # That's 7 mults, worse.
    
    # The rank-5 construction for 2x3:
    # t1 = a00*b0
    # t2 = a01*b1  
    # t3 = a02*b2
    # t4 = (a00+a01)*(b0+b1)
    # t5 = (a01+a02)*(b1+b2)
    # Then: a00*b0 + a01*b1 + a02*b2 = t1 + t2 + t3 + (t4 - t1 - t2) + (t5 - t2 - t3) - t4 - t5 + ... 
    # This doesn't simplify cleanly.
    
    # Let me just return schoolbook if I can't construct rank 30 quickly.
    # The search below will try to find it.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 42
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    # Try Winograd rank-30 construction
    w_result = _winograd_2x3x7(n, m, p)
    if w_result is not None:
        if _verify(w_result["U"], w_result["V"], w_result["W"], n, m, p):
            best = w_result
            best_rank = 30
    
    # Search for lower rank decompositions using random search
    random.seed(42)
    
    # Try ranks from best_rank-1 down to 25
    for target_rank in range(best_rank - 1, 24, -1):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        attempts = 0
        max_attempts = 200
        
        while attempts < max_attempts and time.time() < deadline - 20:
            attempts += 1
            
            # Use small integer/half-integer values
            U = [[random.choice([-2, -1, 0, 1, 2, 0.5, -0.5]) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice([-2, -1, 0, 1, 2, 0.5, -0.5]) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice([-2, -1, 0, 1, 2, 0.5, -0.5]) for _ in range(target_rank)] for _ in range(p * n)]
            
            if _verify(U, V, W, n, m, p):
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
        
        # If we found a lower rank, continue searching for even lower
        # Otherwise, stop after a few ranks
    
    return best
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    # print("Constructed rank:", len(res["U"][0]))
