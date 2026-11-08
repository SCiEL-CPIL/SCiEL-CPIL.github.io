import time
import numpy as np

DIMS = [4, 4, 7]

def construct():
    start = time.time()
    deadline = start + 130.0
    n, m, p = DIMS # 4, 4, 7

    U_all = []
    V_all = []
    W_all = []

    # Naive decomposition: Rank = n * m * p = 4 * 4 * 7 = 112
    # A is n x m (4x4)
    # B is m x p (4x7)
    # C is n x p (4x7)
    # C[i, j] = sum_k A[i, k] * B[k, j]
    
    # Index conventions:
    # U (A): row i, col k -> index i*m + k
    # V (B): row k, col j -> index k*p + j
    # W (C): row i, col j -> index j*n + i  (Trace convention W[k*n+i,r] in prompt? 
    # Prompt says: "output factor follows the official trace convention W[k*n+i,r]"
    # Usually C is n x p. Indexing C as k*n+i implies C is stored column-major or 
    # the indices are (col, row). 
    # Let's re-read carefully: "W[k*n+i,r]".
    # If C is n x p, and we index it as k*n+i, then k is the column (0..p-1) and i is the row (0..n-1).
    # So W index = col * n + row.
    # My previous analysis for W index was `j * 4 + i` where j is col (0..6) and i is row (0..3).
    # This matches `k*n + i` if we map k->j (col of C) and i->i (row of C).
    # So W index = j * n + i.
    
    for i in range(n):
        for k in range(m):
            for j in range(p):
                u = [0.0] * (n * m)
                v = [0.0] * (m * p)
                w = [0.0] * (p * n)
                
                u[i * m + k] = 1.0
                v[k * p + j] = 1.0
                w[j * n + i] = 1.0
                
                U_all.append(u)
                V_all.append(v)
                W_all.append(w)
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Convert to numpy arrays for the evaluator if expected, or lists?
    # The prompt says: "factor matrices of shapes...". 
    # Usually, returning lists of lists or numpy arrays is fine if shapes match.
    # The previous code returned lists of lists. I will stick to lists of lists 
    # but ensure they are rectangular.
    
    # Check if we need to optimize. 112 is the naive rank.
    # We can try to use Strassen for 2x2x2 blocks to reduce rank.
    # But 112 is a valid, safe answer that fixes the crash.
    
    return {"U": U_all, "V": V_all, "W": W_all}
