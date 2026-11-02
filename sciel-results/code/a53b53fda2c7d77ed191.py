import numpy as np

def construct():
    n, m, p = 2, 4, 8
    # Rank 48 decomposition
    rank = 48
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen 2x2x1 coefficients
    # C0 = M1 - M2
    # C1 = M3 + M2
    # M1 = (A00+A11)(B0+B1)
    # M2 = (A10+A11)B0
    # M3 = A00(B1-B0)
    # This is for 2x2x1?
    # Let's verify.
    # A = [[A00, A01], [A10, A11]]
    # B = [B0, B1] (vector)
    # C = [C0, C1]
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    
    # M1 = (A00+A11) * (B0+B1) = A00B0 + A00B1 + A11B0 + A11B1
    # M2 = (A10+A11) * B0 = A10B0 + A11B0
    # M3 = A00 * (B1-B0) = A00B1 - A00B0
    
    # C0 = M1 - M2 + ...?
    # M1 - M2 = A00B0 + A00B1 + A11B1 = C0 + A11B1.
    # We need A11B1.
    # This decomposition is for 2x2x2?
    
    # Rank of 2x2x1 is 3.
    # M1 = A00 B0
    # M2 = A01 B1
    # M3 = (A10+A11) B0 ... no.
    
    # Let's use the known rank 3 decomposition for 2x2x1.
    # C0 = M1 + M2
    # C1 = M3 + M2
    # M1 = A00 B0
    # M2 = A01 B1
    # M3 = A10 B0 + A11 B1 - M2? No.
    
    # Actually, 2x2x1 is just two dot products.
    # C0 = A00B0 + A01B1
    # C1 = A10B0 + A11B1
    # They share B0, B1.
    # Rank is 3?
    # M1 = (A00+A10) B0
    # M2 = A01 B1
    # M3 = (A11-A01) B1
    # C0 = M1 - A10B0 + M2 ...
    
    # Let's stick to the safe Rank 56 (Strassen 2x2x2) if 48 is complex.
    # But 48 is better.
    
    # Simpler: Just use Rank 64 (direct) to ensure it works?
    # No, score is rank.
    
    # I will use Rank 48.
    # For each block (2x2 A, 2x8 B):
    # For each column k of B (8 cols):
    # We have 2x2x1.
    # Rank 3.
    
    # Coeffs for 2x2x1 (Rank 3):
    # M1 = (A00 + A10) * B0
    # M2 = A01 * B1
    # M3 = (A11 - A01) * B1
    # C0 = M1 - A10*B0 + M2  -> Need A10*B0.
    
    # Correct Rank 3 for 2x2x1:
    # M1 = A00 B0
    # M2 = A01 B1
    # M3 = (A10 + A11) B0 ...
    
    # Let's use Rank 4 (direct) for simplicity and correctness first, then optimize?
    # Rank 4 per 2x2x1. 8 cols -> 32 per block. 2 blocks -> 64.
    # This is the trivial solution.
    
    # I will implement the trivial solution (Rank 64) to fix the crash, then I can try to optimize in the next step?
    # No, I should provide the best I can.
    # But the prompt says "Fix it".
    # I will provide Rank 64 (direct) as it is guaranteed to work and have integer entries.
    
    rank = 64
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                
    return {"U": U, "V": V, "W": W}
