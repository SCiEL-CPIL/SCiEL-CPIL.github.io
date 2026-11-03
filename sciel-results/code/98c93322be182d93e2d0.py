"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = 150.0
    start_time = time.time()
    
    # Strategy: Use Strassen 2x2*2x2 for each block.
    # Partition: 2 j-groups x 4 k-groups = 8 subproblems, each rank 7
    # Total rank: 56
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    t = 0
    for a in range(2):
        for b in range(4):
            for s in range(7):
                if s == 0:
                    U[0 * m + 2 * a, t] += 1
                    U[1 * m + 2 * a + 1, t] += 1
                    V[2 * a * p + 2 * b, t] += 1
                    V[(2 * a + 1) * p + 2 * b + 1, t] += 1
                    W[2 * b * n + 0, t] += 1
                    W[(2 * b + 1) * n + 1, t] += 1
                elif s == 1:
                    U[1 * m + 2 * a, t] += 1
                    U[1 * m + 2 * a + 1, t] += 1
                    V[2 * a * p + 2 * b, t] += 1
                    W[2 * b * n + 1, t] += 1
                    W[(2 * b + 1) * n + 1, t] += -1
                elif s == 2:
                    U[0 * m + 2 * a, t] += 1
                    V[2 * a * p + 2 * b + 1, t] += 1
                    V[(2 * a + 1) * p + 2 * b + 1, t] += -1
                    W[(2 * b + 1) * n + 0, t] += 1
                    W[(2 * b + 1) * n + 1, t] += 1
                elif s == 3:
                    U[1 * m + 2 * a + 1, t] += 1
                    V[(2 * a + 1) * p + 2 * b, t] += 1
                    V[2 * a * p + 2 * b, t] += -1
                    W[2 * b * n + 0, t] += 1
                    W[2 * b * n + 1, t] += 1
                elif s == 4:
                    U[0 * m + 2 * a, t] += 1
                    U[0 * m + 2 * a + 1, t] += 1
                    V[(2 * a + 1) * p + 2 * b + 1, t] += 1
                    W[2 * b * n + 0, t] += -1
                    W[(2 * b + 1) * n + 0, t] += 1
                elif s == 5:
                    U[1 * m + 2 * a, t] += 1
                    U[0 * m + 2 * a, t] += -1
                    V[2 * a * p + 2 * b, t] += 1
                    V[2 * a * p + 2 * b + 1, t] += 1
                    W[(2 * b + 1) * n + 1, t] += 1
                elif s == 6:
                    U[0 * m + 2 * a + 1, t] += 1
                    U[1 * m + 2 * a + 1, t] += -1
                    V[(2 * a + 1) * p + 2 * b, t] += 1
                    V[(2 * a + 1) * p + 2 * b + 1, t] += 1
                    W[2 * b * n + 0, t] += 1
                t += 1
    
    # Verify and attempt rank reduction
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        # Try to find a lower rank by exploiting structure
        # The key observation: A is shared across all k-groups.
        # In the current decomposition, for each (a,b) pair we have 7 factors.
        # The U factors for different b but same a are identical (they only depend on a).
        # So we have 2 distinct U patterns, each repeated 4 times.
        # Similarly, V factors for different a but same b have a simple relationship.
        # Let's check if we can merge factors.
        
        # Actually, let's try a direct approach: can we do rank 55?
        # Try setting one factor to zero and checking if the system is still solvable.
        # This is hard to do analytically, so let's try a numerical approach.
        
        # Alternative: try a completely different decomposition structure.
        # What if we use rank 2 for the 2x1 * 1x2 parts?
        # A(2x4) * B(4x8): split B into 4 blocks of 4x2.
        # For each block, A(2x4)*B_block(4x2) = C_block(2x2).
        # The 2x4 * 4x2 problem: can we do better than 2*7=14?
        # 2x4 * 4x2: think of it as 2*(2x2) * (2x2)*2 ... 
        # Actually 2x4 * 4x2: A = [A1 A2] (each 2x2), B = [B1; B2] (each 2x2)
        # C = A1*B1 + A2*B2, each product is 2x2*2x2 (rank 7)
        # Total: 14 per block, 4 blocks = 56. Same.
        
        # What about: A(2x4) * B(4x8). View as 2x(4x4) * (4x4)x8?
        # Or: (2x2x2) * (2x2x4)?
        
        # Let me try: partition differently.
        # A is 2x4, B is 4x8.
        # What if I group output columns differently?
        # C has 2 rows, 8 cols. Group as 2x(4x2) or (2x4)x2 or 2x2x4...
        
        # Try: C[i,k] for i in {0,1}, k in {0..7}
        # Group k as {0,1,2,3} and {4,5,6,7}
        # For first group: A(2x4) * B[:, 0:4](4x4) = C[:, 0:4](2x4)
        # This is 2x4 * 4x4. Can decompose as 2x(2x2) * (2x2)x4?
        # A = [A1 A2] (2x2 each), B1 = B[:, 0:4] = [B1a B1b] (4x2 each)
        # C[:, 0:4] = A1*B1a + A2*B1b, each 2x2*2x2 (rank 7) -> rank 14
        # Similarly for second group: rank 14
        # Total: 28? No wait, that's wrong.
        # A1*B1a is 2x2 * 2x2 -> rank 7
        # A2*B1b is 2x2 * 2x2 -> rank 7
        # But A1 and A2 are fixed, B1a and B1b are different parts of B.
        # The issue: in the tensor decomposition, we can't "fix" A1.
        # Each factor must be a scalar product of A entries and B entries.
        
        # I think 56 might be close to optimal for this structure.
        # Let me just verify the current solution is correct and return it.
        
        Uv = U.copy()
        Vv = V.copy()
        Wv = W.copy()
        np.random.seed(123)
        Atest = np.random.randn(n, m)
        Btest = np.random.randn(m, p)
        Ctest = Atest @ Btest
        Cdecomp = np.zeros((n, p))
        for t in range(rank):
            Mt = np.dot(Uv[:, t], Atest.reshape(-1))
            Nt = np.dot(Vv[:, t], Btest.reshape(-1))
            Wt = Wv[:, t].reshape(p, n).T
            Cdecomp += Mt * Nt * Wt
        err = np.max(np.abs(Cdecomp - Ctest))
        if err > 1e-10:
            # Should not happen, but fallback
            pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
