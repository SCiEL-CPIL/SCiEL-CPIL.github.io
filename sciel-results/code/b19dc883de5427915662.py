"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = 150.0
    start_time = time.time()
    
    # Strategy: Partition m into two groups of 2, and p into eight groups of 1.
    # Each block is 2x2 * 2x1 = 2x1.
    # Rank of 2x2 * 2x1 is 2.
    # Total rank: 2 (m-groups) * 8 (p-groups) * 2 = 32.
    
    rank = 32
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    t = 0
    for j_group in range(2):
        # j indices are 2*j_group and 2*j_group+1
        j0 = 2 * j_group
        j1 = 2 * j_group + 1
        
        for k in range(8):
            # Term 1: A[0, j0]*B[j0, k] + A[0, j1]*B[j1, k] -> C[0, k]
            # This is not a single rank-1 term.
            # We need to decompose C[0, k] and C[1, k].
            # C[0, k] = A[0, j0]*B[j0, k] + A[0, j1]*B[j1, k]
            # C[1, k] = A[1, j0]*B[j0, k] + A[1, j1]*B[j1, k]
            
            # Rank 2 decomposition:
            # M1 = (A[0, j0] + A[1, j0]) * (B[j0, k] + B[j1, k]) ? No.
            
            # Let's use the standard rank-2 for 2x2 * 2x1.
            # C0 = A00 B0 + A01 B1
            # C1 = A10 B0 + A11 B1
            
            # Term 1:
            # U[0*m + j0, t] = 1
            # U[1*m + j0, t] = 1
            # V[j0*p + k, t] = 1
            # V[j1*p + k, t] = 1
            # W[k*n + 0, t] = 1
            # W[k*n + 1, t] = 1
            # This gives (A0j0 + A1j0)(Bj0k + Bj1k) = A0j0Bj0k + A0j0Bj1k + A1j0Bj0k + A1j0Bj1k
            # This is not what we want.
            
            # Correct Rank 2 decomposition for 2x2 * 2x1:
            # We can just use 2 terms corresponding to the two rows of A?
            # No, that would be rank 4 (2x2x1 is not separable in that way).
            
            # Actually, rank(2x2x1) is 2.
            # C0 = A00 B0 + A01 B1
            # C1 = A10 B0 + A11 B1
            
            # Let M1 = A00 B0 + A01 B1
            # Let M2 = A10 B0 + A11 B1
            # This requires 4 multiplications if we count A00*B0 etc.
            # But in tensor rank, we count the number of terms in the sum of outer products.
            
            # Term 1:
            # U[0*m + j0, t] = 1
            # V[j0*p + k, t] = 1
            # W[k*n + 0, t] = 1
            # This gives A0j0 * Bj0k.
            
            # Term 2:
            # U[0*m + j1, t+1] = 1
            # V[j1*p + k, t+1] = 1
            # W[k*n + 0, t+1] = 1
            # This gives A0j1 * Bj1k.
            
            # This is rank 4 for the block.
            
            # Wait, is rank(2x2x1) = 2?
            # T = e0^T (e0^T e0^T + e1^T e1^T) + e1^T (e0^T e0^T + e1^T e1^T)?
            # T_{i,j,k} = delta_{i,0} (delta_{j,0} delta_{k,0} + delta_{j,1} delta_{k,1}) + ...
            # No, k is fixed.
            
            # The tensor is T_{i,j} for fixed k.
            # T = [[1, 1], [1, 1]]? No.
            # T_{0,0} = 1 (A0j0 B_j0k)
            # T_{0,1} = 1 (A0j1 B_j1k)
            # T_{1,0} = 1 (A1j0 B_j0k)
            # T_{1,1} = 1 (A1j1 B_j1k)
            
            # This is a 2x2 matrix of coefficients.
            # Rank of this matrix is 1?
            # [[1, 1], [1, 1]] has rank 1.
            # So T = u v^T where u = [1, 1]^T, v = [1, 1]^T.
            # T_{i,j} = u_i v_j.
            # C_i = sum_j A_{i,j} B_{j,k} u_i v_j?
            # No.
            
            # C_i = sum_j T_{i,j} A_{i,j} B_{j,k}?
            # No, T_{i,j} is the coefficient.
            # C_i = A_{i,0} B_{0,k} + A_{i,1} B_{1,k}.
            
            # If we set:
            # M1 = (A_{0,0} + A_{1,0}) (B_{0,k} + B_{1,k})
            # M2 = ...
            
            # Actually, rank(2x2x1) is 2.
            # C0 = A00 B0 + A01 B1
            # C1 = A10 B0 + A11 B1
            
            # Let M1 = A00 B0 + A01 B1 + A10 B0 + A11 B1
            # Let M2 = A00 B0 - A01 B1 - A10 B0 + A11 B1
            # This is not helpful.
            
            # Let's stick to the 56 rank solution which is known to be correct.
            # The 32 rank solution is risky if my rank calculation is wrong.
            # Rank of 2x2x1 is indeed 2.
            # C0 = A00 B0 + A01 B1
            # C1 = A10 B0 + A11 B1
            
            # Decomposition:
            # Term 1:
            # U[0, t] = 1, U[1, t] = 1
            # V[0, t] = 1, V[1, t] = 1
            # W[0, t] = 1, W[1, t] = 1
            # This gives (A0+A1)(B0+B1) = A0B0 + A0B1 + A1B0 + A1B1.
            # This contributes to both C0 and C1.
            
            # Term 2:
            # U[0, t+1] = 1, U[1, t+1] = -1
            # V[0, t+1] = 1, V[1, t+1] = -1
            # W[0, t+1] = 1, W[1, t+1] = -1
            # This gives (A0-A1)(B0-B1) = A0B0 - A0B1 - A1B0 + A1B1.
            
            # Sum: 2 A0B0 + 2 A1B1.
            # Difference: 2 A0B1 + 2 A1B0.
            
            # We want:
            # C0 = A0B0 + A0B1
            # C1 = A1B0 + A1B1
            
            # We can form these from the 4 terms:
            # T1 = A0B0
            # T2 = A0B1
            # T3 = A1B0
            # T4 = A1B1
            
            # C0 = T1 + T2
            # C1 = T3 + T4
            
            # This is rank 4.
            
            # So rank(2x2x1) is 4?
            # Yes, because the matrix [[1, 1], [1, 1]] is not the tensor.
            # The tensor is T_{i,j,k}.
            # T_{0,0,0} = 1, T_{0,1,0} = 1, T_{1,0,0} = 1, T_{1,1,0} = 1.
            # This is 4 rank-1 terms.
            # So rank is 4.
            
            # Thus, the 32 rank strategy is invalid.
            # The 56 rank strategy is correct.
            
            pass
            
    # Fallback to the known correct 56 rank solution
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
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
