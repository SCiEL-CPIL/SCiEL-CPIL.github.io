"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # We will build a decomposition using Strassen for the 4x4 part and a direct sum approach.
    # Actually, let's just use the known optimal or near-optimal ranks.
    # Rank of 4x4 matrix multiplication is known to be 18 (Strassen).
    # Rank of 4x7 matrix multiplication?
    # Let's try to construct a decomposition.
    
    # Strategy: Use Strassen for 4x4 * 4x4 (rank 18) and 4x4 * 4x3 (remaining columns).
    # The 4x3 part can be done with schoolbook rank 4*4*3 = 48? No, that's too high.
    # We can split the 7 columns into 4 and 3.
    # C = A * B1 + A * B2 where B1 is 4x4 and B2 is 4x3.
    # Rank(A*B1) <= 18 (Strassen)
    # Rank(A*B2) <= 4*4*3 = 48 (schoolbook) -> Total 66.
    
    # Can we do better for 4x3?
    # 4x3 multiplication rank is at most 4*3*4 = 48.
    # Is there a better bound?
    # For 4x4 it's 18. For 4x3, maybe we can use a partial Strassen?
    
    # Let's just implement the direct sum of Strassen(4x4) and Schoolbook(4x3).
    # This gives rank 18 + 48 = 66.
    # The seed is 4*4*7 = 112. So 66 is a huge improvement.
    
    # Let's try to be smarter.
    # Maybe use Strassen for 4x4 and then handle the remaining 3 columns.
    # Or maybe there is a known rank for 4x7.
    # According to literature, rank of 4x4 is 18.
    # Rank of 4x5? 4x6? 4x7?
    
    # Let's stick with the block decomposition:
    # B = [B_left (4x4), B_right (4x3)]
    # C = A * B_left + A * B_right
    # We compute A*B_left using Strassen (rank 18).
    # We compute A*B_right using schoolbook (rank 48).
    # Total rank = 66.
    
    # Let's implement Strassen for 4x4.
    # Strassen splits 4x4 into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]] where each is 2x2.
    # B = [[B11, B12], [B21, B22]] where each is 2x2.
    # C = [[C11, C12], [C21, C22]]
    # C11 = M1 - M4 + M6 + M7
    # C12 = M2 + M4
    # C21 = M3 + M5
    # C22 = M1 - M2 + M3 + M8
    # Where:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # M8 = (A11 + A12) * (B22 - B21)
    
    # Wait, the standard Strassen formulas for 2x2 are:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # M8 = (A11 + A12)(B22 - B21)
    # C11 = M1 - M4 + M6 + M7
    # C12 = M2 + M4
    # C21 = M3 + M5
    # C22 = M1 - M2 + M3 + M8
    
    # This requires 8 multiplications of 2x2 matrices.
    # Each 2x2 multiplication has rank 7 (Strassen) or 4 (schoolbook).
    # If we use Strassen for the 2x2 multiplications, the rank would be 8 * 7 = 56. That's worse than 18.
    # Ah, Strassen for 4x4 uses 18 multiplications of 1x1? No.
    # Strassen's algorithm for 4x4 reduces the number of scalar multiplications from 16 to 18?
    # No, Strassen for 2x2 uses 7 scalar multiplications.
    # For 4x4, we can view it as 2x2 block matrices.
    # If we use Strassen on the blocks, we need 7 block multiplications.
    # Each block multiplication is 2x2 * 2x2.
    # If we use Strassen for the 2x2 block multiplications, we need 7 scalar multiplications per block.
    # So 7 blocks * 7 scalar mults = 49 scalar multiplications.
    # This is worse than the 16 scalar multiplications of schoolbook for 4x4.
    
    # So, how is the rank of 4x4 matrix multiplication 18?
    # The rank of the tensor is the minimum number of rank-1 tensors.
    # For 2x2, rank is 7.
    # For 4x4, the rank is known to be 18.
    # This is not simply 7 * (something).
    # The 18-rank decomposition for 4x4 is a specific construction.
    
    # Since I don't have the explicit 18-rank decomposition for 4x4 memorized, and implementing it from scratch is risky,
    # let's stick to a safe, verifiable decomposition.
    
    # Option 1: Schoolbook for 4x4 (rank 16) + Schoolbook for 4x3 (rank 48) = 64.
    # Option 2: Strassen for 4x4 (if I can find a rank-18 decomposition) + Schoolbook for 4x3.
    
    # Let's check if I can implement the rank-18 decomposition for 4x4.
    # It is known that the rank of 4x4 matrix multiplication is 18.
    # A decomposition exists.
    
    # However, to be safe and ensure correctness within the time limit, I will use the schoolbook method for the 4x4 part as well,
    # but optimize the 4x3 part if possible.
    # Actually, 16 + 48 = 64.
    # The seed is 112. 64 is a significant improvement.
    
    # Can we do better for 4x3?
    # 4x3 multiplication: A (4x4) * B (4x3) -> C (4x3).
    # Rank is at most 4*4*3 = 48.
    # Is there a lower rank?
    # The lower bound for 4x3 is not 7.
    # It is likely higher.
    
    # Let's try to construct the 64-rank decomposition.
    # U, V, W matrices.
    # n=4, m=4, p=7.
    # U shape: (4*4, rank) = (16, rank)
    # V shape: (4*7, rank) = (28, rank)
    # W shape: (7*4, rank) = (28, rank)
    
    # Wait, the task says:
    # U: (n*m, rank) = (4*4, rank) = (16, rank)
    # V: (m*p, rank) = (4*7, rank) = (28, rank)
    # W: (p*n, rank) = (7*4, rank) = (28, rank)
    
    # The decomposition is:
    # C[i, k] = sum_{j=0}^{m-1} A[i, j] * B[j, k]
    # In tensor notation: T_{i, j, k} = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
    # Wait, the convention is W[k*n+i, r].
    # So C[i, k] = sum_j A[i, j] B[j, k] = sum_r (sum_j U_{i*m+j, r} W_{k*n+i, r} V_{j*p+k, r} ?)
    # No, the standard form is:
    # M(A, B) = sum_r U_r * V_r * W_r^T ?
    # Let's check the seed code.
    # U[i*m+j][c] = 1
    # V[j*p+k][c] = 1
    # W[k*n+i][c] = 1
    # This implies that for each c, we have a term.
    # The product is sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    # For the schoolbook term (i, j, k), U=1, V=1, W=1.
    # So the term is 1.
    # Summing over c, we get the number of terms where (i,j,k) matches.
    # This reconstructs A[i,j] * B[j,k] if we sum over j?
    # No, the tensor T_{i,j,k} represents the coefficient of A_{i,j} B_{j,k} in the output C_{i,k}.
    # T_{i,j,k} = 1.
    # So C_{i,k} = sum_{j} A_{i,j} B_{j,k}.
    # The decomposition is T = sum_r U_r \otimes V_r \otimes W_r.
    # Where (U_r)_{i*m+j} = U[i*m+j, r], etc.
    # So T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    
    # So for the block decomposition:
    # We split k into k1 (0..3) and k2 (4..6).
    # T_{i,j,k} = 1 if k < 4, else 1.
    # We can write T = T1 + T2.
    # T1_{i,j,k} = 1 if k < 4, else 0.
    # T2_{i,j,k} = 1 if k >= 4, else 0.
    
    # T1 is the 4x4x4 multiplication tensor.
    # T2 is the 4x4x3 multiplication tensor (with indices shifted).
    
    # We can use a rank-16 decomposition for T1 (schoolbook).
    # And a rank-48 decomposition for T2 (schoolbook).
    # Total rank 64.
    
    # Let's implement this.
    
    rank = 16 + 48
    U = np.zeros((16, rank))
    V = np.zeros((28, rank))
    W = np.zeros((28, rank))
    
    # Part 1: 4x4x4 (k=0..3)
    # Schoolbook: for each i, j, k in 0..3
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*4 + j, col] = 1
                V[j*7 + k, col] = 1
                W[k*4 + i, col] = 1
                col += 1
    # col is now 16.
    
    # Part 2: 4x4x3 (k=4..6)
    # Schoolbook: for each i, j in 0..3, k in 4..6
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U[i*4 + j, col] = 1
                V[j*7 + k, col] = 1
                W[k*4 + i, col] = 1
                col += 1
    # col is now 16 + 48 = 64.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
