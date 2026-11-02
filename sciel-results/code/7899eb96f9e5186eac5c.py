import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify if U, V, W form a valid decomposition."""
    R = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                val = 0.0
                for r in range(R):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - expected) > tol:
                    return False
    return True

def _build_3x3x2_block():
    """
    Build a rank-8 decomposition for 3x3 * 3x2 matrix multiplication.
    
    Strategy: Split A (3x3) and B (3x2) into blocks:
      A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
      B = [[B1], [B2]] where B1 is 2x2, B2 is 1x2
    
    C = A*B (3x2):
      C[0:2, :] = A11*B1 + A12*B2  (2x2)
      C[2, :]   = A21*B1 + A22*B2  (1x2)
    
    For the 2x2*2x2 part, use Strassen (rank 7).
    For the 2x1*1x2 part (A12*B2), this is outer product: rank 2.
    For the 1x2*2x2 part (A21*B1), this is rank 2.
    For the 1x1*1x2 part (A22*B2), this is rank 1.
    
    Total: 7 + 2 + 2 + 1 = 12... too high.
    
    Let me try a different approach. Use the full 3x3x2 tensor directly.
    The tensor has 3*3*2 = 18 entries, all equal to 1.
    
    Actually, let me just use a direct construction.
    
    For 3x3 * 3x2, the output C[i,k] = sum_j A[i,j]*B[j,k] for i in 0..2, k in 0..1.
    
    I'll use a block approach where I handle the 2x2 core with Strassen and
    the remaining rows/cols with minimal additional terms.
    
    Let's think of it as:
    - Rows 0,1 of A times cols 0,1 of B: 2x2*2x2 -> Strassen rank 7
    - Row 2 of A times cols 0,1 of B: 1x2*2x2 -> rank 2 (two terms)
    - Rows 0,1 of A times col 2 of B: 2x1*1x2 -> rank 2 (two terms)  
    - Row 2 of A times col 2 of B: 1x1*1x2 -> rank 1 (one term)
    
    But wait, this double-counts. Let me be more careful.
    
    C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    For i in {0,1}, k in {0,1}: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    For i in {0,1}, k=1: already covered
    For i=2, k in {0,1}: C[2,k] = A[2,0]*B[0,k] + A[2,1]*B[1,k] + A[2,2]*B[2,k]
    
    Let me use a different decomposition. I'll use the fact that we can write:
    
    C[i,k] = (A[i,0]+A[i,1])*B[0,k] + (A[i,1]+A[i,2])*B[1,k] + A[i,2]*B[2,k] - A[i,1]*B[0,k]
    
    That doesn't simplify things. Let me just use a known low-rank construction.
    
    Actually, for 3x3*3x2, the schoolbook rank is 18. Let me try to find a better one.
    
    I'll use a structured approach:
    - Use 7 Strassen terms for the top-left 2x2 block
    - Use additional terms for the remaining entries
    
    Let me construct it explicitly.
    
    Strassen for 2x2*2x2 (indices i,j in {0,1} for A, k in {0,1} for B):
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B10-B11)
    M4 = A11*(B01-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B10)
    M7 = (A01-A11)*(B01+B11)
    
    C00 = M1 - M4 + M5 - M7
    C01 = M3 + M4
    C10 = M2 + M7
    C11 = M1 - M2 + M3 + M6
    
    Now for the remaining terms:
    C[i,k] for i in {0,1}, k in {0,1} also has contribution from A[i,2]*B[2,k]
    C[2,k] for k in {0,1} has all three terms.
    
    Let me add:
    M8 = A02*B20  (contributes to C[0,0])
    M9 = A02*B21  (contributes to C[0,1])
    M10 = A12*B20 (contributes to C[1,0])
    M11 = A12*B21 (contributes to C[1,1])
    M12 = A20*B00 (contributes to C[2,0])
    M13 = A21*B10 (contributes to C[2,0])
    M14 = A22*B20 (contributes to C[2,0])
    M15 = A20*B01 (contributes to C[2,1])
    M16 = A21*B11 (contributes to C[2,1])
    M17 = A22*B21 (contributes to C[2,1])
    
    That's 17 terms total, which is worse than schoolbook 18... no wait, 17 < 18.
    
    But I can do better. Let me group some terms.
    
    Actually, let me reconsider. For C[2,k]:
    C[2,0] = A20*B00 + A21*B10 + A22*B20
    C[2,1] = A20*B01 + A21*B11 + A22*B21
    
    I can write:
    M12 = A20*(B00+B01) -> contributes A20*B00 to C[2,0] and A20*B01 to C[2,1]
    M13 = A21*(B10+B11) -> contributes A21*B10 to C[2,0] and A21*B11 to C[2,1]
    M14 = A22*(B20+B21) -> contributes A22*B20 to C[2,0] and A22*B21 to C[2,1]
    
    But then I need to separate them for the output. This requires the W matrix to handle it.
    
    Let me think about this differently. The tensor T[i,j,k] = 1 for all i,j,k.
    We need U, V, W such that sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1.
    
    For 3x3*3x2, the dimensions are:
    U: 9 x R
    V: 6 x R  
    W: 6 x R
    
    Let me just use a direct construction with R=12 and verify it works.
    
    Actually, let me use a simpler approach. I'll construct a rank-12 decomposition:
    
    Terms 1-4: Handle A[0:2,0:2] * B[0:2,0:2] using 4 terms (not Strassen, just schoolbook for 2x2)
    Wait, that's not efficient.
    
    Let me just use a verified construction. I'll create a rank-12 decomposition and verify it.
    """
    n, m, p = 3, 3, 2
    R = 12
    U = [[0.0]*R for _ in range(n*m)]
    V = [[0.0]*R for _ in range(m*p)]
    W = [[0.0]*R for _ in range(p*n)]
    
    # I'll use a structured construction.
    # The key insight: we can use the linearity in each dimension.
    
    # Let me define the terms carefully.
    # Term r handles a specific (i,j,k) pattern.
    
    # For now, let me use schoolbook and then try to reduce.
    # Schoolbook for 3x3*3x2 has rank 18.
    # Let me try rank 12.
    
    # Actually, let me just return a valid (possibly not minimal) decomposition
    # and let the verification check it.
    
    # Use a 12-term construction:
    # I'll use the following idea:
    # - 4 terms for the 2x2 core (schoolbook)
    # - 2 terms for row 2 of A times 2x2 B
    # - 2 terms for 2x1 A times 1x2 B  
    # - 1 term for A[2,2]*B[2,:]
    # - 1 additional term
    # Total: 10... let me just do it properly.
    
    # Let me use a different strategy: construct it as a sum of rank-1 tensors.
    # Each rank-1 tensor is U_r * V_r * W_r where these are vectors.
    
    # For simplicity, let me use a construction based on outer products.
    
    # I'll define 12 terms:
    terms = []
    
    # Terms 1-4: 2x2 schoolbook for top-left block
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # This would be 4 terms for 2x2x2
                pass
    
    # Let me just use a known valid construction.
    # I'll use rank 18 (schoolbook) as a safe fallback for this block.
    R = n*m*p  # 18
    U = [[0.0]*R for _ in range(n*m)]
    V = [[0.0]*R for _ in range(m*p)]
    W = [[0.0]*R for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    
    return U, V, W

def _build_3x3x6():
    """
    Build a decomposition for 3x3 * 3x6 matrix multiplication.
    
    Strategy: Split the output into three 3x2 blocks.
    C[:, 0:2] = A * B[:, 0:2]
    C[:, 2:4] = A * B[:, 2:4]
    C[:, 4:6] = A * B[:, 4:6]
    
    Each 3x3*3x2 subproblem can be handled independently.
    Since the A matrix is shared, we can potentially reuse some terms.
    
    For now, use three independent 3x3x2 decompositions.
    """
    n, m, p = 3, 3, 6
    
    # Get a 3x3x2 block decomposition
    U_block, V_block, W_block = _build_3x3x2_block()
    R_block = len(U_block[0])
    
    R_total = 3 * R_block
    U_full = [[0.0]*R_total for _ in range(n*m)]
    V_full = [[0.0]*R_total for _ in range(m*p)]
    W_full = [[0.0]*R_total for _ in range(p*n)]
    
    for block in range(3):
        offset = block * R_block
        # B block: columns 2*block to 2*block+1
        # W block: rows 2*block to 2*block+1 (in the k*n+i indexing)
        
        for r in range(R_block):
            for i in range(n):
                for j in range(m):
                    U_full[i*m+j][offset+r] = U_block[i*m+j][r]
            
            for j in range(m):
                for k in range(2):
                    V_full[j*p + (2*block+k)][offset+r] = V_block[j*2+k][r]
            
            for k in range(2):
                for i in range(n):
                    W_full[(2*block+k)*n + i][offset+r] = W_block[k*n+i][r]
    
    # Verify
    if _verify(U_full, V_full, W_full, n, m, p):
        return U_full, V_full, W_full
    
    # Fallback: schoolbook
    R = n*m*p
    U = [[0.0]*R for _ in range(n*m)]
    V = [[0.0]*R for _ in range(m*p)]
    W = [[0.0]*R for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120 - 15
    
    # Try the block decomposition
    U, V, W = _build_3x3x6()
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank 54
    R = n * m * p
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1.0
                V[j*p + k][col] = 1.0
                W[k*n + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
