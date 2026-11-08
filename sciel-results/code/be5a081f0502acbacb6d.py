"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import itertools

def _time_ok(deadline, margin=5.0):
    return time.time() < deadline - margin

def _validate(U, V, W, n, m, p):
    """Check that U, V, W exactly decompose the matrix multiplication tensor.
    The tensor T[i,j,k] = delta_{i,j,k} (only 1 when i==j... no wait)
    Actually for matrix multiplication C = A * B where A is n x m and B is m x p,
    C[i,k] = sum_j A[i,j] * B[j,k]
    The tensor has T[i,j,k] = 1 for all valid i,j,k (it's the identity in the middle index).
    Wait, the multiplication tensor M_{i,j,k} = 1 for all i,j,k in their ranges.
    A rank-r decomposition gives M_{i,j,k} = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    (using the trace convention W[k*n+i, r])
    """
    rank = len(U[0]) if U else 0
    # Build the target tensor: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
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

def _try_split_i(n, m, p):
    """Split i-dimension: i = i1 + i2, with i1 + i2 = n, i1,i2 >= 1
    C[i,k] for i < i1: uses A[0:i1, :] * B
    C[i,k] for i >= i1: uses A[i1:n, :] * B
    This gives rank = i1*m*p + (n-i1)*m*p = n*m*p (no gain)
    Unless we can share terms...
    """
    # No gain from simple splitting of one dimension
    return None

def _try_strassen_like(n, m, p):
    """Try to find a decomposition with rank < n*m*p using structured patterns.
    
    For 4x4 * 4x9, schoolbook is 4*4*9 = 144.
    
    Let's try: split the 4x4 matrix A into blocks.
    A = [[A11, A12], [A21, A22]] where each block is 2x2.
    B is 4x9, split as B = [[B1], [B2]] where B1, B2 are 2x9.
    
    C = A*B = [[A11*B1 + A12*B2], [A21*B1 + A22*B2]]
    
    This is 4 multiplications of 2x2 * 2x9, each of rank 2*2*9=36.
    Total rank = 4*36 = 144. No gain.
    
    Try Strassen-style for the 2x2 * 2x2 part (ignoring the 9 for a moment):
    M1 = (A11+A22)*(B11+B22)
    M2 = (A21+A22)*B22
    M3 = A11*(B11-B22)
    M4 = A22*(B22-B11)
    M5 = (A11+A12)*B22
    M6 = (A21-A11)*(B11+B21)
    M7 = (A12-A22)*(B21+B22)
    
    C11 = M1 + M4 - M2 + M6
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M3 + M5 - M7
    
    Each M is a product of a 2x2 and a 2x2 matrix.
    For the full problem, B is 4x9, so we need to adapt.
    
    Let's think differently. The tensor for 4x4*4x9 multiplication.
    We can try to find a rank-r decomposition by searching over small r.
    """
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # Leave 10s margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find better decompositions
    # The multiplication tensor T[i,j,k] = 1 for all i,j,k
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Try rank r from a lower bound up
    # Lower bound: max(n*p, m*p, n*m) by slice rank arguments
    # For 4x4*4x9: max(36, 36, 16) = 36
    # But actual lower bound might be higher.
    
    # Let's try a few structured approaches:
    
    # Approach 1: Use the fact that 4 = 2+2 and try block decomposition
    # with Strassen-like tricks on the 2x2 blocks.
    
    # For A (4x4) and B (4x9):
    # A = [[A1, A2], [A3, A4]] (each 2x2)
    # B = [[B1], [B2]] (each 2x9)
    # C = [[A1*B1 + A2*B2], [A3*B1 + A4*B2]]
    
    # Each A_i * B_j is a 2x2 * 2x9 multiplication.
    # Rank of 2x2*2x9 = 2*2*9 = 36 (schoolbook)
    # Total: 4 * 36 = 144
    
    # Can we do better on the 2x2*2x9 part?
    # 2x2*2x2 has rank 7 (Strassen). For 2x2*2x9:
    # We can think of it as 2x2*2x2 done 9 times independently: 7*9 = 63
    # Or we can use the structure: C[i,k] = sum_j A[i,j]*B[j,k]
    # For fixed k, it's a 2x2*2x1 multiplication (rank 2 schoolbook, or... )
    # Actually 2x2*2x1: C[i] = A[i,0]*B[0] + A[i,1]*B[1], rank 2.
    # So 2x2*2x9 has rank at most 2*9 = 18? No, that's not right.
    # The tensor is T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,8}
    # Rank = ? Schoolbook is 4*9 = 36.
    # Lower bound: max(2*9, 2*9, 2*2) = max(18,18,4) = 18
    # Can we achieve 18? 
    # T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # For 2x2*2x9, if we set r=18:
    # Let's try: for each (i,j) pair, we need to cover all k.
    # This is tricky. Let me just try to find any rank < 144.
    
    # Actually, let me try a different approach: 
    # The 4x4*4x9 multiplication can be seen as 4 independent 4x9 vector-matrix products
    # (each row of A times B). Each is a 4x9 outer product... no, it's a dot product.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This is like computing 4 row vectors (length 4) dotted with 9 column vectors (length 4) of B.
    # Each dot product has rank 4 (schoolbook for 1x4*4x1).
    # Total: 4*9*4 = 144.
    
    # I don't have a clear path to beat 144 with simple structured methods.
    # Let me just return the schoolbook decomposition.
    
    if not _time_ok(deadline):
        return best
    
    return best
# EVOLVE-BLOCK-END
