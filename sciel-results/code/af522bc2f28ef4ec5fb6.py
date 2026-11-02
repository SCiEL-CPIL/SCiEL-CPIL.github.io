import time
from fractions import Fraction

DIMENSIONS = [3, 3, 8]

def _validate(U, V, W, n=3, m=3, p=8):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    # For each (i,j,k) in the tensor, the entry should be 1
    # T[i*m+j, j*p+k, k*n+i] = 1
    # Decomposition: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                total = 0
                for r in range(rank):
                    total += U[a][r] * V[b][r] * W[c][r]
                if total != 1:
                    return False
    # Check entries are integers or half-integers
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if isinstance(val, Fraction):
                    if val.denominator not in (1, 2):
                        return False
                elif isinstance(val, (int, float)):
                    f = Fraction(val)
                    if f.denominator not in (1, 2):
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = 3, 3, 8
    
    # We decompose the 3x3 x 3x8 multiplication into 4 blocks of 3x3 x 3x2.
    # Each block handles 2 columns of B.
    # For each 3x3 x 3x2 block, we use a rank-17 decomposition.
    # Total rank: 4 * 17 = 68.
    
    rank = 68
    U = [[0] * rank for _ in range(n * m)]  # 9 x 68
    V = [[0] * rank for _ in range(m * p)]  # 24 x 68
    W = [[0] * rank for _ in range(p * n)]  # 24 x 68
    
    # Process 4 blocks: columns (0,1), (2,3), (4,5), (6,7)
    for block in range(4):
        k0 = block * 2  # first column of this block
        k1 = block * 2 + 1  # second column of this block
        base = block * 17  # base index for this block's rank components
        
        # Strassen for A[0:2, 0:2] * B[0:2, k0:k1+1]
        # A sub-matrix: [[A00, A01], [A10, A11]] where indices are (0,0),(0,1),(1,0),(1,1)
        # B sub-matrix: [[B0k0, B0k1], [B1k0, B1k1]]
        
        # P1 = (A00 + A11)(B0k0 + B1k0) for column k0
        # P1 = (A00 + A11)(B0k1 + B1k1) for column k1
        # We need separate rank-1 terms for each column
        
        # For column k0:
        # P1_0 = (A00 + A11)(B0k0 + B1k0)
        U[0][base + 0] = 1  # A00
        U[4][base + 0] = 1  # A11
        V[0 * p + k0][base + 0] = 1  # B0k0
        V[1 * p + k0][base + 0] = 1  # B1k0
        W[k0 * n + 0][base + 0] = 1  # C[0,k0] += P1
        W[k0 * n + 1][base + 0] = -1  # C[1,k0] -= P1 (from Strassen: C11 = P1 - P2 + P3 + P6)
        
        # P2_0 = (A10 + A11)B0k0
        U[3][base + 1] = 1  # A10
        U[4][base + 1] = 1  # A11
        V[0 * p + k0][base + 1] = 1  # B0k0
        W[k0 * n + 1][base + 1] = 1  # C[1,k0] += P2
        W[k0 * n + 0][base + 1] = -1  # C[0,k0] -= P2 (from Strassen: C11 = P1 - P2 + P3 + P6)
        
        # P3_0 = A00(B0k1 - B1k1) -> This is for C[0, k1]
        # Wait, Strassen P3 = a(f-h). Here f=B0k1, h=B1k1.
        # P3 contributes to C[0, k1] and C[1, k1].
        # C[0, k1] += P3
        # C[1, k1] += P3
        U[0][base + 2] = 1  # A00
        V[0 * p + k1][base + 2] = 1  # B0k1
        V[1 * p + k1][base + 2] = -1  # B1k1
        W[k1 * n + 0][base + 2] = 1  # C[0, k1] += P3
        W[k1 * n + 1][base + 2] = 1  # C[1, k1] += P3
        
        # P4_0 = A11(B1k0 - B0k0)
        # P4 contributes to C[0, k0] and C[1, k0].
        # C[0, k0] += P4
        # C[1, k0] += P4
        U[4][base + 3] = 1  # A11
        V[1 * p + k0][base + 3] = 1  # B1k0
        V[0 * p + k0][base + 3] = -1  # B0k0
        W[k0 * n + 0][base + 3] = 1  # C[0, k0] += P4
        W[k0 * n + 1][base + 3] = 1  # C[1, k0] += P4
        
        # P5_0 = (A00 + A01)B1k0
        # P5 contributes to C[0, k0] and C[0, k1].
        # C[0, k0] -= P5
        # C[0, k1] += P5
        U[0][base + 4] = 1  # A00
        U[1][base + 4] = 1  # A01
        V[1 * p + k0][base + 4] = 1  # B1k0
        W[k0 * n + 0][base + 4] = -1  # C[0, k0] -= P5
        W[k1 * n + 0][base + 4] = 1  # C[0, k1] += P5
        
        # P6_0 = (A10 - A00)(B0k0 + B0k1)
        # P6 contributes to C[1, k0] and C[1, k1].
        # C[1, k0] += P6
        # C[1, k1] += P6
        U[3][base + 5] = 1  # A10
        U[0][base + 5] = -1  # A00
        V[0 * p + k0][base + 5] = 1  # B0k0
        V[0 * p + k1][base + 5] = 1  # B0k1
        W[k0 * n + 1][base + 5] = 1  # C[1, k0] += P6
        W[k1 * n + 1][base + 5] = 1  # C[1, k1] += P6
        
        # P7_0 = (A01 - A11)(B1k0 + B1k1)
        # P7 contributes to C[0, k0] and C[0, k1].
        # C[0, k0] += P7
        # C[0, k1] += P7
        U[1][base + 6] = 1  # A01
        U[4][base + 6] = -1  # A11
        V[1 * p + k0][base + 6] = 1  # B1k0
        V[1 * p + k1][base + 6] = 1  # B1k1
        W[k0 * n + 0][base + 6] = 1  # C[0, k0] += P7
        W[k1 * n + 0][base + 6] = 1  # C[0, k1] += P7
        
        # Cross terms:
        # C[0, k0] += A02 * B2k0
        U[2][base + 7] = 1  # A02
        V[2 * p + k0][base + 7] = 1  # B2k0
        W[k0 * n + 0][base + 7] = 1  # C[0, k0] += P
        
        # C[0, k1] += A02 * B2k1
        U[2][base + 8] = 1  # A02
        V[2 * p + k1][base + 8] = 1  # B2k1
        W[k1 * n + 0][base + 8] = 1  # C[0, k1] += P
        
        # C[1, k0] += A12 * B2k0
        U[5][base + 9] = 1  # A12
        V[2 * p + k0][base + 9] = 1  # B2k0
        W[k0 * n + 1][base + 9] = 1  # C[1, k0] += P
        
        # C[1, k1] += A12 * B2k1
        U[5][base + 10] = 1  # A12
        V[2 * p + k1][base + 10] = 1  # B2k1
        W[k1 * n + 1][base + 10] = 1  # C[1, k1] += P
        
        # C[2, k0] = A20 * B0k0 + A21 * B1k0 + A22 * B2k0
        U[6][base + 11] = 1  # A20
        V[0 * p + k0][base + 11] = 1  # B0k0
        W[k0 * n + 2][base + 11] = 1  # C[2, k0] += P
        
        U[7][base + 12] = 1  # A21
        V[1 * p + k0][base + 12] = 1  # B1k0
        W[k0 * n + 2][base + 12] = 1  # C[2, k0] += P
        
        U[8][base + 13] = 1  # A22
        V[2 * p + k0][base + 13] = 1  # B2k0
        W[k0 * n + 2][base + 13] = 1  # C[2, k0] += P
        
        # C[2, k1] = A20 * B0k1 + A21 * B1k1 + A22 * B2k1
        U[6][base + 14] = 1  # A20
        V[0 * p + k1][base + 14] = 1  # B0k1
        W[k1 * n + 2][base + 14] = 1  # C[2, k1] += P
        
        U[7][base + 15] = 1  # A21
        V[1 * p + k1][base + 15] = 1  # B1k1
        W[k1 * n + 2][base + 15] = 1  # C[2, k1] += P
        
        U[8][base + 16] = 1  # A22
        V[2 * p + k1][base + 16] = 1  # B2k1
        W[k1 * n + 2][base + 16] = 1  # C[2, k1] += P
        
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    valid = _validate(result["U"], result["V"], result["W"])
    print(f"Valid: {valid}")
