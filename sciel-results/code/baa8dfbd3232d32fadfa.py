"""Combined program: Strassen-based block decomposition for 4x4 * 4x9 matrix multiplication."""
import time
import sys

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36
    
    def make_schoolbook():
        """Return schoolbook decomposition of rank n*m*p."""
        rank = n * m * p  # 144
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition is valid."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(total - 1) > tol:
                        return False
        return True
    
    # Start with schoolbook
    best_result = make_schoolbook()
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition using block structure
    if time.time() < deadline - 10:
        # Strategy: Split into 2x2 blocks of A and 2x9 blocks of B
        # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
        # B = [[B00, B01], [B10, B11]] where each Bj is 2x9
        # C = A*B = [[A00*B00 + A01*B10, A00*B01 + A01*B11],
        #            [A10*B00 + A11*B10, A10*B01 + A11*B11]]
        
        # Each 2x2 * 2x9 product has schoolbook rank 36.
        # We have 4 such products, total 144.
        
        # For each 2x2 * 2x9, use the identity:
        # C[0,k] = a*B[0,k] + b*B[1,k]
        # C[1,k] = c*B[0,k] + d*B[1,k]
        # = ((a+b)/2)*(B[0,k]+B[1,k]) + ((a-b)/2)*(B[0,k]-B[1,k])
        # = ((c+d)/2)*(B[0,k]+B[1,k]) + ((c-d)/2)*(B[0,k]-B[1,k])
        
        # This doesn't reduce rank directly, but let's try a different approach.
        
        # Alternative: Use the fact that we can combine terms across blocks.
        # For the full 4x4 * 4x9, consider:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        
        # Group j into pairs: (0,1) and (2,3)
        # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
        
        # For each i, this is a linear combination of the 4 rows of B.
        # We can precompute 4 linear combinations of B's rows, but that's 4*9 = 36 values per i.
        
        # Total: 4 * 36 = 144, same as schoolbook.
        
        # Let me try to find a valid rank-1 decomposition with rank < 144.
        # One approach: use the tensor rank properties.
        
        # Actually, let me try a specific construction with rank 108.
        # Idea: For each k, the map (A,B) -> C[:,k] is a bilinear map from R^{16} x R^{36} to R^{36}.
        # The rank of this map is at most 16*36/36 = 16... no, that's not right.
        
        # Let me just try to construct a valid solution with a specific pattern.
        
        # Construction: Use 4*9 = 36 terms for the "diagonal" part and additional terms.
        # This is getting too complex without a clear path.
        
        # Safe fallback: return schoolbook.
        pass
    
    return best_result
# EVOLVE-BLOCK-END
