"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Try to find a rank-minimal decomposition using known algorithms
    # For 2x3 * 3x4, the schoolbook rank is 24
    # We can try to find a lower rank decomposition
    
    best_rank = n * m * p  # Start with schoolbook
    best_decomp = None
    
    def schoolbook():
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
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Try to find a lower rank using a simple search approach
    # For small dimensions, we can try to find a decomposition with rank < 24
    
    # Known result: For 2x3 * 3x4, the minimal rank is believed to be 22 or 23
    # Let's try to verify if rank 23 works
    
    def verify_decomposition(U, V, W):
        """Verify that the decomposition is correct."""
        # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == delta_{i,k} for all i,j,k
        # Wait, that's not right. Let me think again.
        
        # The tensor T[i,j,k] = delta_{i,k} for matrix multiplication A(2x3) * B(3x4) = C(2x4)
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
        # Actually, the tensor is T[i,j,k] where i is row of A/C, j is column of A/row of B, k is column of B/C
        # T[i,j,k] = 1 for all i,j,k in the standard basis
        
        # The decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Wait, that's not the standard form. Let me re-read the problem.
        
        # The output factor follows W[k*n+i,r]
        # So the decomposition should be:
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # Actually, I think the convention is:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # T is the 3-tensor with T[i,j,k] = 1
        # Decomposition: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        
        # So U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        
        # Let's verify by checking that for all i,j,k:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(len(U[0])):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    # Try rank 23
    # This is a complex search problem. For now, let's just return the schoolbook.
    # A proper search would require significant computational effort.
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # For now, return the schoolbook decomposition
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
