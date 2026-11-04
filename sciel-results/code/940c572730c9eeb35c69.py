"""Matrix multiplication 2x4x7 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    def build_rank48_decomposition():
        """
        Build a rank-48 decomposition.
        
        Structure: 
        - Split into two 2x2x7 blocks (j=0,1 and j=2,3)
        - Each 2x2x7: split k into (0,1),(2,3),(4,5) as 2x2x2 blocks (rank 7 each) 
          and k=6 as 2x2x1 (rank 4)
        - Per 2x2x7: 3*7 + 4 = 25
        - Total: 2*25 = 50
        
        To get rank 48, we need to share 2 terms.
        
        Alternative: Use the fact that 2x2x1 has rank 4, but we can potentially
        merge the k=6 terms across the two j-blocks.
        
        Actually, let's just build rank 50 correctly first, then try to reduce.
        """
        rank = 50
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        
        def add_winograd_2x2x2(i0, j0, k0):
            """Add Winograd's 7-term decomposition for C[i0:i0+2, k0:k0+2] += A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]"""
            nonlocal col
            # M1 = (A00+A11)*(B00+B11)
            U[i0*m + j0][col] = 1
            U[(i0+1)*m + (j0+1)][col] = 1
            V[j0*p + k0][col] = 1
            V[(j0+1)*p + (k0+1)][col] = 1
            W[k0*n + i0][col] = 1
            W[(k0+1)*n + (i0+1)][col] = 1
            col += 1
            # M2 = (A10+A11)*B00
            U[(i0+1)*m + j0][col] = 1
            U[(i0+1)*m + (j0+1)][col] = 1
            V[j0*p + k0][col] = 1
            W[k0*n + (i0+1)][col] = 1
            W[(k0+1)*n + (i0+1)][col] = -1
            col += 1
            # M3 = A00*(B01-B11)
            U[i0*m + j0][col] = 1
            V[j0*p + (k0+1)][col] = 1
            V[(j0+1)*p + (k0+1)][col] = -1
            W[(k0+1)*n + i0][col] = 1
            W[(k0+1)*n + (i0+1)][col] = 1
            col += 1
            # M4 = A11*(B10-B00)
            U[(i0+1)*m + (j0+1)][col] = 1
            V[(j0+1)*p + k0][col] = 1
            V[j0*p + k0][col] = -1
            W[k0*n + i0][col] = 1
            W[k0*n + (i0+1)][col] = 1
            col += 1
            # M5 = (A00+A01)*B11
            U[i0*m + j0][col] = 1
            U[i0*m + (j0+1)][col] = 1
            V[(j0+1)*p + (k0+1)][col] = 1
            W[k0*n + i0][col] = -1
            W[(k0+1)*n + i0][col] = 1
            col += 1
            # M6 = (A10-A00)*(B00+B01)
            U[(i0+1)*m + j0][col] = 1
            U[i0*m + j0][col] = -1
            V[j0*p + k0][col] = 1
            V[j0*p + (k0+1)][col] = 1
            W[(k0+1)*n + (i0+1)][col] = 1
            col += 1
            # M7 = (A01-A11)*(B10+B11)
            U[i0*m + (j0+1)][col] = 1
            U[(i0+1)*m + (j0+1)][col] = -1
            V[(j0+1)*p + k0][col] = 1
            V[(j0+1)*p + (k0+1)][col] = 1
            W[k0*n + i0][col] = 1
            col += 1
        
        def add_2x2x1(i0, j0, k0):
            """Add rank-4 decomposition for C[i0:i0+2, k0] += A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0]"""
            nonlocal col
            # This is 2x2x1 matrix mult: C[2x1] = A[2x2] * B[2x1]
            # Rank of 2x2x1 tensor is 4
            # M1 = A00*B0
            U[i0*m + j0][col] = 1
            V[j0*p + k0][col] = 1
            W[k0*n + i0][col] = 1
            col += 1
            # M2 = A01*B1
            U[i0*m + (j0+1)][col] = 1
            V[(j0+1)*p + k0][col] = 1
            W[k0*n + i0][col] = 1
            col += 1
            # M3 = A10*B0
            U[(i0+1)*m + j0][col] = 1
            V[j0*p + k0][col] = 1
            W[k0*n + (i0+1)][col] = 1
            col += 1
            # M4 = A11*B1
            U[(i0+1)*m + (j0+1)][col] = 1
            V[(j0+1)*p + k0][col] = 1
            W[k0*n + (i0+1)][col] = 1
            col += 1
        
        # Block 1: i=0,1, j=0,1, k=0..6
        # k=0,1: 2x2x2
        add_winograd_2x2x2(0, 0, 0)
        # k=2,3: 2x2x2
        add_winograd_2x2x2(0, 0, 2)
        # k=4,5: 2x2x2
        add_winograd_2x2x2(0, 0, 4)
        # k=6: 2x2x1
        add_2x2x1(0, 0, 6)
        
        # Block 2: i=0,1, j=2,3, k=0..6
        # k=0,1: 2x2x2
        add_winograd_2x2x2(0, 2, 0)
        # k=2,3: 2x2x2
        add_winograd_2x2x2(0, 2, 2)
        # k=4,5: 2x2x2
        add_winograd_2x2x2(0, 2, 4)
        # k=6: 2x2x1
        add_2x2x1(0, 2, 6)
        
        return U, V, W, col
    
    def verify(U, V, W, rank):
        """Verify the decomposition is correct."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Build rank-50 decomposition
    U, V, W, rank = build_rank48_decomposition()
    
    # Verify correctness
    if not verify(U, V, W, rank):
        # Fallback to trivial decomposition
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U[i*m + j][idx] = 1
                    V[j*p + k][idx] = 1
                    W[k*n + i][idx] = 1
        return {"U": U, "V": V, "W": W}
    
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try to reduce rank by removing redundant terms
    # Convert to numpy for efficient operations
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Try removing each term and checking if the decomposition still works
    # This is expensive, so do it selectively
    for r in range(rank - 1, -1, -1):
        if time.time() > deadline - 30:
            break
        
        # Try zeroing out term r
        U_test = U_np.copy()
        V_test = V_np.copy()
        W_test = W_np.copy()
        U_test[:, r] = 0
        V_test[:, r] = 0
        W_test[:, r] = 0
        
        # Quick check: compute tensor for a few entries
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_test[i*m+j], V_test[j*p+k] * W_test[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Remove this term
            U_np = np.delete(U_np, r, axis=1)
            V_np = np.delete(V_np, r, axis=1)
            W_np = np.delete(W_np, r, axis=1)
            rank -= 1
    
    # Final verification
    U_final = U_np.tolist()
    V_final = V_np.tolist()
    W_final = W_np.tolist()
    
    if not verify(U_final, V_final, rank):
        # Return the original if reduction broke something
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
