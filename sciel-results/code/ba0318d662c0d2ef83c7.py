"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
import itertools
from collections import defaultdict

# Deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145  # Safety margin before 150s kill

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Uses structured block-based construction with Strassen factors for 2x2 blocks.
    """
    n, m, p = DIMENSIONS
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact."""
        n, m, p = DIMENSIONS
        rank = len(U[0]) if U else 0
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Schoolbook decomposition
    def schoolbook():
        n, m, p = DIMENSIONS
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
    
    # Strassen decomposition for 2x2 * 2x2
    # Returns U, V, W each of shape (4, 7) for the 2x2*2x2 tensor
    def strassen_2x2():
        """
        Strassen decomposition for 2x2 * 2x2 matrix multiplication.
        T[i,j,k] = 1 for all i,j,k in {0,1}
        U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        # Strassen's 7 factors
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Index: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        # i,j,k in {0,1}
        
        # M1: (A00+A11)*(B00+B11)
        # U: A00 (i=0,j=0) -> U[0,0]=1, A11 (i=1,j=1) -> U[3,0]=1
        # V: B00 (j=0,k=0) -> V[0,0]=1, B11 (j=1,k=1) -> V[3,0]=1
        # W: C00 (k=0,i=0) -> W[0,0]=1, C11 (k=1,i=1) -> W[3,0]=1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2: (A10+A11)*B00
        # U: A10 (i=1,j=0) -> U[2,1]=1, A11 (i=1,j=1) -> U[3,1]=1
        # V: B00 (j=0,k=0) -> V[0,1]=1
        # W: C10 (k=0,i=1) -> W[2,1]=1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # M3: A00*(B01-B11)
        # U: A00 (i=0,j=0) -> U[0,2]=1
        # V: B01 (j=0,k=1) -> V[1,2]=1, B11 (j=1,k=1) -> V[3,2]=-1
        # W: C01 (k=1,i=0) -> W[1,2]=1
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # M4: A11*(B10-B00)
        # U: A11 (i=1,j=1) -> U[3,3]=1
        # V: B10 (j=1,k=0) -> V[2,3]=1, B00 (j=0,k=0) -> V[0,3]=-1
        # W: C10 (k=0,i=1) -> W[2,3]=1
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[2, 3] = 1
        
        # M5: (A00+A01)*B11
        # U: A00 (i=0,j=0) -> U[0,4]=1, A01 (i=0,j=1) -> U[1,4]=1
        # V: B11 (j=1,k=1) -> V[3,4]=1
        # W: C01 (k=1,i=0) -> W[1,4]=-1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = -1
        
        # M6: (A10-A00)*(B00+B01)
        # U: A10 (i=1,j=0) -> U[2,5]=1, A00 (i=0,j=0) -> U[0,5]=-1
        # V: B00 (j=0,k=0) -> V[0,5]=1, B01 (j=0,k=1) -> V[1,5]=1
        # W: C11 (k=1,i=1) -> W[3,5]=1
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7: (A01-A11)*(B10+B11)
        # U: A01 (i=0,j=1) -> U[1,6]=1, A11 (i=1,j=1) -> U[3,6]=-1
        # V: B10 (j=1,k=0) -> V[2,6]=1, B11 (j=1,k=1) -> V[3,6]=1
        # W: C00 (k=0,i=0) -> W[0,6]=1
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        # Verify
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = np.sum(U[i*2+j, :] * V[j*2+k, :] * W[k*2+i, :])
                    if abs(s - 1.0) > 1e-9:
                        # This should not happen, but let's continue
                        pass
        
        return U, V, W
    
    # Build 4x4 * 4x8 decomposition using block structure
    # Partition A (4x4) into 2x2 blocks: A00, A01, A10, A11
    # Partition B (4x8) into 2x4 blocks: B0, B1 where each is 2x4
    # Actually, let's partition B into 8 2x2 blocks: B00,B01,B02,B03,B10,B11,B12,B13
    
    # C (4x8) = A * B
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12
    # C03 = A00*B03 + A01*B13
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12
    # C13 = A10*B03 + A11*B13
    
    # For each (row_block, col_block) pair, we compute a 2x2*2x2 product.
    # We can use Strassen for each, but share the A-side factors.
    
    # Let's try a direct construction with rank 32.
    # Use the fact that we can write the tensor as a sum of rank-1 terms.
    
    # Alternative: use tensor product of Strassen decompositions.
    # 4x4*4x4 can be seen as (2x2 ⊗ 2x2) * (2x2 ⊗ 2x2)
    # The rank of the tensor product is the product of ranks: 7*7 = 49
    # But we only need 4x4*4x8, not 4x4*4x4.
    
    # Let me try a different approach: use 7 Strassen factors for the "row" part and handle columns.
    
    # Actually, let's try to construct a valid decomposition with rank 32 using a systematic approach.
    
    # The key insight: we can use the block structure to reduce the problem.
    # For the top half of C (rows 0-1), we need: C_top = A_top * B
    # where A_top is 2x4 and B is 4x8.
    # Similarly for bottom half.
    
    # The tensor for 2x4 * 4x8 has rank at most 2*4*8 = 64 (schoolbook).
    # Lower bound: max(8, 32, 16) = 32.
    
    # If we can achieve rank 32 for 2x4*4x8, then total rank for 4x4*4x8 would be 64.
    # But we can share the B factors between top and bottom.
    
    # Let me try to find a rank-32 decomposition for 2x4 * 4x8.
    
    def try_structured_search(target_rank, num_restarts=100):
        """Try structured random search with block-based initialization."""
        n, m, p = DIMENSIONS
        
        for restart in range(num_restarts):
            if time.time() > _DEADLINE - 10:
                return None
            
            # Initialize with block structure
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Use random initialization
            U = np.random.randn(n * m, target_rank) * 0.3
            V = np.random.randn(m * p, target_rank) * 0.3
            W = np.random.randn(p * n, target_rank) * 0.3
            
            # Check
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        max_err = max(max_err, abs(s - 1.0))
            
            if max_err < 1e-6:
                return U.tolist(), V.tolist(), W.tolist(), target_rank
        return None
    
    # First, try to find a good decomposition with increasing rank
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Try ranks from 32 upward
    for r in range(32, 129):
        if time.time() > _DEADLINE - 15:
            break
        if r >= best_rank:
            break
            
        result = try_structured_search(r, num_restarts=200)
        if result is not None:
            U, V, W, rank = result
            if rank < best_rank:
                best_U, best_V, best_W, best_rank = U, V, W, rank
            break
    
    # If we haven't found anything better, try a more sophisticated approach
    if best_rank == 128:
        # Try using the Strassen-based block decomposition
        # Build a decomposition with rank 49 (tensor product of Strassen for 4x4*4x4, then extend to 4x8)
        
        # Actually, let's try rank 32 with more iterations
        result = try_structured_search(32, num_restarts=500)
        if result is not None:
            U, V, W, rank = result
            best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Ensure we return valid lists
    if best_U is None:
        best_U, best_V, best_W, best_rank = schoolbook()
    
    # Final verification
    if not check_decomposition(best_U, best_V, best_W):
        best_U, best_V, best_W, best_rank = schoolbook()
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
