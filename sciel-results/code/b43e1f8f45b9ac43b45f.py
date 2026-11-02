"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 3x4x7 matrix multiplication."""
    n, m, p = 3, 4, 7
    deadline = time.time() + 140  # 140s deadline (10s safety margin)
    
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
        return {"U": U, "V": V, "W": W}
    
    def validate(U, V, W):
        """Check if decomposition is exact using integer/rational arithmetic."""
        rank = len(U[0])
        # Compute tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Should equal 1 if i,j,k form valid multiplication, 0 otherwise
        # Actually the tensor is: T[i,j,k] = delta(i,j,k) where it's the multiplication tensor
        # The multiplication tensor has T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B
        # contributes to (i,k) entry of C. So T[i,j,k] = 1 for all valid (i,j,k).
        # Wait, let me reconsider. The tensor for matrix multiplication is:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor T has indices (i,j,k) and T[i,j,k] = 1 for all i,j,k
        # No wait, that's not right either.
        
        # The matrix multiplication tensor M_{n,m,p} has:
        # M[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
        # So M[i,j,k] = 1 for all valid i in [0,n), j in [0,m), k in [0,p)
        # And the decomposition is: M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # So we need: for all i in range(n), j in range(m), k in range(p):
        #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    # Try structured decompositions
    
    # Strategy 1: Strassen-like 2x2 block decomposition for the 4x4 middle part
    # Partition 4x7 into [2x7; 2x7] and 3x4 into [3x2 | 3x2]
    # Use 2x2x2 Strassen (rank 7) for overlapping blocks
    
    # Let's try a direct structured approach:
    # C (3x7) = A (3x4) * B (4x7)
    # Split B into B1 (4x3) and B2 (4x4): C = A*B1 + A*B2
    # For A*B1 (3x4 * 4x3): schoolbook rank = 36
    # For A*B2 (3x4 * 4x4): schoolbook rank = 48
    # Total: 84 (no improvement)
    
    # Better: Use Strassen on the 4x4 part
    # A (3x4) * B (4x4) where we can apply Strassen to 2x2 blocks
    # Split A into [A11 A12; A21 A22] where A11,A12 are 2x2, A21,A22 are 1x2... 
    # This doesn't work cleanly for 3x4.
    
    # Alternative: Try rank 80, 79, ... by searching for sparse decompositions
    
    # Use a smarter search: start with schoolbook and try to merge/remove terms
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank by random search with structure
    # Use the fact that many entries can be 0, 1, -1
    
    np.random.seed(42)
    
    def random_sparse_decomposition(rank, time_limit=5.0):
        """Try to find a valid decomposition of given rank using sparse random search."""
        start = time.time()
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Initialize with schoolbook-like structure, then perturb
        for c in range(rank):
            i = np.random.randint(n)
            j = np.random.randint(m)
            k = np.random.randint(p)
            U[i*m+j, c] = np.random.choice([-1, 1])
            V[j*p+k, c] = np.random.choice([-1, 1])
            W[k*n+i, c] = np.random.choice([-1, 1])
        
        # Try local search: flip signs or zero out entries
        best_err = float('inf')
        for iteration in range(5000):
            if time.time() - start > time_limit:
                break
            
            # Compute error
            err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                        err += (s - 1) ** 2
            
            if err == 0:
                # Found valid decomposition
                return U.tolist(), V.tolist(), W.tolist()
            
            # Random perturbation
            if np.random.random() < 0.3:
                # Flip sign
                row = np.random.randint(n*m if np.random.random() < 0.33 else 
                                       m*p if np.random.random() < 0.5 else p*n)
                col = np.random.randint(rank)
                mat = U if row < n*m else (V if row < n*m + m*p else W)
                idx = row if row < n*m else (row - n*m if row < n*m + m*p else row - n*m - m*p)
                if mat[idx, col] != 0:
                    mat[idx, col] *= -1
            else:
                # Try to add a new term
                col = np.random.randint(rank)
                i = np.random.randint(n)
                j = np.random.randint(m)
                k = np.random.randint(p)
                U[i*m+j, col] = np.random.choice([-1, 1])
                V[j*p+k, col] = np.random.choice([-1, 1])
                W[k*n+i, col] = np.random.choice([-1, 1])
        
        return None
    
    # Try structured Strassen-like decomposition
    # For 3x4 * 4x7, we can think of it as:
    # Split columns of B: B = [B_left (4x3) | B_right (4x4)]
    # C = A*B_left (3x3) + A*B_right (3x4)
    
    # For the 3x4 * 4x4 part, use a modified Strassen:
    # Split A into 2x2 + 1x2 blocks, B into 2x2 + 2x2 blocks
    # This is complex; let's try a simpler structured approach.
    
    # Known result: rank(3x4x4) <= 26 (vs schoolbook 48)
    # rank(3x4x3) <= 20 (vs schoolbook 36)
    # If we can achieve these, total rank = 26 + 20 = 46
    
    # But we need to verify these bounds and construct them.
    # For now, let's do a focused search for rank < 84.
    
    # Try ranks from 83 down
    for target_rank in range(83, 70, -1):
        if time.time() > deadline - 20:
            break
        
        result = random_sparse_decomposition(target_rank, time_limit=8.0)
        if result is not None:
            U_list, V_list, W_list = result
            # Convert to list of lists
            U_out = [[int(U_list[i][j]) for j in range(target_rank)] for i in range(n*m)]
            V_out = [[int(V_list[i][j]) for j in range(target_rank)] for i in range(m*p)]
            W_out = [[int(W_list[i][j]) for j in range(target_rank)] for i in range(p*n)]
            
            if validate(U_out, V_out, W_out):
                return {"U": U_out, "V": V_out, "W": W_out}
    
    # If no improvement found, return schoolbook
    return best_result
# EVOLVE-BLOCK-END
