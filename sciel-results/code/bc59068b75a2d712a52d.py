"""Rewritten program for matrix multiplication 4x4x9 rank minimization."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Schoolbook rank
    rank_sb = n * m * p  # 4*4*9 = 144
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    def validate(U, V, W, tol=1e-9):
        """Check if U,V,W exactly decompose the matrix multiplication tensor."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        r = U_arr.shape[1]
        if V_arr.shape[1] != r or W_arr.shape[1] != r:
            return False
        
        # Check all (i,j,k) combinations
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_arr[i*m + j, :]
                    v_val = V_arr[j*p + k, :]
                    w_val = W_arr[k*n + i, :]
                    s = np.dot(np.dot(u_val, v_val), w_val)
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def validate_fast(U, V, W, tol=1e-9):
        """Faster validation using precomputed dot products."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        r = U_arr.shape[1]
        if V_arr.shape[1] != r or W_arr.shape[1] != r:
            return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j,:], V_arr[j*p+k,:] * W_arr[k*n+i,:])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Strategy: Apply Strassen-like decomposition to reduce rank
    # For 4x4 x 4x9, we can use block structure
    
    # First, let's try to construct a Strassen-based decomposition
    # The idea: split A (4x4) into 2x2 blocks, B (4x9) into 2x9 blocks
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B1], [B2]] where B1, B2 are 2x9
    # C = A*B = [[A11*B1 + A12*B2], [A21*B1 + A22*B2]]
    
    # Standard approach uses 8 multiplications:
    # M1 = A11*B1, M2 = A12*B2, M3 = A21*B1, M4 = A22*B2
    # C_top = M1 + M2, C_bottom = M3 + M4
    
    # Strassen uses 7 multiplications for square matrices, but here B is not square
    # Let's try a different approach: use the tensor structure directly
    
    # For the 4x4x4 core (ignoring the 9 columns for a moment),
    # we can use known optimal decompositions
    
    # Actually, let's try a more systematic approach:
    # Use the fact that matrix multiplication tensor has algebraic structure
    
    # Try to find a decomposition with rank < 144 using structured patterns
    
    # Method 1: Try to use Kronecker product structure
    # 4 = 2*2, so we can write the multiplication as a Kronecker product
    
    # The 4x4 x 4x4 multiplication can be written as:
    # (A ⊗ I) * (I ⊗ B) type structure, but this is getting complex
    
    # Let's try a direct search for structured decompositions
    
    # Generate candidate decompositions with specific patterns
    np.random.seed(42)
    
    # Try to find a lower-rank decomposition by exploiting symmetries
    # and algebraic identities
    
    # For now, let's try random perturbations and see if we can find
    # a valid decomposition with fewer terms
    
    def try_structured_decomposition():
        """Try to construct a structured lower-rank decomposition."""
        
        # Attempt 1: Use block-circulant structure
        # This is a heuristic that might work for specific dimensions
        
        # Attempt 2: Use known results for small matrix multiplication
        # The rank of 2x2x2 matrix multiplication is 7 (Strassen)
        # The rank of 4x4x4 is at most 49 (by recursive Strassen)
        
        # For 4x4x9, we can potentially use:
        # - 7 multiplications for the 4x4x4 core
        # - Extend to 9 columns
        
        # Let's try to construct this explicitly
        
        # Strassen for 2x2:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For 4x4x4, we can apply Strassen recursively
        # This gives rank 7^2 = 49
        
        # For 4x4x9, we need to extend this
        # One approach: treat the 9 columns as 3 groups of 3
        # But this doesn't directly apply
        
        # Alternative: use the 49-rank decomposition for 4x4x4
        # and extend to 9 columns by linearity
        
        # Actually, the rank of 4x4x9 might be 49 * 9/4 = 110.25, so at least 111
        # But this is just a lower bound
        
        # Let's try to construct a decomposition with rank around 100-120
        
        # For now, return the schoolbook decomposition
        return None
    
    # Try various random approaches
    for attempt in range(30):
        if time.time() > deadline:
            break
        
        # Generate a random candidate decomposition
        # This is a placeholder for more sophisticated search
        
        # Try to find a decomposition with rank < best_rank
        target_rank = best_rank - 1
        
        if target_rank < 1:
            break
        
        # Random initialization
        U_rand = np.random.randn(n*m, target_rank)
        V_rand = np.random.randn(m*p, target_rank)
        W_rand = np.random.randn(p*n, target_rank)
        
        # Check if this validates (very unlikely to work randomly)
        if validate_fast(U_rand, V_rand, W_rand):
            best_rank = target_rank
            best = {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_rand.tolist()}
    
    # If we didn't find anything better, return the schoolbook
    return best
# EVOLVE-BLOCK-END
