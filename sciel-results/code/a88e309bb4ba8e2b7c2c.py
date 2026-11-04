"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    
    # Target tensor T[i, j, k] = delta_{i,j,k} for matrix multiplication
    # T is a 3x4x5 tensor where T[i,j,k] = 1 if we're computing (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # Actually, the tensor for matrix multiplication is T[i,j,k] which represents the coefficient
    # such that C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor notation, the multiplication tensor has components T_{ijk} where
    # T_{ijk} = 1 if (i,j,k) corresponds to a valid multiplication term
    
    # The standard matrix multiplication tensor for n x m * m x p -> n x p
    # has rank at most n*m*p (schoolbook)
    # We want to find a lower rank decomposition
    
    # For 3x4x5, the schoolbook rank is 3*4*5 = 60
    # Known lower bounds and upper bounds:
    # - Lower bound: max(n*p, m*p, n*m) = max(15, 20, 12) = 20
    # - Actually the Strassen-like lower bounds are more complex
    # - For 3x4x5, we can try to find rank < 60
    
    # Let's try a direct search for low-rank decompositions
    # The tensor T has entries T[i,j,k] = 1 for all valid (i,j,k) in the multiplication
    # Actually, for matrix multiplication A (n x m) * B (m x p) = C (n x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Wait, that's not right. The tensor is defined such that:
    # T(A,B) = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * C^*[i,k]
    # where C^* is the conjugate/dual
    
    # Actually, let me think more carefully. The matrix multiplication tensor
    # M_{n,m,p} is defined by:
    # M_{n,m,p} = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_{i,j} \otimes e_{j,k} \otimes e_{i,k}
    # where e_{i,j} is the basis vector for the (i,j) entry of the first matrix
    
    # So T[i,j,k] = 1 if the indices form a valid multiplication term
    # In our case, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Wait, that gives n*m*p ones, which is the schoolbook
    
    # Let me reconsider. The tensor for multiplying (n x m) by (m x p) matrix:
    # The input has n*m entries from A and m*p entries from B
    # The output has n*p entries for C
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # So the bilinear form is: sum_{i,j,k} A[i,j] * B[j,k] * delta_{i,k}^output
    # The tensor T is a 3-tensor with dimensions (n*m, m*p, n*p)
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[a,b,c] = 1 if a, b, c correspond to valid indices
    
    # Let's build the target tensor
    def build_target_tensor(n, m, p):
        """Build the matrix multiplication tensor as a 3D array."""
        # T has shape (n*m, m*p, n*p)
        T = np.zeros((n*m, m*p, n*p), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j  # index in A
                    b = j * p + k  # index in B
                    c = k * n + i  # index in C (using W[k*n+i,r] convention)
                    T[a, b, c] = 1
        return T
    
    T_target = build_target_tensor(n, m, p)
    
    # Schoolbook solution as fallback
    def schoolbook_solution(n, m, p):
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook_solution(n, m, p)
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Verify a decomposition
    def verify_decomposition(U, V, W, T_target, tol=1e-6):
        """Check if U, V, W give a valid decomposition of T_target."""
        # T_approx[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # This is the tensor product of the three matrices
        # T_approx = sum_r U[:,r] \otimes V[:,r] \otimes W[:,r]
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        
        # Compute T_approx
        # For efficiency, use einsum
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Check if it matches T_target
        diff = np.abs(T_approx - T_target)
        return np.max(diff) < tol, np.max(diff)
    
    # Try to find lower rank decompositions using random search
    # Strategy: try ranks from lower bound up to schoolbook rank
    # For each rank, try random initializations and optimize
    
    def try_random_decomposition(rank, n, m, p, T_target, max_iters=1000, seed=None):
        """Try to find a rank-r decomposition using random search."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Lower bound for rank
        # The rank must be at least max(n*p, m*p, n*m) for matrix multiplication
        # Actually, this is a lower bound on the rank of the tensor
        lower_bound = max(n*p, m*p, n*m)
        if rank < lower_bound:
            return None
        
        # Random initialization
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Try to optimize using gradient-free methods
        # For small problems, we can try direct optimization
        
        # Actually, for exact decompositions, we need integer or half-integer entries
        # Let's try a different approach: use the fact that for matrix multiplication,
        # there are known structured decompositions
        
        # For 3x4x5, let's try to use block structure
        # Split the 3x4 matrix into blocks and use Strassen-like ideas
        
        return None  # Placeholder, will implement below
    
    # For 3x4x5, let's try specific structured approaches
    # Approach 1: Use the fact that 3x4x5 can be decomposed using smaller multiplications
    
    # Actually, let's try a direct numerical approach
    # We want to find U, V, W such that:
    # sum_r U[a,r] * V[b,r] * W[c,r] = T_target[a,b,c] for all a,b,c
    
    # This is a system of equations. Let's try to solve it for small ranks.
    
    # For rank r, we have r variables per entry of U, V, W
    # Total variables: r*(n*m + m*p + p*n)
    # Constraints: n*m*p (one for each nonzero entry of T)
    
    # Let's try ranks from lower bound upward
    lower_bound = max(n*p, m*p, n*m)  # = max(15, 20, 12) = 20
    
    # Try a few specific ranks
    for rank in range(lower_bound, min(rank_sb, lower_bound + 20)):
        if time.time() > deadline:
            break
        
        # Try random search for this rank
        found = False
        for trial in range(10):  # 10 trials per rank
            if time.time() > deadline:
                break
            
            np.random.seed(trial * 1000 + rank)
            
            # Random initialization
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Simple iterative refinement
            # For each (a,b,c) where T_target[a,b,c] = 1:
            # We need sum_r U[a,r]*V[b,r]*W[c,r] = 1
            # For each (a,b,c) where T_target[a,b,c] = 0:
            # We need sum_r U[a,r]*V[b,r]*W[c,r] = 0
            
            # This is hard to solve directly. Let's try a different approach.
            
            # Use the fact that we can fix some entries and solve for others
            # Or use a greedy approach
            
            # For now, just check if random works (unlikely)
            valid, err = verify_decomposition(U, V, W, T_target)
            if valid:
                best_rank = rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                found = True
                break
        
        if found:
            break
    
    # Convert to lists with integer/half-integer entries if possible
    # For now, return the best found
    def to_list(mat):
        return mat.tolist()
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
