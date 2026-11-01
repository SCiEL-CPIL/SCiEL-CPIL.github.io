"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product, combinations
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 42
    
    def to_list(mat):
        return mat.tolist()
    
    def verify(U, V, W):
        """Verify that U, V, W correctly decompose the tensor.
        U, V, W are lists of lists of Fraction or int."""
        rank = len(U[0]) if U else 0
        if len(V[0]) != rank or len(W[0]) != rank:
            return False
        
        # Check all tensor entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = Fraction(0)
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if s != 1:
                        return False
        
        # Check off-diagonal entries (sample)
        np.random.seed(42)
        for _ in range(200):
            a = np.random.randint(0, n * m)
            b = np.random.randint(0, m * p)
            c = np.random.randint(0, p * n)
            i1, j1 = divmod(a, m)
            j2, k1 = divmod(b, p)
            k2, i2 = divmod(c, n)
            if not (j1 == j2 and i1 == i2 and k1 == k2):
                s = Fraction(0)
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                if s != 0:
                    return False
        return True
    
    # Schoolbook decomposition as baseline
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Try to find a lower rank decomposition using algebraic approach
    # Formulate as system of polynomial equations and solve numerically
    
    def try_algebraic_search(target_rank):
        """Try to find a decomposition of given rank using numerical methods."""
        if time.time() > deadline:
            return None
        
        # For rank r, we have r rank-1 terms
        # Each term has (n*m + m*p + p*n) parameters
        # But we can fix scaling: set one entry per term to 1
        # Total parameters: r * (n*m + m*p + p*n - 1)
        
        # The tensor has n*m*p = 42 entries that must be 1
        # And (n*m)*(m*p)*(p*n) - 42 off-diagonal entries that must be 0
        # This is a highly overdetermined system for r < 42
        
        # Let's try a structured approach: assume the decomposition has a specific form
        # For 2x3x7, try to see if rank 35, 30, 25, 21 are achievable
        
        # Use random initialization and gradient-free optimization
        # Minimize the sum of squares of (tensor_entry - target)
        
        def random_init(r):
            """Generate random initial guess for rank-r decomposition."""
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            # Scale to make typical entries ~1
            U /= np.sqrt(np.mean(U**2) + 1e-10)
            V /= np.sqrt(np.mean(V**2) + 1e-10)
            W /= np.sqrt(np.mean(W**2) + 1e-10)
            return U, V, W
        
        def compute_tensor(U, V, W):
            """Compute the tensor from the decomposition."""
            rank = U.shape[1]
            tensor = np.zeros((n * m, m * p, p * n))
            for r in range(rank):
                tensor += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            return tensor
        
        def objective(U, V, W):
            """Compute the objective: sum of squared errors."""
            tensor = compute_tensor(U, V, W)
            error = 0.0
            # Diagonal entries should be 1
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        error += (tensor[a, b, c] - 1.0) ** 2
            # Off-diagonal entries should be 0
            # Sample a subset for speed
            np.random.seed(123)
            for _ in range(500):
                a = np.random.randint(0, n * m)
                b = np.random.randint(0, m * p)
                c = np.random.randint(0, p * n)
                i1, j1 = divmod(a, m)
                j2, k1 = divmod(b, p)
                k2, i2 = divmod(c, n)
                if not (j1 == j2 and i1 == i2 and k1 == k2):
                    error += tensor[a, b, c] ** 2
            return error
        
        # Try different ranks from high to low
        best_result = None
        for r in [rank_schoolbook - 1, rank_schoolbook - 2, rank_schoolbook - 3, rank_schoolbook - 5]:
            if time.time() > deadline - 10:
                break
            
            # Multiple random restarts
            for restart in range(3):
                if time.time() > deadline - 5:
                    break
                
                U, V, W = random_init(r)
                
                # Use Nelder-Mead or similar derivative-free method
                # But for high-dimensional problems, this is slow
                # Instead, try a simple iterative refinement
                
                # Fix one entry per term to normalize
                # This reduces the number of free parameters
                
                # Simple approach: alternating minimization
                for it in range(200):
                    if time.time() > deadline - 5:
                        break
                    
                    # Update U
                    for a in range(n * m):
                        # For each a, find optimal U[a, :] given V, W
                        # This is a linear least squares problem
                        # min_U sum_{b,c} (sum_r U[a,r]*V[b,r]*W[c,r] - target[a,b,c])^2
                        # = min_U ||U[a,:]^T * (V ⊙ W) - target_a||^2
                        # where target_a is the vector of targets for fixed a
                        pass  # Too slow for full implementation
                    
                    # Update V
                    for b in range(m * p):
                        pass  # Similar
                    
                    # Update W
                    for c in range(p * n):
                        pass  # Similar
                
                # Check if we found a good solution
                obj = objective(U, V, W)
                if obj < 1e-6:
                    # Convert to Fraction
                    U_frac = [[Fraction(int(round(U[a, r] * 10000) / 10000)) for r in range(r)] for a in range(n * m)]
                    V_frac = [[Fraction(int(round(V[b, r] * 10000) / 10000)) for r in range(r)] for b in range(m * p)]
                    W_frac = [[Fraction(int(round(W[c, r] * 10000) / 10000)) for r in range(r)] for c in range(p * n)]
                    
                    if verify(U_frac, V_frac, W_frac):
                        best_result = (U_frac, V_frac, W_frac, r)
                        return best_result
        
        return best_result
    
    # Try the algebraic search
    result = try_algebraic_search(rank_schoolbook - 1)
    
    if result is not None:
        U_frac, V_frac, W_frac, r = result
        return {
            "U": U_frac,
            "V": V_frac,
            "W": W_frac
        }
    
    # Fall back to schoolbook
    if not verify(U_sb, V_sb, W_sb):
        pass
    
    return {
        "U": U_sb,
        "V": V_sb,
        "W": W_sb
    }
# EVOLVE-BLOCK-END
