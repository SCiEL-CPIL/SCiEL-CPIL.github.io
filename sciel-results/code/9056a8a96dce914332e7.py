"""Systematic rank-1 peeling for 2x3x5 matrix multiplication tensor."""
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Build the tensor: T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try to find a low-rank decomposition using rank-1 peeling
    # We'll search for rank-1 tensors with small integer/half-integer coefficients
    
    # First, let's try known low-rank constructions
    # For 2x3x5, the schoolbook rank is 30
    # Known results suggest rank might be around 24-28
    
    # Strategy: Try to peel off rank-1 tensors with rational coefficients
    # A rank-1 tensor is u ⊗ v ⊗ w where u ∈ R^{n*m}, v ∈ R^{m*p}, w ∈ R^{p*n}
    
    # We'll use a systematic approach:
    # 1. Start with the full tensor
    # 2. At each step, find a rank-1 tensor that is "contained" in the residual
    # 3. Subtract it and continue
    
    # For exact peeling, we need the rank-1 tensor to have entries that are
    # linear combinations of the residual entries with rational coefficients
    
    # Let's try a different approach: use the fact that matrix multiplication
    # tensors have structure. We can try to combine terms.
    
    # For 2x3x5, let's try to find a decomposition with rank < 30
    # by combining schoolbook terms in clever ways.
    
    # One approach: use the identity (a+b)(c+d) = ac + ad + bc + bd - (a-b)(c-d)
    # This can reduce the rank for certain patterns.
    
    # Let's try to systematically search for low-rank decompositions
    # by trying to combine pairs of schoolbook terms.
    
    # Schoolbook terms: for each (i,j,k), we have term T_{i,j,k} = e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # There are 2*3*5 = 30 such terms.
    
    # Can we combine two terms into fewer?
    # For example, if we have terms for (i,j,k) and (i,j',k), can we combine them?
    # e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i} + e_{i*m+j'} ⊗ e_{j'*p+k} ⊗ e_{k*n+i}
    # = (e_{i*m+j} + e_{i*m+j'}) ⊗ ... no, that doesn't work directly.
    
    # Let's try a more systematic approach using linear algebra.
    # We want to find vectors u_c, v_c, w_c for c = 1, ..., r such that
    # sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1 if (i,j,k) is valid, 0 otherwise
    
    # This is equivalent to finding a low-rank decomposition of the tensor.
    
    # Let's try to use the known fact that for matrix multiplication,
    # the rank is related to the number of multiplications needed.
    
    # For 2x3x5, let's try to find a decomposition with rank 24 or less.
    
    # Approach: Use integer linear programming or systematic search
    # to find rank-1 tensors with small integer coefficients.
    
    # Let's try a greedy approach:
    # 1. Start with residual = T
    # 2. Find a rank-1 tensor u⊗v⊗w that best approximates residual
    # 3. Subtract it
    # 4. Repeat
    
    # For the rank-1 tensor, we'll search over small integer vectors.
    
    # Actually, let's try a known construction. For multiplying an n x m matrix
    # by an m x p matrix, there are known lower bounds on the rank.
    
    # For 2x3x5, let's try to construct a decomposition with rank 25.
    
    # One approach: use the fact that 2*3*5 = 30, and try to save 5 terms
    # by combining them in pairs using Strassen-like tricks.
    
    # Let's try to use a systematic search for rank-1 tensors with coefficients
    # in {-1, 0, 1} or {-1, 0, 1, 2} or half-integers.
    
    # For efficiency, let's try to find a decomposition with rank 24.
    
    # We'll use a random search with rational coefficients.
    
    def find_rank1_approximation(residual, max_trials=1000):
        """Find a rank-1 tensor that best approximates the residual."""
        best_score = -np.inf
        best_u, best_v, best_w = None, None, None
        
        np.random.seed(42)
        for trial in range(max_trials):
            # Random initialization with small integers
            scale = np.random.choice([0.5, 1, 1.5, 2])
            u = np.random.choice([-scale, -scale/2, 0, scale/2, scale], size=n*m)
            v = np.random.choice([-scale, -scale/2, 0, scale/2, scale], size=m*p)
            w = np.random.choice([-scale, -scale/2, 0, scale/2, scale], size=p*n)
            
            # Compute the contraction
            score = np.einsum('ijk,i,j,k', residual, u, v, w)
            
            if score > best_score:
                best_score = score
                best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
        
        return best_u, best_v, best_w, best_score
    
    # Try to find a decomposition with rank < 30
    for target_rank in range(24, 30):
        residual = T.copy()
        U_list, V_list, W_list = [], [], []
        
        success = True
        for rank in range(target_rank):
            # Find a rank-1 tensor to subtract
            u, v, w, score = find_rank1_approximation(residual, max_trials=200)
            
            if score < 1e-10:
                success = False
                break
            
            # Subtract the rank-1 tensor
            residual -= np.einsum('i,j,k->ijk', u, v, w)
            
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)
        
        if not success:
            continue
        
        # Check if residual is close to zero
        if np.linalg.norm(residual) < 1e-8:
            # Verify the decomposition
            U_mat = np.array(U_list).T
            V_mat = np.array(V_list).T
            W_mat = np.array(W_list).T
            
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_mat[i*m+j] * V_mat[j*p+k] * W_mat[k*n+i])
                        expected = 1.0
                        err = abs(val - expected)
                        max_err = max(max_err, err)
            
            # Also check non-diagonal entries
            for i1 in range(n):
                for j1 in range(m):
                    for k1 in range(p):
                        for i2 in range(n):
                            for j2 in range(m):
                                for k2 in range(p):
                                    if i1 != i2 or j1 != j2 or k1 != k2:
                                        val = np.sum(U_mat[i1*m+j1] * V_mat[j2*p+k2] * W_mat[k2*n+i2])
                                        err = abs(val)
                                        max_err = max(max_err, err)
            
            if max_err < 1e-6:
                # Found a valid decomposition!
                # Convert to list of lists
                U_frac = [[float(x) for x in row] for row in U_mat]
                V_frac = [[float(x) for x in row] for row in V_mat]
                W_frac = [[float(x) for x in row] for row in W_mat]
                
                return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    # Fall back to schoolbook decomposition
    U = [[0] * (n*m*p) for _ in range(n*m)]
    V = [[0] * (n*m*p) for _ in range(m*p)]
    W = [[0] * (n*m*p) for _ in range(p*n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][column] = 1
                V[j*p+k][column] = 1
                W[k*n+i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
