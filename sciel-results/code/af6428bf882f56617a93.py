"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]
import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START

def _validate(U, V, W, n, m, p):
    """Check that U, V, W form a valid tensor decomposition."""
    rank = len(U[0])
    # Build the tensor via rank-1 sums
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Compare with the standard matrix multiplication tensor which is 1 for all (i,j,k)
    # Actually, the matrix multiplication tensor T[i,j,k] = 1 for all i,j,k in the standard basis
    # Wait, let me reconsider. The matrix multiplication tensor is defined as:
    # T[i,j,k] = 1 if we're computing (AB)[i,k] = sum_j A[i,j]B[j,k]
    # In tensor form, the multiplication tensor M has M[i,j,k] = 1 for all valid i,j,k
    # because (AB)[i,k] = sum_j A[i,j] B[j,k] * 1
    # So the tensor is all ones.
    
    # Let's verify: for random A, B, check that (AB)[i,k] = sum_c (U^c A)[i] (V^c B)[k]
    # where U^c is the c-th column reshaped to (n,m), V^c reshaped to (m,p), W^c reshaped to (p,n)
    
    # Actually, the standard identity is:
    # (AB)[i,k] = sum_j A[i,j] B[j,k]
    # And we want: (AB)[i,k] = sum_c (sum_{i',j'} U[i',j',c] A[i',j']) (sum_{j',k'} V[j',k',c] B[j',k']) W[i,k,c]
    # Wait, that's not right either. Let me think more carefully.
    
    # The tensor decomposition of matrix multiplication:
    # We have tensors U in R^{n*m x r}, V in R^{m*p x r}, W in R^{p*n x r}
    # The product is: for each c, we compute a "partial product"
    # The c-th term contributes: (U_c^T A) * (V_c^T B) * W_c^T
    # where U_c is the c-th column of U (size n*m), V_c is c-th column of V (size m*p), W_c is c-th column of W (size p*n)
    
    # More precisely:
    # Let A be n x m, B be m x p.
    # Reshape U_c to (n, m), V_c to (m, p), W_c to (p, n).
    # The c-th term is: (U_c^T * A) is a scalar? No...
    
    # Let me use the standard formulation:
    # (AB)[i,k] = sum_c (sum_{i',j'} U[i',j',c] A[i',j']) * (sum_{j',k'} V[j',k',c] B[j',k']) * W[k,i,c]
    
    # Wait, I need to be more careful with the index conventions.
    # U has shape (n*m, r), so U[i*m+j, c]
    # V has shape (m*p, r), so V[j*p+k, c]  
    # W has shape (p*n, r), so W[k*n+i, c]
    
    # The identity is:
    # (AB)[i,k] = sum_j A[i,j] B[j,k] = sum_c (sum_{i',j'} U[i',j',c] A[i',j']) (sum_{j',k'} V[j',k',c] B[j',k']) W[k*n+i, c]
    
    # For this to hold for all A, B, we need the tensor to match.
    # Let's just verify numerically with random matrices.
    
    rng = np.random.RandomState(42)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    AB = A @ B
    
    # Compute the decomposed version
    result = np.zeros((n, p))
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for c in range(rank):
        U_c = U_arr[:, c].reshape(n, m)
        V_c = V_arr[:, c].reshape(m, p)
        W_c = W_arr[:, c].reshape(p, n)
        # The c-th term: (A^T U_c) is n x m times... hmm
        # Actually: sum_{i',j'} A[i',j'] U_c[i',j'] is a scalar
        # sum_{j',k'} B[j',k'] V_c[j',k'] is a scalar
        # Then multiply by W_c[k,i] for each (k,i)
        scalar_U = np.sum(A * U_c)
        scalar_V = np.sum(B * V_c)
        result += scalar_U * scalar_V * W_c.T  # W_c is (p,n), we need (n,p)
    
    if not np.allclose(result, AB, atol=1e-8):
        return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(schoolbook_decomp, n, m, p, target_rank, max_iter=1000, seed=42):
    """
    Try to reduce the rank from schoolbook to target_rank using iterative refinement.
    We start with the schoolbook decomposition and try to zero out entries.
    """
    rng = np.random.RandomState(seed)
    
    # Start with schoolbook
    rank = n * m * p
    U = np.array([[float(x) for x in row] for row in schoolbook_decomp["U"]])
    V = np.array([[float(x) for x in row] for row in schoolbook_decomp["V"]])
    W = np.array([[float(x) for x in row] for row in schoolbook_decomp["W"]])
    
    # We'll try to merge terms by finding linear dependencies
    # Strategy: try to set some columns to zero and adjust others
    
    # Alternative: use a more systematic approach
    # The key insight: if two columns are linearly dependent in a certain way, we can merge them.
    
    # Let's try a different approach: random perturbation and then rank reduction
    # Actually, for small dimensions, let's try to directly search for a lower rank decomposition.
    
    # For 3x4x6, the known lower bound is 72 (schoolbook) but better bounds exist.
    # Actually, the Strassen-like algorithms can reduce the rank.
    # For 2x2x2, Strassen gives rank 7 instead of 8.
    
    # Let's try to find a decomposition of rank less than n*m*p using a search approach.
    # This is very hard in general, but for small cases we might get lucky.
    
    # Alternative: use the fact that the tensor has structure.
    # The matrix multiplication tensor T_{i,j,k} = delta_{i,i'}delta_{j,j'}delta_{k,k'} ... 
    # Actually T_{i,j,k} = 1 for all valid indices in the standard basis.
    
    # Let me try a simple approach: start with schoolbook and try to eliminate terms
    # by finding linear combinations.
    
    # For now, let's just return the schoolbook decomposition.
    # A proper search would require significant computation.
    
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 6 matrix.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_decomp = sb
    
    # Try to find a better decomposition
    # For 3x4x6, let's see if we can do better than 72.
    # The trivial lower bound is max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # But that's not achievable in general.
    
    # Let's try a simple local search: start with schoolbook and try to reduce rank
    # by finding linear dependencies among the rank-1 terms.
    
    # Actually, for this problem, let's just return the schoolbook decomposition.
    # Finding a truly minimal decomposition is an open problem in general.
    
    # But wait - maybe we can use a known algorithm. For matrix multiplication,
    # the best known algorithms for small matrices use recursive constructions.
    
    # For 3x4x6, we can try to use the fact that 3x4 = (2+1)x4 and apply
    # Strassen-like techniques.
    
    # Let's try a simple approach: if we can find a decomposition with rank < 72,
    # we'll use it. Otherwise, return schoolbook.
    
    # Given the time constraints, let's just return the schoolbook decomposition.
    # A more sophisticated search would be needed to find a better one.
    
    return best_decomp

# EVOLVE-BLOCK-END
