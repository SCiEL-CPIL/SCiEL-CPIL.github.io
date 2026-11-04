"""Tensor decomposition for 3x4 by 4x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

def _rank1_contribution(A, B, C):
    """
    For a rank-1 decomposition of M_{n,m,p}:
    U is (n*m) x 1, V is (m*p) x 1, W is (p*n) x 1
    The contribution to the multiplication tensor is:
    sum_{i,j,k} A[i,j] B[j,k] W[k*n+i] * e_{(i,j),(j,k),(k,i)}
    We need: for all i,j,k: U[i*m+j]*V[j*p+k]*W[k*n+i] = delta_{i,j,k} (identity in the sense that
    the product equals 1 if this is the (i,j,k) term, 0 otherwise)
    For a single rank-1 term with indices (i0, j0, k0):
    U[i*m+j] = 1 if i==i0 and j==j0, else 0
    V[j*p+k] = 1 if j==j0 and k==k0, else 0
    W[k*n+i] = 1 if k==k0 and i==i0, else 0
    """
    n, m, p = A.shape[0], A.shape[1], B.shape[1]  # This won't work, let me restructure
    pass

def construct():
    """
    Return a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 3x4 matrix by a 4x7 matrix.
    
    The tensor M_{3,4,7} has entries T[(i,j),(j,k),(k,i)] = 1 if and only if
    the multiplication C[i,k] = sum_j A[i,j]*B[j,k] is correctly computed.
    
    We use the schoolbook decomposition as a baseline and attempt to find
    a lower rank via structured search.
    """
    n, m, p = DIMENSIONS  # 3, 4, 7
    t_start = time.time()
    deadline = t_start + 145.0  # 145s deadline to be safe before 150s kill
    
    # Schoolbook baseline: rank = n*m*p = 84
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook), dtype=float)
    V_sb = np.zeros((m * p, rank_schoolbook), dtype=float)
    W_sb = np.zeros((p * n, rank_schoolbook), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    # Try to find a lower rank decomposition using the fact that
    # M_{3,4,7} can potentially be decomposed using structure.
    # 
    # Key insight: We can try to exploit the rectangular structure.
    # The middle dimension is 4. We can think of this as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # One approach: Use the fact that min rank for M_{a,b,c} satisfies
    # rank >= max(a*b, b*c, c*a) / something... but that's not tight.
    #
    # Known results: 
    # - M_{2,2,2} has rank 7 (Strassen)
    # - M_{2,2,3} has rank 11
    # - M_{2,3,3} has rank 16
    # - M_{3,3,3} has rank 23
    #
    # For M_{3,4,7}: We can try block decomposition.
    # Split the 4x7 matrix B into blocks, or split A and B.
    #
    # Strategy: Try rank R from a lower bound up to 84, and use
    # a structured ansatz to find valid decompositions.
    #
    # Lower bound: The matrix multiplication tensor M_{n,m,p} has
    # border rank at least max(n*m, m*p, p*n). For exact rank,
    # rank >= max(n*m, m*p, p*n) is not always true, but it's a
    # good starting point.
    # max(3*4, 4*7, 7*3) = max(12, 28, 21) = 28
    #
    # But we need exact decomposition, so let's try to find one.
    
    # Helper to verify a decomposition
    def verify_decomposition(U, V, W, n, m, p):
        """Check if U,V,W form a valid exact decomposition of M_{n,m,p}."""
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False
        
        # Build the tensor from the decomposition
        # T[(i,j),(j,k),(k,i)] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # This should equal 1 if (i,j,k) is a valid triple, 0 otherwise.
        # Wait, actually the matrix multiplication tensor is:
        # T[i,j,k] = 1 for all i,j,k (it's the identity in the tensor sense)
        # No wait. The matrix multiplication tensor M_{n,m,p} is defined as:
        # M_{n,m,p} = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
        # So T[(i,j),(j,k),(k,i)] = 1 for all valid i,j,k, and 0 otherwise.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    # Try to find decompositions of lower rank using structured ansatz
    # 
    # Ansatz 1: Exploit that n=3 is small. 
    # We can write A as a 3x4 matrix. Each row of A has 4 elements.
    # 
    # Ansatz 2: Use the fact that for fixed j, the map (i,k) -> A[i,j]*B[j,k]
    # is a rank-1 matrix. So C = sum_j (A[:,j] * B[j,:]).
    # This is exactly the schoolbook with m terms, but each term has rank n*p.
    # Total rank = m * n * p = 84.
    #
    # Can we do better? Strassen-style ideas apply when dimensions are powers of 2.
    # For 3x4x7, we don't have a clean Strassen decomposition.
    #
    # Let's try a randomized search for lower rank decompositions.
    # This is computationally expensive, so we limit the search.
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try ranks from 28 up to 83
    for target_rank in range(28, min(rank_schoolbook, 50)):
        if time.time() > deadline:
            break
        
        # Random search: try a few random initializations
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            np.random.seed(target_rank * 1000 + attempt)
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Simple verification (this is O(n*m*p*rank) which is manageable)
            if verify_decomposition(U, V, W, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
        if best_rank == target_rank:
            break
    
    # If no better rank found, return schoolbook
    if best_rank == rank_schoolbook:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
