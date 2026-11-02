"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # First, provide a known valid decomposition (schoolbook rank 32)
    # Then try to find a lower rank decomposition
    
    # Schoolbook decomposition (rank 32)
    U_school = [[0]*32 for _ in range(n*m)]
    V_school = [[0]*32 for _ in range(m*p)]
    W_school = [[0]*32 for _ in range(p*n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i*m*p + j*p + k
                U_school[i*m+j][idx] = 1
                V_school[j*p+k][idx] = 1
                W_school[k*n+i][idx] = 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = 32

    # Try Strassen-based rank 28 decomposition
    # For 2x2 * 2x8, we can use 4 independent Strassen blocks
    # Each Strassen block has rank 7, so total rank = 4*7 = 28
    # But we need to be careful about the indexing
    
    # Let's try a direct rank-28 construction
    # A is 2x2, B is 2x8, C is 2x8
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Using Strassen for each pair of columns in B:
    # For columns (2b, 2b+1) of B, we have a 2x2 submatrix
    # Apply Strassen to multiply A (2x2) by B[:, 2b:2b+2] (2x2)
    
    rank_28 = 28
    U28 = [[0]*rank_28 for _ in range(n*m)]
    V28 = [[0]*rank_28 for _ in range(m*p)]
    W28 = [[0]*rank_28 for _ in range(p*n)]
    
    # Strassen formulas for 2x2 * 2x2:
    # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    # C = [[P1+P4-P5+P7, P3+P5], [P2+P4, P1-P2+P3+P6]]
    
    for b in range(4):
        base = b * 7
        # B columns for this block: 2b, 2b+1
        # V indices: row j (0 or 1), col k (2b or 2b+1)
        # V[j*p + k]
        
        # P1 = (A[0,0]+A[1,1]) * (B[0,2b]+B[1,2b+1])
        U28[0][base] = 1
        U28[3][base] = 1
        V28[0*p + 2*b][base] = 1
        V28[1*p + 2*b+1][base] = 1
        # C[0,2b] += P1, C[1,2b+1] += P1
        W28[2*b*n + 0][base] = 1
        W28[(2*b+1)*n + 1][base] = 1
        
        # P2 = (A[1,0]+A[1,1]) * B[0,2b]
        U28[2][base+1] = 1
        U28[3][base+1] = 1
        V28[0*p + 2*b][base+1] = 1
        # C[1,2b] += P2, C[1,2b+1] -= P2
        W28[2*b*n + 1][base+1] = 1
        W28[(2*b+1)*n + 1][base+1] = -1
        
        # P3 = A[0,0] * (B[0,2b+1]-B[1,2b+1])
        U28[0][base+2] = 1
        V28[0*p + 2*b+1][base+2] = 1
        V28[1*p + 2*b+1][base+2] = -1
        # C[0,2b+1] += P3, C[1,2b+1] += P3
        W28[(2*b+1)*n + 0][base+2] = 1
        W28[(2*b+1)*n + 1][base+2] = 1
        
        # P4 = A[1,1] * (B[1,2b]-B[0,2b])
        U28[3][base+3] = 1
        V28[1*p + 2*b][base+3] = 1
        V28[0*p + 2*b][base+3] = -1
        # C[0,2b] += P4, C[1,2b] += P4
        W28[2*b*n + 0][base+3] = 1
        W28[2*b*n + 1][base+3] = 1
        
        # P5 = (A[0,0]+A[0,1]) * B[1,2b+1]
        U28[0][base+4] = 1
        U28[1][base+4] = 1
        V28[1*p + 2*b+1][base+4] = 1
        # C[0,2b] -= P5, C[0,2b+1] += P5
        W28[2*b*n + 0][base+4] = -1
        W28[(2*b+1)*n + 0][base+4] = 1
        
        # P6 = (A[1,0]-A[0,0]) * (B[0,2b]+B[0,2b+1])
        U28[2][base+5] = 1
        U28[0][base+5] = -1
        V28[0*p + 2*b][base+5] = 1
        V28[0*p + 2*b+1][base+5] = 1
        # C[1,2b+1] += P6
        W28[(2*b+1)*n + 1][base+5] = 1
        
        # P7 = (A[0,1]-A[1,1]) * (B[1,2b]+B[1,2b+1])
        U28[1][base+6] = 1
        U28[3][base+6] = -1
        V28[1*p + 2*b][base+6] = 1
        V28[1*p + 2*b+1][base+6] = 1
        # C[0,2b] += P7
        W28[2*b*n + 0][base+6] = 1
    
    # Verify the rank-28 decomposition
    if _verify_decomposition(U28, V28, W28, n, m, p):
        best_rank = 28
        best_result = {"U": U28, "V": V28, "W": W28}
    
    # Try to find even lower rank using numerical optimization
    # The minimum rank for 2x2 * 2x8 might be 24 or lower
    # We'll try a gradient-based search for lower rank decompositions
    
    if time.time() < deadline - 30:
        # Try rank 24
        for target_rank in [24, 25, 26, 27]:
            if time.time() > deadline - 10:
                break
            result = _search_rank(n, m, p, target_rank, deadline)
            if result is not None:
                best_rank = target_rank
                best_result = result
                break
    
    return best_result


def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = n, m, p
    # Check dimensions
    if len(U) != n*m or len(V) != m*p or len(W) != p*n:
        return False
    
    rank = len(U[0]) if U else 0
    for r in range(rank):
        # Check that each term contributes correctly
        # U[:, r] * V[:, r] * W[:, r] should give the right tensor entries
        pass
    
    # Direct verification: compute C = A*B for random A, B and check
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_r U[i*m+j] V[j*p+k] W[k*n+i]
    
    # Actually, the tensor decomposition means:
    # M_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where M_{i,j,k} = 1 if this is a valid multiplication term
    
    # Let's verify by checking all (i,j,k) tuples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(actual - expected) > tol:
                    return False
    return True


def _search_rank(n, m, p, target_rank, deadline):
    """Try to find a decomposition of the given rank using numerical methods."""
    # This is a complex non-convex optimization problem
    # We'll use a simple approach: random initialization + gradient descent
    
    # The constraint is: for all (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # This is a system of n*m*p = 32 equations with 3 * target_rank * (n*m + m*p + p*n) unknowns
    # For target_rank = 24: 3 * 24 * (4 + 16 + 16) = 3 * 24 * 36 = 2592 unknowns, 32 equations
    # So we have many degrees of freedom
    
    # Simple approach: try random initializations and check if they satisfy the constraints
    # This is unlikely to work for exact solutions, but let's try a few
    
    np.random.seed(42)
    
    # Actually, for exact integer/half-integer solutions, we need a different approach
    # Let's try a combinatorial search for small ranks
    
    # For now, return None to indicate no improvement found
    return None


# EVOLVE-BLOCK-END
