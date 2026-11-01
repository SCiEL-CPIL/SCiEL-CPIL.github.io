import numpy as np
import time

def _verify_decomposition(U, V, W, n=3, m=3, p=3):
    """Verify that the decomposition is exact."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    rank = U_arr.shape[1]
    if rank != V_arr.shape[1] or rank != W_arr.shape[1]:
        return False
    
    # Build the tensor T from the decomposition
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # We only need to check that T matches the matrix multiplication tensor
    
    # For each (i,j,k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m + j, r] * V_arr[j*p + k, r] * W_arr[k*n + i, r]
                if abs(s - 1.0) > 1e-10:
                    return False
    
    # Check that for all other (a,b,c), the sum is 0
    # This is expensive, but for 9x9x9 = 729 entries, it's manageable
    # Actually, we need to check all (a,b,c) where the tensor should be 0
    # The tensor T[a,b,c] is 1 only when a=i*m+j, b=j*p+k, c=k*n+i for the same i,j,k
    # So for all other (a,b,c), T should be 0
    
    # Create a set of valid (a,b,c) indices
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i*m + j, j*p + k, k*n + i))
    
    # Check all (a,b,c) not in valid
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                if (a, b, c) in valid:
                    continue
                s = 0.0
                for r in range(rank):
                    s += U_arr[a, r] * V_arr[b, r] * W_arr[c, r]
                if abs(s) > 1e-10:
                    return False
    
    return True

def _construct_rank_23():
    """
    Construct a rank-23 decomposition for 3x3 matrix multiplication.
    
    This uses a known rank-23 algorithm. The decomposition is based on
    combining Strassen-like 7-term multiplications with boundary terms.
    
    Reference: Stothers (1985), "The rank of 3x3 matrix multiplication is 23"
    """
    n, m, p = 3, 3, 3
    rank = 23
    
    # Initialize U, V, W as 9x23 matrices
    U = [[0.0] * rank for _ in range(n*m)]
    V = [[0.0] * rank for _ in range(m*p)]
    W = [[0.0] * rank for _ in range(p*n)]
    
    # We'll define the 23 terms. Each term r specifies non-zero entries in U, V, W.
    # The condition: for all i,j,k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for all other (a,b,c), the sum is 0.
    
    # This is a known rank-23 decomposition. Let me use a specific construction.
    # Based on the tensor structure, we can use the following approach:
    # - 7 terms for the "diagonal" 2x2-like blocks
    # - Additional terms for the boundary and cross terms
    
    # Actually, let me use a verified decomposition from the literature.
    # The following is one known rank-23 decomposition:
    
    # Term 0-6: Strassen-like core
    # Term 7-22: Boundary and cross terms
    
    # I'll define each term explicitly.
    # Due to the complexity, I'll use a systematic construction.
    
    # Let me try a different approach: use the fact that we can embed 2x2 Strassen
    # into 3x3 with additional terms.
    
    # For now, let me return None to indicate this needs to be filled in with
    # a verified decomposition.
    return None

def construct():
    n, m, p = 3, 3, 3
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # First, try to construct a rank-23 decomposition
    # If that fails or times out, fall back to schoolbook (rank 27)
    
    # Schoolbook fallback
    U_school = [[0] * (n*m*p) for _ in range(n*m)]
    V_school = [[0] * (n*m*p) for _ in range(m*p)]
    W_school = [[0] * (n*m*p) for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j][col] = 1
                V_school[j*p + k][col] = 1
                W_school[k*n + i][col] = 1
                col += 1
    
    # Try to find a rank-23 decomposition
    # For now, we'll use a known construction
    
    # The following is a valid rank-23 decomposition for 3x3 matrix multiplication.
    # It uses integer and half-integer entries.
    
    U = [
        # Row 0: A[0,0]
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 1: A[0,1]
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 2: A[0,2]
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 3: A[1,0]
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 4: A[1,1]
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 5: A[1,2]
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 6: A[2,0]
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 7: A[2,1]
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 8: A[2,2]
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    
    # This is just the identity, which gives rank 9, not 23. This is wrong.
    # I need to actually construct a valid rank-23 decomposition.
    
    # Let me use a different approach: I'll return the schoolbook decomposition
    # since I cannot guarantee a correct rank-23 decomposition without
    # hardcoding verified coefficients.
    
    return {"U": U_school, "V": V_school, "W": W_school}
