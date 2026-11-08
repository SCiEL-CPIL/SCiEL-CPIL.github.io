import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 15
    pn = p * n  # 10

    # Known rank-30 decomposition: C[i,k] = sum_j A[i,j]*B[j,k]
    # We try to find a lower-rank decomposition using structured search.
    
    # Start with the naive rank-30
    rank = 30
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, r] = 1
                V[j*p + k, r] = 1
                W[k*n + i, r] = 1
                r += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try to find lower rank decompositions
    # Using a structured approach: try rank 27, 28, 29
    # with random initialization and local search
    
    deadline = time.time() + 100  # 100 second deadline
    
    # Values to try: integers and half-integers
    vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    # More focused set for the search
    focused_vals = [-1, -0.5, 0, 0.5, 1]
    
    def check_decomposition(U, V, W, rank):
        """Check if the decomposition exactly reproduces matrix multiplication."""
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    expected = 1.0 if True else 0.0
                    actual = 0.0
                    for rr in range(rank):
                        actual += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                    if abs(actual - expected) > 1e-9:
                        return False
                # For C[i,k] = sum_j A[i,j] B[j,k], we need:
                # sum_rr U[i*m+j,rr] * V[j*p+k,rr] * W[k*n+i,rr] = delta_{j} for each j? 
                # No, we need sum_j (U[i*m+j] * V[j*p+k]) * W[k*n+i] = 1
                # Wait, the tensor is T[i,j,k] = 1 if... no.
                # The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k? No.
                # C[i,k] = sum_j A[i,j] B[j,k]
                # So the tensor is sum_{i,j,k} A[i,j] B[j,k] C[i,k]
                # The structure tensor has entries T[i,j,k] = 1 for all i,j,k? 
                # No, it's that we need: for each (i,k), sum_j U[i*m+j,rr] V[j*p+k,rr] W[k*n+i,rr] = 1
                # Wait, that's not right either. Let me re-read the problem.
                
                # "exactly decompose the matrix multiplication tensor"
                # The matrix multiplication tensor for C=AB is:
                # T = sum_{i,j,k} A[i,j] * B[j,k] * C[i,k]
                # A rank-1 decomposition is A[i,j] * B[j,k] * C[i,k]
                # So we need: T[i,j,k] = sum_rr U[i*m+j,rr] * V[j*p+k,rr] * W[k*n+i,rr]
                # And T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]B[j,k])
                # Wait no. The tensor is the multilinear form.
                # Actually, the matrix multiplication tensor is defined such that:
                # For the decomposition to be exact, we need:
                # For each (i,j,k): sum_rr U[i*m+j,rr] * V[j*p+k,rr] * W[k*n+i,rr] = 1
                # This is because the tensor T[i,j,k] = 1 for all i,j,k in the 
                # standard matrix multiplication tensor (the "generic" one).
                
        return True
    
    # Correct validation:
    def validate(U, V, W, rank):
        """Check if the decomposition exactly reproduces the matrix multiplication tensor.
        T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        """
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for rr in range(rank):
                        s += U[i*m + j, rr] * V[j*p + k, rr] * W[k*n + i, rr]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Verify our rank-30 works
    if not validate(best_U, best_V, best_W, 30):
        # Fallback: should not happen
        pass
    
    # Try rank 28 using the known decomposition from the inspiration
    # Build the rank-28 decomposition
    U28 = np.zeros((nm, 28))
    V28 = np.zeros((mp, 28))
    W28 = np.zeros((pn, 28))
    
    r = 0
    # Part 1: A2 B2 (j=2, k=0..4)
    for i in range(n):
        for k in range(p):
            U28[i*m + 2, r] = 1
            V28[2*p + k, r] = 1
            W28[k*n + i, r] = 1
            r += 1
            
    # Part 2: A1 B13 (j=0,1, k=4)
    for i in range(n):
        for j in range(2):
            U28[i*m + j, r] = 1
            V28[j*p + 4, r] = 1
            W28[4*n + i, r] = 1
            r += 1
            
    # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen
    # M1
    U28[0*m + 0, r] = 1
    U28[1*m + 1, r] = 1
    V28[0*p + 0, r] = 1
    V28[1*p + 1, r] = 1
    W28[0*n + 0, r] = 1
    W28[1*n + 1, r] = 1
    r += 1
    # M2
    U28[1*m + 0, r] = 1
    U28[1*m + 1, r] = 1
    V28[0*p + 0, r] = 1
    W28[0*n + 1, r] = 1
    W28[1*n + 1, r] = -1
    r += 1
    # M3
    U28[0*m + 0, r] = 1
    V28[0*p + 1, r] = 1
    V28[1*p + 1, r] = -1
    W28[1*n + 0, r] = 1
    W28[1*n + 1, r] = 1
    r += 1
    # M4
    U28[1*m + 1, r] = 1
    V28[1*p + 0, r] = 1
    V28[0*p + 0, r] = -1
    W28[0*n + 0, r] = 1
    W28[0*n + 1, r] = 1
    r += 1
    # M5
    U28[0*m + 0, r] = 1
    U28[0*m + 1, r] = 1
    V28[1*p + 1, r] = 1
    W28[0*n + 0, r] = -1
    W28[1*n + 0, r] = 1
    r += 1
    # M6
    U28[1*m + 0, r] = 1
    U28[0*m + 0, r] = -1
    V28[0*p + 0, r] = 1
    V28[0*p + 1, r] = 1
    W28[1*n + 1, r] = 1
    r += 1
    # M7
    U28[0*m + 1, r] = 1
    U28[1*m + 1, r] = -1
    V28[1*p + 0, r] = 1
    V28[1*p + 1, r] = 1
    W28[0*n + 0, r] = 1
    r += 1
    
    # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen
    # M1
    U28[0*m + 0, r] = 1
    U28[1*m + 1, r] = 1
    V28[0*p + 2, r] = 1
    V28[1*p + 3, r] = 1
    W28[2*n + 0, r] = 1
    W28[3*n + 1, r] = 1
    r += 1
    # M2
    U28[1*m + 0, r] = 1
    U28[1*m + 1, r] = 1
    V28[0*p + 2, r] = 1
    W28[2*n + 1, r] = 1
    W28[3*n + 1, r] = -1
    r += 1
    # M3
    U28[0*m + 0, r] = 1
    V28[0*p + 3, r] = 1
    V28[1*p + 3, r] = -1
    W28[3*n + 0, r] = 1
    W28[3*n + 1, r] = 1
    r += 1
    # M4
    U28[1*m + 1, r] = 1
    V28[1*p + 2, r] = 1
    V28[0*p + 2, r] = -1
    W28[2*n + 0, r] = 1
    W28[2*n + 1, r] = 1
    r += 1
    # M5
    U28[0*m + 0, r] = 1
    U28[0*m + 1, r] = 1
    V28[1*p + 3, r] = 1
    W28[2*n + 0, r] = -1
    W28[3*n + 0, r] = 1
    r += 1
    # M6
    U28[1*m + 0, r] = 1
    U28[0*m + 0, r] = -1
    V28[0*p + 2, r] = 1
    V28[0*p + 3, r] = 1
    W28[3*n + 1, r] = 1
    r += 1
    # M7
    U28[0*m + 1, r] = 1
    U28[1*m + 1, r] = -1
    V28[1*p + 2, r] = 1
    V28[1*p + 3, r] = 1
    W28[2*n + 0, r] = 1
    r += 1
    
    if validate(U28, V28, W28, 28):
        best_U, best_V, best_W = U28, V28, W28
        best_rank = 28
    
    # Now try to find rank 27 or lower
    # Use random search with the focused value set
    for target_rank in range(27, best_rank):
        if time.time() > deadline:
            break
        found = False
        for attempt in range(500):
            if time.time() > deadline:
                break
            U_t = np.random.choice(focused_vals, size=(nm, target_rank))
            V_t = np.random.choice(focused_vals, size=(mp, target_rank))
            W_t = np.random.choice(focused_vals, size=(pn, target_rank))
            if validate(U_t, V_t, W_t, target_rank):
                best_U, best_V, best_W = U_t, V_t, W_t
                best_rank = target_rank
                found = True
                break
        if found:
            # Try even lower
            pass
    
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
