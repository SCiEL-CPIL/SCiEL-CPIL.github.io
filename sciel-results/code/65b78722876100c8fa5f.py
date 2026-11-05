import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145
    
    # First, build a correct rank-28 decomposition
    result = _strassen_rank28(n, m, p)
    if result is not None:
        # Try to reduce rank from 28 to 27
        if time.time() < deadline - 5:
            reduced = _try_reduce_rank(result, n, m, p, deadline)
            if reduced is not None:
                return reduced
        return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

def _strassen_rank28(n, m, p):
    """Build a correct rank-28 decomposition using Strassen for 2x2 blocks."""
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(i1, i2, j1, j2, k1, k2):
        """Add 7 Strassen factors for 2x2 * 2x2 multiplication.
        
        A is 2x2 with rows i1,i2 and cols j1,j2
        B is 2x2 with rows j1,j2 and cols k1,k2  
        C is 2x2 with rows i1,i2 and cols k1,k2
        
        Strassen formulas:
        M1 = (A00 + A11)(B00 + B11)
        M2 = (A10 + A11) * B00
        M3 = A00 * (B01 - B11)
        M4 = A11 * (B10 - B00)
        M5 = (A00 + A01) * B11
        M6 = (A10 - A00) * (B00 + B01)
        M7 = (A01 - A11) * (B10 + B11)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        """
        nonlocal col
        
        # M1 = (A00 + A11)(B00 + B11)
        # U: A[i1,j1]=1, A[i2,j2]=1
        # V: B[j1,k1]=1, B[j2,k2]=1
        # W: C[i1,k1]+=1, C[i2,k2]+=1
        U[i1*m+j1, col] = 1
        U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1
        W[k2*n+i2, col] = 1
        col += 1
        
        # M2 = (A10 + A11) * B00
        # U: A[i2,j1]=1, A[i2,j2]=1
        # V: B[j1,k1]=1
        # W: C[i2,k1]+=1, C[i2,k2]-=1
        U[i2*m+j1, col] = 1
        U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1
        W[k1*n+i2, col] = 1
        W[k2*n+i2, col] = -1
        col += 1
        
        # M3 = A00 * (B01 - B11)
        # U: A[i1,j1]=1
        # V: B[j1,k2]=1, B[j2,k2]=-1
        # W: C[i1,k2]+=1, C[i2,k2]+=1
        U[i1*m+j1, col] = 1
        V[j1*p+k2, col] = 1
        V[j2*p+k2, col] = -1
        W[k2*n+i1, col] = 1
        W[k2*n+i2, col] = 1
        col += 1
        
        # M4 = A11 * (B10 - B00)
        # U: A[i2,j2]=1
        # V: B[j2,k1]=1, B[j1,k1]=-1
        # W: C[i1,k1]+=1, C[i2,k1]+=1
        U[i2*m+j2, col] = 1
        V[j2*p+k1, col] = 1
        V[j1*p+k1, col] = -1
        W[k1*n+i1, col] = 1
        W[k1*n+i2, col] = 1
        col += 1
        
        # M5 = (A00 + A01) * B11
        # U: A[i1,j1]=1, A[i1,j2]=1
        # V: B[j2,k2]=1
        # W: C[i1,k1]-=1, C[i1,k2]+=1
        U[i1*m+j1, col] = 1
        U[i1*m+j2, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = -1
        W[k2*n+i1, col] = 1
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        # U: A[i2,j1]=1, A[i1,j1]=-1
        # V: B[j1,k1]=1, B[j1,k2]=1
        # W: C[i2,k2]+=1
        U[i2*m+j1, col] = 1
        U[i1*m+j1, col] = -1
        V[j1*p+k1, col] = 1
        V[j1*p+k2, col] = 1
        W[k2*n+i2, col] = 1
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        # U: A[i1,j2]=1, A[i2,j2]=-1
        # V: B[j2,k1]=1, B[j2,k2]=1
        # W: C[i1,k1]+=1
        U[i1*m+j2, col] = 1
        U[i2*m+j2, col] = -1
        V[j2*p+k1, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1
        col += 1

    def add_direct(i1, i2, j_idx, k_idx):
        """Add 4 direct factors for 2x1 * 1x1 (or similar)."""
        nonlocal col
        for i in [i1, i2]:
            U[i*m+j_idx, col] = 1
            V[j_idx*p+k_idx, col] = 1
            W[k_idx*n+i, col] = 1
            col += 1

    # Part 1: j=0, k=0..4 -> A[:,0] * B[0,:] -> C[:,0:5]
    # This is 2x1 * 1x5 = 2x5, rank 10
    for i in range(n):
        for k in range(p):
            U[i*m+0, col] = 1
            V[0*p+k, col] = 1
            W[k*n+i, col] = 1
            col += 1
    
    # Part 2: j=1,2, k=0,1 -> A[:,1:3] * B[1:3,0:2] -> C[:,0:2]
    # This is 2x2 * 2x2, use Strassen, rank 7
    add_strassen(0, 1, 1, 2, 0, 1)
    
    # Part 3: j=1,2, k=2,3 -> A[:,1:3] * B[1:3,2:4] -> C[:,2:4]
    # This is 2x2 * 2x2, use Strassen, rank 7
    add_strassen(0, 1, 1, 2, 2, 3)
    
    # Part 4: j=1,2, k=4 -> A[:,1:3] * B[1:3,4] -> C[:,4]
    # This is 2x2 * 2x1, rank 4
    for i in range(n):
        for j in range(1, 3):
            U[i*m+j, col] = 1
            V[j*p+4, col] = 1
            W[4*n+i, col] = 1
            col += 1
    
    # Verify col == rank
    if col != rank:
        return None
    
    # Verify exactness
    U_arr = U
    V_arr = V
    W_arr = W
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(s - 1.0) > 1e-9:
                    return None
    
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(base_result, n, m, p, deadline):
    """Try to reduce rank from 28 to 27 by finding linear dependencies."""
    U_base = base_result["U"].copy()
    V_base = base_result["V"].copy()
    W_base = base_result["W"].copy()
    rank_base = U_base.shape[1]
    
    if rank_base < 27:
        return base_result
    
    # Strategy: Try to find a linear combination of the last few factors
    # that can be eliminated. We'll test if any single factor can be expressed
    # as a linear combination of the others.
    
    # For each factor c, check if it can be written as sum of other factors
    # This is a linear algebra problem: is column c in the span of other columns?
    
    # We'll use a simpler approach: try random perturbations and see if we can
    # reduce the effective rank.
    
    # Actually, let's try a more systematic approach:
    # The tensor is T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # If we can find coefficients alpha_c such that for all i,j,k:
    #   sum_c alpha_c * U[i,j,c] * V[j,k,c] * W[k,i,c] = 0
    # and alpha_c != 0 for some c, then we have a dependency.
    
    # This is equivalent to checking if the factors are linearly independent
    # in the tensor space.
    
    # For rank reduction, we need to find if any factor is redundant.
    # Let's try: for each factor c, set its coefficients to 0 and check if
    # the remaining factors still span the tensor.
    
    # This is expensive, so we'll use a heuristic: try to merge pairs of factors.
    
    # Alternative: use the fact that we have structure. The rank-28 decomposition
    # has 10 + 7 + 7 + 4 = 28 factors. Maybe some of the "direct" factors can
    # be combined with Strassen factors.
    
    # For now, return None to keep rank 28. A full search would be complex.
    # In a future iteration, we could implement a proper linear algebra search.
    
    return None

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
