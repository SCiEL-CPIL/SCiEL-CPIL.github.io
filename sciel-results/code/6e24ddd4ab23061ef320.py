"""Matrix multiplication tensor decomposition for 4x5 by 5x6."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130.0  # Safety margin before 150s kill
    
    nm = n * m  # 20
    mp = m * p  # 30
    pn = p * n  # 24
    
    # Schoolbook solution (rank 120)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, r_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, r_sb
    
    def verify_full(U, V, W, rank, tol=1e-8):
        """Full verification: check all (a,b,c) entries of the tensor."""
        # Tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Should be 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some (i,j,k)
        # And 0 otherwise
        
        # Check the "1" entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(s - 1.0) > tol:
                        return False
        
        # Check a sample of "0" entries (all combinations that don't match)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    for j2 in range(m):
                        for k2 in range(p):
                            for i2 in range(n):
                                if j != j2 or i != i2 or k != k2:
                                    b = j2 * p + k
                                    c = k2 * n + i
                                    s = np.dot(U[a, :], V[b, :] * W[c, :])
                                    if abs(s) > tol:
                                        return False
        return True
    
    def verify_quick(U, V, W, rank, tol=1e-4):
        """Quick check: only verify the '1' entries."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def polish(U, V, W, rank, max_iters=50, tol=1e-10):
        """Iteratively polish the solution by alternating least-squares solves."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U: for each (i,j), solve for U[i*m+j,:]
            # Given V and W, for each (i,j,k): sum_r U[a,r]*V[b,r]*W[c,r] = 1
            # This is a linear system in U[a,:] for each (i,j)
            U_new = U.copy()
            for i in range(n):
                for j in range(m):
                    a = i * m + j
                    # Build system: for each k, sum_r U[a,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1
                    M = np.zeros((p, rank))
                    for k in range(p):
                        b = j * p + k
                        c = k * n + i
                        M[k, :] = V[b, :] * W[c, :]
                    rhs = np.ones(p)
                    try:
                        U_new[a, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                    except:
                        pass
            
            # Update V: for each (j,k), solve for V[j*p+k,:]
            V_new = V.copy()
            for j in range(m):
                for k in range(p):
                    b = j * p + k
                    M = np.zeros((n, rank))
                    for i in range(n):
                        a = i * m + j
                        c = k * n + i
                        M[i, :] = U[a, :] * W[c, :]
                    rhs = np.ones(n)
                    try:
                        V_new[b, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                    except:
                        pass
            
            # Update W: for each (k,i), solve for W[k*n+i,:]
            W_new = W.copy()
            for k in range(p):
                for i in range(n):
                    c = k * n + i
                    M = np.zeros((m, rank))
                    for j in range(m):
                        a = i * m + j
                        b = j * p + k
                        M[j, :] = U[a, :] * V[b, :]
                    rhs = np.ones(m)
                    try:
                        W_new[c, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                    except:
                        pass
            
            U, V, W = U_new, V_new, W_new
            
            # Check convergence
            if verify_quick(U, V, W, rank, tol=1e-6):
                # Check if full verification passes
                if verify_full(U, V, W, rank, tol=1e-8):
                    return U, V, W, True
        
        return U, V, W, False
    
    def try_random_rank(target_rank, max_attempts=30):
        """Try to find a valid decomposition of the given rank using random initialization."""
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
            
            # Random initialization with different strategies
            if attempt % 3 == 0:
                # Standard random
                V_try = np.random.randn(mp, target_rank)
                W_try = np.random.randn(pn, target_rank)
            elif attempt % 3 == 1:
                # Sparse-ish initialization
                V_try = np.random.randn(mp, target_rank)
                V_try[np.random.random((mp, target_rank)) < 0.5] = 0
                W_try = np.random.randn(pn, target_rank)
                W_try[np.random.random((pn, target_rank)) < 0.5] = 0
            else:
                # Block-structured: group by j for V, by (k,i) for W
                V_try = np.zeros((mp, target_rank))
                W_try = np.zeros((pn, target_rank))
                for j in range(m):
                    V_try[j*p:(j+1)*p, :] = np.random.randn(p, target_rank)
                for k in range(p):
                    for i in range(n):
                        W_try[k*n+i, :] = np.random.randn(1, target_rank)[0]
            
            # Scale to reasonable magnitude
            V_try /= np.sqrt(target_rank)
            W_try /= np.sqrt(target_rank)
            
            # Solve for U
            U_try = np.zeros((nm, target_rank))
            valid = True
            for i in range(n):
                for j in range(m):
                    M_ij = np.zeros((p, target_rank))
                    for k in range(p):
                        M_ij[k, :] = V_try[j*p+k, :] * W_try[k*n+i, :]
                    rhs = np.ones(p)
                    try:
                        U_try[i*m+j, :], _, _, _ = np.linalg.lstsq(M_ij, rhs, rcond=None)
                    except:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Quick check
            if not verify_quick(U_try, V_try, W_try, target_rank, tol=1e-3):
                continue
            
            # Polish
            U_pol, V_pol, W_pol, success = polish(U_try, V_try, W_try, target_rank)
            
            if success:
                return U_pol, V_pol, W_pol
            
            # Even if polish didn't fully converge, check if it's close enough
            if verify_full(U_pol, V_pol, W_pol, target_rank, tol=1e-6):
                return U_pol, V_pol, W_pol
        
        return None
    
    def try_structured_rank(target_rank):
        """Try structured decompositions that exploit the problem's symmetry."""
        
        # Strategy 1: Use the fact that for fixed j, we need to multiply (n x 1) by (1 x p)
        # The tensor has a block structure where each j-block is independent in a sense
        
        # Strategy 2: Try low-rank approximations of the identity-like structure
        
        # For now, try a few different structured initializations
        for variant in range(5):
            if time.time() > deadline - 5:
                return None
            
            if variant == 0:
                # Diagonal-like structure
                V_try = np.zeros((mp, target_rank))
                W_try = np.zeros((pn, target_rank))
                for j in range(m):
                    for k in range(p):
                        idx = j * p + k
                        # Assign a random direction in the rank space
                        if idx < target_rank:
                            V_try[idx, idx] = 1.0
                        else:
                            V_try[idx, :] = np.random.randn(target_rank) / np.sqrt(target_rank)
                for k in range(p):
                    for i in range(n):
                        idx = k * n + i
                        if idx < target_rank:
                            W_try[idx, idx] = 1.0
                        else:
                            W_try[idx, :] = np.random.randn(target_rank) / np.sqrt(target_rank)
                
            elif variant == 1:
                # Fourier-like structure
                V_try = np.zeros((mp, target_rank))
                W_try = np.zeros((pn, target_rank))
                for j in range(m):
                    for k in range(p):
                        idx = j * p + k
                        for r in range(target_rank):
                            V_try[idx, r] = np.cos(2 * np.pi * idx * r / target_rank)
                for k in range(p):
                    for i in range(n):
                        idx = k * n + i
                        for r in range(target_rank):
                            W_try[idx, r] = np.sin(2 * np.pi * idx * r / target_rank)
                
            elif variant == 2:
                # Kronecker product structure
                V_try = np.zeros((mp, target_rank))
                W_try = np.zeros((pn, target_rank))
                # Try to use tensor product of smaller decompositions
                for j in range(m):
                    for k in range(p):
                        idx = j * p + k
                        V_try[idx, :] = np.random.randn(target_rank)
                for k in range(p):
                    for i in range(n):
                        idx = k * n + i
                        W_try[idx, :] = np.random.randn(target_rank)
                        
            else:
                # Random with normalization
                V_try = np.random.randn(mp, target_rank)
                W_try = np.random.randn(pn, target_rank)
                # Normalize rows
                for row in range(mp):
                    norm = np.linalg.norm(V_try[row, :])
                    if norm > 0:
                        V_try[row, :] /= norm
                for row in range(pn):
                    norm = np.linalg.norm(W_try[row, :])
                    if norm > 0:
                        W_try[row, :] /= norm
            
            # Solve for U
            U_try = np.zeros((nm, target_rank))
            valid = True
            for i in range(n):
                for j in range(m):
                    M_ij = np.zeros((p, target_rank))
                    for k in range(p):
                        M_ij[k, :] = V_try[j*p+k, :] * W_try[k*n+i, :]
                    rhs = np.ones(p)
                    try:
                        U_try[i*m+j, :], _, _, _ = np.linalg.lstsq(M_ij, rhs, rcond=None)
                    except:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            if not verify_quick(U_try, V_try, W_try, target_rank, tol=1e-3):
                continue
            
            U_pol, V_pol, W_pol, success = polish(U_try, V_try, W_try, target_rank)
            
            if success:
                return U_pol, V_pol, W_pol
            
            if verify_full(U_pol, V_pol, W_pol, target_rank, tol=1e-6):
                return U_pol, V_pol, W_pol
        
        return None
    
    # Try to find a better rank than schoolbook
    # Start from rank 100 and go down, trying both random and structured approaches
    for target_rank in range(110, 85, -1):
        if time.time() > deadline - 10:
            break
        
        # Try structured first (fewer attempts but more targeted)
        result = try_structured_rank(target_rank)
        if result is not None:
            best_U, best_V, best_W, best_rank = result
            break
        
        # Try random
        result = try_random_rank(target_rank, max_attempts=15)
        if result is not None:
            best_U, best_V, best_W, best_rank = result
            break
    
    # Final verification of the best result
    if not verify_full(best_U, best_V, best_W, best_rank, tol=1e-6):
        # Fallback to schoolbook
        best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, r_sb
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
