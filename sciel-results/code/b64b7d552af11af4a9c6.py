import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    n, m, p = 3, 3, 6
    
    # Build the 3x3x6 tensor T
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def verify(U, V, W, tol=1e-6):
        """Verify decomposition of 3x3x6 tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        if U.shape[0] != n*m: return False
        if V.shape[0] != m*p: return False
        if W.shape[0] != p*n: return False
        rank = U.shape[1]
        if rank != V.shape[1] or rank != W.shape[1]: return False
        
        approx = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Check required entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    if abs(approx[a, b, c] - 1.0) > tol:
                        return False
        
        # Check that non-required entries are zero
        for a in range(n*m):
            i_a = a // m
            j_a = a % m
            for b in range(m*p):
                j_b = b // p
                k_b = b % p
                for c in range(p*n):
                    k_c = c // n
                    i_c = c % n
                    if not (j_a == j_b and k_b == k_c and i_c == i_a):
                        if abs(approx[a, b, c]) > tol:
                            return False
        return True
    
    def als_decompose(rank, max_iter=2000, tol=1e-10, seed=None):
        """Alternating Least Squares for tensor decomposition."""
        if seed is not None:
            np.random.seed(seed)
        
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        prev_err = np.inf
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Update W
            for r in range(rank):
                R = T - np.einsum('ar,br,cr->abc', U, V, W) + np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                coeff = U[:, r] * V[:, r]
                num = np.einsum('ab,abc->c', coeff, R)
                den = np.sum(coeff**2)
                if den > 1e-14:
                    W[:, r] = num / den
                else:
                    W[:, r] = 0.0
            
            # Update U
            for r in range(rank):
                R = T - np.einsum('ar,br,cr->abc', U, V, W) + np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                coeff = V[:, r] * W[:, r]
                num = np.einsum('bc,abc->a', coeff, R)
                den = np.sum(coeff**2)
                if den > 1e-14:
                    U[:, r] = num / den
                else:
                    U[:, r] = 0.0
            
            # Update V
            for r in range(rank):
                R = T - np.einsum('ar,br,cr->abc', U, V, W) + np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                coeff = U[:, r] * W[:, r]
                num = np.einsum('ac,abc->b', coeff, R)
                den = np.sum(coeff**2)
                if den > 1e-14:
                    V[:, r] = num / den
                else:
                    V[:, r] = 0.0
            
            # Compute error
            approx = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.sum((T - approx)**2)
            
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        
        if verify(U, V, W, tol=1e-4):
            return U, V, W
        return None
    
    # Strategy 1: Try direct ALS on 3x3x6 with various ranks
    # Start with rank 30 and go up
    for rank in range(30, 40):
        if time.time() > deadline - 10:
            break
        
        for seed in range(15):
            if time.time() > deadline - 5:
                break
            
            result = als_decompose(rank, max_iter=500, seed=seed * 1000 + rank)
            if result is not None:
                U, V, W = result
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # Strategy 2: Build from 3x3x3 decomposition
    # Find rank-19 decomposition of 3x3x3
    T333 = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                T333[a, b, c] = 1.0
    
    def als_333(rank, max_iter=2000, tol=1e-10, seed=None):
        """ALS for 3x3x3 tensor."""
        if seed is not None:
            np.random.seed(seed)
        
        U = np.random.randn(9, rank) * 0.1
        V = np.random.randn(9, rank) * 0.1
        W = np.random.randn(9, rank) * 0.1
        
        prev_err = np.inf
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Update W
            for r in range(rank):
                R = T333 - np.einsum('ar,br,cr->abc', U, V, W) + np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                coeff = U[:, r] * V[:, r]
                num = np.einsum('ab,abc->c', coeff, R)
                den = np.sum(coeff**2)
                if den > 1e-14:
                    W[:, r] = num / den
                else:
                    W[:, r] = 0.0
            
            # Update U
            for r in range(rank):
                R = T333 - np.einsum('ar,br,cr->abc', U, V, W) + np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                coeff = V[:, r] * W[:, r]
                num = np.einsum('bc,abc->a', coeff, R)
                den = np.sum(coeff**2)
                if den > 1e-14:
                    U[:, r] = num / den
                else:
                    U[:, r] = 0.0
            
            # Update V
            for r in range(rank):
                R = T333 - np.einsum('ar,br,cr->abc', U, V, W) + np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                coeff = U[:, r] * W[:, r]
                num = np.einsum('ac,abc->b', coeff, R)
                den = np.sum(coeff**2)
                if den > 1e-14:
                    V[:, r] = num / den
                else:
                    V[:, r] = 0.0
            
            approx = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.sum((T333 - approx)**2)
            
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        
        # Verify 3x3x3
        approx = np.einsum('ar,br,cr->abc', U, V, W)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    a = i*3 + j
                    b = j*3 + k
                    c = k*3 + i
                    if abs(approx[a, b, c] - 1.0) > 1e-4:
                        return None
        
        for a in range(9):
            i_a = a // 3
            j_a = a % 3
            for b in range(9):
                j_b = b // 3
                k_b = b % 3
                for c in range(9):
                    k_c = c // 3
                    i_c = c % 3
                    if not (j_a == j_b and k_b == k_c and i_c == i_a):
                        if abs(approx[a, b, c]) > 1e-4:
                            return None
        return U, V, W
    
    U333 = V333 = W333 = None
    for rank in range(19, 23):
        if time.time() > deadline - 15:
            break
        
        for seed in range(20):
            if time.time() > deadline - 10:
                break
            
            U333, V333, W333 = als_333(rank, max_iter=800, seed=seed * 1000 + rank)
            if U333 is not None:
                break
        
        if U333 is not None:
            break
    
    # Build 3x3x6 from 3x3x3
    if U333 is not None:
        rank_333 = U333.shape[1]
        rank_total = 2 * rank_333
        
        U = np.zeros((9, rank_total))
        V = np.zeros((18, rank_total))
        W = np.zeros((18, rank_total))
        
        # First half: k in [0,3)
        for r in range(rank_333):
            U[:, r] = U333[:, r]
            for j in range(3):
                for k in range(3):
                    V[j*6 + k, r] = V333[j*3 + k, r]
            for k in range(3):
                for i in range(3):
                    W[k*3 + i, r] = W333[k*3 + i, r]
        
        # Second half: k in [3,6)
        for r in range(rank_333):
            U[:, rank_333 + r] = U333[:, r]
            for j in range(3):
                for k in range(3):
                    V[j*6 + (k+3), rank_333 + r] = V333[j*3 + k, r]
            for k in range(3):
                for i in range(3):
                    W[(k+3)*3 + i, rank_333 + r] = W333[k*3 + i, r]
        
        if verify(U, V, W, tol=1e-4):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback: schoolbook rank 54
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
