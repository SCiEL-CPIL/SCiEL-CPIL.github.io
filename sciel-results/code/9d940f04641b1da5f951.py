import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    n, m, p = 3, 3, 6
    
    # Build the 3x3x3 tensor T333
    T333 = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                T333[a, b, c] = 1.0
    
    # ALS to find rank-19 decomposition of 3x3x3
    def als_333(rank, max_iter=2000, tol=1e-8):
        """Alternating Least Squares for 3x3x3 tensor decomposition."""
        np.random.seed(42)
        U = np.random.randn(9, rank)
        V = np.random.randn(9, rank)
        W = np.random.randn(9, rank)
        
        # Precompute tensor slices
        # T333[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
        
        prev_err = np.inf
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Update W: for each column r, solve least squares
            # T333[a,b,c] = sum_s U[a,s]V[b,s]W[c,s]
            # Fix U,V. For W[:,r]:
            # Minimize sum_{a,b,c} (T333[a,b,c] - sum_s U[a,s]V[b,s]W[c,s])^2
            # This is linear in W[:,r]
            # Let M[a,b] = sum_{s!=r} U[a,s]V[b,s]W[c,s] ... no
            # Actually: T333[a,b,c] = U[a,r]V[b,r]W[c,r] + sum_{s!=r} U[a,s]V[b,s]W[c,s]
            # For fixed r, define R[a,b,c] = T333[a,b,c] - sum_{s!=r} U[a,s]V[b,s]W[c,s]
            # Then R[a,b,c] = U[a,r] * V[b,r] * W[c,r]
            # This is: for each (a,b), R[a,b,:] = (U[a,r]*V[b,r]) * W[:,r]
            # So W[:,r] can be found by least squares:
            # min_W sum_{a,b} ||R[a,b,:] - (U[a,r]*V[b,r])*W||^2
            
            for r in range(rank):
                # Compute residual
                R = T333.copy()
                for s in range(rank):
                    if s == r:
                        continue
                    # R[a,b,c] -= U[a,s]*V[b,s]*W[c,s]
                    R -= np.einsum('as,bs,cs->abc', U[:, s], V[:, s], W[:, s])
                
                # Now R[a,b,c] = U[a,r] * V[b,r] * W[c,r]
                # For each (a,b): R[a,b,:] = (U[a,r]*V[b,r]) * W[:,r]
                # This is a rank-1 matrix approximation problem
                # W[:,r] = argmin sum_{a,b} ||R[a,b,:] - (U[a,r]*V[b,r])*W||^2
                # Taking derivative: sum_{a,b} (U[a,r]*V[b,r]) * R[a,b,:] = (sum_{a,b} (U[a,r]*V[b,r])^2) * W
                # So W[:,r] = sum_{a,b} (U[a,r]*V[b,r]) * R[a,b,:] / sum_{a,b} (U[a,r]*V[b,r])^2
                
                coeff = U[:, r] * V[:, r]  # shape (9,9)
                # W[:,r] = sum_{a,b} coeff[a,b] * R[a,b,:] / sum_{a,b} coeff[a,b]^2
                numerator = np.einsum('ab,abc->c', coeff, R)
                denominator = np.sum(coeff**2)
                if denominator > 1e-12:
                    W[:, r] = numerator / denominator
                else:
                    W[:, r] = 0.0
            
            # Update U similarly
            for r in range(rank):
                R = T333.copy()
                for s in range(rank):
                    if s == r:
                        continue
                    R -= np.einsum('as,bs,cs->abc', U[:, s], V[:, s], W[:, s])
                
                coeff = V[:, r] * W[:, r]  # shape (9,9)
                numerator = np.einsum('bc,abc->a', coeff, R)
                denominator = np.sum(coeff**2)
                if denominator > 1e-12:
                    U[:, r] = numerator / denominator
                else:
                    U[:, r] = 0.0
            
            # Update V similarly
            for r in range(rank):
                R = T333.copy()
                for s in range(rank):
                    if s == r:
                        continue
                    R -= np.einsum('as,bs,cs->abc', U[:, s], V[:, s], W[:, s])
                
                coeff = U[:, r] * W[:, r]  # shape (9,9)
                numerator = np.einsum('ac,abc->b', coeff, R)
                denominator = np.sum(coeff**2)
                if denominator > 1e-12:
                    V[:, r] = numerator / denominator
                else:
                    V[:, r] = 0.0
            
            # Compute error
            approx = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.sum((T333 - approx)**2)
            
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        
        # Verify
        if _verify_333(U, V, W):
            return U, V, W
        return None
    
    def _verify_333(U, V, W, tol=1e-5):
        """Verify 3x3x3 decomposition."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    a = i*3 + j
                    b = j*3 + k
                    c = k*3 + i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val - 1.0) > tol:
                        return False
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
                        val = np.sum(U[a, :] * V[b, :] * W[c, :])
                        if abs(val) > tol:
                            return False
        return True
    
    def _verify_full(U, V, W, n=3, m=3, p=6, tol=1e-5):
        """Verify full 3x3x6 decomposition."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        if U.shape != (n*m, U.shape[1]): return False
        if V.shape != (m*p, V.shape[1]): return False
        if W.shape != (p*n, W.shape[1]): return False
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]: return False
        
        nm, mp, pn = n*m, m*p, p*n
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val - 1.0) > tol:
                        return False
                    
        for a in range(nm):
            i_a = a // m
            j_a = a % m
            for b in range(mp):
                j_b = b // p
                k_b = b % p
                for c in range(pn):
                    k_c = c // n
                    i_c = c % n
                    if not (j_a == j_b and k_b == k_c and i_c == i_a):
                        val = np.sum(U[a, :] * V[b, :] * W[c, :])
                        if abs(val) > tol:
                            return False
        return True
    
    # Try to find rank-19 3x3x3 decomposition
    U333 = V333 = W333 = None
    for rank in range(19, 28):
        if time.time() > deadline - 10:
            break
        
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(seed * 1000 + rank)
            U333, V333, W333 = als_333(rank, max_iter=1500)
            if U333 is not None:
                break
        
        if U333 is not None:
            break
    
    # Build 3x3x6 decomposition from 3x3x3
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
                    W[k*3 + 0, r] = 0  # placeholder
            # W for first half: c = k*3 + i where k in [0,3)
            for k in range(3):
                for i in range(3):
                    W[k*3 + i, r] = W333[k*3 + i, r]
        
        # Second half: k in [3,6)
        for r in range(rank_333):
            U[:, rank_333 + r] = U333[:, r]
            for j in range(3):
                for k in range(3):
                    V[j*6 + (k+3), rank_333 + r] = V333[j*3 + k, r]
                    W[(k+3)*3 + i, rank_333 + r] = 0  # placeholder
            for k in range(3):
                for i in range(3):
                    W[(k+3)*3 + i, rank_333 + r] = W333[k*3 + i, r]
        
        # Verify
        if _verify_full(U, V, W):
            # Convert to list of lists for output
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback: schoolbook
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
