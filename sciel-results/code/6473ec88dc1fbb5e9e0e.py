import time
import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    nmp = n * m * p
    
    start_time = time.time()
    deadline = start_time + 130  # Safety margin for 150s limit
    
    def schoolbook():
        rank = nmp
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = schoolbook()
    best_rank = nmp
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def round_to_half_int(X):
        """Round matrix to nearest half-integer."""
        return np.round(X * 2) / 2.0
    
    def try_rank(rank, max_iterations=200, max_restarts=3):
        """Try to find a decomposition of given rank using ALS."""
        for restart in range(max_restarts):
            if time.time() > deadline - 2:
                return None
            
            seed = rank * 1000 + restart * 7919
            np.random.seed(seed)
            
            # Initialize with small random values
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            # Scale to make initial products closer to 1
            U /= np.sqrt(np.sum(U**2, axis=1, keepdims=True) + 1e-10)
            V /= np.sqrt(np.sum(V**2, axis=1, keepdims=True) + 1e-10)
            W /= np.sqrt(np.sum(W**2, axis=1, keepdims=True) + 1e-10)
            
            for it in range(max_iterations):
                if time.time() > deadline - 1:
                    return None
                
                # Update U: for each (i,j), solve for U[i*m+j]
                for i in range(n):
                    for j in range(m):
                        # Build system: sum_k V[j*p+k] * W[k*n+i] * U[i*m+j] = 1
                        # This is: U[i*m+j] . (V[j*p+k] * W[k*n+i]) = 1 for all k
                        A = np.zeros((p, rank))
                        for k in range(p):
                            A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V: for each (j,k), solve for V[j*p+k]
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, rank))
                        for i in range(n):
                            A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Update W: for each (k,i), solve for W[k*n+i]
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, rank))
                        for j in range(m):
                            A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                        except:
                            pass
                
                # Check convergence
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                
                if max_err < 1e-6:
                    # Try to round to half-integers
                    U_r = round_to_half_int(U)
                    V_r = round_to_half_int(V)
                    W_r = round_to_half_int(W)
                    
                    if verify_decomposition(U_r, V_r, W_r):
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    
                    # If rounding failed, try with tighter ALS iterations
                    for polish_it in range(50):
                        if time.time() > deadline - 1:
                            break
                        
                        for i in range(n):
                            for j in range(m):
                                A = np.zeros((p, rank))
                                for k in range(p):
                                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                                try:
                                    U[i*m+j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                                except:
                                    pass
                        
                        for j in range(m):
                            for k in range(p):
                                A = np.zeros((n, rank))
                                for i in range(n):
                                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                                try:
                                    V[j*p+k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                                except:
                                    pass
                        
                        for k in range(p):
                            for i in range(n):
                                A = np.zeros((m, rank))
                                for j in range(m):
                                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                                try:
                                    W[k*n+i, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                                except:
                                    pass
                        
                        U_r = round_to_half_int(U)
                        V_r = round_to_half_int(V)
                        W_r = round_to_half_int(W)
                        
                        if verify_decomposition(U_r, V_r, W_r):
                            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    
                    break
        
        return None
    
    def try_kronecker_structure():
        """
        Try to exploit Kronecker product structure by factoring m=4 as 2*2.
        This decomposes the problem into smaller sub-problems.
        
        We can write the 2x4 matrix as [A1; A2] where A1, A2 are 2x2.
        The 4x7 matrix as [B1 B2] where B1, B2 are 2x7.
        Then C = A1*B1 + A2*B2.
        
        This gives us two independent 2x2x7 multiplications.
        """
        # For 2x2x7: schoolbook rank = 2*2*7 = 28
        # But we can potentially do better.
        # Try to find decomposition for 2x2x7 first, then combine.
        
        n_sub = 2
        m_sub = 2
        p_sub = 7
        
        # Try various ranks for the sub-problem
        for sub_rank in range(n_sub * m_sub * p_sub // 2, n_sub * m_sub * p_sub + 1):
            if time.time() > deadline - 5:
                break
            
            # Try to find 2x2x7 decomposition
            sub_result = try_rank_2x2x7(sub_rank, n_sub, m_sub, p_sub)
            if sub_result is not None:
                # Combine two copies for the full 2x4x7 problem
                # U_full has shape (8, 2*sub_rank)
                # V_full has shape (28, 2*sub_rank)  
                # W_full has shape (14, 2*sub_rank)
                
                U_sub, V_sub, W_sub = sub_result['U'], sub_result['V'], sub_result['W']
                total_rank = 2 * sub_rank
                
                U_full = np.zeros((8, total_rank))
                V_full = np.zeros((28, total_rank))
                W_full = np.zeros((14, total_rank))
                
                # First half: A1 * B1
                U_full[0:4, 0:sub_rank] = U_sub
                V_full[0:14, 0:sub_rank] = V_sub
                W_full[0:14, 0:sub_rank] = W_sub
                
                # Second half: A2 * B2
                U_full[4:8, sub_rank:2*sub_rank] = U_sub
                V_full[14:28, sub_rank:2*sub_rank] = V_sub
                W_full[0:14, sub_rank:2*sub_rank] = W_sub
                
                # Verify
                if verify_decomposition(U_full, V_full, W_full):
                    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
        
        return None
    
    def try_rank_2x2x7(rank, n_s, m_s, p_s):
        """Try to find decomposition for 2x2x7 matrix multiplication."""
        nm_s = n_s * m_s
        mp_s = m_s * p_s
        pn_s = p_s * n_s
        
        for restart in range(2):
            if time.time() > deadline - 3:
                return None
            
            seed = rank * 100 + restart * 7919
            np.random.seed(seed)
            
            U = np.random.randn(nm_s, rank) * 0.5
            V = np.random.randn(mp_s, rank) * 0.5
            W = np.random.randn(pn_s, rank) * 0.5
            
            for it in range(100):
                if time.time() > deadline - 2:
                    return None
                
                # Update U
                for i in range(n_s):
                    for j in range(m_s):
                        A = np.zeros((p_s, rank))
                        for k in range(p_s):
                            A[k, :] = V[j*p_s+k, :] * W[k*n_s+i, :]
                        try:
                            U[i*m_s+j, :] = np.linalg.lstsq(A, np.ones(p_s), rcond=None)[0]
                        except:
                            pass
                
                # Update V
                for j in range(m_s):
                    for k in range(p_s):
                        A = np.zeros((n_s, rank))
                        for i in range(n_s):
                            A[i, :] = U[i*m_s+j, :] * W[k*n_s+i, :]
                        try:
                            V[j*p_s+k, :] = np.linalg.lstsq(A, np.ones(n_s), rcond=None)[0]
                        except:
                            pass
                
                # Update W
                for k in range(p_s):
                    for i in range(n_s):
                        A = np.zeros((m_s, rank))
                        for j in range(m_s):
                            A[j, :] = U[i*m_s+j, :] * V[j*p_s+k, :]
                        try:
                            W[k*n_s+i, :] = np.linalg.lstsq(A, np.ones(m_s), rcond=None)[0]
                        except:
                            pass
                
                # Check
                max_err = 0.0
                for i in range(n_s):
                    for j in range(m_s):
                        for k in range(p_s):
                            val = np.dot(U[i*m_s+j, :], V[j*p_s+k, :] * W[k*n_s+i, :])
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                
                if max_err < 1e-6:
                    U_r = round_to_half_int(U)
                    V_r = round_to_half_int(V)
                    W_r = round_to_half_int(W)
                    
                    # Verify for sub-problem
                    valid = True
                    for i in range(n_s):
                        for j in range(m_s):
                            for k in range(p_s):
                                val = np.dot(U_r[i*m_s+j, :], V_r[j*p_s+k, :] * W_r[k*n_s+i, :])
                                if abs(val - 1.0) > 1e-9:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    break
        
        return None
    
    # First try Kronecker structure approach
    if time.time() < deadline - 10:
        kronecker_result = try_kronecker_structure()
        if kronecker_result is not None:
            best_result = kronecker_result
            best_rank = kronecker_result['U'][0].__len__() if isinstance(kronecker_result['U'][0], list) else len(kronecker_result['U'][0])
    
    # Then try direct ALS search for lower ranks
    lower_bound = max(nm, mp, pn)
    
    for rank in range(lower_bound, best_rank):
        if time.time() > deadline - 5:
            break
        
        result = try_rank(rank, max_iterations=100, max_restarts=2)
        if result is not None:
            best_result = result
            best_rank = rank
            break
    
    return best_result
