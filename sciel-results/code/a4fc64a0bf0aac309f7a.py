<<<<<<< SEARCH
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    n, m, p = 2, 3, 4
    
    def verify(U, V, W):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        T = np.zeros((n, m, p))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T[i,j,k] - 1.0) > 1e-10:
                        return False
        return True
    
    def make_schoolbook():
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Start with schoolbook
    best_result = make_schoolbook()
    best_rank = n * m * p
    
    # Try to find a lower rank by searching over sparse patterns
    # Key insight: each rank-1 term is defined by which (i,j) entries in U,
    # (j,k) entries in V, and (k,i) entries in W are non-zero.
    # For a "simple" term, U has one non-zero at (i,j), V at (j,k), W at (k,i).
    # More complex terms can have multiple non-zeros.
    
    # Try rank 20: use a structured decomposition
    # Split k into two groups: {0,1} and {2,3}
    # For each group, use a 2x3x2 decomposition
    
    def try_structured_rank20():
        """Try a rank-20 decomposition based on 2x3x2 blocks."""
        if time.time() > deadline - 15:
            return None
        
        r = 20
        U = np.zeros((n*m, r), dtype=float)
        V = np.zeros((m*p, r), dtype=float)
        W = np.zeros((p*n, r), dtype=float)
        
        # For 2x3x2, try to find rank-10 decomposition
        # A is 2x3, B is 3x2, C is 2x2
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
        
        # Try 10 terms:
        # t0: (0,0,0) - covers C[0,0] part
        # t1: (0,1,0) - covers C[0,0] part  
        # t2: (0,2,0) - covers C[0,0] part
        # t3: (1,0,0) - covers C[1,0] part
        # t4: (1,1,0) - covers C[1,0] part
        # t5: (1,2,0) - covers C[1,0] part
        # t6: combined for C[0,1] and C[1,1]
        # t7: combined for C[0,1] and C[1,1]
        # t8: correction
        # t9: correction
        
        # This is getting complex. Let me try a different approach:
        # Use the fact that we can have terms with multiple non-zeros
        
        # For the first 2x3x2 block (k=0,1):
        # Use 10 terms
        for col in range(10):
            if col < 6:
                # Schoolbook for k=0
                i = col // 3
                j = col % 3
                U[i*m+j, col] = 1
                V[j*p+0, col] = 1
                W[0*n+i, col] = 1
            else:
                # For k=1, use combined terms
                c = col - 6
                if c == 0:
                    # (A[0,0]+A[0,1]+A[0,2]) * (B[0,1]+B[1,1]+B[2,1]) * 1
                    U[0, col] = 1
                    U[1, col] = 1
                    U[2, col] = 1
                    V[0*p+1, col] = 1
                    V[1*p+1, col] = 1
                    V[2*p+1, col] = 1
                    W[0, col] = 1
                    W[1, col] = 1
                elif c == 1:
                    # (A[0,0]+A[0,1]) * (B[0,1]+B[1,1]) * 1
                    U[0, col] = 1
                    U[1, col] = 1
                    V[0*p+1, col] = 1
                    V[1*p+1, col] = 1
                    W[0, col] = 1
                elif c == 2:
                    # (A[1,0]+A[1,1]) * (B[0,1]+B[1,1]) * 1
                    U[3, col] = 1
                    U[4, col] = 1
                    V[0*p+1, col] = 1
                    V[1*p+1, col] = 1
                    W[1, col] = 1
                elif c == 3:
                    # (A[0,1]+A[0,2]) * (B[1,1]+B[2,1]) * 1
                    U[1, col] = 1
                    U[2, col] = 1
                    V[1*p+1, col] = 1
                    V[2*p+1, col] = 1
                    W[0, col] = 1
                elif c == 4:
                    # (A[1,1]+A[1,2]) * (B[1,1]+B[2,1]) * 1
                    U[4, col] = 1
                    U[5, col] = 1
                    V[1*p+1, col] = 1
                    V[2*p+1, col] = 1
                    W[1, col] = 1
        
        # For the second 2x3x2 block (k=2,3):
        for col in range(10, 20):
            c = col - 10
            if c < 6:
                # Schoolbook for k=2
                i = c // 3
                j = c % 3
                U[i*m+j, col] = 1
                V[j*p+2, col] = 1
                W[2*n+i, col] = 1
            else:
                # For k=3, use combined terms
                cc = c - 6
                if cc == 0:
                    U[0, col] = 1
                    U[1, col] = 1
                    U[2, col] = 1
                    V[0*p+3, col] = 1
                    V[1*p+3, col] = 1
                    V[2*p+3, col] = 1
                    W[0, col] = 1
                    W[1, col] = 1
                elif cc == 1:
                    U[0, col] = 1
                    U[1, col] = 1
                    V[0*p+3, col] = 1
                    V[1*p+3, col] = 1
                    W[0, col] = 1
                elif cc == 2:
                    U[3, col] = 1
                    U[4, col] = 1
                    V[0*p+3, col] = 1
                    V[1*p+3, col] = 1
                    W[1, col] = 1
                elif cc == 3:
                    U[1, col] = 1
                    U[2, col] = 1
                    V[1*p+3, col] = 1
                    V[2*p+3, col] = 1
                    W[0, col] = 1
                elif cc == 4:
                    U[4, col] = 1
                    U[5, col] = 1
                    V[1*p+3, col] = 1
                    V[2*p+3, col] = 1
                    W[1, col] = 1
        
        result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        if verify(result["U"], result["V"], result["W"]):
            return result
        return None
    
    result = try_structured_rank20()
    if result is not None:
        best_result = result
        best_rank = 20
    else:
        # Try rank 22
        def try_rank22():
            if time.time() > deadline - 15:
                return None
            r = 22
            U = np.zeros((n*m, r), dtype=float)
            V = np.zeros((m*p, r), dtype=float)
            W = np.zeros((p*n, r), dtype=float)
            
            # 6 terms for k=0, 6 for k=1, 5 for k=2, 5 for k=3
            col = 0
            for i in range(n):
                for j in range(m):
                    U[i*m+j, col] = 1
                    V[j*p+0, col] = 1
                    W[0*n+i, col] = 1
                    col += 1
            for i in range(n):
                for j in range(m):
                    U[i*m+j, col] = 1
                    V[j*p+1, col] = 1
                    W[1*n+i, col] = 1
                    col += 1
            # k=2: 5 terms, one shared
            for i in range(n):
                for j in range(m):
                    if i == 0 and j == 0:
                        # Modify first term to also cover k=2
                        V[0*p+2, 0] = 1
                        W[2*n+0, 0] = 1
                    else:
                        U[i*m+j, col] = 1
                        V[j*p+2, col] = 1
                        W[2*n+i, col] = 1
                        col += 1
            # k=3: 5 terms, one shared
            for i in range(n):
                for j in range(m):
                    if i == 0 and j == 0:
                        V[0*p+3, 6] = 1
                        W[3*n+0, 6] = 1
                    else:
                        U[i*m+j, col] = 1
                        V[j*p+3, col] = 1
                        W[3*n+i, col] = 1
                        col += 1
            
            result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            if verify(result["U"], result["V"], result["W"]):
                return result
            return None
        
        result = try_rank22()
        if result is not None:
            best_result = result
            best_rank = 22
    
    return best_result
=======
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    n, m, p = 2, 3, 4
    
    def verify(U, V, W):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        T = np.zeros((n, m, p))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T[i,j,k] - 1.0) > 1e-10:
                        return False
        return True
    
    def make_schoolbook():
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Start with schoolbook
    best_result = make_schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions using a search approach
    # We'll try to find valid decompositions with rank < 24
    
    def search_rank(r, max_iters=50000):
        """Search for a valid rank-r decomposition."""
        if time.time() > deadline - 20:
            return None
        
        # We represent each rank-1 term by its support pattern
        # U_support: which (i,j) pairs are non-zero
        # V_support: which (j,k) pairs are non-zero  
        # W_support: which (k,i) pairs are non-zero
        # For simplicity, use binary patterns with values in {-1, 0, 1}
        
        import itertools
        
        # For each term, the contribution to T[i,j,k] is 
        # U[i,j,r] * V[j,k,r] * W[k,i,r]
        # We need sum over r to equal 1 for all (i,j,k)
        
        # Try random search with local improvement
        np.random.seed(42)
        
        U = np.zeros((n*m, r), dtype=float)
        V = np.zeros((m*p, r), dtype=float)
        W = np.zeros((p*n, r), dtype=float)
        
        # Initialize with random sparse patterns
        for col in range(r):
            # Pick a random pattern
            u_pat = np.random.choice([0, 1], size=n*m)
            v_pat = np.random.choice([0, 1], size=m*p)
            w_pat = np.random.choice([0, 1], size=p*n)
            # Ensure at least one non-zero in each
            if u_pat.sum() == 0:
                u_pat[np.random.randint(n*m)] = 1
            if v_pat.sum() == 0:
                v_pat[np.random.randint(m*p)] = 1
            if w_pat.sum() == 0:
                w_pat[np.random.randint(p*n)] = 1
            U[:, col] = u_pat
            V[:, col] = v_pat
            W[:, col] = w_pat
        
        # Compute current tensor
        T = np.zeros((n, m, p))
        for col in range(r):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += U[i*m+j, col] * V[j*p+k, col] * W[k*n+i, col]
        
        # Target is all 1s
        target = np.ones((n, m, p))
        error = np.sum((T - target)**2)
        
        # Try to improve
        for it in range(max_iters):
            if time.time() > deadline - 25:
                break
            
            if error < 1e-10:
                break
            
            # Pick a random term and modify it
            col = np.random.randint(r)
            
            # Try modifying one entry
            which = np.random.randint(3)  # 0=U, 1=V, 2=W
            if which == 0:
                idx = np.random.randint(n*m)
                old_val = U[idx, col]
                U[idx, col] = 1.0 if old_val == 0 else 0.0
            elif which == 1:
                idx = np.random.randint(m*p)
                old_val = V[idx, col]
                V[idx, col] = 1.0 if old_val == 0 else 0.0
            else:
                idx = np.random.randint(p*n)
                old_val = W[idx, col]
                W[idx, col] = 1.0 if old_val == 0 else 0.0
            
            # Recompute error (fast approximation)
            T_new = np.zeros((n, m, p))
            for c in range(r):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_new[i,j,k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            new_error = np.sum((T_new - target)**2)
            
            if new_error < error:
                error = new_error
                T = T_new
            else:
                # Revert
                if which == 0:
                    U[idx, col] = old_val
                elif which == 1:
                    V[idx, col] = old_val
                else:
                    W[idx, col] = old_val
        
        if error < 1e-10:
            result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            if verify(result["U"], result["V"], result["W"]):
                return result
        return None
    
    # Try ranks from 23 down
    for r in range(best_rank - 1, 15, -1):
        if time.time() > deadline - 30:
            break
        result = search_rank(r)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    return best_result
>>>>>>> REPLACE

The random search approach is too slow and unlikely to find a solution. Let me take a much more targeted approach based on known mathematical results.

For 2×3×4 matrix multiplication, I'll try to construct a valid rank-20 decomposition based on the structure of the problem.

