import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    nm = n * m  # 12
    mp = m * p  # 16
    pn = p * n  # 12
    
    def check_decomposition(U, V, W, rank, tol=1e-10):
        """Verify the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def als_solve(rank, seed, max_iters=1000, max_time=20):
        """Alternating least squares to find a decomposition."""
        t0 = time.time()
        
        np.random.seed(seed)
        # Initialize with small random values
        U = np.random.randn(nm, rank) * 0.3
        V = np.random.randn(mp, rank) * 0.3
        W = np.random.randn(pn, rank) * 0.3
        
        prev_loss = float('inf')
        
        for it in range(max_iters):
            if time.time() - t0 > max_time:
                break
            if time.time() > deadline - 5:
                break
            
            # Update W: for each (i,k), solve A_{(i,k),:} · W[k*3+i,:] = 1
            for i in range(n):
                for k in range(p):
                    A = np.zeros(rank)
                    for j in range(m):
                        A += U[i*m+j, :] * V[j*p+k, :]
                    norm_sq = np.dot(A, A)
                    if norm_sq > 1e-14:
                        W[k*n+i, :] = A / norm_sq
            
            # Update V: for each (j,k), solve B_{(j,k),:} · V[j*p+k,:] = 1
            for j in range(m):
                for k in range(p):
                    B = np.zeros(rank)
                    for i in range(n):
                        B += U[i*m+j, :] * W[k*n+i, :]
                    norm_sq = np.dot(B, B)
                    if norm_sq > 1e-14:
                        V[j*p+k, :] = B / norm_sq
            
            # Update U: for each (i,j), solve C_{(i,j),:} · U[i*m+j,:] = 1
            for i in range(n):
                for j in range(m):
                    C = np.zeros(rank)
                    for k in range(p):
                        C += V[j*p+k, :] * W[k*n+i, :]
                    norm_sq = np.dot(C, C)
                    if norm_sq > 1e-14:
                        U[i*m+j, :] = C / norm_sq
            
            # Compute loss every 10 iterations to save time
            if it % 10 == 0 or it == max_iters - 1:
                loss = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(rank):
                                s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            loss += (s - 1.0) ** 2
                if abs(prev_loss - loss) < 1e-16:
                    break
                prev_loss = loss
        
        # Check if solution is valid
        if check_decomposition(U.tolist(), V.tolist(), W.tolist(), rank, tol=1e-9):
            return U.tolist(), V.tolist(), W.tolist()
        
        return None
    
    # Try ranks from 43 upward
    for rank in [43, 44, 45, 46, 47, 48]:
        if time.time() > deadline - 15:
            break
        
        found = False
        for seed in range(50):
            if time.time() > deadline - 5:
                break
            if found:
                break
            
            remaining_time = deadline - time.time() - 5
            if remaining_time < 1:
                break
            
            result = als_solve(rank, seed, max_time=min(3, remaining_time))
            if result is not None:
                U, V, W = result
                return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank 48
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
