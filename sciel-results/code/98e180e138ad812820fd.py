"""Matrix multiplication 2x3 x 3x6 rank-minimal decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 18
    pn = p * n  # 12
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 36
    
    # Build the tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def verify_fast(U, V, W):
        """Faster verification using tensor contraction."""
        T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
        if not np.allclose(T_decomp, T, atol=1e-6):
            return False
        return True
    
    def optimize(U, V, W, max_steps=2000, tol=1e-8):
        """Optimize using adaptive gradient descent with momentum."""
        r = U.shape[1]
        lr = 0.01
        momentum = 0.9
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        best_loss = np.inf
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T_decomp - T
            loss = np.sum(diff**2)
            
            if loss < best_loss:
                best_loss = loss
            
            if np.max(np.abs(diff)) < tol:
                break
            
            grad_U = np.einsum('abc,br,cr->ar', diff, V, W) * 2
            grad_V = np.einsum('abc,ar,cr->br', diff, U, W) * 2
            grad_W = np.einsum('abc,ar,br->cr', diff, U, V) * 2
            
            vU = momentum * vU + (1 - momentum) * grad_U
            vV = momentum * vV + (1 - momentum) * grad_V
            vW = momentum * vW + (1 - momentum) * grad_W
            
            U -= lr * vU
            V -= lr * vV
            W -= lr * vW
            
            # Adaptive learning rate
            if step % 100 == 0 and step > 0:
                if loss < best_loss * 0.99:
                    lr *= 0.8
                elif loss > best_loss * 1.01:
                    lr *= 1.2
                    lr = min(lr, 0.1)
        
        return U, V, W, loss
    
    # Try to find lower rank decompositions
    best_rank = schoolbook_rank
    best_result = None
    
    # First, create the schoolbook solution
    U_school = np.zeros((nm, schoolbook_rank))
    V_school = np.zeros((mp, schoolbook_rank))
    W_school = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    # Try ranks from lower to higher, with more attempts for lower ranks
    for rank in range(25, schoolbook_rank):
        if time.time() > deadline - 5:
            break
        
        # More attempts for lower ranks
        num_attempts = 30 if rank < 30 else 20
        
        np.random.seed(rank * 1000)
        
        found = False
        for attempt in range(num_attempts):
            if time.time() > deadline - 5:
                break
            
            # Different initialization strategies
            if attempt % 3 == 0:
                U = np.random.randn(nm, rank)
                V = np.random.randn(mp, rank)
                W = np.random.randn(pn, rank)
            elif attempt % 3 == 1:
                U = np.random.randn(nm, rank) * 0.5
                V = np.random.randn(mp, rank) * 0.5
                W = np.random.randn(pn, rank) * 0.5
            else:
                U = np.zeros((nm, rank))
                V = np.zeros((mp, rank))
                W = np.zeros((pn, rank))
                # Initialize with partial schoolbook structure
                for i in range(min(n, 2)):
                    for j in range(min(m, 3)):
                        for k in range(min(p, 6)):
                            idx = (i*m+j)*p + (j*p+k)*n + (k*n+i)
                            if idx < rank:
                                U[i*m+j, idx % rank] = 1.0
                                V[j*p+k, idx % rank] = 1.0
                                W[k*n+i, idx % rank] = 1.0
                # Add random perturbation
                U += np.random.randn(nm, rank) * 0.1
                V += np.random.randn(mp, rank) * 0.1
                W += np.random.randn(pn, rank) * 0.1
            
            U, V, W, loss = optimize(U, V, W)
            
            if verify_fast(U, V, W):
                found = True
                best_rank = rank
                U_list = [[float(U[i,j]) for j in range(rank)] for i in range(nm)]
                V_list = [[float(V[i,j]) for j in range(rank)] for i in range(mp)]
                W_list = [[float(W[i,j]) for j in range(rank)] for i in range(pn)]
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                break
        
        if found:
            break
    
    if best_result is None:
        # Return schoolbook
        U_list = [[float(U_school[i,j]) for j in range(schoolbook_rank)] for i in range(nm)]
        V_list = [[float(V_school[i,j]) for j in range(schoolbook_rank)] for i in range(mp)]
        W_list = [[float(W_school[i,j]) for j in range(schoolbook_rank)] for i in range(pn)]
        best_result = {"U": U_list, "V": V_list, "W": W_list}
    
    return best_result
