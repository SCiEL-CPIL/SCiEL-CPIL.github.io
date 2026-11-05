import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Target tensor entries: T[a, b, c] = 1 if a=im+j, b=jp+k, c=kn+i
    target_entries = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_entries.add((a, b, c))
    
    # Precompute target tensor
    target = np.zeros((nm, mp, pn))
    for (a, b, c) in target_entries:
        target[a, b, c] = 1.0
    
    # Schoolbook fallback
    def get_schoolbook():
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_solution = get_schoolbook()
    best_rank = n * m * p
    
    def compute_tensor(U, V, W):
        return np.einsum('ar,br,cr->abc', U, V, W)
    
    def check_validity(U, V, W, tol=1e-9):
        T = compute_tensor(U, V, W)
        return np.allclose(T, target, atol=tol)
    
    def try_solution(U, V, W, rank):
        nonlocal best_solution, best_rank
        if check_validity(U, V, W):
            if rank < best_rank:
                best_solution = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                best_rank = rank
                return True
        return False
    
    # Try various rank values with different initialization strategies
    # Start from a reasonable lower bound
    for R in range(30, best_rank):
        if time.time() > deadline:
            break
        
        found = False
        for restart in range(30):
            if time.time() > deadline:
                break
            
            # Different initialization strategies
            if restart % 4 == 0:
                # Random initialization with small values
                U = np.random.randn(nm, R) * 0.1
                V = np.random.randn(mp, R) * 0.1
                W = np.random.randn(pn, R) * 0.1
            elif restart % 4 == 1:
                # Structured initialization based on Kronecker products
                U = np.zeros((nm, R))
                V = np.zeros((mp, R))
                W = np.zeros((pn, R))
                for r in range(R):
                    i = r % n
                    j = (r // n) % m
                    k = (r // (n*m)) % p
                    U[i*m+j, r] = 1.0
                    V[j*p+k, r] = 1.0
                    W[k*n+i, r] = 1.0
                # Add random perturbations
                U += np.random.randn(nm, R) * 0.01
                V += np.random.randn(mp, R) * 0.01
                W += np.random.randn(pn, R) * 0.01
            elif restart % 4 == 2:
                # Sparse random initialization
                U = np.zeros((nm, R))
                V = np.zeros((mp, R))
                W = np.zeros((pn, R))
                for r in range(R):
                    # Randomly select a few non-zero entries
                    for _ in range(3):
                        a = np.random.randint(nm)
                        b = np.random.randint(mp)
                        c = np.random.randint(pn)
                        U[a, r] = np.random.choice([-1, 1])
                        V[b, r] = np.random.choice([-1, 1])
                        W[c, r] = np.random.choice([-1, 1])
            else:
                # Mixed initialization
                U = np.random.randn(nm, R) * 0.5
                V = np.random.randn(mp, R) * 0.5
                W = np.random.randn(pn, R) * 0.5
                # Add some structure
                for r in range(min(R, n*m)):
                    i = r % n
                    j = (r // n) % m
                    U[i*m+j, r] += 1.0
            
            # Gradient descent with adaptive learning rate
            lr = 0.1
            prev_loss = float('inf')
            for step in range(1000):
                if time.time() > deadline:
                    break
                
                T = compute_tensor(U, V, W)
                err = T - target
                loss = np.sum(err**2)
                
                if loss < 1e-12:
                    break
                
                # Check for convergence
                if abs(prev_loss - loss) < 1e-15:
                    break
                prev_loss = loss
                
                grad_U = 2 * np.einsum('abc,br,cr->ar', err, V, W, optimize=True)
                grad_V = 2 * np.einsum('abc,ar,cr->br', err, U, W, optimize=True)
                grad_W = 2 * np.einsum('abc,ar,br->cr', err, U, V, optimize=True)
                
                # Adaptive learning rate
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Decay learning rate
                lr *= 0.999
            
            # Try to snap to half-integers with multiple precision levels
            for snap_factor in [1, 2, 4, 8]:
                U_snap = np.round(U * snap_factor) / snap_factor
                V_snap = np.round(V * snap_factor) / snap_factor
                W_snap = np.round(W * snap_factor) / snap_factor
                
                if try_solution(U_snap, V_snap, W_snap, R):
                    found = True
                    break
            
            if found:
                break
        
        if found:
            continue
    
    return best_solution
