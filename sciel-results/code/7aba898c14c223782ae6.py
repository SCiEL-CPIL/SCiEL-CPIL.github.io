"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Build the multiplication tensor
    # T[a, b, c] where a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try to find low-rank decomposition
    # Start from rank 35 and go down
    best_result = None
    start_time = time.time()
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    # Try rank 35
    for rank in [35, 34, 33]:
        if time.time() - start_time > 100:
            break
        
        # Initialize
        np.random.seed(42)
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        # Gradient descent
        lr = 0.01
        for it in range(5000):
            if time.time() - start_time > 110:
                break
            
            # Compute current decomposition
            # T_approx[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
            # Gradient w.r.t. U[a,r] = sum_{b,c} T[a,b,c] * V[b,r] * W[c,r] - sum_{b,c} (sum_r' U[a,r']*V[b,r']*W[c,r']) * V[b,r] * W[c,r]
            
            # More efficiently:
            # Let M_VW = V^T @ W (rank x rank) -- no, that's not right
            # T_approx[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # dT/da = sum_r U[a,r] V[b,r] W[c,r]
            
            # Gradient for U[a,r]: sum_{b,c} (T[a,b,c] - T_approx[a,b,c]) * V[b,r] * W[c,r]
            # = sum_{b,c} T[a,b,c] V[b,r] W[c,r] - sum_{b,c} T_approx[a,b,c] V[b,r] W[c,r]
            
            # sum_{b,c} T[a,b,c] V[b,r] W[c,r] = for each a, sum over b,c where T[a,b,c]=1 of V[b,r]*W[c,r]
            # T[a,b,c]=1 when a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
            # For fixed a=i*m+j, the non-zero entries are for b=j*p+k, c=k*n+i for k=0..p-1
            
            # Let me precompute
            # For each a, list of (b,c) pairs where T[a,b,c]=1
            T_pairs = {}
            for a in range(n*m):
                pairs = []
                for b in range(m*p):
                    for c in range(p*n):
                        if T[a,b,c] != 0:
                            pairs.append((b,c))
                T_pairs[a] = pairs
            
            # Compute gradient for U
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for a in range(n*m):
                for r in range(rank):
                    s = 0
                    for (b,c) in T_pairs[a]:
                        s += V[b,r] * W[c,r]
                    # T_approx[a,b,c] = sum_r' U[a,r'] V[b,r'] W[c,r']
                    # sum_{b,c} T_approx[a,b,c] V[b,r] W[c,r] = sum_r' U[a,r'] * sum_{b,c} V[b,r'] V[b,r] W[c,r'] W[c,r]
                    # = sum_r' U[a,r'] * (V[:,r']^T V[:,r]) * (W[:,r']^T W[:,r]) -- no, that's not right either
                    # Actually sum_{b,c} T_approx[a,b,c] V[b,r] W[c,r] = sum_{b,c} (sum_r' U[a,r'] V[b,r'] W[c,r']) V[b,r] W[c,r]
                    # = sum_r' U[a,r'] * (sum_b V[b,r'] V[b,r]) * (sum_c W[c,r'] W[c,r])
                    # = sum_r' U[a,r'] * dot(V[:,r'], V[:,r]) * dot(W[:,r'], W[:,r])
                    
                    s2 = 0
                    for r2 in range(rank):
                        s2 += U[a,r2] * np.dot(V[:,r2], V[:,r]) * np.dot(W[:,r2], W[:,r])
                    grad_U[a,r] = s - s2
            
            # This is getting too slow. Let me use a vectorized approach.
            # T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            # This is a 4x18x18 tensor, which is small enough.
            
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T - T_approx
            
            # grad_U[a,r] = sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
            grad_U = np.einsum('abc,br,cr->ar', diff, V, W)
            grad_V = np.einsum('abc,ar,cr->br', diff, U, W)
            grad_W = np.einsum('abc,ar,br->cr', diff, U, V)
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            if it % 500 == 0:
                err = np.linalg.norm(diff)
                if err < 1e-6:
                    break
        
        err = np.linalg.norm(T - np.einsum('ar,br,cr->abc', U, V, W))
        if err < 1e-6:
            best_result = (U, V, W, rank)
            break
    
    if best_result is None:
        # Return schoolbook
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    else:
        U, V, W, rank = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
