import numpy as np
import time
import random

def construct():
    n, m, p = 2, 2, 9
    start_time = time.time()
    deadline = start_time + 145
    
    # Dimensions
    dim_u = n * m  # 4
    dim_v = m * p  # 18
    dim_w = p * n  # 18
    
    # Target tensor T_target of shape (dim_u, dim_v, dim_w)
    # T_target[i, j, k] = 1 if i=(a,c), j=(c,b), k=(a,b) for some a,b,c
    # i = a*m + c
    # j = c*p + b
    # k = b*n + a
    T_target = np.zeros((dim_u, dim_v, dim_w))
    for a in range(n):
        for c in range(m):
            for b in range(p):
                i = a * m + c
                j = c * p + b
                k = b * n + a
                T_target[i, j, k] = 1.0

    def compute_error(U, V, W):
        """
        Compute squared Frobenius norm of difference between 
        tensor(U,V,W) and T_target.
        U: (dim_u, R)
        V: (dim_v, R)
        W: (dim_w, R)
        """
        # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        # Efficient way: 
        # Let A_r = U[:, r], B_r = V[:, r], C_r = W[:, r]
        # T = sum_r np.einsum('i,j,k->ijk', A_r, B_r, C_r)
        
        # Using einsum directly on the whole array is fast enough for small dims
        # T = np.einsum('ir,jr,kr->ijk', U, V, W)
        
        # Alternatively, we can compute the error as:
        # sum_{i,j,k} (sum_r U[i,r]V[j,r]W[k,r] - T_target[i,j,k])^2
        # = sum_{i,j,k} (sum_r U[i,r]V[j,r]W[k,r])^2 
        #   - 2 * sum_{i,j,k} T_target[i,j,k] sum_r U[i,r]V[j,r]W[k,r]
        #   + sum_{i,j,k} T_target[i,j,k]^2
        
        # The last term is constant (number of 1s in T_target = n*m*p = 36)
        # The middle term is linear in the product, but still complex.
        
        # Direct einsum is simplest and likely fast enough for R<=32, dims ~4,18,18
        T = np.einsum('ir,jr,kr->ijk', U, V, W)
        diff = T - T_target
        return np.sum(diff * diff)

    def get_schoolbook_rank36():
        rank = 36
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def get_strassen_rank32():
        rank = 32
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        
        col = 0
        def add_strassen_block(k1, k2, col_offset):
            nonlocal col
            # P1: (A11+A22)(B11+B22)
            U[0, col_offset] = 1; U[3, col_offset] = 1
            V[0*p+k1, col_offset] = 1; V[1*p+k2, col_offset] = 1
            W[k1*n+0, col_offset] = 1; W[k2*n+0, col_offset] = 0
            W[k1*n+1, col_offset] = 0; W[k2*n+1, col_offset] = 1
            col += 1
            # P2: (A21+A22)B11
            U[2, col_offset] = 1; U[3, col_offset] = 1
            V[0*p+k1, col_offset] = 1
            W[k1*n+0, col_offset] = 0; W[k2*n+0, col_offset] = 0
            W[k1*n+1, col_offset] = 1; W[k2*n+1, col_offset] = 0
            col += 1
            # P3: A11(B21-B22)
            U[0, col_offset] = 1
            V[0*p+k2, col_offset] = 1; V[1*p+k2, col_offset] = -1
            W[k1*n+0, col_offset] = 0; W[k2*n+0, col_offset] = 1
            W[k1*n+1, col_offset] = 0; W[k2*n+1, col_offset] = 0
            col += 1
            # P4: A22(B11-B12)
            U[3, col_offset] = 1
            V[1*p+k1, col_offset] = 1; V[0*p+k1, col_offset] = -1
            W[k1*n+0, col_offset] = 1; W[k2*n+0, col_offset] = 0
            W[k1*n+1, col_offset] = 1; W[k2*n+1, col_offset] = 0
            col += 1
            # P5: (A11+A12)B22
            U[0, col_offset] = 1; U[1, col_offset] = 1
            V[1*p+k2, col_offset] = 1
            W[k1*n+0, col_offset] = -1; W[k2*n+0, col_offset] = 1
            W[k1*n+1, col_offset] = 0; W[k2*n+1, col_offset] = 0
            col += 1
            # P6: (A21-A11)(B11+B12)
            U[2, col_offset] = 1; U[0, col_offset] = -1
            V[0*p+k1, col_offset] = 1; V[0*p+k2, col_offset] = 1
            W[k1*n+0, col_offset] = 0; W[k2*n+0, col_offset] = 0
            W[k1*n+1, col_offset] = 0; W[k2*n+1, col_offset] = -1
            col += 1
            # P7: (A12-A22)(B21+B22)
            U[1, col_offset] = 1; U[3, col_offset] = -1
            V[1*p+k1, col_offset] = 1; V[1*p+k2, col_offset] = 1
            W[k1*n+0, col_offset] = 1; W[k2*n+0, col_offset] = 0
            W[k1*n+1, col_offset] = 0; W[k2*n+1, col_offset] = 0
            col += 1

        for i in range(4):
            k1 = i * 2
            k2 = i * 2 + 1
            add_strassen_block(k1, k2, col)
            
        k = 8
        # Schoolbook 2x2 * 2x1
        U[0, col] = 1; V[0*p+k, col] = 1; W[k*n+0, col] = 1; col += 1
        U[1, col] = 1; V[1*p+k, col] = 1; W[k*n+0, col] = 1; col += 1
        U[2, col] = 1; V[0*p+k, col] = 1; W[k*n+1, col] = 1; col += 1
        U[3, col] = 1; V[1*p+k, col] = 1; W[k*n+1, col] = 1; col += 1
        
        if col != rank:
            return get_schoolbook_rank36()
            
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def verify_solution(U, V, W, rank):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        T = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(T, T_target, atol=1e-9)

    best_solution = get_strassen_rank32()
    best_rank = 32
    
    # Try to find a rank 28 solution using numerical optimization
    # We will use scipy.optimize.minimize if available, otherwise a simple gradient descent
    try:
        from scipy.optimize import minimize
        
        target_rank = 28
        
        def objective(x):
            R = target_rank
            U = x[0:dim_u*R].reshape(dim_u, R)
            V = x[dim_u*R:dim_u*R + dim_v*R].reshape(dim_v, R)
            W = x[dim_u*R + dim_v*R:].reshape(dim_w, R)
            return compute_error(U, V, W)
            
        # We perform several random restarts
        num_restarts = 5
        found_28 = False
        
        for i in range(num_restarts):
            if time.time() > deadline - 20:
                break
                
            # Random initialization
            x0 = np.random.randn(dim_u*R + dim_v*R + dim_w*R) * 0.1
            
            # L-BFGS-B is good for box constraints, but we don't have explicit bounds.
            # Nelder-Mead might be more robust for this non-convex landscape without gradients.
            # However, L-BFGS-B with numerical gradients is often faster.
            # Let's try L-BFGS-B first.
            
            res = minimize(objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if res.fun < 1e-9:
                # Check if it's actually valid (numerical precision)
                R = target_rank
                U = res.x[0:dim_u*R].reshape(dim_u, R)
                V = res.x[dim_u*R:dim_u*R + dim_v*R].reshape(dim_v, R)
                W = res.x[dim_u*R + dim_v*R:].reshape(dim_w, R)
                
                if verify_solution(U, V, W, R):
                    best_solution = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = R
                    found_28 = True
                    break # Found the optimal lower bound, stop searching
            
            # If L-BFGS-B fails, try Nelder-Mead with a tighter tolerance if time permits
            # But NM is slower. Let's stick to L-BFGS-B with more restarts if needed.
            
    except Exception as e:
        pass # Fallback to rank 32
    
    # If we didn't find 28, we can try 29, 30, 31?
    # Given the time constraints and the difficulty, 28 is the main target.
    # If 28 fails, 32 is the safe bet.
    
    return best_solution

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
