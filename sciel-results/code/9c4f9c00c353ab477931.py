import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115
    
    # Target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=np.float64)
    V_school = np.zeros((m * p, rank_school), dtype=np.float64)
    W_school = np.zeros((p * n, rank_school), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1
                V_school[b, col] = 1
                W_school[c, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = rank_school
    
    # Try to find lower rank
    rng = np.random.default_rng(123)
    
    # Try ranks from 104 down to 80
    for target_rank in range(rank_school - 1, max(rank_school - 25, 70), -1):
        if time.time() > deadline - 15:
            break
        
        for trial in range(10):
            if time.time() > deadline - 10:
                break
            
            U = rng.standard_normal((n * m, target_rank)) * 0.5
            V = rng.standard_normal((m * p, target_rank)) * 0.5
            W = rng.standard_normal((p * n, target_rank)) * 0.5
            
            # Gradient descent
            lr = 0.01
            for it in range(500):
                if time.time() > deadline - 5:
                    break
                
                # Compute T_hat
                # T_hat[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # This is expensive: 15*35*21*target_rank operations
                
                # Efficient computation:
                # For each r: outer product U[:,r] @ V[:,r]^T @ W[:,r] ... no
                # T_hat = sum_r np.einsum('ar,br,cr->abc', U, V, W)
                
                # Let's compute residual
                # residual[a,b,c] = T_hat[a,b,c] - T[a,b,c]
                
                # Gradient w.r.t. U[a,r]: V[b,r]*W[c,r] summed over b,c with residual
                # grad_U[a,r] = sum_{b,c} residual[a,b,c] * V[b,r] * W[c,r]
                # = sum_{b,c} (T_hat[a,b,c] - T[a,b,c]) * V[b,r] * W[c,r]
                
                # This is: for each a,r: sum_b sum_c residual[a,b,c] * V[b,r] * W[c,r]
                # = sum_b V[b,r] * (sum_c residual[a,b,c] * W[c,r])
                
                # Let's compute it
                # residual = T_hat - T
                # But computing T_hat fully is 15*35*21 = 11025 elements, times target_rank ~ 100
                # That's ~1 million operations per gradient step, times 500 steps = 500M
                # Might be too slow in Python. Let's use numpy efficiently.
                
                # Compute T_hat
                T_hat = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                residual = T_hat - T
                
                # Gradients
                grad_U = np.einsum('abc,br,cr->ar', residual, V, W, optimize=True)
                grad_V = np.einsum('abc,ar,cr->br', residual, U, W, optimize=True)
                grad_W = np.einsum('abc,ar,br->cr', residual, U, V, optimize=True)
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                if it % 50 == 0:
                    res_norm = np.linalg.norm(residual)
                    if res_norm < 1e-10:
                        break
            
            # Check if we found a good solution
            T_hat_final = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            res_norm = np.linalg.norm(T_hat_final - T)
            
            if res_norm < 1e-8:
                # Try to round to exact values
                # Round to nearest 0.5
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                T_hat_rounded = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded, optimize=True)
                res_rounded = np.linalg.norm(T_hat_rounded - T)
                
                if res_rounded < 1e-10:
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    best_rank = target_rank
                    break
                else:
                    # Use the unrounded solution
                    best_U, best_V, best_W = U, V, W
                    best_rank = target_rank
            
            if best_rank < target_rank:
                break
    
    return {
        'U': best_U.tolist(),
        'V': best_V.tolist(),
        'W': best_W.tolist()
    }

if __name__ == '__main__':
    res = construct()
    print(f"Rank: {len(res['U'][0]) if res['U'] else 0}")
