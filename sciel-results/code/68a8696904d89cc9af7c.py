import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Precompute index arrays
    i_list, j_list, jp_list, k_list = [], [], [], []
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    i_list.append(i)
                    j_list.append(j)
                    jp_list.append(jp)
                    k_list.append(k)
    
    i_arr = np.array(i_list)
    j_arr = np.array(j_list)
    jp_arr = np.array(jp_list)
    k_arr = np.array(k_list)
    
    U_idx = i_arr * m + j_arr  # (N,)
    V_idx = jp_arr * p + k_arr  # (N,)
    W_idx = k_arr * n + i_arr  # (N,)
    target = (j_arr == jp_arr).astype(float)  # (N,)
    
    N = len(i_list)
    
    # Try to find lower rank decompositions
    for rank in [50, 45, 40, 38, 36, 34, 32]:
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + rank * 100 + attempt)
            U = np.random.randn(n * m, rank) * 0.3
            V = np.random.randn(m * p, rank) * 0.3
            W = np.random.randn(p * n, rank) * 0.3
            
            lr = 0.005
            prev_loss = float('inf')
            
            for step in range(300):
                if time.time() > deadline:
                    break
                
                # Compute loss and gradients vectorized
                U_rows = U[U_idx]  # (N, rank)
                V_rows = V[V_idx]  # (N, rank)
                W_rows = W[W_idx]  # (N, rank)
                
                prod = (U_rows * V_rows * W_rows).sum(axis=1)  # (N,)
                err = prod - target  # (N,)
                loss = (err ** 2).sum()
                
                # Gradients
                # grad_U[U_idx[t], :] += err[t] * V_rows[t, :] * W_rows[t, :]
                # Use scatter-add
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                # For each element, accumulate gradients
                # This is O(N*rank) which is 384*rank ~ 19200 for rank=50, fine.
                for t in range(N):
                    e = err[t]
                    if abs(e) < 1e-10:
                        continue
                    grad_U[U_idx[t]] += e * V_rows[t] * W_rows[t]
                    grad_V[V_idx[t]] += e * U_rows[t] * W_rows[t]
                    grad_W[W_idx[t]] += e * U_rows[t] * V_rows[t]
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                if abs(prev_loss - loss) < 1e-12:
                    break
                prev_loss = loss
            
            # Check max error
            U_rows = U[U_idx]
            V_rows = V[V_idx]
            W_rows = W[W_idx]
            prod = (U_rows * V_rows * W_rows).sum(axis=1)
            max_err = np.max(np.abs(prod - target))
            
            if max_err < 1e-5:
                # Rationalize
                def rationalize(matrix):
                    result = []
                    for row in matrix:
                        r = []
                        for val in row:
                            if abs(val) < 1e-10:
                                r.append(Fraction(0))
                            else:
                                r.append(Fraction(val).limit_denominator(10000))
                        result.append(r)
                    return result
                
                U_rat = rationalize(U)
                V_rat = rationalize(V)
                W_rat = rationalize(W)
                
                # Verify exactness
                exact = True
                for t in range(N):
                    s = Fraction(0)
                    for tt in range(rank):
                        s += U_rat[U_idx[t]][tt] * V_rat[V_idx[t]][tt] * W_rat[W_idx[t]][tt]
                    target_val = 1 if j_list[t] == jp_list[t] else 0
                    if s != target_val:
                        exact = False
                        break
                
                if exact:
                    best_result = {"U": U_rat, "V": V_rat, "W": W_rat}
                    found = True
                    break
        
        if found:
            break
    
    return best_result
