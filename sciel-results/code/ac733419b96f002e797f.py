"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 10
    dim_v = m * p  # 30
    dim_w = p * n  # 12
    
    # Build support
    support = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                support.append((u_idx, v_idx, w_idx))
    support_set = set(support)
    
    # Precompute constraints per index
    support_by_u = [[] for _ in range(dim_u)]
    support_by_v = [[] for _ in range(dim_v)]
    support_by_w = [[] for _ in range(dim_w)]
    for (u, v, w) in support:
        support_by_u[u].append((v, w))
        support_by_v[v].append((u, w))
        support_by_w[w].append((u, v))
    
    non_support = []
    non_support_by_u = [[] for _ in range(dim_u)]
    non_support_by_v = [[] for _ in range(dim_v)]
    non_support_by_w = [[] for _ in range(dim_w)]
    for u in range(dim_u):
        for v in range(dim_v):
            for w in range(dim_w):
                if (u, v, w) not in support_set:
                    non_support.append((u, v, w))
                    non_support_by_u[u].append((v, w))
                    non_support_by_v[v].append((u, w))
                    non_support_by_w[w].append((u, v))
    
    deadline = time.time() + 130
    best_result = None
    best_rank = float('inf')
    
    for rank in range(15, 50):
        if time.time() > deadline:
            break
        if best_result is not None and rank >= best_rank:
            break
        
        result = _als(dim_u, dim_v, dim_w, support, non_support,
                      support_by_u, support_by_v, support_by_w,
                      non_support_by_u, non_support_by_v, non_support_by_w,
                      rank, deadline)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, support, non_support):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                break
    
    if best_result is None:
        rank = n * m * p
        U = [[0] * rank for _ in range(dim_u)]
        V = [[0] * rank for _ in range(dim_v)]
        W = [[0] * rank for _ in range(dim_w)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result

def _als(dim_u, dim_v, dim_w, support, non_support,
         support_by_u, support_by_v, support_by_w,
         non_support_by_u, non_support_by_v, non_support_by_w,
         rank, deadline):
    np.random.seed(999)
    
    best_loss = float('inf')
    best_U = best_V = best_W = None
    
    for restart in range(15):
        if time.time() > deadline:
            break
        
        U = np.random.randn(dim_u, rank) * 0.1
        V = np.random.randn(dim_v, rank) * 0.1
        W = np.random.randn(dim_w, rank) * 0.1
        
        for it in range(300):
            if time.time() > deadline:
                break
            
            # Update U
            for u in range(dim_u):
                A_list = []
                b_list = []
                for (v, w) in support_by_u[u]:
                    A_list.append(V[v] * W[w])
                    b_list.append(1.0)
                for (v, w) in non_support_by_u[u]:
                    A_list.append(V[v] * W[w])
                    b_list.append(0.0)
                if A_list:
                    A = np.array(A_list)
                    b = np.array(b_list)
                    U[u], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            # Update V
            for v in range(dim_v):
                A_list = []
                b_list = []
                for (u, w) in support_by_v[v]:
                    A_list.append(U[u] * W[w])
                    b_list.append(1.0)
                for (u, w) in non_support_by_v[v]:
                    A_list.append(U[u] * W[w])
                    b_list.append(0.0)
                if A_list:
                    A = np.array(A_list)
                    b = np.array(b_list)
                    V[v], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            # Update W
            for w in range(dim_w):
                A_list = []
                b_list = []
                for (u, v) in support_by_w[w]:
                    A_list.append(U[u] * V[v])
                    b_list.append(1.0)
                for (u, v) in non_support_by_w[w]:
                    A_list.append(U[u] * V[v])
                    b_list.append(0.0)
                if A_list:
                    A = np.array(A_list)
                    b = np.array(b_list)
                    W[w], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            # Compute loss
            loss = 0.0
            for (u, v, w) in support:
                val = np.dot(U[u], V[v] * W[w])
                loss += (val - 1.0) ** 2
            for (u, v, w) in non_support:
                val = np.dot(U[u], V[v] * W[w])
                loss += val ** 2
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-12:
                break
        
        if best_loss < 1e-10:
            break
    
    if best_U is not None and best_loss < 1e-8:
        return (best_U.tolist(), best_V.tolist(), best_W.tolist())
    return None

def _verify(U, V, W, support, non_support):
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    for (u, v, w) in support:
        val = np.dot(U_arr[u], V_arr[v] * W_arr[w])
        if abs(val - 1.0) > 1e-8:
            return False
    for (u, v, w) in non_support:
        val = np.dot(U_arr[u], V_arr[v] * W_arr[w])
        if abs(val) > 1e-8:
            return False
    return True

# EVOLVE-BLOCK-END
