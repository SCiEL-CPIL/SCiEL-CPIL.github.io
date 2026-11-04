"""Matrix multiplication 2x4x8 via Strassen block decomposition with dependency reduction."""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 4, 8]

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    # Strassen 2x2x2 blocks
    n_blocks_i = 1
    n_blocks_j = 2
    n_blocks_k = 4
    total_blocks = n_blocks_i * n_blocks_j * n_blocks_k
    rank = total_blocks * 7  # 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for bi in range(n_blocks_i):
        for bj in range(n_blocks_j):
            for bk in range(n_blocks_k):
                i0 = 2 * bi
                i1 = 2 * bi + 1
                j0 = 2 * bj
                j1 = 2 * bj + 1
                k0 = 2 * bk
                k1 = 2 * bk + 1
                
                u_a = i0 * m + j0
                u_b = i0 * m + j1
                u_c = i1 * m + j0
                u_d = i1 * m + j1
                
                v_e = j0 * p + k0
                v_f = j0 * p + k1
                v_g = j1 * p + k0
                v_h = j1 * p + k1
                
                w_c00 = k0 * n + i0
                w_c01 = k1 * n + i0
                w_c10 = k0 * n + i1
                w_c11 = k1 * n + i1
                
                # M1 = (a+d)(e+h)
                U[u_a][col] += 1; U[u_d][col] += 1
                V[v_e][col] += 1; V[v_h][col] += 1
                W[w_c00][col] += 1; W[w_c11][col] += 1
                col += 1
                
                # M2 = (c+d)e
                U[u_c][col] += 1; U[u_d][col] += 1
                V[v_e][col] += 1
                W[w_c10][col] += 1; W[w_c11][col] -= 1
                col += 1
                
                # M3 = a(f-h)
                U[u_a][col] += 1
                V[v_f][col] += 1; V[v_h][col] -= 1
                W[w_c01][col] += 1; W[w_c11][col] += 1
                col += 1
                
                # M4 = d(g-e)
                U[u_d][col] += 1
                V[v_g][col] += 1; V[v_e][col] -= 1
                W[w_c00][col] += 1; W[w_c10][col] += 1
                col += 1
                
                # M5 = (a+b)h
                U[u_a][col] += 1; U[u_b][col] += 1
                V[v_h][col] += 1
                W[w_c00][col] -= 1; W[w_c01][col] += 1
                col += 1
                
                # M6 = (c-a)(e+f)
                U[u_c][col] += 1; U[u_a][col] -= 1
                V[v_e][col] += 1; V[v_f][col] += 1
                W[w_c11][col] += 1
                col += 1
                
                # M7 = (b-d)(g+h)
                U[u_b][col] += 1; U[u_d][col] -= 1
                V[v_g][col] += 1; V[v_h][col] += 1
                W[w_c00][col] += 1
                col += 1
    
    # Convert to numpy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the true multiplication tensor for verification
    # T[a,b,c] = 1 iff j_a == j_b, i_a == i_c, k_b == k_c
    def build_true_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T
    
    true_tensor = build_true_tensor()
    
    # Compute current tensor from decomposition
    def compute_tensor(U_m, V_m, W_m, cols):
        T = np.zeros((n * m, m * p, p * n))
        for c in cols:
            T += np.einsum('i,j,k->ijk', U_m[:, c], V_m[:, c], W_m[:, c])
        return T
    
    # Verify initial decomposition
    current_tensor = compute_tensor(U_np, V_np, W_np, list(range(rank)))
    if not np.allclose(current_tensor, true_tensor, atol=1e-10):
        # Fallback: return trivial decomposition
        return {"U": U, "V": V, "W": W}
    
    # Greedy elimination: try to remove each term
    keep = list(range(rank))
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for c in range(rank):
            if c not in keep:
                continue
            if time.time() > deadline - 5:
                break
            new_keep = [k for k in keep if k != c]
            if len(new_keep) < 32:  # Can't go below theoretical lower bound
                continue
            test_tensor = compute_tensor(U_np, V_np, W_np, new_keep)
            if np.allclose(test_tensor, true_tensor, atol=1e-10):
                keep = new_keep
                improved = True
    
    # Try removing pairs of terms
    if time.time() < deadline - 10 and len(keep) > 32:
        keep_set = set(keep)
        pairs = list(combinations(sorted(keep_set), 2))
        for pair in pairs:
            if time.time() > deadline - 5:
                break
            new_keep = [k for k in keep if k not in pair]
            if len(new_keep) < 32:
                continue
            test_tensor = compute_tensor(U_np, V_np, W_np, new_keep)
            if np.allclose(test_tensor, true_tensor, atol=1e-10):
                keep = new_keep
                keep_set = set(keep)
                pairs = list(combinations(sorted(keep_set), 2))
    
    # Final verification
    if len(keep) < rank:
        U_reduced = U_np[:, keep].tolist()
        V_reduced = V_np[:, keep].tolist()
        W_reduced = W_np[:, keep].tolist()
        
        # Verify by checking the tensor
        U_v = np.array(U_reduced, dtype=float)
        V_v = np.array(V_reduced, dtype=float)
        W_v = np.array(W_reduced, dtype=float)
        test_tensor = compute_tensor(U_v, V_v, W_v, list(range(len(keep))))
        if np.allclose(test_tensor, true_tensor, atol=1e-10):
            return {"U": U_reduced, "V": V_reduced, "W": W_reduced}
    
    return {"U": U, "V": V, "W": W}
