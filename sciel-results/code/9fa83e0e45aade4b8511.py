import numpy as np
import time

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Fallback rank-28 solution
    def get_rank_28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        U_coeffs = [
            {0: 1, 3: 1},
            {2: 1, 3: 1},
            {0: 1},
            {3: 1},
            {0: 1, 1: 1},
            {2: 1, 0: -1},
            {1: 1, 3: -1}
        ]
        V_coeffs = [
            {0: 1, 3: 1},
            {0: 1},
            {1: 1, 3: -1},
            {2: 1, 0: -1},
            {3: 1},
            {0: 1, 1: 1},
            {2: 1, 3: 1}
        ]
        W_coeffs = [
            {0: 1, 3: 1},
            {1: 1, 3: -1},
            {2: 1, 3: 1},
            {0: 1, 1: 1},
            {0: -1, 2: 1},
            {3: 1},
            {0: 1}
        ]
        
        for q in range(4):
            for s in range(7):
                r = q * 7 + s
                for u_idx, val in U_coeffs[s].items():
                    U[u_idx][r] = val
                for v_local_idx, val in V_coeffs[s].items():
                    j = v_local_idx // 2
                    k_local = v_local_idx % 2
                    k = 2 * q + k_local
                    v_idx = j * p + k
                    V[v_idx][r] = val
                for w_local_idx, val in W_coeffs[s].items():
                    i = w_local_idx % 2
                    k_local = w_local_idx // 2
                    k = 2 * q + k_local
                    w_idx = k * n + i
                    W[w_idx][r] = val
        return {"U": U, "V": V, "W": W}

    # Attempt to find rank-12 solution for 2x2x4
    # Then combine two such solutions for 2x2x8 (rank 24)
    
    start_time = time.time()
    time_limit = 10.0 # seconds for search
    
    def find_rank_12_2x2x4():
        # Target tensor T[i, j, k] for C = A B
        # A is 2x2 (indices 0,1,2,3), B is 2x4 (indices 0..7), C is 2x4 (indices 0..7)
        # T[i, j, k] is 1 if C[i,k] depends on A[i,j] and B[j,k]
        # C[i,k] = sum_j A[i,j] B[j,k]
        # So T[i, j, k] = 1 if (i,j) is a valid A index and (j,k) is a valid B index?
        # Wait, the tensor decomposition is for the multiplication map.
        # T is a 3-tensor of size 4 x 8 x 8?
        # No, the tensor is defined on the space of A, B, C.
        # The standard tensor M_{n,m,p} has dimensions n*m x m*p x p*n.
        # For 2x2x4:
        # U dim: 2*2 = 4
        # V dim: 2*4 = 8
        # W dim: 4*2 = 8
        
        T = np.zeros((4, 8, 8))
        # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # B indices: j*4 + k. j in 0,1. k in 0..3.
        # W indices: k*2 + i. k in 0..3. i in 0,1.
        
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    a_idx = i * 2 + j
                    b_idx = j * 4 + k
                    w_idx = k * 2 + i
                    T[a_idx, b_idx, w_idx] = 1.0
        
        rank = 12
        best_residual = float('inf')
        best_solution = None
        
        # Try a few random initializations
        for restart in range(20):
            if time.time() - start_time > time_limit:
                break
                
            U = np.random.randn(4, rank)
            V = np.random.randn(8, rank)
            W = np.random.randn(8, rank)
            
            # Normalize to avoid scaling issues
            U = U / np.linalg.norm(U, axis=0, keepdims=True)
            V = V / np.linalg.norm(V, axis=0, keepdims=True)
            W = W / np.linalg.norm(W, axis=0, keepdims=True)
            
            for it in range(50):
                # Update W
                # T[a, b, w] = sum_r U[a, r] V[b, r] W[w, r]
                # For each r, we want to find W[:, r]
                # This is a linear system:
                # Let A_r[a, b] = U[a, r] V[b, r]
                # Then T[:, :, w] = sum_r A_r[:, :, w]? No.
                # T[a, b, w] = sum_r (U[a, r] V[b, r]) W[w, r]
                # Fix r. We have a system for W[:, r]?
                # Actually, we can solve for all W at once.
                # Flatten T to vector t.
                # The matrix for W is a Khatri-Rao product of U and V?
                # Let's use lstsq for each column of W? No, W is 8x12.
                # We can solve for W by treating each row of W as a variable vector?
                # Better: Solve for W[:, r] for each r independently?
                # No, the terms are coupled in the sum.
                # But ALS updates one factor at a time.
                # When updating W, U and V are fixed.
                # T[a, b, w] = sum_r (U[a, r] V[b, r]) W[w, r]
                # Let M[a, b, r] = U[a, r] V[b, r].
                # Then T[a, b, w] = sum_r M[a, b, r] W[w, r].
                # This is a linear system for W.
                # We can reshape T to (32, 8) ? No.
                # Let's iterate over w? No.
                # We can solve for W by flattening the (a,b) indices.
                # T_flat[ab, w] = sum_r M_flat[ab, r] W[w, r].
                # This is not a standard matrix product because W is indexed by w and r.
                # Let's swap indices: T[w, ab] = sum_r W[w, r] M[ab, r].
                # So T_swapped = W @ M.T ?
                # T_swapped[w, ab] = sum_r W[w, r] M[ab, r].
                # Yes!
                # T_swapped is 8x32.
                # M is 32x12.
                # W is 8x12.
                # So T_swapped = W @ M.T.
                # We solve W = T_swapped @ (M.T)^\dagger.
                
                M = np.einsum('ar,br->abr', U, V) # 4x8x12
                M_flat = M.reshape(32, 12) # 32x12
                T_flat = T.reshape(32, 8) # 32x8
                T_swapped = T_flat.T # 8x32
                
                W, _, _, _ = np.linalg.lstsq(M_flat.T, T_swapped.T, rcond=None)
                W = W.T # 8x12
                
                # Update V
                # T[a, b, w] = sum_r U[a, r] V[b, r] W[w, r]
                # Fix r.
                # T[a, b, w] = sum_r (U[a, r] W[w, r]) V[b, r]
                # Let N[a, w, r] = U[a, r] W[w, r].
                # T[a, b, w] = sum_r N[a, w, r] V[b, r].
                # Swap indices: T[b, aw] = sum_r V[b, r] N[aw, r].
                # T_swapped2[b, aw] = V[b, r] N[aw, r].
                # T_swapped2 = V @ N.T.
                N = np.einsum('ar,wr->awr', U, W) # 4x8x12
                N_flat = N.reshape(32, 12)
                T_flat2 = T.reshape(8, 32) # b, aw
                T_swapped2 = T_flat2 # 8x32
                V, _, _, _ = np.linalg.lstsq(N_flat.T, T_swapped2.T, rcond=None)
                V = V.T # 8x12
                
                # Update U
                # T[a, b, w] = sum_r U[a, r] V[b, r] W[w, r]
                # Fix r.
                # T[a, b, w] = sum_r U[a, r] (V[b, r] W[w, r])
                # Let P[b, w, r] = V[b, r] W[w, r].
                # T[a, b, w] = sum_r U[a, r] P[bw, r].
                # T[a, bw] = sum_r U[a, r] P[bw, r].
                # T_swapped3[a, bw] = U[a, r] P[bw, r].
                # T_swapped3 = U @ P.T.
                P = np.einsum('br,wr->bwr', V, W) # 8x8x12
                P_flat = P.reshape(64, 12)
                T_flat3 = T.reshape(4, 64) # a, bw
                T_swapped3 = T_flat3 # 4x64
                U, _, _, _ = np.linalg.lstsq(P_flat.T, T_swapped3.T, rcond=None)
                U = U.T # 4x12
                
            # Check residual
            approx = np.einsum('ar,br,wr->abw', U, V, W)
            residual = np.linalg.norm(T - approx)
            
            if residual < best_residual:
                best_residual = residual
                best_solution = (U, V, W)
                
            if residual < 1e-9:
                break

        if best_solution is not None and best_residual < 1e-9:
            return best_solution
        return None

    sol_2x2x4 = find_rank_12_2x2x4()
    
    if sol_2x2x4 is not None:
        U_12, V_12, W_12 = sol_2x2x4
        rank = 24
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Block 0: cols 0-3 of B, rows 0-3 of C
        # Block 1: cols 4-7 of B, rows 4-7 of C
        
        for r in range(12):
            # Block 0
            for a in range(4):
                U[a][r] = U_12[a, r]
            for b in range(8):
                V[b][r] = V_12[b, r]
            for w in range(8):
                W[w][r] = W_12[w, r]
                
            # Block 1
            r2 = 12 + r
            for a in range(4):
                U[a][r2] = U_12[a, r]
            # V indices for block 1: b in 8..15 (cols 4-7)
            # V_12 indices are 0..7.
            # Map V_12[b] to V[b+8]
            for b in range(8):
                V[b + 8][r2] = V_12[b, r]
            # W indices for block 1: w in 8..15 (rows 4-7 of C)
            # W_12 indices are 0..7.
            # Map W_12[w] to W[w+8]
            for w in range(8):
                W[w + 8][r2] = W_12[w, r]
                
        return {"U": U, "V": V, "W": W}
    else:
        return get_rank_28()

# EVOLVE-BLOCK-END
