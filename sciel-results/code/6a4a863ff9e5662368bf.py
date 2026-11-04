import time
import numpy as np
import random

DIMENSIONS = [3, 3, 8]
DEADLINE_S = 115

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n

    def time_left():
        return DEADLINE_S - (time.time() - start_time)

    def schoolbook():
        rank = n*m*p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def validate(U, V, W, tol=1e-9):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        
        T_expected = np.ones((n, m, p))
        T_computed = np.zeros((n, m, p))
        
        for c in range(rank):
            u_c = U[:, c].reshape(n, m)
            v_c = V[:, c].reshape(m, p)
            w_c = W[:, c].reshape(p, n)
            
            uv = np.einsum('ij,jk->ijk', u_c, v_c)
            T_computed += uv * w_c.T[:, np.newaxis, :]
        
        return np.max(np.abs(T_computed - T_expected)) < tol

    def get_68_rank_block():
        # Returns U, V, W for a 3x3x2 block with rank 17
        rank = 17
        U = np.zeros((9, rank))
        V = np.zeros((6, rank))
        W = np.zeros((6, rank))
        
        col = 0
        # Strassen for top-left 2x2
        U_coeffs = [
            {0: 1, 3: 1}, {2: 1, 3: 1}, {0: 1}, {3: 1},
            {0: 1, 1: 1}, {2: 1, 0: -1}, {1: 1, 3: -1}
        ]
        V_coeffs = [
            {0: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}, {2: 1, 0: -1},
            {3: 1}, {0: 1, 1: 1}, {2: 1, 3: 1}
        ]
        W_coeffs = [
            {0: 1, 3: 1}, {2: 1, 3: -1}, {1: 1, 3: 1}, {0: 1, 2: 1},
            {0: -1, 1: 1}, {3: 1}, {0: 1}
        ]
        
        for l in range(7):
            for li, coeff in U_coeffs[l].items():
                U[li, col] += coeff
            for lj, coeff in V_coeffs[l].items():
                V[lj, col] += coeff
            for lc, coeff in W_coeffs[l].items():
                W[lc, col] += coeff
            col += 1
            
        # Schoolbook for remaining parts
        # i in {0,1}, j=2
        for i in range(2):
            for k in range(2):
                U[i*3 + 2, col] = 1.0
                V[2*3 + k, col] = 1.0
                W[k*3 + i, col] = 1.0
                col += 1
        # i=2, j in {0,1}
        for j in range(2):
            for k in range(2):
                U[2*3 + j, col] = 1.0
                V[j*3 + k, col] = 1.0
                W[k*3 + 2, col] = 1.0
                col += 1
        # i=2, j=2
        for k in range(2):
            U[2*3 + 2, col] = 1.0
            V[2*3 + k, col] = 1.0
            W[k*3 + 2, col] = 1.0
            col += 1
            
        return U, V, W

    def assemble_68():
        rank = 68
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        for g in range(4):
            U_block, V_block, W_block = get_68_rank_block()
            # Map block indices to global indices
            # U: i, j (0..2) -> i*3+j
            # V: j, k (0..1) -> j*3 + (2*g + k)
            # W: k (0..1), i (0..2) -> (2*g + k)*3 + i
            
            for r in range(17):
                for u_idx in range(9):
                    if U_block[u_idx, r] != 0:
                        U[u_idx, col + r] = U_block[u_idx, r]
                
                for v_idx in range(6):
                    j = v_idx // 3
                    k_local = v_idx % 3
                    v_global = j*3 + (2*g + k_local)
                    V[v_global, col + r] = V_block[v_idx, r]
                    
                for w_idx in range(6):
                    k_local = w_idx // 3
                    i = w_idx % 3
                    w_global = (2*g + k_local)*3 + i
                    W[w_global, col + r] = W_block[w_idx, r]
            
            col += 17
            
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to find rank 16 block
    def try_find_rank_16_block(timeout=5.0):
        target_time = start_time + timeout
        rank = 16
        # Random search with local refinement
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        # Initialize with random values
        for _ in range(10):
            U = np.random.randn(9, rank)
            V = np.random.randn(6, rank)
            W = np.random.randn(6, rank)
            
            # Scale to approximate magnitude
            U /= np.linalg.norm(U, axis=0, keepdims=True)
            V /= np.linalg.norm(V, axis=0, keepdims=True)
            W /= np.linalg.norm(W, axis=0, keepdims=True)
            
            # Local search
            for _ in range(100):
                if time.time() > target_time:
                    return None
                
                # Compute error
                T_comp = np.zeros((3,3,2))
                for c in range(rank):
                    u_c = U[:, c].reshape(3,3)
                    v_c = V[:, c].reshape(3,2)
                    w_c = W[:, c].reshape(2,3)
                    uv = np.einsum('ij,jk->ijk', u_c, v_c)
                    T_comp += uv * w_c.T[:, np.newaxis, :]
                
                err = np.sum((T_comp - 1.0)**2)
                
                if err < best_err:
                    best_err = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    
                    if err < 1e-10:
                        # Try to snap to half-integers
                        U_snap = np.round(best_U * 2) / 2
                        V_snap = np.round(best_V * 2) / 2
                        W_snap = np.round(best_W * 2) / 2
                        
                        T_snap = np.zeros((3,3,2))
                        for c in range(rank):
                            u_c = U_snap[:, c].reshape(3,3)
                            v_c = V_snap[:, c].reshape(3,2)
                            w_c = W_snap[:, c].reshape(2,3)
                            uv = np.einsum('ij,jk->ijk', u_c, v_c)
                            T_snap += uv * w_c.T[:, np.newaxis, :]
                        
                        if np.max(np.abs(T_snap - 1.0)) < 1e-9:
                            return U_snap, V_snap, W_snap

                # Update one variable (simple coordinate descent)
                var = np.random.choice(3)
                if var == 0:
                    idx = np.random.randint(0, 9*rank)
                    r, c = idx // rank, idx % rank
                    U[r, c] += np.random.randn() * 0.1
                elif var == 1:
                    idx = np.random.randint(0, 6*rank)
                    r, c = idx // rank, idx % rank
                    V[r, c] += np.random.randn() * 0.1
                else:
                    idx = np.random.randint(0, 6*rank)
                    r, c = idx // rank, idx % rank
                    W[r, c] += np.random.randn() * 0.1

        # Check if best found is close enough to snap
        if best_U is not None and best_err < 1e-6:
            U_snap = np.round(best_U * 2) / 2
            V_snap = np.round(best_V * 2) / 2
            W_snap = np.round(best_W * 2) / 2
            
            T_snap = np.zeros((3,3,2))
            for c in range(rank):
                u_c = U_snap[:, c].reshape(3,3)
                v_c = V_snap[:, c].reshape(3,2)
                w_c = W_snap[:, c].reshape(2,3)
                uv = np.einsum('ij,jk->ijk', u_c, v_c)
                T_snap += uv * w_c.T[:, np.newaxis, :]
            
            if np.max(np.abs(T_snap - 1.0)) < 1e-9:
                return U_snap, V_snap, W_snap

        return None

    # Attempt to find rank 16 block
    block_16 = try_find_rank_16_block(timeout=10.0)
    
    if block_16 is not None:
        U_block, V_block, W_block = block_16
        rank = 64
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        for g in range(4):
            for r in range(16):
                for u_idx in range(9):
                    if U_block[u_idx, r] != 0:
                        U[u_idx, col + r] = U_block[u_idx, r]
                
                for v_idx in range(6):
                    j = v_idx // 3
                    k_local = v_idx % 3
                    v_global = j*3 + (2*g + k_local)
                    V[v_global, col + r] = V_block[v_idx, r]
                    
                for w_idx in range(6):
                    k_local = w_idx // 3
                    i = w_idx % 3
                    w_global = (2*g + k_local)*3 + i
                    W[w_global, col + r] = W_block[w_idx, r]
            
            col += 16
            
        result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        if validate(U, V, W):
            return result

    # Fallback to rank 68
    return assemble_68()

if __name__ == "__main__":
    construct()
