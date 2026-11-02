import time
import numpy as np

DIMENSIONS = [3, 4, 4]
_START = time.time()
_TIME_LIMIT = 140.0

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    r0 = n * m * p  # 48 schoolbook

    # Schoolbook decomposition as fallback
    U0, V0, W0 = _get_schoolbook(n, m, p)
    best_rank = r0
    best_U, best_V, best_W = U0, V0, W0

    # Strategy: Decompose M_3,4,4 = M_3,3,4 + M_3,1,4
    # M_3,1,4 has rank 12 (schoolbook, which is minimal for this shape)
    # M_3,3,4 = M_3,3,2 + M_3,3,2
    # So Total Rank = 2 * Rank(M_3,3,2) + 12
    
    # Try to find Rank(M_3,3,2) < 18
    # Known schoolbook is 18.
    # Try 17, then 16.
    
    r_332_found = None
    for target_r in [17, 16]:
        if time.time() - _START > _TIME_LIMIT - 30:
            break
        res = _find_decomposition_332(target_r)
        if res is not None:
            r_332_found = target_r
            break
            
    if r_332_found is not None:
        total_rank = 2 * r_332_found + 12
        if total_rank < best_rank:
            U_new, V_new, W_new = _assemble_344_from_332(r_332_found)
            if U_new is not None and _verify_full(U_new, V_new, W_new, n, m, p):
                best_rank = total_rank
                best_U, best_V, best_W = U_new, V_new, W_new

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def _get_schoolbook(n, m, p):
    nm, mp, pn = n * m, m * p, p * n
    r = n * m * p
    U = np.zeros((nm, r))
    V = np.zeros((mp, r))
    W = np.zeros((pn, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _find_decomposition_332(rank, n_trials=100):
    # Solve for M_3,3,2
    # n=3, m=3, p=2
    n, m, p = 3, 3, 2
    nm, mp, pn = n * m, m * p, p * n
    idx = [(i * m + j, j * p + k, k * n + i)
           for i in range(n) for j in range(m) for k in range(p)]
    
    best_loss = float('inf')
    best_sol = None
    
    for t in range(n_trials):
        if time.time() - _START > _TIME_LIMIT - 30:
            break
        
        rng = np.random.RandomState(t * 12345 + rank * 7)
        
        # Initialize
        U = rng.randn(nm, rank) * 0.1
        V = rng.randn(mp, rank) * 0.1
        W = rng.randn(pn, rank) * 0.1
        
        # Gradient Descent with Momentum
        v_U = np.zeros_like(U)
        v_V = np.zeros_like(V)
        v_W = np.zeros_like(W)
        momentum = 0.9
        lr = 0.01
        
        for it in range(5000):
            if time.time() - _START > _TIME_LIMIT - 30:
                break
                
            Ug = np.zeros_like(U)
            Vg = np.zeros_like(V)
            Wg = np.zeros_like(W)
            loss = 0.0
            
            for ui, vi, wi in idx:
                vw = V[vi] * W[wi]
                uv = U[ui] * V[vi]
                uw = U[ui] * W[wi]
                s = np.dot(U[ui], vw)
                e = s - 1.0
                loss += e * e
                Ug[ui] += e * vw
                Vg[vi] += e * uw
                Wg[wi] += e * uv
                
            if loss < 1e-8:
                # Check for half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if _verify_332(U_r, V_r, W_r):
                    return (U_r, V_r, W_r)
                # If float solution is good, try to snap
                # But we need exact half-integers.
                # For now, if loss is tiny, it's likely a valid decomposition in reals.
                # The task requires integer/half-integer entries.
                # So we must find a solution on the grid.
                # This is hard.
                # However, if we find a real solution, we can try to project to grid.
                # But projection might break the equations.
                # So we need to search on the grid or find a grid solution directly.
                # Given the small size, maybe we can use a different strategy.
                break
            
            # Update
            v_U = momentum * v_U - lr * Ug
            v_V = momentum * v_V - lr * Vg
            v_W = momentum * v_W - lr * Wg
            
            U += v_U
            V += v_V
            W += v_W
            
            if loss < best_loss:
                best_loss = loss
                best_sol = (U.copy(), V.copy(), W.copy())
                
            # Decay LR
            lr *= 0.999
            
    if best_sol is not None and best_loss < 1e-4:
        U_b, V_b, W_b = best_sol
        U_r = np.round(U_b * 2) / 2
        V_r = np.round(V_b * 2) / 2
        W_r = np.round(W_b * 2) / 2
        if _verify_332(U_r, V_r, W_r):
            return (U_r, V_r, W_r)
            
    return None

def _verify_332(U, V, W):
    n, m, p = 3, 3, 2
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = i * m + j
                vi = j * p + k
                wi = k * n + i
                val = np.dot(U[ui], V[vi] * W[wi])
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _assemble_344_from_332(r_332):
    # Get the 332 decomposition
    res = _find_decomposition_332(r_332)
    if res is None:
        return None, None, None
    U_332, V_332, W_332 = res
    
    n, m, p = 3, 4, 4
    R_total = 2 * r_332 + 12
    U_full = np.zeros((12, R_total))
    V_full = np.zeros((16, R_total))
    W_full = np.zeros((12, R_total))
    
    # Part 1: A[0:3, 0:3] * B[0:3, 0:2] -> C[0:2, 0:3] ?
    # Wait, the assembly in the original code was:
    # M_3,4,4 = M_3,3,2 (cols 0-1 of B) + M_3,3,2 (cols 2-3 of B) + M_3,1,4 (col 3 of A)
    # Let's trace indices carefully.
    
    # M_3,4,4: A is 3x4, B is 4x4, C is 3x4.
    # C[i,k] = sum_{j=0}^3 A[i,j] B[j,k].
    
    # Split j into 0,1,2,3.
    # Term j=0: A[i,0] B[0,k]
    # Term j=1: A[i,1] B[1,k]
    # Term j=2: A[i,2] B[2,k]
    # Term j=3: A[i,3] B[3,k]
    
    # Group j=0,1 into M_3,2,4? No, M_3,3,2 was used in original.
    # Original: M_3,3,2 + M_3,3,2 + M_3,1,4.
    # This implies splitting B's columns? Or A's columns?
    # M_3,3,2: A(3x3), B(3x2).
    # If we use two M_3,3,2, we cover B(3x4)?
    # B is 4x4.
    # Let's assume the split is on B's rows (which are A's cols).
    # B rows 0,1,2,3.
    # Split B rows into 0,1 and 2,3?
    # Then A cols 0,1 and 2,3.
    # So A1 = A[:, 0:2] (3x2), B1 = B[0:2, :] (2x4).
    # This is M_3,2,4.
    # But original used M_3,3,2.
    # M_3,3,2 is A(3x3), B(3x2).
    # This doesn't match M_3,2,4.
    
    # Let's re-read the original assembly logic.
    # "Part 1: A1 (rows 0-2) * B1a (cols 0-1 of B, rows 0-2)"
    # This description is confusing.
    
    # Let's build a correct assembly for M_3,4,4 using two M_3,2,4 and one M_3,2,4?
    # No, let's stick to a valid split.
    # M_3,4,4 = M_3,2,4 + M_3,2,4.
    # If we find M_3,2,4 rank R, total 2R.
    # Schoolbook M_3,2,4 is 24. Total 48.
    
    # Let's try to find M_3,2,4 rank < 24.
    # M_3,2,4 is isomorphic to M_4,2,3.
    
    # Given the complexity, I will stick to the original assembly structure if it is valid,
    # or switch to a simpler one if I can solve the sub-problem.
    
    # Let's assume the original assembly was:
    # C = A1 B1 + A2 B2 + A3 B3
    # A1: 3x3 (cols 0-2 of A), B1: 3x4 (rows 0-2 of B). Rank of M_3,3,4?
    # A2: 3x1 (col 3 of A), B2: 1x4 (row 3 of B). Rank of M_3,1,4 is 12.
    # So we need M_3,3,4.
    # M_3,3,4 = M_3,3,2 + M_3,3,2.
    # This requires splitting B's columns?
    # B1 is 3x4. Split into B1a (3x2) and B1b (3x2).
    # Then A1 B1a is M_3,3,2?
    # A1 is 3x3. B1a is 3x2.
    # Yes, M_3,3,2.
    # So this assembly is valid.
    
    # U_full (12 x R)
    # V_full (16 x R)
    # W_full (12 x R)
    
    # Part 1: M_3,3,2 (A cols 0-2, B rows 0-2, B cols 0-1)
    # A indices: 0-8 (3x3)
    # B indices: rows 0-2, cols 0-1 -> j*p+k.
    # j in 0,1,2. k in 0,1.
    # Indices: 0,1, 4,5, 8,9.
    # C indices: k*n+i. k in 0,1. i in 0,1,2.
    # Indices: 0,1,2, 3,4,5.
    
    U_full[0:9, 0:r_332] = U_332
    v_idx_1 = [0, 1, 4, 5, 8, 9]
    V_full[v_idx_1, 0:r_332] = V_332
    w_idx_1 = [0, 1, 2, 3, 4, 5]
    W_full[w_idx_1, 0:r_332] = W_332
    
    # Part 2: M_3,3,2 (A cols 0-2, B rows 0-2, B cols 2-3)
    # A indices: 0-8
    # B indices: j in 0,1,2. k in 2,3.
    # Indices: 2,3, 6,7, 10,11.
    # C indices: k in 2,3. i in 0,1,2.
    # Indices: 6,7,8, 9,10,11.
    
    U_full[0:9, r_332:2*r_332] = U_332
    v_idx_2 = [2, 3, 6, 7, 10, 11]
    V_full[v_idx_2, r_332:2*r_332] = V_332
    w_idx_2 = [6, 7, 8, 9, 10, 11]
    W_full[w_idx_2, r_332:2*r_332] = W_332
    
    # Part 3: M_3,1,4 (A col 3, B row 3)
    # A indices: i*4+3. i in 0,1,2. -> 3, 7, 11.
    # B indices: 3*4+k. k in 0..3. -> 12, 13, 14, 15.
    # C indices: k*3+i.
    
    u_idx_3 = [3, 7, 11]
    v_idx_3 = [12, 13, 14, 15]
    
    for c in range(12):
        i = c // 4
        k = c % 4
        U_full[u_idx_3[i], 2*r_332 + c] = 1
        V_full[v_idx_3[k], 2*r_332 + c] = 1
        W_full[k*3 + i, 2*r_332 + c] = 1
        
    return U_full, V_full, W_full

def _verify_full(U, V, W, n, m, p):
    nm, mp, pn = n * m, m * p, p * n
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = i * m + j
                vi = j * p + k
                wi = k * n + i
                val = np.dot(U[ui], V[vi] * W[wi])
                if abs(val - 1.0) > 1e-6:
                    return False
    return True
