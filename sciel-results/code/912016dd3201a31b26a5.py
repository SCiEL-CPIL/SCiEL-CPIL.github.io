import numpy as np
import time
from itertools import product

def construct():
    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12
    
    # Build the target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    T_flat = T.flatten()
    nnz = np.count_nonzero(T_flat)
    
    # Deadline guard: 150s wall clock limit, keep 10s safety margin
    deadline = time.time() + 140.0
    
    # We will search for rank 20 first. If not found, fall back to 21.
    # The theoretical lower bound is likely 20 (ceil(2*2*6 / 2) = 12? No, 
    # for matrix multiplication M_{n x m} * M_{m x p}, 
    # rank is at least max(n*m, m*p, p*n) ? No.
    # For 2x2 * 2x2, rank is 7.
    # For 2x2 * 2x6, we can view it as 3 blocks of 2x2 * 2x2.
    # Block diagonal rank is 3*7 = 21.
    # Can we do better?
    # The tensor is T_{i,j,k} = delta_{j, j'} ... no.
    # A_{i,j} B_{j,k} -> C_{i,k}.
    # Indices: i in [0,1], j in [0,1], k in [0,5].
    # T_{i, j, k} = 1 if (j is the middle index).
    # Actually, the tensor entries are T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i.
    # This is the standard matrix multiplication tensor.
    
    # Let's try to find a rank 20 decomposition using a modified ALS or direct search.
    # Since rank 20 is very close to the known 21, it might be impossible or require specific structure.
    # However, the prompt asks for rank-minimal. If 20 is not possible, 21 is the best we can guarantee easily.
    # But let's try to verify if 20 is possible.
    # If we use Strassen for 2x2, we save 1 multiplication per 2x2 block.
    # 3 blocks -> 21.
    # Can we share computations between blocks?
    # The input B is 2x6. It can be seen as three 2x2 blocks B1, B2, B3.
    # A is 2x2.
    # C = A * B = [A*B1, A*B2, A*B3].
    # Since A is the same, we can compute the Strassen products of A with B1, B2, B3.
    # The Strassen algorithm for A*B1 uses:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # If we process B1, B2, B3 separately, we do 7*3 = 21 multiplications.
    # Can we combine?
    # Note that M2 depends on A21+A22 and B11.
    # M6 depends on A21-A11 and B11+B12.
    # The terms involving B's are linear.
    # Let's look at the multiplications.
    # The multiplications are scalar products of linear forms of A and linear forms of B.
    # Let A_forms be a set of 7 linear forms of A.
    # Let B_forms be a set of 7 linear forms of B (for one block).
    # For 3 blocks, we have B_forms1, B_forms2, B_forms3.
    # Total multiplications: 7 * (1 + 1 + 1) = 21.
    # To reduce to 20, we would need to share a multiplication or use a lower rank algorithm for the whole 2x2 * 2x6.
    # Is there a rank 20 algorithm?
    # It is an open question for some small matrices, but 2x2 * 2x6 is small enough.
    # Let's try a random search for rank 20 with a tight time limit.
    # If it fails, we return rank 21.
    
    def try_find_decomposition(rank, max_iters=50, time_limit=10.0):
        start_time = time.time()
        np.random.seed(42)
        
        # Initialize U, V, W
        # U: (nm, rank), V: (mp, rank), W: (pn, rank)
        # We can initialize with the rank 21 solution and zero out one column?
        # Or random.
        # Let's try random init with a few restarts.
        
        best_U, best_V, best_W = None, None, None
        best_err = np.inf
        
        for restart in range(5):
            if time.time() > start_time + time_limit:
                break
                
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            for i in range(max_iters):
                if time.time() > start_time + time_limit:
                    break
                
                # ALS step 1: Solve for U
                # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # Fix V, W.
                # For each a, T[a,:,:] = sum_r U[a,r] (V[:,r] outer W[:,r])
                # B_mat (mp*pn, rank) where B_mat[(b,c), r] = V[b,r] W[c,r]
                B_mat = np.einsum('br,cr->bcr', V, W).reshape(mp * pn, rank)
                U_new = np.zeros((nm, rank))
                for a in range(nm):
                    T_a_flat = T[a, :, :].flatten()
                    U_a, _, _, _ = np.linalg.lstsq(B_mat, T_a_flat, rcond=None)
                    U_new[a, :] = U_a
                U = U_new
                
                # ALS step 2: Solve for V
                # For each b, T[:,b,:] = sum_r V[b,r] (U[:,r] outer W[:,r])
                C_mat = np.einsum('ar,cr->acr', U, W).reshape(nm * pn, rank)
                V_new = np.zeros((mp, rank))
                for b in range(mp):
                    T_b_flat = T[:, b, :].flatten()
                    V_b, _, _, _ = np.linalg.lstsq(C_mat, T_b_flat, rcond=None)
                    V_new[b, :] = V_b
                V = V_new
                
                # ALS step 2: Solve for W
                # For each c, T[:,:,c] = sum_r W[c,r] (U[:,r] outer V[:,r])
                D_mat = np.einsum('ar,br->abr', U, V).reshape(nm * mp, rank)
                W_new = np.zeros((pn, rank))
                for c in range(pn):
                    T_c_flat = T[:, :, c].flatten()
                    W_c, _, _, _ = np.linalg.lstsq(D_mat, T_c_flat, rcond=None)
                    W_new[c, :] = W_c
                W = W_new
                
                # Check error
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(T - T_approx)
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    
                if err < 1e-10:
                    # Snap to half-integers
                    U_s = np.round(2*U)/2
                    V_s = np.round(2*V)/2
                    W_s = np.round(2*W)/2
                    T_check = np.einsum('ar,br,cr->abc', U_s, V_s, W_s)
                    if np.allclose(T_check, T, atol=1e-9):
                        return U_s, V_s, W_s
                    else:
                        # Maybe not half-integer? Try integer?
                        U_i = np.round(U)
                        V_i = np.round(V)
                        W_i = np.round(W)
                        T_check_i = np.einsum('ar,br,cr->abc', U_i, V_i, W_i)
                        if np.allclose(T_check_i, T, atol=1e-9):
                            return U_i, V_i, W_i
                        # If not, return the floating point version? 
                        # The problem says "integer/half-integer entries".
                        # If we can't snap, we might need to search more.
                        # But for now, if error is 0, it's a valid decomposition.
                        # The interface says "exactly decompose".
                        # Floating point exactness is hard.
                        # Let's assume if error is 1e-10, it's close enough for snapping.
                        pass
        
        return None

    # Try rank 20
    if time.time() < deadline - 20:
        result = try_find_decomposition(20, max_iters=100, time_limit=10.0)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to rank 21 (Strassen block diagonal)
    rank = 21
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Strassen steps for 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    # Indices for A (2x2): a=00, b=01, c=10, d=11
    # Indices for B (2x2): e=00, f=01, g=10, h=11
    # Indices for C (2x2): 00, 01, 10, 11
    
    # Map local indices to global
    # A: 0->0, 1->1, 2->2, 3->3 (row-major)
    # B: for block g, 0->6g+0, 1->6g+1, 2->6g+2, 3->6g+3
    # C: for block g, 0->2g+0, 1->2g+1, 2->2g+2, 3->2g+3
    
    strassen_coeffs = [
        # M1: (a+d)(e+h)
        {'a': {0: 1, 3: 1}, 'b': {0: 1, 3: 1}, 'c': {0: 1, 1: 1, 2: -1, 3: 1}},
        # M2: (c+d)e
        {'a': {2: 1, 3: 1}, 'b': {0: 1}, 'c': {1: 1, 3: -1}}, # Wait, C11 = M2+M4? No.
        # Let's re-verify Strassen output mapping.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So:
        # M1 contributes to C00 (+1) and C11 (+1)
        # M2 contributes to C10 (+1) and C11 (-1)
        # M3 contributes to C01 (+1) and C11 (+1)
        # M4 contributes to C00 (+1) and C10 (+1)
        # M5 contributes to C00 (-1) and C01 (+1)
        # M6 contributes to C11 (+1)
        # M7 contributes to C00 (+1)
        
        # Let's define the list of tuples (coeff_a, coeff_b, coeff_c) for each multiplication.
        # Actually, it's easier to just fill the matrices.
    ]
    
    # Let's just hardcode the Strassen decomposition for 3 blocks.
    # Block g:
    # A indices: 0,1,2,3
    # B indices: 6g, 6g+1, 6g+2, 6g+3
    # C indices: 2g, 2g+1, 2g+2, 2g+3
    
    # Multiplications:
    # 1: (A0+A3)(B0+B3) -> C0 += 1, C3 += 1
    # 2: (A2+A3)B0     -> C2 += 1, C3 -= 1
    # 3: A0(B1-B3)     -> C1 += 1, C3 += 1
    # 4: A3(B2-B0)     -> C0 += 1, C2 += 1
    # 5: (A0+A1)B3     -> C0 -= 1, C1 += 1
    # 6: (A2-A0)(B0+B1) -> C3 += 1
    # 7: (A1-A3)(B2+B3) -> C0 += 1
    
    for g in range(3):
        b_base = 6 * g
        c_base = 2 * g
        
        # M1
        r = 7 * g + 0
        U[0, r] += 1; U[3, r] += 1
        V[b_base+0, r] += 1; V[b_base+3, r] += 1
        W[c_base+0, r] += 1; W[c_base+3, r] += 1
        
        # M2
        r = 7 * g + 1
        U[2, r] += 1; U[3, r] += 1
        V[b_base+0, r] += 1
        W[c_base+2, r] += 1; W[c_base+3, r] -= 1
        
        # M3
        r = 7 * g + 2
        U[0, r] += 1
        V[b_base+1, r] += 1; V[b_base+3, r] -= 1
        W[c_base+1, r] += 1; W[c_base+3, r] += 1
        
        # M4
        r = 7 * g + 3
        U[3, r] += 1
        V[b_base+2, r] += 1; V[b_base+0, r] -= 1
        W[c_base+0, r] += 1; W[c_base+2, r] += 1
        
        # M5
        r = 7 * g + 4
        U[0, r] += 1; U[1, r] += 1
        V[b_base+3, r] += 1
        W[c_base+0, r] -= 1; W[c_base+1, r] += 1
        
        # M6
        r = 7 * g + 5
        U[2, r] += 1; U[0, r] -= 1
        V[b_base+0, r] += 1; V[b_base+1, r] += 1
        W[c_base+3, r] += 1
        
        # M7
        r = 7 * g + 6
        U[1, r] += 1; U[3, r] -= 1
        V[b_base+2, r] += 1; V[b_base+3, r] += 1
        W[c_base+0, r] += 1

    # Verify
    T_check = np.einsum('ar,br,cr->abc', U, V, W)
    if not np.allclose(T_check, T, atol=1e-10):
        # Fallback to schoolbook
        rank_school = 24
        U_s = np.zeros((nm, rank_school))
        V_s = np.zeros((mp, rank_school))
        W_s = np.zeros((pn, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U_s[a, col] = 1
                    V_s[b, col] = 1
                    W_s[c, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
