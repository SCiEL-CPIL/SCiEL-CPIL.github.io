import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    nm = n * m  # 20
    mp = m * p  # 30
    pn = p * n  # 24
    
    deadline = time.time() + 145
    
    # Start with schoolbook
    best_rank = nm * p  # 120
    schoolbook_U = np.zeros((nm, best_rank))
    schoolbook_V = np.zeros((mp, best_rank))
    schoolbook_W = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                schoolbook_U[i * m + j, col] = 1
                schoolbook_V[j * p + k, col] = 1
                schoolbook_W[k * n + i, col] = 1
                col += 1
    best_result = {
        "U": schoolbook_U.tolist(),
        "V": schoolbook_V.tolist(),
        "W": schoolbook_W.tolist()
    }
    
    # Try Strassen-based construction
    # Split A (4x5) into A_core (4x4) and A_tail (4x1)
    # Split B (5x6) into B_core (4x6) and B_tail (1x6)
    # AB = A_core * B_core + A_tail * B_tail^T
    
    # A_core * B_core is 4x4 x 4x6
    # Split B_core into B1 (4x4) and B2 (4x2)
    # A_core * B1: 4x4x4, Strassen rank 49
    # A_core * B2: 4x4x2, rank 28
    # A_tail * B_tail^T: 4x1x6, rank 4
    
    # Total: 49 + 28 + 4 = 81
    
    try:
        U_list = []
        V_list = []
        W_list = []
        
        # Helper: add a term
        def add_term(u_entries, v_entries, w_entries):
            """u_entries: dict mapping (i,j) -> coeff
               v_entries: dict mapping (j,k) -> coeff
               w_entries: dict mapping (k,i) -> coeff"""
            u = np.zeros(nm)
            v = np.zeros(mp)
            w = np.zeros(pn)
            for (i, j), c in u_entries.items():
                u[i * m + j] = c
            for (j, k), c in v_entries.items():
                v[j * p + k] = c
            for (k, i), c in w_entries.items():
                w[k * n + i] = c
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)
        
        # 2x2x2 Strassen terms (for local 2x2 blocks)
        # Local indices: A has (0,0),(0,1),(1,0),(1,1)
        # B has (0,0),(0,1),(1,0),(1,1)
        # C has (0,0),(0,1),(1,0),(1,1)
        
        # For a 2x2x2 block at position (i0, j0, k0) in the larger matrix
        # where i0 is the row block start, j0 is the col/row block start, k0 is the output col block start
        
        # Strassen 2x2x2 terms:
        # m1: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]
        # m2: U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,0]
        # m3: U=[1,0,0,0], V=[0,1,0,-1], W=[0,1,0,0]
        # m4: U=[0,0,0,1], V=[0,0,1,-1], W=[1,0,1,0]
        # m5: U=[1,1,0,0], V=[0,0,0,1], W=[-1,1,0,0]
        # m6: U=[-1,0,1,0], V=[1,1,0,0], W=[0,0,0,1]
        # m7: U=[0,1,0,-1], V=[0,0,1,1], W=[1,0,0,1]
        
        strassen_2x2x2 = [
            # (U_coeffs, V_coeffs, W_coeffs) for local 2x2 indices
            # U: (0,0),(0,1),(1,0),(1,1)
            # V: (0,0),(0,1),(1,0),(1,1)
            # W: (0,0),(0,1),(1,0),(1,1)
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),  # m1
            ([0,0,1,1], [1,0,0,0], [0,0,1,0]),  # m2
            ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),  # m3
            ([0,0,0,1], [0,0,1,-1], [1,0,1,0]),  # m4
            ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),  # m5
            ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),  # m6
            ([0,1,0,-1], [0,0,1,1], [1,0,0,1]),  # m7
        ]
        
        def add_strassen_2x2x2(i0, j0, k0, scale_u=1, scale_v=1, scale_w=1):
            """Add a 2x2x2 Strassen decomposition at position (i0, j0, k0)
            i0: starting row index in A (and C)
            j0: starting col index in A / row index in B
            k0: starting col index in B (and C)
            """
            for u_c, v_c, w_c in strassen_2x2x2:
                u_entries = {}
                v_entries = {}
                w_entries = {}
                # Map local 2x2 indices to global
                # A local (li, lj) -> global (i0+li, j0+lj)
                # B local (lj, lk) -> global (j0+lj, k0+lk)
                # C local (li, lk) -> global (i0+li, k0+lk)
                for li in range(2):
                    for lj in range(2):
                        if u_c[li*2+lj] != 0:
                            gi = i0 + li
                            gj = j0 + lj
                            if gi < n and gj < m:
                                u_entries[(gi, gj)] = u_c[li*2+lj] * scale_u
                for lj in range(2):
                    for lk in range(2):
                        if v_c[lj*2+lk] != 0:
                            gj = j0 + lj
                            gk = k0 + lk
                            if gj < m and gk < p:
                                v_entries[(gj, gk)] = v_c[lj*2+lk] * scale_v
                for li in range(2):
                    for lk in range(2):
                        if w_c[li*2+lk] != 0:
                            gi = i0 + li
                            gk = k0 + lk
                            if gi < n and gk < p:
                                w_entries[(gk, gi)] = w_c[li*2+lk] * scale_w
                add_term(u_entries, v_entries, w_entries)
        
        # 4x4x4 Strassen: split into 2x2 blocks
        # A_core is 4x4, split into 4 2x2 blocks:
        # A00 at (0,0), A01 at (0,2), A10 at (2,0), A11 at (2,2)
        # B1 is 4x4, split similarly
        # C is 4x4, split similarly
        
        # Strassen on 4x4x4:
        # m1 = (A00+A11)(B00+B11): 2x2x2 at various positions
        # This requires summing 2x2 matrices before multiplying
        
        # Actually, for the 4x4x4 Strassen, I need to:
        # 1. Form the sums/differences of 2x2 blocks
        # 2. Multiply them using 2x2x2 Strassen
        # 3. Combine results
        
        # This is complex. Let me use a different approach.
        # I'll directly enumerate the 49 rank-1 terms for 4x4x4.
        
        # Alternative: just use the schoolbook for 4x4x4 (rank 16) 
        # and 4x4x2 (rank 8), total 24, plus 4 for the tail = 28.
        # That's much simpler and gives rank 28 for the core + 4 = 32 total.
        # Wait, that's not right either.
        
        # Let me just use schoolbook for everything and see if I can improve.
        # Schoolbook for 4x4x4: rank 16
        # Schoolbook for 4x4x2: rank 8
        # Schoolbook for 4x1x6: rank 4
        # Total: 28
        
        # That's already much better than 120!
        
        # Let me verify: 
        # A_core (4x4) * B1 (4x4): 4*4*4 = 64 products, rank 16 with schoolbook? 
        # No! Schoolbook for 4x4x4 has rank 4*4*4 = 64, not 16.
        # I was confusing with the number of 2x2 blocks.
        
        # OK so schoolbook for 4x4x4 is rank 64.
        # Strassen gives rank 49.
        
        # Let me just try to get the Strassen 4x4x4 working.
        
        # For 4x4x4 Strassen:
        # Split A into 2x2 blocks: A = [[A00, A01], [A10, A11]]
        # Split B into 2x2 blocks: B = [[B00, B01], [B10, B11]]
        # 
        # m1 = (A00+A11) * (B00+B11)  [2x2x2, rank 7]
        # m2 = (A10+A11) * B00        [2x2x2, rank 7]
        # m3 = A00 * (B01-B11)        [2x2x2, rank 7]
        # m4 = A11 * (B10-B00)        [2x2x2, rank 7]
        # m5 = (A00+A01) * B11        [2x2x2, rank 7]
        # m6 = (A10-A00) * (B00+B01)  [2x2x2, rank 7]
        # m7 = (A01-A11) * (B10+B11)  [2x2x2, rank 7]
        
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        # Each mj is a 2x2x2 multiplication, which itself has 7 rank-1 terms.
        # So total 7*7 = 49 rank-1 terms.
        
        # To implement this, I need to:
        # 1. For each mj, determine the linear combinations of A and B blocks
        # 2. Apply the 2x2x2 Strassen decomposition
        # 3. Track how the results combine into C
        
        # This is very involved. Let me try a simpler approach first.
        
        # Simple approach: just use schoolbook for the whole thing but try to find
        # a better rank using ALS.
        
        pass  # Skip the complex Strassen construction for now
        
    except Exception as e:
        print(f"Strassen construction failed: {e}")
    
    # Use ALS to try to find a better rank
    np.random.seed(42)
    
    def check_solution(U, V, W, tol=1e-9):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    max_err = max(max_err, abs(val - 1.0))
        return max_err < tol
    
    def get_max_error(U, V, W):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    max_err = max(max_err, abs(val - 1.0))
        return max_err
    
    def als_solve(U, V, W, rank, max_iters=100, tol=1e-12):
        prev_err = float('inf')
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Fix V and W, solve for U (vectorized)
            for i in range(n):
                for j in range(m):
                    A_mat = np.array([V[j * p + k, :] * W[k * n + i, :] for k in range(p)])
                    b = np.ones(p)
                    try:
                        x, _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                        U[i * m + j, :] = x
                    except:
                        pass
            
            # Fix U and W, solve for V
            for j in range(m):
                for k in range(p):
                    A_mat = np.array([U[i * m + j, :] * W[k * n + i, :] for i in range(n)])
                    b = np.ones(n)
                    try:
                        x, _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                        V[j * p + k, :] = x
                    except:
                        pass
            
            # Fix U and V, solve for W
            for k in range(p):
                for i in range(n):
                    A_mat = np.array([U[i * m + j, :] * V[j * p + k, :] for j in range(m)])
                    b = np.ones(m)
                    try:
                        x, _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                        W[k * n + i, :] = x
                    except:
                        pass
            
            err = get_max_error(U, V, W)
            if err < tol:
                return True
            if abs(prev_err - err) < 1e-14:
                U += np.random.randn(nm, rank) * 0.01
                V += np.random.randn(mp, rank) * 0.01
                W += np.random.randn(pn, rank) * 0.01
            prev_err = err
        
        return False
    
    def try_exact_recovery(U, V, W, rank):
        for scale in [2, 4, 8, 16]:
            U_round = np.round(U * scale) / scale
            V_round = np.round(V * scale) / scale
            W_round = np.round(W * scale) / scale
            if check_solution(U_round, V_round, W_round, 1e-10):
                return U_round, V_round, W_round
        return None
    
    # Try ranks from 70 to 119
    for rank in range(70, 120):
        if time.time() > deadline - 30:
            break
        
        for init_scale in [0.3, 0.6, 1.0]:
            if time.time() > deadline - 10:
                break
            
            U = np.random.randn(nm, rank) * init_scale
            V = np.random.randn(mp, rank) * init_scale
            W = np.random.randn(pn, rank) * init_scale
            
            if als_solve(U, V, W, rank):
                result = try_exact_recovery(U, V, W, rank)
                if result is not None:
                    U_round, V_round, W_round = result
                    best_rank = rank
                    best_result = {
                        "U": U_round.tolist(),
                        "V": V_round.tolist(),
                        "W": W_round.tolist()
                    }
                    break
        
        if best_rank < nm * p:
            break
    
    return best_result
