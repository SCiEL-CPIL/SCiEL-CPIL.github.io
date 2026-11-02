"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    nm, mp, pn = n*m, m*p, p*n  # 4, 10, 10
    
    # Build support set
    support = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                support.append((a, b, c))
    support_set = set(support)
    
    # Schoolbook decomposition (rank = n*m*p = 20)
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    for idx, (a, b, c) in enumerate(support):
        U_school[a, idx] = 1.0
        V_school[b, idx] = 1.0
        W_school[c, idx] = 1.0
    
    def to_list(arr):
        return arr.tolist()
    
    def compute_error(U, V, W):
        max_err = 0.0
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if (a, b, c) in support_set:
                        err = abs(val - 1.0)
                    else:
                        err = abs(val)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def build_rank18():
        """Construct a rank-18 decomposition using Strassen-like structure."""
        r = 18
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        
        # Strassen's algorithm for 2x2 matrix multiplication uses rank 7
        # For 2x2 x 2x5, we can use block structure
        # Split the 2x5 matrix into two 2x2 blocks and one 2x1 block
        
        # Indices:
        # U rows: (i,j) where i in {0,1}, j in {0,1} -> 0,1,2,3
        # V rows: (j,k) where j in {0,1}, k in {0,1,2,3,4} -> 0..9
        # W rows: (k,i) where k in {0,1,2,3,4}, i in {0,1} -> 0..9
        
        # Use 7 Strassen terms for each 2x2 block pair, plus adjustments
        # This gives us a structured rank-18 decomposition
        
        # For the first 2x2 x 2x2 block (k=0,1):
        # Strassen formulas for C = A * B where A,B are 2x2
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-e)
        # M5 = (a+b)h, M6 = (c-d)(g+h), M7 = (b-d)(e+f)
        
        # Term 1: M1 = (a+d)(e+h)
        U[0, 0] = 1; U[2, 0] = 1  # a, d
        V[0, 0] = 1; V[1, 0] = 1  # e, h (k=0,1)
        W[0, 0] = 1; W[1, 0] = 1  # (k,i) for i=0,1
        
        # Term 2: M2 = (c+d)e
        U[2, 1] = 1; U[3, 1] = 1  # c, d
        V[0, 1] = 1  # e
        W[0, 1] = 1  # k=0, i=0
        
        # Term 3: M3 = a(f-h)
        U[0, 2] = 1  # a
        V[1, 2] = 1; V[3, 2] = -1  # f (k=1), -h... wait, need to be careful
        
        # Let me use a different approach: direct construction with known rank-18
        # For 2x2 x 2x5, we can achieve rank 18
        
        # Actually, let me just use a well-known construction
        # The tensor rank of [2,2,5] is known to be 18
        
        # I'll construct it directly
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        
        # Terms 0-6: Strassen for first 2x2 x 2x2 (k=0,1)
        # A = [[A00, A01], [A10, A11]], B1 = [[B00, B01], [B10, B11]]
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[1, 0] = 1
        W[0, 0] = 1; W[2, 0] = 1  # k*n+i: (0,0)=0, (1,0)=2
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[0, 1] = 1
        
        # M3 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1  # B01 at k=0,i=1 -> V[1], B11 at k=1,i=1 -> V[3]
        W[1, 2] = 1  # k=0, i=1 -> 0*2+1=1
        
        # M4 = A11(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1  # B10 at k=1,i=0 -> V[2], B00 at k=0,i=0 -> V[0]
        W[2, 3] = 1  # k=1, i=0 -> 1*2+0=2
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[3, 4] = 1  # k=1, i=1 -> 1*2+1=3
        
        # M6 = (A10-A11)(B10+B11)
        U[2, 5] = 1; U[3, 5] = -1
        V[2, 5] = 1; V[3, 5] = 1
        W[0, 5] = 1; W[2, 5] = 1
        
        # M7 = (A01-A11)(B00+B01)
        U[1, 6] = 1; U[3, 6] = -1
        V[0, 6] = 1; V[1, 6] = 1
        W[1, 6] = 1; W[3, 6] = 1
        
        # Terms 7-13: Strassen for second 2x2 x 2x2 (k=2,3)
        # B2 = [[B20, B21], [B30, B31]]
        
        # M8 = (A00+A11)(B20+B31)
        U[0, 7] = 1; U[3, 7] = 1
        V[4, 7] = 1; V[5, 7] = 1
        W[4, 7] = 1; W[6, 7] = 1  # k=2,i=0 -> 4, k=3,i=0 -> 6
        
        # M9 = (A10+A11)B20
        U[2, 8] = 1; U[3, 8] = 1
        V[4, 8] = 1
        W[4, 8] = 1
        
        # M10 = A00(B21-B31)
        U[0, 9] = 1
        V[5, 9] = 1; V[7, 9] = -1
        W[5, 9] = 1  # k=2, i=1 -> 5
        
        # M11 = A11(B30-B20)
        U[3, 10] = 1
        V[6, 10] = 1; V[4, 10] = -1
        W[6, 10] = 1  # k=3, i=0 -> 6
        
        # M12 = (A00+A01)B31
        U[0, 11] = 1; U[1, 11] = 1
        V[7, 11] = 1
        W[7, 11] = 1  # k=3, i=1 -> 7
        
        # M13 = (A10-A11)(B30+B31)
        U[2, 12] = 1; U[3, 12] = -1
        V[6, 12] = 1; V[7, 12] = 1
        W[4, 12] = 1; W[6, 12] = 1
        
        # M14 = (A01-A11)(B20+B21)
        U[1, 13] = 1; U[3, 13] = -1
        V[4, 13] = 1; V[5, 13] = 1
        W[5, 13] = 1; W[7, 13] = 1
        
        # Terms 14-17: For the last 2x1 block (k=4)
        # B3 = [[B40], [B41]] (2x1)
        # C contribution: [[A00*B40 + A01*B41], [A10*B40 + A11*B41]]
        
        # M15 = A00*B40
        U[0, 14] = 1
        V[8, 14] = 1
        W[8, 14] = 1  # k=4, i=0 -> 8
        
        # M16 = A01*B40
        U[1, 15] = 1
        V[8, 15] = 1
        W[8, 15] = 1
        
        # M17 = A10*B40
        U[2, 16] = 1
        V[8, 16] = 1
        W[8, 16] = 1
        
        # M18 = A11*B40
        U[3, 17] = 1
        V[8, 17] = 1
        W[8, 17] = 1
        
        # Wait, I need to handle B41 as well
        # Let me reconsider: for k=4, we have B40 and B41
        # C[0,0] += A00*B40 + A01*B41
        # C[1,0] += A10*B40 + A11*B41
        
        # This needs 4 more terms, but we only have 4 slots left
        # Let me use a different approach for the last block
        
        # Actually, let me just verify the current construction
        # and if it doesn't work, fall back to search
        
        return U, V, W
    
    def try_rank_als(r, num_attempts, time_budget, init=None):
        """Try to find a rank-r decomposition using ALS."""
        deadline = time.time() + time_budget
        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None
            
            if init is not None and attempt == 0:
                U, V, W = init.copy(), init[1].copy(), init[2].copy()
            else:
                np.random.seed(12345 + r * 1000 + attempt)
                U = np.random.randn(nm, r)
                V = np.random.randn(mp, r)
                W = np.random.randn(pn, r)
            
            # Alternating minimization with better convergence
            for iteration in range(1000):
                if time.time() > deadline:
                    return None
                
                # Update U
                for a in range(nm):
                    rows = []
                    targets = []
                    for (a2, b, c) in support:
                        if a2 == a:
                            rows.append(V[b, :] * W[c, :])
                            targets.append(1.0)
                    for b in range(mp):
                        for c in range(pn):
                            if (a, b, c) not in support_set:
                                rows.append(V[b, :] * W[c, :])
                                targets.append(0.0)
                    if rows:
                        A_mat = np.array(rows)
                        b_vec = np.array(targets)
                        sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        U[a, :] = sol
                
                # Update V
                for b in range(mp):
                    rows = []
                    targets = []
                    for (a, b2, c) in support:
                        if b2 == b:
                            rows.append(U[a, :] * W[c, :])
                            targets.append(1.0)
                    for a in range(nm):
                        for c in range(pn):
                            if (a, b, c) not in support_set:
                                rows.append(U[a, :] * W[c, :])
                                targets.append(0.0)
                    if rows:
                        A_mat = np.array(rows)
                        b_vec = np.array(targets)
                        sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        V[b, :] = sol
                
                # Update W
                for c in range(pn):
                    rows = []
                    targets = []
                    for (a, b, c2) in support:
                        if c2 == c:
                            rows.append(U[a, :] * V[b, :])
                            targets.append(1.0)
                    for a in range(nm):
                        for b in range(mp):
                            if (a, b, c) not in support_set:
                                rows.append(U[a, :] * V[b, :])
                                targets.append(0.0)
                    if rows:
                        A_mat = np.array(rows)
                        b_vec = np.array(targets)
                        sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        W[c, :] = sol
                
                # Check convergence
                if iteration % 5 == 0:
                    err = compute_error(U, V, W)
                    if err < 1e-12:
                        return U, V, W
        
        return None
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    deadline = time.time() + 130
    
    # First, try to construct rank-18
    U18, V18, W18 = build_rank18()
    err18 = compute_error(U18, V18, W18)
    if err18 < 1e-10:
        best_rank = 18
        best_U, best_V, best_W = U18, V18, W18
    
    # Try to improve to rank 17
    if time.time() < deadline - 20:
        remaining = deadline - 20 - time.time()
        if remaining > 10:
            # Try ALS from rank-18 initialization (truncated)
            r_try = 17
            U_init = U18[:, :r_try] if best_rank == 18 else None
            V_init = V18[:, :r_try] if best_rank == 18 else None
            W_init = W18[:, :r_try] if best_rank == 18 else None
            
            if U_init is not None:
                result = try_rank_als(r_try, 5, remaining * 0.3, init=(U_init, V_init, W_init))
                if result is not None:
                    best_rank = r_try
                    best_U, best_V, best_W = result
            
            # Try random initializations for rank 17
            if best_rank > 17 and time.time() < deadline - 10:
                remaining = deadline - 10 - time.time()
                if remaining > 5:
                    result = try_rank_als(17, 20, remaining)
                    if result is not None:
                        best_rank = 17
                        best_U, best_V, best_W = result
    
    # Try rank 16 if time allows
    if best_rank > 16 and time.time() < deadline - 10:
        remaining = deadline - 10 - time.time()
        if remaining > 5:
            result = try_rank_als(16, 10, remaining)
            if result is not None:
                best_rank = 16
                best_U, best_V, best_W = result
    
    if best_rank < rank_school:
        return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
    else:
        return {"U": to_list(U_school), "V": to_list(V_school), "W": to_list(W_school)}
# EVOLVE-BLOCK-END
