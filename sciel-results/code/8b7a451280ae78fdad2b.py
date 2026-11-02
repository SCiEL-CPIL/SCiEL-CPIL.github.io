import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145

    n, m, p = 2, 2, 13

    # Schoolbook fallback (Rank 52)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U_school[i * m + k, col] = 1.0
                V_school[k * p + j, col] = 1.0
                W_school[j * n + i, col] = 1.0
                col += 1

    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school

    # Strassen terms for 2x2x2x2
    strassen_terms = [
        (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),
        (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1])),
        (np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1])),
        (np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0])),
        (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0])),
        (np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1])),
        (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0])),
    ]

    # Try to find rank-10 decomposition for 2x2 x 2x3
    def find_rank10_2x2x2x3():
        """Find rank-10 decomposition for 2x2 x 2x3 matrix multiplication."""
        # We'll use a numerical approach to solve the system
        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        
        # Nonzero entries of T:
        nonzero = [
            (0,0,0), (0,1,2), (0,2,4),
            (1,3,0), (1,4,2), (1,5,4),
            (2,0,1), (2,1,3), (2,2,5),
            (3,3,1), (3,4,3), (3,5,5)
        ]
        
        # All possible (a,b,c) triples:
        all_triples = [(a,b,c) for a in range(4) for b in range(6) for c in range(6)]
        
        # We want to find U (4x10), V (6x10), W (6x10)
        # This is a nonlinear system. Let's try to solve it.
        
        # Strategy: fix W, solve for U and V
        # For fixed W, the equations are:
        # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all (a,b,c)
        
        # This is linear in the products U[a,r]*V[b,r].
        # Let X[a,b,r] = U[a,r]*V[b,r]
        # Then sum_r X[a,b,r]*W[c,r] = T[a,b,c]
        
        # For each (a,b), this gives: sum_r X[a,b,r]*W[c,r] = T[a,b,c] for c=0..5
        # i.e., X[a,b,:] @ W.T = T[a,b,:]
        
        # But X[a,b,r] = U[a,r]*V[b,r] is nonlinear.
        
        # Let's try a different approach: use the fact that we can write
        # the tensor as a sum of rank-1 tensors and solve for the coefficients.
        
        # I'll use a simple iterative method:
        # 1. Initialize W randomly
        # 2. For each (a,b), solve for X[a,b,:] from X[a,b,:] @ W.T = T[a,b,:]
        # 3. Recover U and V from X
        
        # But step 3 is nonlinear. Let's try a different approach.
        
        # Actually, let me just try to find a known decomposition.
        # After some research, the rank of 2x2 x 2x3 is indeed 10.
        # Here's a construction:
        
        # I'll construct it by solving a system of equations.
        # Let's assume the decomposition has the following structure:
        # For r=0..9, we have (u_r, v_r, w_r)
        
        # I'll use the following ansatz:
        # The k=0 and k=1 parts are coupled through the A matrix.
        
        # Let me try:
        # r=0: u=[1,0,0,0], v=[1,0,0,0,0,0], w=[1,0,0,0,0,0]  -> A[0,0]*B[0,0]->C[0,0]
        # r=1: u=[1,0,0,0], v=[0,1,0,0,0,0], w=[0,0,1,0,0,0]  -> A[0,0]*B[0,1]->C[0,1]
        # r=2: u=[1,0,0,0], v=[0,0,1,0,0,0], w=[0,0,0,0,1,0]  -> A[0,0]*B[0,2]->C[0,2]
        # r=3: u=[0,0,1,0], v=[1,0,0,0,0,0], w=[0,1,0,0,0,0]  -> A[1,0]*B[0,0]->C[1,0]
        # r=4: u=[0,0,1,0], v=[0,1,0,0,0,0], w=[0,0,0,1,0,0]  -> A[1,0]*B[0,1]->C[1,1]
        # r=5: u=[0,0,1,0], v=[0,0,1,0,0,0], w=[0,0,0,0,0,1]  -> A[1,0]*B[0,2]->C[1,2]
        # r=6: u=[0,1,0,0], v=[0,0,0,1,0,0], w=[1,0,0,0,0,0]  -> A[0,1]*B[1,0]->C[0,0]
        # r=7: u=[0,1,0,0], v=[0,0,0,0,1,0], w=[0,0,1,0,0,0]  -> A[0,1]*B[1,1]->C[0,1]
        # r=8: u=[0,1,0,0], v=[0,0,0,0,0,1], w=[0,0,0,0,1,0]  -> A[0,1]*B[1,2]->C[0,2]
        # r=9: u=[0,0,0,1], v=[0,0,0,1,0,0], w=[0,1,0,0,0,0]  -> A[1,1]*B[1,0]->C[1,0]
        
        # But this only covers 10 of the 12 nonzero entries. We're missing:
        # (1,4,2): A[0,1]*B[1,1]->C[0,1] - covered by r=7
        # (1,5,4): A[0,1]*B[1,2]->C[0,2] - covered by r=8
        # (3,4,3): A[1,1]*B[1,1]->C[1,1] - NOT covered
        # (3,5,5): A[1,1]*B[1,2]->C[1,2] - NOT covered
        
        # So we need to add terms for A[1,1]*B[1,1] and A[1,1]*B[1,2].
        # But we only have 10 terms total. We need to modify the above to include these.
        
        # Let me try a different construction. I'll use the following 10 terms:
        
        # Actually, let me just solve the system numerically.
        
        # I'll use the following approach:
        # 1. Set W to a random matrix
        # 2. Solve for U and V
        # 3. Check if the solution is valid
        
        # For fixed W, the problem is:
        # Find U (4x10) and V (6x10) such that
        # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all (a,b,c)
        
        # This is equivalent to: for each r, U[:,r]*V[:,r] is a 4x6 matrix,
        # and sum_r (U[:,r]*V[:,r]) * W[c,r] = T[:,:,c] for each c
        
        # Let M_c = T[:,:,c] (4x6 matrix for each c)
        # Then sum_r (U[:,r]*V[:,r]) * W[c,r] = M_c
        
        # This is: sum_r W[c,r] * (U[:,r] * V[:,r]) = M_c
        # where U[:,r]*V[:,r] is the outer product of U[:,r] and V[:,r]
        
        # Let P_r = U[:,r] * V[:,r] (4x6 matrix)
        # Then sum_r W[c,r] * P_r = M_c for c=0..5
        
        # This is a linear system in P_r. But P_r = U[:,r]*V[:,r] is nonlinear.
        
        # Let me try a different approach: use the fact that we can fix U and solve for V.
        
        # For fixed U and W, the equations are:
        # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
        # For fixed (a,c): sum_r (U[a,r]*W[c,r])*V[b,r] = T[a,b,c] for all b
        # i.e., V @ (U[a,:]*W[c,:])^T = T[a,:,c]
        
        # This is: V @ D_{a,c} = T[a,:,c] where D_{a,c} = U[a,:]*W[c,:] (10-vector)
        
        # So V @ [D_{a,0}, D_{a,1}, ..., D_{a,5}] = T[a,:,:]
        
        # This is a linear system in V. Let's set it up.
        
        # For fixed U and W:
        # Let A_mat be a 6x60 matrix where A_mat[b, a*6+c] = U[a,r]*W[c,r] summed over r
        # Wait, this is getting too complicated.
        
        # Let me just try a simple numerical solver.
        
        # I'll use scipy.optimize or a simple iterative method.
        
        # Actually, let me try to find the decomposition by hand.
        
        # Here's a known rank-10 decomposition for 2x2 x 2x3:
        # (I'm constructing this based on the structure of the tensor)
        
        # Let me denote the entries of A as a00, a01, a10, a11
        # B as b00, b01, b02, b10, b11, b12
        # C as c00, c01, c02, c10, c11, c12
        
        # The equations are:
        # c00 = a00*b00 + a01*b10
        # c01 = a00*b01 + a01*b11
        # c02 = a00*b02 + a01*b12
        # c10 = a10*b00 + a11*b10
        # c11 = a10*b01 + a11*b11
        # c12 = a10*b02 + a11*b12
        
        # We need to express these as sum of 10 products.
        
        # Let me try:
        # m1 = (a00+a10)*(b00+b10) = a00*b00 + a00*b10 + a10*b00 + a10*b10
        # m2 = (a00+a10)*(b01+b11) = a00*b01 + a00*b11 + a10*b01 + a10*b11
        # m3 = (a00+a10)*(b02+b12) = a00*b02 + a00*b12 + a10*b02 + a10*b12
        # m4 = (a01+a11)*(b00+b10) = a01*b00 + a01*b10 + a11*b00 + a11*b10
        # m5 = (a01+a11)*(b01+b11) = a01*b01 + a01*b11 + a11*b01 + a11*b11
        # m6 = (a01+a11)*(b02+b12) = a01*b02 + a01*b12 + a11*b02 + a11*b12
        # m7 = a00*b10
        # m8 = a00*b11
        # m9 = a00*b12
        # m10 = a10*b10
        
        # This doesn't seem to work directly.
        
        # Let me try a different approach. I'll just return the rank-46 construction
        # and note that finding a rank-45 construction requires finding a rank-10
        # decomposition of 2x2 x 2x3, which is non-trivial.
        
        return None

    # Build Strassen-based (rank 46)
    try:
        rank_strassen = 6 * 7 + 4
        U = np.zeros((n * m, rank_strassen))
        V = np.zeros((m * p, rank_strassen))
        W = np.zeros((p * n, rank_strassen))

        col = 0
        for block in range(6):
            j0 = block * 2
            j1 = block * 2 + 1

            for t in range(7):
                u_local, v_local, w_local = strassen_terms[t]

                U[0, col] = u_local[0]
                U[1, col] = u_local[1]
                U[2, col] = u_local[2]
                U[3, col] = u_local[3]

                V[0 * p + j0, col] = v_local[0]
                V[0 * p + j1, col] = v_local[1]
                V[1 * p + j0, col] = v_local[2]
                V[1 * p + j1, col] = v_local[3]

                W[j0 * n + 0, col] = w_local[0]
                W[j0 * n + 1, col] = w_local[1]
                W[j1 * n + 0, col] = w_local[2]
                W[j1 * n + 1, col] = w_local[3]

                col += 1

        # Last column (j=12): schoolbook 2x2 x 2x1, rank 4
        j_last = 12
        for i in range(n):
            for k in range(m):
                U[i * m + k, col] = 1.0
                V[k * p + j_last, col] = 1.0
                W[j_last * n + i, col] = 1.0
                col += 1

        # Verify correctness
        T_target = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    T_target[i * m + k, k * p + j, j * n + i] = 1.0

        T_approx = np.einsum('it,jt,kt->ijk', U, V, W)
        error = np.max(np.abs(T_approx - T_target))

        if error < 1e-10:
            best_rank = rank_strassen
            best_U, best_V, best_W = U, V, W

    except Exception:
        pass

    if time.time() > deadline:
        pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
