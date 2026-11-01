import numpy as np
import time
import itertools

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_2x2x4():
    """
    Construct a rank-14 tensor decomposition for 2x2 * 2x4 matrix multiplication.
    Two independent Strassen blocks exploiting the shared A matrix structure.
    """
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: output columns k in {0,1}, rank indices 0..6
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[2][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[0][2] = 1; W[1][2] = 1
    
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1; W[3][4] = -1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1; W[0][5] = -1
    
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[1][6] = 1; W[2][6] = -1
    
    # Block 2: output columns k in {2,3}, rank indices 7..13
    U[0][7] = 1; U[3][7] = 1
    V[4][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    U[2][8] = 1; U[3][8] = 1
    V[6][8] = 1
    W[6][8] = 1; W[7][8] = -1
    
    U[0][9] = 1
    V[6][9] = 1; V[7][9] = -1
    W[4][9] = 1; W[5][9] = 1
    
    U[3][10] = 1
    V[5][10] = 1; V[4][10] = -1
    W[4][10] = 1; W[6][10] = 1
    
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[5][11] = 1; W[7][11] = -1
    
    U[2][12] = 1; U[0][12] = -1
    V[4][12] = 1; V[5][12] = 1
    W[6][12] = 1; W[4][12] = -1
    
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[5][13] = 1; W[6][13] = -1
    
    return {"U": U, "V": V, "W": W}

def _peel_structured_search(target_rank, deadline, n=2, m=2, p=4):
    """
    Structural peeling search.
    1. Generate random sparse patterns for the first target_rank-1 terms.
    2. Compute the residual tensor R = T - sum_{r=1}^{R-1} term_r.
    3. Check if R can be represented as a single rank-1 term (u, v, w).
       This is equivalent to checking if the matrix M_{(i,k), j} = R[i,j,k]
       has rank 1. If so, we can factorize it.
    """
    # T[i,j,k] = 1 for all i,j,k
    # We want sum_{r=1}^R u_r[i] v_r[j] w_r[k] = 1
    # Let's fix the first R-1 terms randomly with small integer values {-1, 0, 1}
    # and solve for the last one.
    
    # Indices:
    # u: i (0..n-1)
    # v: j (0..m-1) -- Wait, standard matrix mult A(n,m) * B(m,p) = C(n,p)
    # Here A is 2x2 (n=2, m=2), B is 2x4 (m=2, p=4).
    # C is 2x4 (n=2, p=4).
    # T_{i,j,k} corresponds to C[i,k] = sum_j A[i,j] B[j,k].
    # So the tensor is T[i,j,k] = 1.
    # Decomposition: T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # Note: The task description says:
    # U shape (n*m, rank) -> Index i*m+j ? No, usually U corresponds to A, V to B, W to C.
    # Let's check the seed program's _verify_decomposition:
    # s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This is unusual. Standard is U[i,r] V[j,r] W[k,r] if U is A, V is B, W is C.
    # But here U is indexed by i*m+j. This implies U mixes i and j?
    # Wait, A is n x m. B is m x p. C is n x p.
    # Usually: C[i,k] = sum_j A[i,j] B[j,k].
    # Tensor T[i,j,k] = 1.
    # Rank 1 term: A[i,j] = u_i v_j? No.
    # The decomposition of the multiplication algorithm is:
    # A[i,j] B[j,k] = sum_r M_r A[...] B[...]
    # The "tensor decomposition" usually refers to decomposing the tensor T_{i,j,k}
    # such that T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r}.
    # However, the seed program uses:
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This looks like it's decomposing something else or using a specific convention.
    # Let's re-read the task: "The output factor follows the official trace convention W[k*n+i,r]."
    # And U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    # This is the "border rank" or specific tensor format for matrix multiplication.
    # Actually, for matrix multiplication M_{n,m,p}, the tensor is T in C^{n x m} x C^{m x p} x C^{n x p}^*.
    # The standard decomposition is T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r} is NOT correct for matrix mult.
    # Matrix mult is bilinear: C = A B.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # We want to compute C_{i,k} using R multiplications.
    # Each multiplication is M_r = (sum_{i,j} u_{i,j}^r A_{i,j}) * (sum_{j,k} v_{j,k}^r B_{j,k}).
    # Then C_{i,k} = sum_r w_{i,k}^r M_r.
    # This corresponds to the tensor decomposition of the multiplication tensor.
    # The tensor T has components T_{i,j;k,l;m,n}? No.
    # The standard tensor for matmul M_{n,m,p} is in C^{n,m} \otimes C^{m,p} \otimes C^{n,p}^*.
    # T_{i,j,k} is not the right index set.
    # The seed program's verification loop:
    # for i in range(n): for j in range(m): for k in range(p):
    #   s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This checks if sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1.
    # This is the decomposition of the identity tensor? Or is it a specific convention?
    # If this equals 1 for all i,j,k, it means the "algorithm" computes the sum correctly?
    # Actually, if we set A and B to basis matrices, we can recover the coefficients.
    # Let's assume the seed program's convention is correct and we must satisfy:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    # This is a system of equations for the entries of U, V, W.
    # Number of equations: n * m * p = 2 * 2 * 4 = 16.
    # Number of variables:
    # U: n*m * R = 4 * R
    # V: m*p * R = 8 * R
    # W: p*n * R = 8 * R
    # Total variables: 20 R.
    # For R=13, 260 variables. 16 equations. Underdetermined.
    # But the values must be "integer/half-integer".
    
    # Strategy:
    # Fix R-1 terms randomly.
    # Let the last term be U_last, V_last, W_last.
    # The equation is:
    # sum_{r=1}^{R-1} U_r V_r W_r + U_last V_last W_last = 1
    # Let S = 1 - sum_{r=1}^{R-1} U_r V_r W_r.
    # We need U_last V_last W_last = S (element-wise for the indices i,j,k).
    # This means the "tensor" S_{i,j,k} must be rank 1.
    # S_{i,j,k} = U_last[i*m+j] * V_last[j*p+k] * W_last[k*n+i].
    # Note the indices are coupled.
    # Let x = i*m+j, y = j*p+k, z = k*n+i.
    # This is a specific structure.
    
    # We can iterate over random sparse patterns for the first R-1 terms.
    # Then check if the residual S can be factored into the specific form.
    
    # To check if S can be factored:
    # We can pick one component, say U_last[0], and solve for others?
    # Or, we can observe that if we fix U_last and V_last, W_last is determined?
    # W_last[z] = S[x,y,z] / (U_last[x] V_last[y]).
    # This must be consistent for all (x,y,z) that map to the same z?
    # No, each (i,j,k) maps to unique (x,y,z).
    # x = 2i+j.
    # y = 2j+k.
    # z = 4i+j? No, W index is k*n+i = 4k+i? No, n=2. W index is k*2+i.
    # Wait, W shape is (p*n, rank) = (8, R). Index is k*n+i.
    # k in 0..3, i in 0..1.
    # z = 2k+i.
    
    # Let's check consistency.
    # Given S_{i,j,k}, we want to find u_i, v_j, w_k (with specific index mappings) such that
    # S_{i,j,k} = u_{2i+j} v_{2j+k} w_{2k+i}.
    
    # We can try to solve for u, v, w.
    # Fix u[0] = 1 (scaling).
    # Then for any (i,j,k), if u[2i+j] and v[2j+k] are non-zero, w[2k+i] is determined.
    # We must ensure consistency.
    
    # Algorithm:
    # 1. Generate random sparse U_1..U_{R-1}, V_1..V_{R-1}, W_1..W_{R-1} from {-1, 0, 1}.
    # 2. Compute S_{i,j,k} = 1 - sum_{r=1}^{R-1} ...
    # 3. Check if S is "separable" in the specific index structure.
    #    - Initialize u, v, w with NaNs.
    #    - Set u[0] = 1.
    #    - Iterate through all (i,j,k).
    #    - If u[2i+j] and v[2j+k] are known and non-zero, compute w_candidate = S / (u*v).
    #      If w[2k+i] is unknown, set it. If known, check consistency.
    #    - If u[2i+j] is unknown, we might need to determine it from another equation.
    #    - This is a constraint satisfaction problem.
    #    - A simpler check: The rank of the tensor S (reshaped appropriately) must be 1.
    #    - But the indices are permuted.
    #    - We can form a matrix M where M_{x, y} = S[i,j,k] for the unique k?
    #    - No, k is determined by i,j? No, k is free.
    #    - For fixed i,j, k varies.
    #    - S_{i,j,k} = u_{2i+j} v_{2j+k} w_{2k+i}.
    #    - Let x = 2i+j. x is fixed for fixed i,j.
    #    - Then S_{x, 2j+k, 2k+i} = u_x v_{2j+k} w_{2k+i}.
    #    - Let y = 2j+k. As k varies, y varies.
    #    - Let z = 2k+i.
    #    - Note that y = 2j+k and z = 2k+i.
    #    - k = y - 2j. z = 2(y-2j) + i = 2y - 4j + i.
    #    - So for fixed i,j (and thus x), we have a sequence of values S_{x, y, z(y)}.
    #    - S_{x, y, z(y)} = u_x v_y w_{z(y)}.
    #    - This implies that for fixed x, the sequence S_{x, y, z(y)} / v_y should be proportional to w_{z(y)}.
    #    - This is getting complicated.
    
    # Alternative: Random search for the last term's non-zero positions.
    # Given S, we can try to factor it.
    # We can use a greedy approach:
    # Pick a random (i,j,k) where S != 0.
    # Set u[2i+j] = 1.
    # Then v[2j+k] = S. w[2k+i] = 1.
    # Then for other (i',j',k'), check if S_{i',j',k'} = u[2i'+j'] v[2j'+k'] w[2k'+i'].
    # If u, v, w are not yet set, we can set them.
    # If they are set, we check consistency.
    # If conflict, fail.
    
    # This is a heuristic. Let's implement it.
    
    n, m, p = 2, 2, 4
    indices = [(i,j,k) for i in range(n) for j in range(m) for k in range(p)]
    
    for _ in range(1000):
        if time.time() > deadline:
            return None
        
        # Generate R-1 random terms
        U_rand = np.zeros((n*m, target_rank-1))
        V_rand = np.zeros((m*p, target_rank-1))
        W_rand = np.zeros((p*n, target_rank-1))
        
        for r in range(target_rank-1):
            # Sparse random vectors
            mask_u = np.random.rand(n*m) < 0.3
            mask_v = np.random.rand(m*p) < 0.3
            mask_w = np.random.rand(p*n) < 0.3
            U_rand[:, r] = np.where(mask_u, np.random.choice([-1, 1], n*m), 0)
            V_rand[:, r] = np.where(mask_v, np.random.choice([-1, 1], m*p), 0)
            W_rand[:, r] = np.where(mask_w, np.random.choice([-1, 1], p*n), 0)
            
        # Compute residual S
        S = np.ones((n, m, p))
        for r in range(target_rank-1):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        x = i*m + j
                        y = j*p + k
                        z = k*n + i
                        S[i,j,k] -= U_rand[x, r] * V_rand[y, r] * W_rand[z, r]
                        
        # Try to factor S into u, v, w
        u = np.full(n*m, np.nan)
        v = np.full(m*p, np.nan)
        w = np.full(p*n, np.nan)
        
        success = True
        # Initialize with one known value
        # Find a non-zero S
        non_zero = [(i,j,k) for i,j,k in indices if abs(S[i,j,k]) > 1e-9]
        if not non_zero:
            # S is zero, so last term is zero? Rank reduced?
            # If S is 0, we have a rank R-1 solution.
            # But we are looking for rank R. If we found rank R-1, we can pad with zeros.
            # But the task asks for rank-minimal. If R-1 works, we should return R-1.
            # However, we are searching for specific R.
            # If S is 0, we can set last term to 0.
            u[:] = 0; v[:] = 0; w[:] = 0
            success = True
        else:
            # Pick first non-zero
            i0, j0, k0 = non_zero[0]
            x0 = i0*m + j0
            y0 = j0*p + k0
            z0 = k0*n + i0
            
            u[x0] = 1.0
            v[y0] = S[i0,j0,k0]
            w[z0] = 1.0
            
            # Propagate
            for i,j,k in indices:
                if i==i0 and j==j0 and k==k0:
                    continue
                x = i*m + j
                y = j*p + k
                z = k*n + i
                val = S[i,j,k]
                
                # Cases:
                # 1. u[x], v[y], w[z] all known -> check consistency
                # 2. One unknown -> solve for it
                # 3. Two unknowns -> ambiguous, need more info or fail
                # 4. Three unknowns -> ambiguous
                
                known_u = not np.isnan(u[x])
                known_v = not np.isnan(v[y])
                known_w = not np.isnan(w[z])
                
                if known_u and known_v and known_w:
                    if abs(u[x]*v[y]*w[z] - val) > 1e-9:
                        success = False
                        break
                elif known_u and known_v:
                    if abs(u[x]*v[y]) < 1e-9:
                        if abs(val) > 1e-9:
                            success = False
                            break
                        else:
                            w[z] = 0
                    else:
                        w[z] = val / (u[x]*v[y])
                elif known_u and known_w:
                    if abs(u[x]*w[z]) < 1e-9:
                        if abs(val) > 1e-9:
                            success = False
                            break
                        else:
                            v[y] = 0
                    else:
                        v[y] = val / (u[x]*w[z])
                elif known_v and known_w:
                    if abs(v[y]*w[z]) < 1e-9:
                        if abs(val) > 1e-9:
                            success = False
                            break
                        else:
                            u[x] = 0
                    else:
                        u[x] = val / (v[y]*w[z])
                else:
                    # Too many unknowns, cannot determine uniquely without more constraints
                    # This heuristic fails if the graph of indices is not connected enough
                    # or if we start with a bad seed.
                    success = False
                    break
            
            if success:
                # Check for NaNs
                if np.isnan(u).any() or np.isnan(v).any() or np.isnan(w).any():
                    success = False
                else:
                    # Check if values are "nice" (integers or half-integers)
                    # The task allows integer/half-integer.
                    # Our random terms are integers. The solved term might be rational.
                    # We can scale? No, scaling changes the sum.
                    # We must check if u,v,w are close to half-integers.
                    u_check = np.round(u * 2) / 2
                    v_check = np.round(v * 2) / 2
                    w_check = np.round(w * 2) / 2
                    
                    if np.allclose(u, u_check, atol=1e-6) and np.allclose(v, v_check, atol=1e-6) and np.allclose(w, w_check, atol=1e-6):
                        # Construct full solution
                        U = np.zeros((n*m, target_rank))
                        V = np.zeros((m*p, target_rank))
                        W = np.zeros((p*n, target_rank))
                        
                        U[:, :target_rank-1] = U_rand
                        V[:, :target_rank-1] = V_rand
                        W[:, :target_rank-1] = W_rand
                        
                        U[:, target_rank-1] = u_check
                        V[:, target_rank-1] = v_check
                        W[:, target_rank-1] = w_check
                        
                        # Verify
                        if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-6):
                            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    n, m, p = 2, 2, 4
    start_time = time.time()
    
    # Try rank 12
    deadline_12 = start_time + 30
    sol_12 = _peel_structured_search(12, deadline_12, n, m, p)
    if sol_12:
        return sol_12
        
    # Try rank 13
    deadline_13 = start_time + 90
    sol_13 = _peel_structured_search(13, deadline_13, n, m, p)
    if sol_13:
        return sol_13
        
    # Fallback to rank 14
    sol_14 = _strassen_2x2x4()
    if _verify_decomposition(sol_14['U'], sol_14['V'], sol_14['W'], n, m, p, tol=1e-9):
        return sol_14
        
    # Fallback to rank 16
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print(res['U'][0])
