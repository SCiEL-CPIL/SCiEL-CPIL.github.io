import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    best_result = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p  # 36

    # Block-repetition strategy: find rank-r for 2x3x2, expand to 2x3x6 (rank 3r)
    n_sub, m_sub, p_sub = 2, 3, 2
    repeats = p // p_sub  # 3

    # Lower bound for sub-problem: max(n_sub*m_sub, m_sub*p_sub, p_sub*n_sub) = 6
    min_sub_rank = max(n_sub * m_sub, m_sub * p_sub, p_sub * n_sub)  # 6

    for sub_rank in range(min_sub_rank, best_rank // repeats + 1):
        if time.time() > deadline:
            break

        full_rank = sub_rank * repeats
        if full_rank >= best_rank:
            break

        # Budget per sub_rank
        max_attempts = 2000 if sub_rank == 6 else 500
        found = _search_sub_decomposition(n_sub, m_sub, p_sub, sub_rank, max_attempts, deadline)

        if found is not None:
            U_sub, V_sub, W_sub = found
            result = _expand_decomposition(U_sub, V_sub, W_sub, n, m, p, p_sub)
            if result is not None and full_rank < best_rank:
                best_result = result
                best_rank = full_rank
                if best_rank <= max(n * m, m * p, p * n):
                    break

    return best_result

def _schoolbook_decomposition(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _search_sub_decomposition(n, m, p, r, max_attempts, deadline):
    np_sub = n * p
    nm = n * m
    mp = m * p

    for attempt in range(max_attempts):
        if time.time() > deadline:
            break

        # Generate random U and V
        # Use values from {-1, 0, 1}
        U_rand = np.random.randint(-1, 2, size=(nm, r))
        V_rand = np.random.randint(-1, 2, size=(mp, r))

        # Build A: shape (n*p, m*r)
        # A[(i,k), (j,c)] = U[i*m+j, c] * V[j*p+k, c]
        # Vectorized construction
        # U_rand shape: (n*m, r) -> reshape to (n, m, r)
        # V_rand shape: (m*p, r) -> reshape to (m, p, r)
        U_reshaped = U_rand.reshape(n, m, r)
        V_reshaped = V_rand.reshape(m, p, r)
        
        # We need A[i, k, j, c] = U[i, j, c] * V[j, k, c]
        # Sum over c? No, A is the matrix of coefficients.
        # The equation is sum_{j,c} A[(i,k), (j,c)] W[(j,c), (i',k')] = delta_{ii'} delta_{kk'}
        # Wait, W index is (k, i) in the task description?
        # Task: W[k*n+i, r]. So row index is k*n+i.
        # Equation: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # This is a linear system for W.
        # Let W_flat be a vector of size p*n*r? No, W has columns c.
        # For each c, we have a term.
        # The system is:
        # For fixed (i, k): sum_{j=1}^m sum_{c=1}^r (U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c] = 1
        # This is NOT a linear system in W entries W[k*n+i, c] because the coefficient of W[k*n+i, c] depends on c and j.
        # Actually, for a fixed (i, k), the sum is over j and c.
        # The variable is W[k*n+i, c].
        # So for each (i, k), we have one equation:
        # sum_{j=1}^m sum_{c=1}^r M_{(i,k), (j,c)} * W_{(k,i), c} = 1 ?
        # No, W index is (k, i). The column is c.
        # So the variable is W_{row, c} where row = k*n+i.
        # The coefficient of W_{row, c} in the equation for (i, k) is sum_{j} U[i*m+j, c] * V[j*p+k, c].
        # Let's define C_{(i,k), c} = sum_{j=1}^m U[i*m+j, c] * V[j*p+k, c].
        # Then the equation is: sum_{c=1}^r C_{(i,k), c} * W[k*n+i, c] = 1.
        # This is a system of n*p equations (one for each i, k) with r * (n*p) variables?
        # No, for each (i, k), the variables are W[k*n+i, c] for c=1..r.
        # So we have n*p independent equations?
        # Equation for (i, k): sum_{c=1}^r C_{(i,k), c} * W_{k*n+i, c} = 1.
        # The variables in this equation are W_{k*n+i, 1}, ..., W_{k*n+i, r}.
        # These variables do not appear in any other equation!
        # Because the row index of W is k*n+i, which is unique for each (i, k).
        # So the system decouples into n*p independent systems!
        # For each (i, k), we need to solve:
        # sum_{c=1}^r C_{(i,k), c} * W_{k*n+i, c} = 1.
        # This is a single linear equation with r unknowns.
        # We need integer/half-integer solutions.
        # This is much easier!
        
        # Let's re-verify this logic.
        # Tensor: T_{i,j,k} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}
        # We want T_{i,j,k} = delta_{i,i'} delta_{k,k'}? No, we want the tensor to be the matmul tensor.
        # The matmul tensor has T_{i,j,k} = 1 if we are computing the (i,k) entry?
        # No, the tensor M_{i,j,k} represents the coefficient of A_{ij} B_{jk} in C_{ik}.
        # So M_{i,j,k} = 1 for all i, j, k.
        # So we need sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} = 1 for all i, j, k.
        
        # My previous derivation:
        # For fixed i, k: sum_{j=1}^m sum_{c=1}^r U_{i,j,c} V_{j,k,c} W_{k,i,c} = 1.
        # This is sum_{c=1}^r (sum_{j=1}^m U_{i,j,c} V_{j,k,c}) W_{k,i,c} = 1.
        # Let C_{i,k,c} = sum_{j=1}^m U_{i,j,c} V_{j,k,c}.
        # Then sum_{c=1}^r C_{i,k,c} W_{k,i,c} = 1.
        # Yes, the equations for different (i, k) pairs involve disjoint sets of W variables.
        # So we just need to find W_{k,i,c} for each (i, k) such that the dot product with C_{i,k,:} is 1.
        # And W entries must be half-integers.
        
        # So for each (i, k), we have a vector C_{i,k} in R^r.
        # We need a half-integer vector w_{i,k} in R^r such that C_{i,k} . w_{i,k} = 1.
        # This is possible if and only if the gcd of the components of C_{i,k} (when scaled to integers) divides 1?
        # If C has integer entries, we need integer/half-integer w.
        # If we scale C by 2, we need 2 C . w = 2.
        # If C is integer, 2 C is even integers.
        # We need a half-integer w. Let w = v/2 where v is integer.
        # Then C . (v/2) = 1 => C . v = 2.
        # So we need an integer vector v such that C . v = 2.
        # This has a solution if and only if gcd(C_1, ..., C_r) divides 2.
        
        # So the algorithm is:
        # 1. Generate random integer U, V.
        # 2. Compute C_{i,k,c} = sum_j U_{i,j,c} V_{j,k,c}.
        # 3. For each (i, k), check if gcd(C_{i,k,1}, ..., C_{i,k,r}) divides 2.
        # 4. If yes, find an integer solution v_{i,k} to C_{i,k} . v_{i,k} = 2.
        # 5. Set W_{k,i,c} = v_{i,k,c} / 2.
        # 6. Verify the full decomposition.
        
        # This is MUCH faster and more reliable!
        
        valid = True
        W_sol = np.zeros((n*p, r))
        for i in range(n):
            for k in range(p):
                # Compute C_{i,k,c}
                C_vec = np.zeros(r)
                for j in range(m):
                    C_vec += U_rand[i*m+j, :] * V_rand[j*p+k, :]
                
                # Check if solution exists
                # C_vec is integer.
                # We need v such that C_vec . v = 2.
                # This requires gcd(C_vec) to divide 2.
                if np.all(C_vec == 0):
                    valid = False
                    break
                
                # Compute gcd of non-zero elements
                non_zero = C_vec[C_vec != 0]
                if len(non_zero) == 0:
                    valid = False
                    break
                
                g = non_zero[0]
                for val in non_zero[1:]:
                    g = np.gcd(g, val)
                
                if 2 % g != 0:
                    valid = False
                    break
                
                # Find a solution v
                # Simple greedy: set v_1 = 2 / C_1 if C_1 divides 2? No.
                # We can just find any integer solution.
                # Since gcd divides 2, a solution exists.
                # We can use a simple backtracking or just set one variable.
                # If C_1 != 0, we can set v_1 = 2 / C_1 if 2 is divisible by C_1.
                # But C_1 might not divide 2.
                # Example: C = [2, 2]. gcd=2. 2%2==0.
                # v=[1,0] works. 2*1+2*0=2.
                # Example: C = [4, 2]. gcd=2.
                # v=[0,1] works. 4*0+2*1=2.
                # Example: C = [6, 4]. gcd=2.
                # v=[1, -2] works. 6-8=-2? No. 6*1 + 4*(-2) = -2.
                # We need sum = 2.
                # 6*1 + 4*0 = 6.
                # 6*0 + 4*?
                # Let's just find the minimum norm solution or any solution.
                # We can use np.linalg.lstsq on the single equation?
                # Or just iterate.
                
                # Simple approach:
                # Try to find v with small entries.
                v = np.zeros(r, dtype=int)
                # Set first non-zero C to 2/g, others to 0?
                # If C_1 = g, then v_1 = 2/g is integer.
                idx = np.argmax(C_vec != 0)
                if C_vec[idx] != 0:
                    if 2 % C_vec[idx] == 0:
                        v[idx] = 2 // C_vec[idx]
                    else:
                        # Need to use another component
                        # Find another non-zero
                        found = False
                        for idx2 in range(r):
                            if idx2 == idx or C_vec[idx2] == 0:
                                continue
                            # Try to solve C1*v1 + C2*v2 = 2
                            # This is a linear Diophantine equation.
                            # Since gcd(C1, C2) divides 2, solution exists.
                            # We can just search small integers for v1, v2.
                            for v1 in range(-10, 11):
                                rem = 2 - C_vec[idx] * v1
                                if rem % C_vec[idx2] == 0:
                                    v2 = rem // C_vec[idx2]
                                    v[idx] = v1
                                    v[idx2] = v2
                                    found = True
                                    break
                            if found:
                                break
                        if not found:
                            valid = False
                            break
                
                W_sol[k*n+i, :] = v / 2.0
        
        if not valid:
            continue
        
        # Final verification
        U_list = U_rand.tolist()
        V_list = V_rand.tolist()
        W_list = W_sol.tolist()
        
        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
            return U_list, V_list, W_list

    return None

def _verify_decomposition(U, V, W, n, m, p):
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    r = U_np.shape[1]

    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U_np[i * m + j, c] * V_np[j * p + k, c] * W_np[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _expand_decomposition(U_sub, V_sub, W_sub, n, m, p, p_sub):
    r_sub = len(U_sub[0])
    repeats = p // p_sub
    r_total = r_sub * repeats

    U_full = [[0] * r_total for _ in range(n * m)]
    V_full = [[0] * r_total for _ in range(m * p)]
    W_full = [[0] * r_total for _ in range(p * n)]

    for block in range(repeats):
        offset = block * r_sub

        for i in range(n * m):
            for c in range(r_sub):
                U_full[i][offset + c] = U_sub[i][c]

        for j in range(m):
            for k in range(p_sub):
                for c in range(r_sub):
                    V_full[j * p + block * p_sub + k][offset + c] = V_sub[j * p_sub + k][c]

        for i in range(n):
            for k in range(p_sub):
                for c in range(r_sub):
                    W_full[(block * p_sub + k) * n + i][offset + c] = W_sub[k * n + i][c]

    if _verify_decomposition(U_full, V_full, W_full, n, m, p):
        return {"U": U_full, "V": V_full, "W": W_full}
    return None
# EVOLVE-BLOCK-END
