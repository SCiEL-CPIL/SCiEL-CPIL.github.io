import numpy as np
import time

def construct():
    """
    Construct a rank-42 tensor decomposition for 2x2 * 2x14 matrix multiplication.
    Uses 3-term decomposition for each of the 14 2x1 blocks (or 7 2x2 blocks with 6 terms).
    Actually, for M_{2,2,1} (2x2 * 2x1 -> 2x1), rank is 3.
    We can decompose the 14 columns into 14 independent M_{2,2,1} problems?
    No, the tensor is M_{2,2,14}.
    The standard Kronecker product gives rank(M_{2,2,14}) = rank(M_{2,2,1}) * rank(M_{1,1,14})? No.
    But if we group the 14 columns into 7 pairs of 2, we have 7 copies of M_{2,2,2}.
    rank(M_{2,2,2}) = 7 (Strassen). Total 49.
    
    Can we do better?
    M_{2,2,14} can be seen as 14 copies of M_{2,2,1} if the A matrix is shared?
    Yes, A is shared.
    The tensor T_{i,j,k} = delta_{i,j} (for A part?) No.
    T_{i,j,k} = 1 if i=row of C, j=col of A, k=col of B.
    C_{i,k} = sum_j A_{i,j} B_{j,k}.
    Tensor indices: (i, j, k).
    If we fix k, we have a tensor T_{i,j,k} which is the M_{2,2,1} tensor.
    Since the A part is shared across all k, we can potentially share terms.
    However, the B part is different for each k.
    
    Let's look at the structure:
    T = sum_{j=0}^{1} sum_{k=0}^{13} e_{i,j} (A part) * e_{j,k} (B part) * e_{k,i} (C part)
    Wait, the tensor is T_{i,j,k} = 1 if we are computing C_{i,k} from A_{i,j} B_{j,k}.
    Actually, the tensor for matrix multiplication M_{n,m,p} is defined by:
    T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    Wait, the standard definition is T = sum_{i,j,k} A_{i,j} B_{j,k} C^*_{k,i} (dual).
    Or simply, the bilinear map (A, B) -> C.
    The tensor is in U * V * W* where U is A-space, V is B-space, W is C-space.
    T_{u,v,w} = coeff of (u in A, v in B) in w of C.
    
    For M_{2,2,14}:
    A is 2x2. Indices (i, j) with i in {0,1}, j in {0,1}.
    B is 2x14. Indices (j, k) with j in {0,1}, k in {0..13}.
    C is 2x14. Indices (i, k) with i in {0,1}, k in {0..13}.
    
    The tensor is T_{(i,j), (j',k), (i',k')} = delta_{i,i'} delta_{j,j'} delta_{k,k'}?
    No. C_{i,k} = sum_j A_{i,j} B_{j,k}.
    So the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    Thus T_{(i,j), (j,k), (i,k)} = 1.
    Note that the middle index j of A and B must match.
    
    This tensor is a "block diagonal" structure in the k dimension?
    For a fixed k, the terms are A_{0,0}B_{0,k} -> C_{0,k}, A_{0,1}B_{1,k} -> C_{0,k},
    A_{1,0}B_{0,k} -> C_{1,k}, A_{1,1}B_{1,k} -> C_{1,k}.
    This is exactly the M_{2,2,1} tensor for each k.
    Since the A part is identical for all k, and the B and C parts are disjoint for different k,
    we can share the A factors?
    
    Let's try to construct a rank-3 decomposition for M_{2,2,1}.
    If we find a rank-3 decomposition for M_{2,2,1}, can we use it for M_{2,2,14}?
    If the rank-3 decomposition of M_{2,2,1} uses terms M_s = (sum a_s A) * (sum b_s B),
    then for M_{2,2,14}, we would have terms M_s,k = (sum a_s A) * (sum b_s B_k).
    But the A part is the same for all k.
    So we can define terms T_s,k = U_s * V_{s,k} * W_{s,k}.
    Wait, the tensor decomposition is T = sum_r u_r \otimes v_r \otimes w_r.
    If we use the same u_s for all k, we are not using a simple product decomposition.
    We need to sum over r.
    If we set u_r = u_s for r = (s, k), then we have 3 * 14 = 42 terms.
    But do they satisfy the tensor equation?
    T_{(i,j), (j',k), (i',k')} = sum_r U_{(i,j), r} V_{(j',k), r} W_{(i',k'), r}.
    If r = (s, k0), then U_{(i,j), (s,k0)} = u_s(i,j).
    V_{(j',k), (s,k0)} = v_s(j', k) if k=k0, else 0?
    If V is non-zero only when k=k0, then V_{(j',k), (s,k0)} = delta_{k,k0} v_s(j', k0).
    Similarly W_{(i',k'), (s,k0)} = delta_{k',k0} w_s(i', k0).
    Then the sum becomes:
    sum_s sum_{k0} u_s(i,j) delta_{k,k0} v_s(j', k0) delta_{k',k0} w_s(i', k0)
    = delta_{k,k'} sum_s u_s(i,j) v_s(j', k) w_s(i', k).
    
    We require this to be 1 if i=i', j=j', k=k', and 0 otherwise.
    Specifically:
    If i != i' or j != j' or k != k', it must be 0.
    The delta_{k,k'} ensures k=k'.
    So we need:
    sum_s u_s(i,j) v_s(j', k) w_s(i', k) = delta_{i,i'} delta_{j,j'} for all i, i', j, j', k.
    
    This is exactly the condition for a rank-3 decomposition of M_{2,2,1} where the coefficients depend on k?
    No, u_s, v_s, w_s are fixed vectors.
    v_s(j', k) is the component of v_s at index (j', k).
    But in M_{2,2,1}, the B vector is 2-dimensional (j' in {0,1}).
    Here, the B vector is 28-dimensional (j' in {0,1}, k in {0..13}).
    If we set v_s(j', k) = delta_{k, k_s} * v'_s(j'), then it's not a single vector v_s.
    
    Wait, in the tensor decomposition, v_r is a vector in the B-space (dim 28).
    If we choose v_{s,k} to be a vector that is zero except at column k, then:
    v_{s,k}[(j', k')] = delta_{k,k'} v_s(j').
    Then the product V_{(j',k), (s,k0)} W_{(i',k'), (s,k0)} becomes:
    delta_{k,k0} v_s(j') * delta_{k',k0} w_s(i').
    Summing over k0:
    sum_s sum_{k0} u_s(i,j) delta_{k,k0} v_s(j') delta_{k',k0} w_s(i')
    = delta_{k,k'} sum_s u_s(i,j) v_s(j') w_s(i').
    
    We need this to equal delta_{i,i'} delta_{j,j'}.
    This is possible if we have a rank-3 decomposition of the M_{2,2,1} tensor:
    sum_s u_s(i,j) v_s(j') w_s(i') = delta_{i,i'} delta_{j,j'}.
    
    Does such a decomposition exist?
    The M_{2,2,1} tensor has rank 3?
    It is known that rank(M_{2,2,1}) = 3.
    Let's find the rank-3 decomposition.
    
    Equations:
    i=0, j=0: sum_s u_s(0,0) v_s(0) w_s(0) = 1
    i=0, j=0: sum_s u_s(0,0) v_s(1) w_s(0) = 0
    i=0, j=1: sum_s u_s(0,1) v_s(0) w_s(0) = 0
    i=0, j=1: sum_s u_s(0,1) v_s(1) w_s(0) = 1
    i=1, j=0: sum_s u_s(1,0) v_s(0) w_s(1) = 1
    i=1, j=0: sum_s u_s(1,0) v_s(1) w_s(1) = 0
    i=1, j=1: sum_s u_s(1,1) v_s(0) w_s(1) = 0
    i=1, j=1: sum_s u_s(1,1) v_s(1) w_s(1) = 1
    
    And cross terms must be 0:
    i=0, j=0, i'=1: sum_s u_s(0,0) v_s(j') w_s(1) = 0 for j'=0,1.
    etc.
    
    Let's try a simple solution.
    u_0 = (1, 0, 0, 0)  [i=0,j=0]
    u_1 = (0, 1, 0, 0)  [i=0,j=1]
    u_2 = (0, 0, 1, 0)  [i=1,j=0]
    ... this is rank 4.
    
    Try rank 3.
    Let w_0 = (1, 0), w_1 = (0, 1), w_2 = (1, 1).
    Then v_s must be such that:
    For i'=0: sum_s u_s(i,j) v_s(j') w_s(0) = delta_{i,0} delta_{j,j'}.
    For i'=1: sum_s u_s(i,j) v_s(j') w_s(1) = delta_{i,1} delta_{j,j'}.
    
    Let's set v_s to be standard basis?
    If v_0 = (1, 0), v_1 = (0, 1), v_2 = (1, 1).
    
    This is a linear system. We can solve it numerically inside construct() if we want to be safe,
    or hardcode if we can find it.
    
    Let's try to find it analytically.
    Assume:
    u_0 = (1, 1, 0, 0)  [A00 + A01]
    u_1 = (0, 0, 1, 1)  [A10 + A11]
    u_2 = (1, 0, -1, 0) [A00 - A10]
    
    Let's check if this works with some v, w.
    
    Actually, there is a known rank-3 decomposition for M_{2,2,1}.
    It is related to the fact that M_{2,2,1} is a sub-tensor of M_{2,2,2}.
    
    Let's just implement a solver for the rank-3 decomposition of M_{2,2,1} in construct().
    If it fails, fall back to rank 49.
    
    Variables:
    u_s (dim 4), v_s (dim 2), w_s (dim 2) for s=0,1,2.
    Total 24 variables.
    Equations: 8 non-zero + 8 zero = 16 equations.
    We can use numpy.linalg.lstsq or just solve a linear system if we fix some variables.
    Since the system is homogeneous in scaling (u_s * c, v_s / c, w_s), we can fix one variable, e.g., w_0(0) = 1.
    
    Let's write a helper to find the rank-3 decomposition.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # First, try to find a rank-3 decomposition for M_{2,2,1}
    # If found, use it to construct rank-42 for M_{2,2,14}
    
    def find_rank3_m221():
        # Variables: u0, u1, u2 (each 4), v0, v1, v2 (each 2), w0, w1, w2 (each 2)
        # Total 24 variables.
        # We fix w0[0] = 1 to break scaling symmetry.
        # Remaining 23 variables.
        # Equations:
        # T(0,0,0) = 1
        # T(0,0,1) = 0
        # T(0,1,0) = 0
        # T(0,1,1) = 1
        # T(1,0,0) = 1
        # T(1,0,1) = 0
        # T(1,1,0) = 0
        # T(1,1,1) = 1
        # T(0,0,0) for i=1 is 0? No, T(i,j,i') = delta(i,i')delta(j,j').
        # So T(1,0,0) = 0? No, i=1, i'=0 => 0.
        # Let's list all 16 equations (i,j,i' in {0,1}^3, j' in {0,1}).
        # Actually, for each k (which is i' here in M_{2,2,1}), we have 4 equations for j, j'.
        # Total 8 equations for non-zero, 8 for zero.
        
        # Let's use a simple iterative method or just try to solve linear system with fixed variables.
        # Since it's small, we can use scipy.optimize or just manual backtracking?
        # Better: use numpy to solve a linear system by fixing some variables.
        
        # Let's fix:
        # w0 = [1, 0]
        # w1 = [0, 1]
        # w2 = [1, 1]
        
        # Then solve for u_s and v_s.
        # Equations for i'=0 (w0):
        # sum_s u_s(i,j) v_s(j') * 1 = delta(i,0) delta(j,j')
        # Equations for i'=1 (w1):
        # sum_s u_s(i,j) v_s(j') * 0 = 0 (satisfied)
        # Wait, w1 = [0,1] means w1(0)=0, w1(1)=1.
        # So for i'=0, we use w_s(0).
        # For i'=1, we use w_s(1).
        
        # Let's define the system.
        # Unknowns: u0, u1, u2 (12 vars), v0, v1, v2 (6 vars). Total 18 vars.
        # Equations:
        # For i'=0:
        # (i=0,j=0,j'=0): u0(0,0)v0(0) + u1(0,0)v1(0) + u2(0,0)v2(0) = 1
        # (i=0,j=0,j'=1): u0(0,0)v0(1) + u1(0,0)v1(1) + u2(0,0)v2(1) = 0
        # (i=0,j=1,j'=0): u0(0,1)v0(0) + u1(0,1)v1(0) + u2(0,1)v2(0) = 0
        # (i=0,j=1,j'=1): u0(0,1)v0(1) + u1(0,1)v1(1) + u2(0,1)v2(1) = 1
        # (i=1,j=0,j'=0): u0(1,0)v0(0) + u1(1,0)v1(0) + u2(1,0)v2(0) = 0
        # (i=1,j=0,j'=1): u0(1,0)v0(1) + u1(1,0)v1(1) + u2(1,0)v2(1) = 0
        # (i=1,j=1,j'=0): u0(1,1)v0(0) + u1(1,1)v1(0) + u2(1,1)v2(0) = 0
        # (i=1,j=1,j'=1): u0(1,1)v0(1) + u1(1,1)v1(1) + u2(1,1)v2(1) = 0
        
        # For i'=1:
        # (i=0,j=0,j'=0): u0(0,0)v0(0) + u1(0,0)v1(0) + u2(0,0)v2(0) = 0
        # ... and so on.
        
        # This is a bilinear system. We can't solve it directly with linear algebra.
        # We need to use a solver like scipy.optimize.fsolve or similar.
        # But we don't have scipy in the standard library? Yes, we do.
        
        from scipy.optimize import fsolve
        
        def equations(vars):
            # vars: [u0(4), u1(4), u2(4), v0(2), v1(2), v2(2)]
            u0 = vars[0:4]
            u1 = vars[4:8]
            u2 = vars[8:12]
            v0 = vars[12:14]
            v1 = vars[14:16]
            v2 = vars[16:18]
            
            w0 = [1, 0]
            w1 = [0, 1]
            w2 = [1, 1]
            
            res = []
            # i'=0
            res.append(u0[0]*v0[0] + u1[0]*v1[0] + u2[0]*v2[0] - 1) # i=0,j=0,j'=0
            res.append(u0[0]*v0[1] + u1[0]*v1[1] + u2[0]*v2[1] - 0) # i=0,j=0,j'=1
            res.append(u0[1]*v0[0] + u1[1]*v1[0] + u2[1]*v2[0] - 0) # i=0,j=1,j'=0
            res.append(u0[1]*v0[1] + u1[1]*v1[1] + u2[1]*v2[1] - 1) # i=0,j=1,j'=1
            res.append(u0[2]*v0[0] + u1[2]*v1[0] + u2[2]*v2[0] - 0) # i=1,j=0,j'=0
            res.append(u0[2]*v0[1] + u1[2]*v1[1] + u2[2]*v2[1] - 0) # i=1,j=0,j'=1
            res.append(u0[3]*v0[0] + u1[3]*v1[0] + u2[3]*v2[0] - 0) # i=1,j=1,j'=0
            res.append(u0[3]*v0[1] + u1[3]*v1[1] + u2[3]*v2[1] - 0) # i=1,j=1,j'=1
            
            # i'=1
            res.append(u0[0]*v0[0] + u1[0]*v1[0] + u2[0]*v2[0] - 0) # i=0,j=0,j'=0
            res.append(u0[0]*v0[1] + u1[0]*v1[1] + u2[0]*v2[1] - 0) # i=0,j=0,j'=1
            res.append(u0[1]*v0[0] + u1[1]*v1[0] + u2[1]*v2[0] - 0) # i=0,j=1,j'=0
            res.append(u0[1]*v0[1] + u1[1]*v1[1] + u2[1]*v2[1] - 0) # i=0,j=1,j'=1
            res.append(u0[2]*v0[0] + u1[2]*v1[0] + u2[2]*v2[0] - 1) # i=1,j=0,j'=0
            res.append(u0[2]*v0[1] + u1[2]*v1[1] + u2[2]*v2[1] - 0) # i=1,j=0,j'=1
            res.append(u0[3]*v0[0] + u1[3]*v1[0] + u2[3]*v2[0] - 0) # i=1,j=1,j'=0
            res.append(u0[3]*v0[1] + u1[3]*v1[1] + u2[3]*v2[1] - 1) # i=1,j=1,j'=1
            
            return res
        
        # Initial guess
        guess = np.array([
            1, 0, 0, 0,  # u0
            0, 1, 0, 0,  # u1
            0, 0, 1, 0,  # u2
            1, 0,        # v0
            0, 1,        # v1
            1, 1         # v2
        ])
        
        try:
            sol, info, ier, msg = fsolve(equations, guess, full_output=True)
            if ier == 1:
                # Verify
                u0, u1, u2 = sol[0:4], sol[4:8], sol[8:12]
                v0, v1, v2 = sol[12:14], sol[14:16], sol[16:18]
                w0, w1, w2 = [1,0], [0,1], [1,1]
                
                # Check residual
                res = equations(sol)
                if np.linalg.norm(res) < 1e-8:
                    return (u0, u1, u2), (v0, v1, v2), (w0, w1, w2)
        except Exception:
            pass
        
        return None
    
    rank3_sol = find_rank3_m221()
    
    if rank3_sol is not None and time.time() < deadline:
        (u0, u1, u2), (v0, v1, v2), (w0, w1, w2) = rank3_sol
        U_blocks = [u0, u1, u2]
        V_blocks = [v0, v1, v2]
        W_blocks = [w0, w1, w2]
        
        n, m, p = 2, 2, 14
        rank = 3 * p  # 42
        
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        for k in range(p):
            for s in range(3):
                col = k * 3 + s
                
                # U part
                for idx in range(4):
                    U[idx, col] = U_blocks[s][idx]
                
                # V part: v_s(j') at column k
                for j in range(2):
                    V[j*p + k, col] = V_blocks[s][j]
                
                # W part: w_s(i') at column k
                for i in range(2):
                    W[k*n + i, col] = W_blocks[s][i]
        
        # Verify
        def tensor_check(Uf, Vf, Wf):
            for i in range(n):
                for j in range(m):
                    u_idx = i * m + j
                    for k in range(p):
                        v_idx = j * p + k
                        w_idx = k * n + i
                        s = np.dot(Uf[u_idx, :], Vf[v_idx, :] * Wf[w_idx, :])
                        if abs(s - 1.0) > 1e-8:
                            return False
            return True
        
        if tensor_check(U, V, W):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback to rank 49 Strassen
    n, m, p = 2, 2, 14
    nm = n * m
    mp = m * p
    pn = p * n
    num_blocks = p // 2  # 7
    rank = num_blocks * 7  # 49

    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))

    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            a_coeffs, b_coeffs, c_coeffs = strassen_terms[s]
            
            for a_idx, coeff in a_coeffs.items():
                a_i = a_idx // m
                a_j = a_idx % m
                U[a_i * m + a_j, col] = coeff
            
            for b_idx, coeff in b_coeffs.items():
                b_j = b_idx // 2
                b_col = b_idx % 2
                b_k = 2 * t + b_col
                V[b_j * p + b_k, col] = coeff
            
            for c_idx, coeff in c_coeffs.items():
                c_i = c_idx // 2
                c_col = c_idx % 2
                c_k = 2 * t + c_col
                W[c_k * n + c_i, col] = coeff

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
