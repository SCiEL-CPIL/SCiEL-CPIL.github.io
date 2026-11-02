import numpy as np
import time
import itertools
import scipy.optimize

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 7
    nm, mp, pn = n * m, m * p, p * n
    
    # Target tensor
    # T[i, j, k, l, r, s] = 1 if i==r, j==l, k==s else 0
    # Actually the tensor for matrix multiplication C = A * B is:
    # T_{i, j, k, l, m, n} = delta_{i, l} delta_{k, m}
    # Here:
    # A is n x m (2x3)
    # B is m x p (3x7)
    # C is n x p (2x7)
    # C_{i, k} = sum_j A_{i, j} B_{j, k}
    # Tensor T has indices (i, j, k, l, m, n) ? No.
    # The decomposition is T = sum_r U_{i, j, r} V_{j, k, r} W_{k, i, r}
    # Wait, the interface says:
    # U: (n*m, rank) -> U[i*m+j, r]
    # V: (m*p, rank) -> V[j*p+k, r]
    # W: (p*n, rank) -> W[k*n+i, r]
    # And the trace convention W[k*n+i, r].
    # The contraction is:
    # C_{i, k} = sum_j A_{i, j} B_{j, k}
    # Decomposition:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] ?
    # Let's check the indices.
    # Sum over r: U_{i,j,r} V_{j,k,r} W_{k,i,r}
    # If we fix i, k, the sum over r and j gives C_{i,k}.
    # So T_target[i, j, j, k, i, k] = 1.
    # Yes, this matches the current code.
    
    T_target = np.zeros((n, m, m, p, n, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i, j, j, k, i, k] = 1.0
    T_target_flat = T_target.flatten()
    
    def residual(params):
        R = len(params) // (nm + mp + pn)
        if R == 0:
            return np.zeros(len(T_target_flat))
        U = params[:nm*R].reshape(nm, R)
        V = params[nm*R:nm*R+mp*R].reshape(mp, R)
        W = params[nm*R+mp*R:].reshape(pn, R)
        
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        
        # A_{i, j, k} = sum_r U_{i, j, r} V_{j, k, r}
        A = np.einsum('ijr,jkr->ijk', U_3d, V_3d)
        # T_{i, j, k, l, m, n} = sum_r A_{i, j, k} W_{k, i, r} ? No.
        # The target tensor is T[i, j, j, k, i, k].
        # The decomposition is sum_r U[i, j, r] V[j, k, r] W[k, i, r].
        # So for fixed i, j, k, the value is sum_r U[i, j, r] V[j, k, r] W[k, i, r].
        # This is exactly einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        T_calc = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        # We need to map T_calc (n, m, p) to the flat target (n, m, m, p, n, p)
        # T_target[i, j, j, k, i, k] = 1
        # T_calc[i, j, k] should equal 1 if j==j (always true) ... wait.
        # The tensor T is 6-dimensional.
        # T[i, j, l, k, r, s] = delta_{i, r} delta_{l, k} delta_{j, l} ? No.
        # C_{i, k} = sum_j A_{i, j} B_{j, k}
        # The "tensor" of multiplication is usually defined as T_{i, j, k} = delta_{i, j} delta_{j, k} ? No.
        # Let's look at the residual in the original code:
        # A = np.einsum('ijr,jkr->ijjkr', U_3d, V_3d)
        # T_calc = np.einsum('ijjkr,kir->ijjkii', A, W_3d)
        # This seems to be constructing the full 6D tensor.
        # Let's stick to the 3D residual for speed, but ensure it's correct.
        # If T_calc[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
        # And we want this to be 1 if j==j (which is always true in this context? No, j is an index of A and B).
        # Wait, the indices are:
        # i: row of C (0..n-1)
        # j: col of A / row of B (0..m-1)
        # k: col of C (0..p-1)
        # The term is A_{i, j} B_{j, k}.
        # The decomposition is sum_r U_{i, j, r} V_{j, k, r} W_{k, i, r}.
        # This sum must equal 1 for all i, j, k? No.
        # It must equal the tensor entry.
        # The tensor entry T[i, j, k] for the multiplication tensor is 1 if the indices align?
        # No, the multiplication tensor is a 3-tensor T_{i, j, k} such that C = A * B is given by contraction.
        # Usually T is defined such that <T, A \otimes B> = C.
        # But here we are decomposing the structure constants.
        # The structure constants are 1 if the indices match the summation path.
        # For C_{i, k} = sum_j A_{i, j} B_{j, k}, the term A_{i, j} B_{j, k} appears with coefficient 1.
        # So the tensor T has non-zero entries at (i, j, k) corresponding to the triplet (i, j, k) in the sum?
        # No, the tensor is over the indices of A, B, and C?
        # The standard tensor decomposition for matrix multiplication [n x m, m x p -> n x p] involves a tensor T of size (n*m) x (m*p) x (n*p).
        # T_{(i,j), (j',k), (i',k')} = 1 if i=i', j=j', k=k'.
        # Here, the indices of A are (i, j), B are (j', k), C are (i', k').
        # The contraction is over j and j'.
        # So T_{(i,j), (j',k), (i',k')} = delta_{j, j'} delta_{i, i'} delta_{k, k'}.
        # This is a 3-way tensor of size (n*m) x (m*p) x (n*p).
        # The decomposition is T = sum_r U_{(i,j), r} V_{(j',k), r} W_{(i',k'), r}.
        # So for each r, we have vectors u_r, v_r, w_r.
        # The condition is:
        # sum_r U[i, j, r] V[j', k, r] W[i', k', r] = delta_{j, j'} delta_{i, i'} delta_{k, k'}
        
        # Let's verify the residual calculation in the current code.
        # T_target_flat is from T_target[i, j, j, k, i, k].
        # This corresponds to delta_{j, j'} (where j'=j), delta_{i, i'} (where i'=i), delta_{k, k'} (where k'=k).
        # So T_target is correct.
        
        # Now, T_calc = einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        # This computes sum_r U[i, j, r] V[j, k, r] W[k, i, r].
        # This sums over r.
        # Does this equal T_target[i, j, j, k, i, k]?
        # T_target[i, j, j, k, i, k] is 1.
        # So we need sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1 for all i, j, k?
        # No, T_target is sparse.
        # T_target[i, j, j, k, i, k] = 1.
        # All other entries are 0.
        # But T_calc[i, j, k] is a scalar for each i, j, k.
        # The full tensor T has size (n*m) x (m*p) x (n*p).
        # Let's flatten the indices.
        # A_idx = i*m + j
        # B_idx = j*p + k
        # C_idx = k*n + i  (Note: W is p*n, index k*n+i)
        
        # The tensor T[A_idx, B_idx, C_idx] = 1 if (i,j) from A, (j',k) from B, (i',k') from C satisfy i=i', j=j', k=k'.
        # So T[(i*m+j), (j*p+k), (k*n+i)] = 1.
        
        # My T_calc[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r].
        # This is the value of the decomposed tensor at A_idx=i*m+j, B_idx=j*p+k, C_idx=k*n+i.
        # So T_calc[i, j, k] should equal 1.
        # For all other combinations of indices, the tensor is 0.
        # But the decomposition is linear.
        # If the decomposition is exact, then for any A, B:
        # C_{i, k} = sum_j A_{i, j} B_{j, k} = sum_r (sum_j A_{i, j} U_{i, j, r}) (sum_j B_{j, k} V_{j, k, r}) W_{k, i, r}
        # This must hold for all A, B.
        # This implies that the tensor T defined by the decomposition must match the multiplication tensor.
        # The multiplication tensor has 1s at (i*m+j, j*p+k, k*n+i).
        # So yes, T_calc[i, j, k] must be 1 for all i, j, k.
        # Wait, is that right?
        # If T_calc[i, j, k] = 1 for all i, j, k, then the tensor is the identity tensor in some sense?
        # Let's check a simple case. 1x1 * 1x1 -> 1x1.
        # A[0,0] B[0,0] = C[0,0].
        # T[0,0,0] = 1.
        # Decomposition: U[0,0] V[0,0] W[0,0] = 1.
        # T_calc[0,0,0] = 1.
        
        # 2x1 * 1x1 -> 2x1.
        # C[0,0] = A[0,0]B[0,0]
        # C[1,0] = A[1,0]B[0,0]
        # T[0,0,0] = 1 (A[0,0], B[0,0], C[0,0])
        # T[1,0,1] = 1 (A[1,0], B[0,0], C[1,0]) -- Wait, C index is k*n+i. Here p=1, n=2.
        # C_idx = k*n + i = 0*2 + 1 = 1.
        # B_idx = j*p + k = 0*1 + 0 = 0.
        # A_idx = i*m + j = 1*1 + 0 = 1.
        # So T[1, 0, 1] = 1.
        
        # My T_calc formula: sum_r U[i, j, r] V[j, k, r] W[k, i, r].
        # For i=1, j=0, k=0: T_calc[1, 0, 0] = sum_r U[1,0,r] V[0,0,r] W[0,1,r].
        # This should be 1.
        # For i=0, j=0, k=0: T_calc[0, 0, 0] = sum_r U[0,0,r] V[0,0,r] W[0,0,r].
        # This should be 1.
        # For i=0, j=0, k=1 (if p>1): T_calc[0, 0, 1] should be 0?
        # In the 2x1 * 1x1 case, p=1, so k is always 0.
        
        # In general, T[A_idx, B_idx, C_idx] is 1 only if the indices match.
        # A_idx = i*m+j, B_idx = j'*p+k, C_idx = k'*n+i'.
        # Match if i=i', j=j', k=k'.
        # So T[(i*m+j), (j*p+k), (k*n+i)] = 1.
        # All other entries are 0.
        
        # My T_calc[i, j, k] computes the value at A_idx=i*m+j, B_idx=j*p+k, C_idx=k*n+i.
        # So T_calc[i, j, k] MUST be 1 for all valid i, j, k.
        # Is this true?
        # Let's check if there are other non-zero entries.
        # T[(i*m+j), (j'*p+k), (k'*n+i')]
        # If j != j', then the entry is 0.
        # But my T_calc only varies i, j, k. It fixes the relationship between indices.
        # Specifically, it assumes the B index uses the same j as the A index, and the C index uses the same k and i.
        # This covers exactly the non-zero entries of the tensor!
        # Because the non-zero entries are exactly those where A's col index (j) matches B's row index (j'), and C's row index (i') matches A's row index (i), and C's col index (k') matches B's col index (k).
        # So yes, T_calc[i, j, k] should be 1 for all i, j, k.
        
        # So the residual is simply T_calc - 1.
        
        # Let's re-verify with the original code's residual.
        # Original:
        # A = np.einsum('ijr,jkr->ijjkr', U_3d, V_3d)
        # This creates a tensor of size n x m x m x p x R? No.
        # 'ijr, jkr -> ijjkr'
        # i: n, j: m, r: R
        # j: m, k: p, r: R
        # Output: i, j, j, k, r. Size n x m x m x p x R.
        # This sums over r? No, r is in the output.
        # So A[i, j, j, k, r] = U[i, j, r] V[j, k, r].
        # Then T_calc = np.einsum('ijjkr, kir -> ijjkii', A, W_3d)
        # A: n x m x m x p x R
        # W: p x n x R
        # Output: i, j, j, k, i.
        # Sums over r and k? No, k is in output.
        # 'ijjkr, kir -> ijjkii'
        # i, j, j, k, r (from A)
        # k, i, r (from W)
        # Output i, j, j, k, i.
        # Sums over r.
        # So T_calc[i, j, j, k, i] = sum_r U[i, j, r] V[j, k, r] W[k, i, r].
        # This is exactly T_calc[i, j, k] from my derivation.
        # And it is mapped to the 6D tensor T_calc[i, j, j, k, i, k] ?
        # The original code returns T_calc.flatten() - T_target_flat.
        # T_calc shape is (n, m, m, p, n).
        # T_target shape is (n, m, m, p, n, p).
        # There is a shape mismatch in the original code's residual return!
        # T_calc.flatten() has size n*m*m*p*n.
        # T_target_flat has size n*m*m*p*n*p.
        # This would cause a broadcast error or incorrect comparison.
        # Let's check the original code again.
        # T_target = np.zeros((n, m, m, p, n, p))
        # T_target_flat = T_target.flatten()
        # ...
        # T_calc = np.einsum('ijjkr,kir->ijjkii', A, W_3d)
        # return T_calc.flatten() - T_target_flat
        
        # If shapes don't match, this code would crash or behave unexpectedly.
        # n=2, m=3, p=7.
        # T_target size: 2*3*3*7*2*7 = 1764.
        # T_calc size: 2*3*3*7*2 = 252.
        # 252 != 1764.
        # So the original code's residual is buggy?
        # Or maybe I am misinterpreting the einsum.
        # 'ijjkr, kir -> ijjkii'
        # Left: i, j, j, k, r
        # Right: k, i, r
        # Output: i, j, j, k, i
        # The last 'i' in output comes from the right operand's 'i'.
        # So the output indices are i, j, j, k, i.
        # This is a 5D tensor.
        # T_target is 6D.
        # So the residual calculation in the current program is definitely flawed or I am missing something.
        # However, the program has a score of 1.0769, which suggests it works.
        # Maybe the evaluator only checks the final matrices U, V, W by multiplying them out?
        # Yes, the evaluator likely constructs the tensor from U, V, W and compares to the target.
        # The residual function inside `construct` is only used for the search.
        # If the residual is wrong, the search will fail, and it will fall back to the hardcoded rank 39.
        # The score 1.0769 corresponds to rank 39 (since 42/39 = 1.0769).
        # So the search is likely failing due to the residual bug, and it always returns rank 39.
        
        # To improve, I need to fix the residual and potentially find a lower rank.
        
        # Correct residual:
        # T_calc[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
        # Target: 1 for all i, j, k.
        # So residual = T_calc - 1.
        # Size of T_calc is n*m*p = 2*3*7 = 42.
        # This is much smaller and faster.
        
        T_calc = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        return T_calc.flatten() - np.ones(n*m*p)

    def try_rank(R, max_restarts=10, timeout=10):
        start_time = time.time()
        for _ in range(max_restarts):
            if time.time() - start_time > timeout:
                break
            params = np.random.randn(R * (nm + mp + pn)) * 0.1
            if R > 0:
                params[0] = 1.0
            try:
                result = scipy.optimize.least_squares(residual, params, method='lm', max_nfev=500)
                if np.linalg.norm(result.fun) < 1e-5:
                    U = result.x[:nm*R].reshape(nm, R)
                    V = result.x[nm*R:nm*R+mp*R].reshape(mp, R)
                    W = result.x[nm*R+mp*R:].reshape(pn, R)
                    
                    # Try to round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    params_r = np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()])
                    res_r = residual(params_r)
                    if np.linalg.norm(res_r) < 1e-5:
                        return U_r, V_r, W_r
            except Exception:
                pass
        return None

    # Fallback: Rank 39 decomposition
    def get_rank_39():
        terms = []
        def add_term(u, v, w):
            u_list = [0.0] * (n * m)
            v_list = [0.0] * (m * p)
            w_list = [0.0] * (p * n)
            for idx, val in u.items(): u_list[idx] = val
            for idx, val in v.items(): v_list[idx] = val
            for idx, val in w.items(): w_list[idx] = val
            terms.append((u_list, v_list, w_list))
        
        def u_idx(i, j): return i * m + j
        def v_idx(j, k): return j * p + k
        def w_idx(k, i): return k * n + i
        
        # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
        add_term({u_idx(0,0): 1, u_idx(1,1): 1}, {v_idx(0,0): 1, v_idx(1,1): 1}, {w_idx(0,0): 1, w_idx(1,1): 1})
        add_term({u_idx(1,0): 1, u_idx(1,1): 1}, {v_idx(0,0): 1}, {w_idx(0,1): 1, w_idx(1,1): -1})
        add_term({u_idx(0,0): 1}, {v_idx(0,1): 1, v_idx(1,1): -1}, {w_idx(1,0): 1, w_idx(1,1): 1})
        add_term({u_idx(1,1): 1}, {v_idx(1,0): 1, v_idx(0,0): -1}, {w_idx(0,0): 1, w_idx(0,1): 1})
        add_term({u_idx(0,0): 1, u_idx(0,1): 1}, {v_idx(1,1): 1}, {w_idx(0,0): -1, w_idx(1,0): 1})
        add_term({u_idx(1,0): 1, u_idx(0,0): -1}, {v_idx(0,0): 1, v_idx(0,1): 1}, {w_idx(1,1): 1})
        add_term({u_idx(0,1): 1, u_idx(1,1): -1}, {v_idx(1,0): 1, v_idx(1,1): 1}, {w_idx(0,0): 1})
        
        # Part 2: Schoolbook for A[0:2, 0:2] * B[0:2, 2]
        add_term({u_idx(0,0): 1}, {v_idx(0,2): 1}, {w_idx(2,0): 1})
        add_term({u_idx(0,1): 1}, {v_idx(1,2): 1}, {w_idx(2,0): 1})
        add_term({u_idx(1,0): 1}, {v_idx(0,2): 1}, {w_idx(2,1): 1})
        add_term({u_idx(1,1): 1}, {v_idx(1,2): 1}, {w_idx(2,1): 1})
        
        # Part 3: Strassen for A[0:2, 0:2] * B[0:2, 3:5]
        add_term({u_idx(0,0): 1, u_idx(1,1): 1}, {v_idx(0,3): 1, v_idx(1,4): 1}, {w_idx(3,0): 1, w_idx(4,1): 1})
        add_term({u_idx(1,0): 1, u_idx(1,1): 1}, {v_idx(0,3): 1}, {w_idx(3,1): 1, w_idx(4,1): -1})
        add_term({u_idx(0,0): 1}, {v_idx(0,4): 1, v_idx(1,4): -1}, {w_idx(4,0): 1, w_idx(4,1): 1})
        add_term({u_idx(1,1): 1}, {v_idx(1,3): 1, v_idx(0,3): -1}, {w_idx(3,0): 1, w_idx(3,1): 1})
        add_term({u_idx(0,0): 1, u_idx(0,1): 1}, {v_idx(1,4): 1}, {w_idx(3,0): -1, w_idx(4,0): 1})
        add_term({u_idx(1,0): 1, u_idx(0,0): -1}, {v_idx(0,3): 1, v_idx(0,4): 1}, {w_idx(4,1): 1})
        add_term({u_idx(0,1): 1, u_idx(1,1): -1}, {v_idx(1,3): 1, v_idx(1,4): 1}, {w_idx(3,0): 1})
        
        # Part 4: Strassen for A[0:2, 0:2] * B[0:2, 5:7]
        add_term({u_idx(0,0): 1, u_idx(1,1): 1}, {v_idx(0,5): 1, v_idx(1,6): 1}, {w_idx(5,0): 1, w_idx(6,1): 1})
        add_term({u_idx(1,0): 1, u_idx(1,1): 1}, {v_idx(0,5): 1}, {w_idx(5,1): 1, w_idx(6,1): -1})
        add_term({u_idx(0,0): 1}, {v_idx(0,6): 1, v_idx(1,6): -1}, {w_idx(6,0): 1, w_idx(6,1): 1})
        add_term({u_idx(1,1): 1}, {v_idx(1,5): 1, v_idx(0,5): -1}, {w_idx(5,0): 1, w_idx(5,1): 1})
        add_term({u_idx(0,0): 1, u_idx(0,1): 1}, {v_idx(1,6): 1}, {w_idx(5,0): -1, w_idx(6,0): 1})
        add_term({u_idx(1,0): 1, u_idx(0,0): -1}, {v_idx(0,5): 1, v_idx(0,6): 1}, {w_idx(6,1): 1})
        add_term({u_idx(0,1): 1, u_idx(1,1): -1}, {v_idx(1,5): 1, v_idx(1,6): 1}, {w_idx(5,0): 1})
        
        # Part 5: Schoolbook for A[0:2, 2] * B[2, 0:7]
        for k in range(p):
            add_term({u_idx(0,2): 1}, {v_idx(2,k): 1}, {w_idx(k,0): 1})
            add_term({u_idx(1,2): 1}, {v_idx(2,k): 1}, {w_idx(k,1): 1})
            
        rank = len(terms)
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        for c, (u_list, v_list, w_list) in enumerate(terms):
            for i in range(n * m):
                U[i][c] = u_list[i]
            for i in range(m * p):
                V[i][c] = v_list[i]
            for i in range(p * n):
                W[i][c] = w_list[i]
        return {"U": U, "V": V, "W": W}

    # Search for lower rank
    start_search_time = time.time()
    max_search_time = 110.0
    
    # Try ranks 38 down to 30
    for R in range(38, 29, -1):
        if time.time() - start_search_time > max_search_time:
            break
        result = try_rank(R, max_restarts=5, timeout=10)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # If no better rank found, return rank 39
    return get_rank_39()

# EVOLVE-BLOCK-END
