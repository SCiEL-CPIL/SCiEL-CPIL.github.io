import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    size_a = n * m
    size_b = m * p
    size_c = p * n
    
    # Construct the tensor T over F_2
    # T is a 3D array of shape (size_a, size_b, size_c)
    # T[a, b, c] = 1 if indices align
    T = np.zeros((size_a, size_b, size_c), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # Greedy Set Cover / Peeling over F_2
    # We want to find U, V, W such that T = sum_r u_r \otimes v_r \otimes w_r
    # We will try to find a decomposition with rank < 105.
    
    U_list = []
    V_list = []
    W_list = []
    T_curr = T.copy()
    
    max_rank = 104
    start_time = time.time()
    
    # Precompute all possible rank-1 terms that are subsets of T?
    # That's too many.
    # Instead, we greedily pick a term that covers the most 1s in T_curr.
    # A rank-1 term is defined by sets I_u, J_v, K_w.
    # The term covers {(i, j, k) | i in I_u, j in J_v, k in K_w}.
    # We need this set to be a subset of Support(T_curr).
    # And we want to maximize the size of this set.
    
    # To speed this up, we can iterate over small subsets of I, J, K.
    # Since n=3, m=5, p=7, the sets are small.
    
    # We will try to find a term that covers at least 2 entries.
    # If none found, we pick a single entry.
    
    for r in range(max_rank):
        if time.time() - start_time > 110:
            break
            
        if not np.any(T_curr):
            break
            
        best_term = None
        best_count = 0
        
        # Try all combinations of small subsets for I, J, K
        # Since n=3, m=5, p=7, we can try subsets of size 1 or 2.
        # Size 1 is always valid (schoolbook).
        # Size 2 might be valid.
        
        # Generate all non-empty subsets of [0, n), [0, m), [0, p)
        # This is 2^3 - 1 = 7, 2^5 - 1 = 31, 2^7 - 1 = 127.
        # Total combos: 7 * 31 * 127 = ~27k.
        # For each combo, check if the term is a subset of T_curr.
        # This is fast enough for 104 iterations? 104 * 27k = 2.8M checks.
        # Each check involves iterating over the product of the sets.
        # Max product size is 3*5*7 = 105.
        # So 2.8M * 105 = 300M ops. Might be too slow in Python.
        
        # Optimization: Only try subsets that are likely to be valid.
        # Or just try to find ANY term with count > 1.
        
        found = False
        # Try to find a term with count > 1
        # We can iterate over pairs of entries in T_curr and see if they can be covered by one term.
        
        # Get list of non-zero entries
        nz = np.argwhere(T_curr == 1)
        
        # If we can't find a multi-cover, we pick the first one
        # But we must ensure we don't exceed time.
        
        # Simple heuristic: Pick the first non-zero entry (a, b, c)
        # This guarantees progress.
        a_idx, b_idx, c_idx = nz[0]
        
        # Check if we can extend this to a larger term
        # i = a_idx // m, j = a_idx % m
        # j' = b_idx // p, k = b_idx % p
        # i' = c_idx % n, k' = c_idx // n
        # We know i=i', j=j', k=k'
        
        i0 = a_idx // m
        j0 = a_idx % m
        k0 = b_idx % p
        
        # Can we add another i?
        # Try i1 != i0
        for i1 in range(n):
            if i1 == i0: continue
            a1 = i1 * m + j0
            # Check if T_curr[a1, b_idx, c1] is 1 for some c1?
            # c1 = k0 * n + i1
            c1 = k0 * n + i1
            if T_curr[a1, b_idx, c1] == 1:
                # We can cover (i0, j0, k0) and (i1, j0, k0)
                # Term: I={i0, i1}, J={j0}, K={k0}
                # Verify all 4? No, only 2 entries.
                # Check if any other entry in this term is 1 in T_curr?
                # The term covers (i0, j0, k0) and (i1, j0, k0).
                # Are there other i?
                # Let's just take this term.
                I_set = [i0, i1]
                J_set = [j0]
                K_set = [k0]
                
                # Verify this term is a subset of T_curr
                valid = True
                for ii in I_set:
                    for jj in J_set:
                        for kk in K_set:
                            aa = ii * m + jj
                            bb = jj * p + kk
                            cc = kk * n + ii
                            if T_curr[aa, bb, cc] != 1:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if valid:
                    # Apply this term
                    for ii in I_set:
                        for jj in J_set:
                            for kk in K_set:
                                aa = ii * m + jj
                                bb = jj * p + kk
                                cc = kk * n + ii
                                T_curr[aa, bb, cc] = 0 # mod 2 subtraction (addition)
                    
                    # Store the term
                    u = np.zeros(size_a, dtype=np.int8)
                    v = np.zeros(size_b, dtype=np.int8)
                    w = np.zeros(size_c, dtype=np.int8)
                    for ii in I_set:
                        u[ii * m + j0] = 1
                    v[j0 * p + k0] = 1
                    for ii in I_set:
                        w[k0 * n + ii] = 1
                    # Wait, w must be the same for all i in the term?
                    # In the term u \otimes v \otimes w, w is a fixed vector.
                    # My w has 1s at k0*n+i0 and k0*n+i1.
                    # So w is not a unit vector, it's a sum.
                    # This is a valid rank-1 term.
                    
                    U_list.append(u)
                    V_list.append(v)
                    W_list.append(w)
                    found = True
                    break
        
        if not found:
            # Fall back to single entry
            I_set = [i0]
            J_set = [j0]
            K_set = [k0]
            
            u = np.zeros(size_a, dtype=np.int8)
            v = np.zeros(size_b, dtype=np.int8)
            w = np.zeros(size_c, dtype=np.int8)
            
            u[i0 * m + j0] = 1
            v[j0 * p + k0] = 1
            w[k0 * n + i0] = 1
            
            T_curr[a_idx, b_idx, c_idx] = 0
            
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)
            
    # If we found a decomposition with rank < 105, return it
    # Otherwise, return schoolbook
    if len(U_list) < 105 and np.all(T_curr == 0):
        # Convert to lists of lists for the interface
        U = [u.tolist() for u in U_list]
        V = [v.tolist() for v in V_list]
        W = [w.tolist() for w in W_list]
        # Note: The interface expects U, V, W to be matrices of shape (size, rank)
        # My U_list is a list of vectors of length size_a.
        # So U should be size_a x rank.
        U_mat = np.array(U_list).T
        V_mat = np.array(V_list).T
        W_mat = np.array(W_list).T
        return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
    else:
        # Schoolbook
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
# EVOLVE-BLOCK-END
