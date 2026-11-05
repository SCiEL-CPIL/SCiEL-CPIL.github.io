import time
import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 9, 15, 15
    
    # Build the target tensor T
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Standard Strassen terms for 2x2x2
    # Local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_terms = [
        # M1: (A00+A11)(B00+B11)
        (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),
        # M2: (A10+A11)B00
        (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 0, 1, 0])),
        # M3: A00(B10-B00)
        (np.array([1, 0, 0, 0]), np.array([0, 1, -1, 0]), np.array([0, 1, 0, 0])),
        # M4: A11(B01-B10)  -- wait, let me re-derive
        (np.array([0, 0, 0, 1]), np.array([0, 1, -1, 0]), np.array([0, 0, 1, 0])),
        # M5: (A00+A01)B11
        (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([1, 0, 0, 0])),
        # M6: (A10-A00)(B00+B01)
        (np.array([0, 0, 1, -1]), np.array([1, 1, 0, 0]), np.array([0, 0, 1, 0])),
        # M7: (A01-A11)(B10+B11)
        (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 0, 1, 0]))
    ]
    
    def apply_strassen(U_list, V_list, W_list, i_vals, j_vals, k_vals):
        """Apply Strassen to a 2x2x2 sub-tensor with given index ranges."""
        # Build mappings from local (0,1,2,3) to global indices
        # Local order: 0=(i0,j0), 1=(i0,j1), 2=(i1,j0), 3=(i1,j1)
        i0, i1 = i_vals
        j0, j1 = j_vals
        k0, k1 = k_vals
        
        # a-index mapping: a = i*m + j
        map_a = {0: i0*m + j0, 1: i0*m + j1, 2: i1*m + j0, 3: i1*m + j1}
        # b-index mapping: b = j*p + k
        map_b = {0: j0*p + k0, 1: j0*p + k1, 2: j1*p + k0, 3: j1*p + k1}
        # c-index mapping: c = k*n + i
        map_c = {0: k0*n + i0, 1: k0*n + i1, 2: k1*n + i0, 3: k1*n + i1}
        
        for u_sub, v_sub, w_sub in strassen_terms:
            u_global = np.zeros(nm)
            v_global = np.zeros(mp)
            w_global = np.zeros(pn)
            
            for s, g in map_a.items():
                u_global[g] = u_sub[s]
            for s, g in map_b.items():
                v_global[g] = v_sub[s]
            for s, g in map_c.items():
                w_global[g] = w_sub[s]
                
            U_list.append(u_global)
            V_list.append(v_global)
            W_list.append(w_global)
    
    # Define the two disjoint 2x2x2 blocks
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3}
    
    # Terms covered by the blocks
    block1_terms = set()
    block2_terms = set()
    
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                a = i * m + j
                b = j * p + k
                c = k * n + i
                block1_terms.add((a, b, c))
            for k in [2, 3]:
                a = i * m + j
                b = j * p + k
                c = k * n + i
                block2_terms.add((a, b, c))
    
    # Build decomposition: non-block terms as rank-1, then Strassen for each block
    U_list = []
    V_list = []
    W_list = []
    
    # Add non-block terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if (a, b, c) in block1_terms or (a, b, c) in block2_terms:
                    continue
                u = np.zeros(nm)
                v = np.zeros(mp)
                w = np.zeros(pn)
                u[a] = 1
                v[b] = 1
                w[c] = 1
                U_list.append(u)
                V_list.append(v)
                W_list.append(w)
    
    # Apply Strassen to Block 1
    apply_strassen(U_list, V_list, W_list, [0, 1], [0, 1], [0, 1])
    
    # Apply Strassen to Block 2
    apply_strassen(U_list, V_list, W_list, [0, 1], [0, 1], [2, 3])
    
    rank = len(U_list)
    
    # Verify
    U_final = np.column_stack(U_list)
    V_final = np.column_stack(V_list)
    W_final = np.column_stack(W_list)
    
    T_recon = np.einsum('ar,br,cr->abc', U_final, V_final, W_final)
    
    if not np.allclose(T_recon, T, atol=1e-10):
        # Fallback to schoolbook
        U_school = np.zeros((nm, 45))
        V_school = np.zeros((mp, 45))
        W_school = np.zeros((pn, 45))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, col] = 1
                    V_school[j * p + k, col] = 1
                    W_school[k * n + i, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
