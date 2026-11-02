"""
Optimized tensor decomposition for 2x2x15 matrix multiplication.
Uses block decomposition: 6 blocks of 2x2x2 (Strassen) and 1 block of 2x2x3.
Total Rank: 6 * 7 + 11 = 53.
"""

import time
import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    
    # We split p=15 into 6 blocks of size 2 and 1 block of size 3.
    # Block sizes: [2, 2, 2, 2, 2, 2, 3]
    # Total p = 15.
    
    # Rank of 2x2x2 (Strassen) = 7
    # Rank of 2x2x3 (Strassen + Schoolbook) = 7 + 4 = 11
    # Total Rank = 6 * 7 + 11 = 42 + 11 = 53
    
    rank = 53
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # --- Strassen 2x2x2 Factors ---
    # Local indices for 2x2 matrices
    # U: 2x2 = 4 entries (indices 0..3)
    # V: 2x2 = 4 entries (indices 0..3)
    # W: 2x2 = 4 entries (indices 0..3)
    
    # Strassen Products for 2x2x2:
    # m0 = (a+d)(e+h)
    # m1 = (c+d)e
    # m2 = a(g-h)
    # m3 = d(f-e)
    # m4 = (a+b)h
    # m5 = (c-a)(e+g)
    # m6 = (b-d)(f+h)
    
    # U factors (rows: A elements)
    # A = [[a, b], [c, d]] -> indices: 0:a, 1:b, 2:c, 3:d
    strassen_U = [
        {0: 1, 3: 1}, # m0: a + d
        {2: 1, 3: 1}, # m1: c + d
        {0: 1},       # m2: a
        {3: 1},       # m3: d
        {0: 1, 1: 1}, # m4: a + b
        {2: 1, 0: -1},# m5: c - a
        {1: 1, 3: -1} # m6: b - d
    ]
    
    # V factors (cols: B elements)
    # B = [[e, f], [g, h]] -> indices: 0:e, 1:f, 2:g, 3:h
    strassen_V = [
        {0: 1, 3: 1}, # m0: e + h
        {0: 1},       # m1: e
        {2: 1, 3: -1},# m2: g - h
        {1: 1, 0: -1},# m3: f - e
        {3: 1},       # m4: h
        {0: 1, 2: 1}, # m5: e + g
        {1: 1, 3: 1}  # m6: f + h
    ]
    
    # W factors (rows: C elements)
    # C = [[C00, C01], [C10, C11]] -> indices: 0:C00, 1:C01, 2:C10, 3:C11
    # C00 = m0 - m3 + m5 - m6
    # C01 = m4 + m6
    # C10 = m1 + m2
    # C11 = m0 - m2 + m3 - m5
    
    strassen_W = [
        {0: 1, 3: -1, 2: 1, 1: -1}, # m0 - m3 + m5 - m6 -> C00
        {1: 1, 6: 1},                # m4 + m6 -> C01 (Wait, indices in W list are product indices)
        # Correction: W[r_loc] maps product index to C index.
        # Let's define W as a list of dicts: {product_idx: coeff} for each C element?
        # No, standard form: W[k*n+i, r]
        # Here r is the product index (0..6).
        # W is a matrix of shape (p_loc * n, rank_loc).
        # p_loc=2, n=2. W size 4x7.
        # W[0, 0]=1 (C00 gets m0)
        # W[0, 3]=-1 (C00 gets -m3)
        # W[0, 5]=1 (C00 gets m5)
        # W[0, 6]=-1 (C00 gets -m6)
        
        # Let's rebuild W as a list of dicts for each product r, mapping to C indices.
        # Or just fill the array directly.
        
        # C00 (idx 0): m0, -m3, m5, -m6
        # C01 (idx 1): m4, m6
        # C10 (idx 2): m1, m2
        # C11 (idx 3): m0, -m2, m3, -m5
        
        # W[r, c_idx]
    ]
    
    # Define W entries for Strassen block
    # r=0 (m0): C00(1), C11(1)
    # r=1 (m1): C10(1)
    # r=2 (m2): C10(1), C11(-1)
    # r=3 (m3): C00(-1), C11(1)
    # r=4 (m4): C01(1)
    # r=5 (m5): C00(1), C11(-1)
    # r=6 (m6): C00(-1), C01(1)
    
    strassen_W_entries = [
        {0: 1, 3: 1},   # r0: C00, C11
        {2: 1},         # r1: C10
        {2: 1, 3: -1},  # r2: C10, C11
        {0: -1, 3: 1},  # r3: C00, C11
        {1: 1},         # r4: C01
        {0: 1, 3: -1},  # r5: C00, C11
        {0: -1, 1: 1}   # r6: C00, C01
    ]

    # --- Schoolbook 2x2x1 Factors ---
    # A 2x2, B 2x1, C 2x1
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    # Rank 4.
    
    schoolbook_U = [
        {0: 1}, # r0: A00
        {1: 1}, # r1: A01
        {2: 1}, # r2: A10
        {3: 1}  # r3: A11
    ]
    
    schoolbook_V = [
        {0: 1}, # r0: B0
        {1: 1}, # r1: B1
        {0: 1}, # r2: B0
        {1: 1}  # r3: B1
    ]
    
    # C0 (idx 0): r0, r1
    # C1 (idx 1): r2, r3
    schoolbook_W_entries = [
        {0: 1}, # r0: C0
        {0: 1}, # r1: C0
        {1: 1}, # r2: C1
        {1: 1}  # r3: C1
    ]

    # --- Fill Matrices ---
    
    # Block 0-5: Strassen 2x2x2
    r_offset = 0
    for b in range(6):
        k_start = 2 * b # Start column of B for this block
        # U is global, so no k_offset for U.
        # V has k_offset.
        # W has k_offset.
        
        for r_loc in range(7):
            r_glob = r_offset + r_loc
            
            # Fill U
            for u_idx, val in strassen_U[r_loc].items():
                U[u_idx, r_glob] += val
                
            # Fill V
            for v_idx, val in strassen_V[r_loc].items():
                # v_idx is local j * 2 + k_loc
                j = v_idx // 2
                k_loc = v_idx % 2
                v_glob_idx = j * 15 + (k_start + k_loc)
                V[v_glob_idx, r_glob] += val
                
            # Fill W
            for w_idx, val in strassen_W_entries[r_loc].items():
                # w_idx is local C index (0..3)
                # C index = k_loc * 2 + i
                k_loc = w_idx // 2
                i = w_idx % 2
                w_glob_idx = (k_start + k_loc) * 2 + i
                W[w_glob_idx, r_glob] += val

    # Block 6: 2x2x3
    # This is a combination of Strassen (2x2x2) and Schoolbook (2x2x1)
    # k_start = 12
    # Rank 11 (7 Strassen + 4 Schoolbook)
    
    k_start = 12
    r_offset = 42
    
    # Part 1: Strassen part (cols 12, 13)
    for r_loc in range(7):
        r_glob = r_offset + r_loc
        
        for u_idx, val in strassen_U[r_loc].items():
            U[u_idx, r_glob] += val
            
        for v_idx, val in strassen_V[r_loc].items():
            j = v_idx // 2
            k_loc = v_idx % 2
            v_glob_idx = j * 15 + (k_start + k_loc)
            V[v_glob_idx, r_glob] += val
            
        for w_idx, val in strassen_W_entries[r_loc].items():
            k_loc = w_idx // 2
            i = w_idx % 2
            w_glob_idx = (k_start + k_loc) * 2 + i
            W[w_glob_idx, r_glob] += val

    # Part 2: Schoolbook part (col 14)
    # r_offset for schoolbook part is 42 + 7 = 49
    r_offset_sb = 49
    k_sb = 14
    
    for r_loc in range(4):
        r_glob = r_offset_sb + r_loc
        
        for u_idx, val in schoolbook_U[r_loc].items():
            U[u_idx, r_glob] += val
            
        for v_idx, val in schoolbook_V[r_loc].items():
            j = v_idx
            k_loc = 0
            v_glob_idx = j * 15 + (k_sb + k_loc)
            V[v_glob_idx, r_glob] += val
            
        for w_idx, val in schoolbook_W_entries[r_loc].items():
            k_loc = 0
            i = w_idx
            w_glob_idx = (k_sb + k_loc) * 2 + i
            W[w_glob_idx, r_glob] += val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    # Simple verification
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    # Test with random matrices
    A = np.random.rand(2, 2)
    B = np.random.rand(2, 15)
    
    # Compute C = A @ B
    C_true = A @ B
    
    # Compute C using tensor
    # C[i, k] = sum_r U[i, r] * V[j, r] * W[k, i]?
    # Convention: W[k*n+i, r]
    # C[i, k] = sum_r U[i*m+j? No, U is (n*m, rank).
    # U index: i*m + j?
    # The problem states: U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # Standard tensor decomposition:
    # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    # Here T is the multiplication tensor.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So T_{i,j,k} = delta? No.
    # The tensor T such that <T, A \otimes B> = C?
    # Usually: C = A B.
    # C_{i,k} = sum_{j} A_{i,j} B_{j,k}.
    # This is a trilinear form? No, bilinear in A, B.
    # The rank of the tensor is the rank of the bilinear map.
    # Map: (A, B) -> C.
    # C_{i,k} = sum_r U_{i,j,r}? No.
    # U is indexed by (i, j)?
    # "U factor matrix of shape (n*m, rank)".
    # So U_{idx_u, r} where idx_u = i*m + j.
    # V_{idx_v, r} where idx_v = j*p + k.
    # W_{idx_w, r} where idx_w = k*n + i.
    
    # C_{i,k} = sum_{j, r} A_{i,j} V_{j*p+k, r} W_{k*n+i, r} ?
    # No, U must be involved.
    # C_{i,k} = sum_{j, r} U_{i*m+j, r} A_{i,j} ... No, A is variable.
    # The decomposition is:
    # C_{i,k} = sum_{j, r} U_{i*m+j, r} * A_{i,j} * ... No.
    
    # Correct interpretation:
    # The tensor T is defined by T[i,j,k] = 1 if it contributes to C?
    # Actually, the bilinear map is:
    # C_{i,k} = sum_{j} A_{i,j} B_{j,k}
    # We want to write this as sum_{r} (sum_{j} U_{i*m+j, r} A_{i,j}) * (sum_{j} V_{j*p+k, r} B_{j,k}) * W_{k*n+i, r}
    # This implies:
    # C_{i,k} = sum_{r} W_{k*n+i, r} * (sum_{j} U_{i*m+j, r} A_{i,j}) * (sum_{j'} V_{j'*p+k, r} B_{j',k})
    # Let M_{i,k,r} = sum_j U_{i*m+j, r} A_{i,j}
    # Let N_{j,k,r} = sum_j' V_{j'*p+k, r} B_{j',k}
    # This doesn't look right.
    
    # Standard Tensor Decomposition for Matrix Mult:
    # C = A B
    # C_{i,k} = sum_{j} A_{i,j} B_{j,k}
    # Decomposition:
    # C_{i,k} = sum_{r} U_{i,j,r} A_{i,j} V_{j,k,r} B_{j,k} W_{k,i,r} ?
    # No, the indices must contract correctly.
    # C_{i,k} = sum_{r} ( sum_{j} U_{i*m+j, r} A_{i,j} ) * ( sum_{j'} V_{j'*p+k, r} B_{j',k} ) * W_{k*n+i, r}
    # This requires that the j index in U and V are the same for each r?
    # No, the sum over j is inside the product?
    # Actually, the rank-r decomposition of the multiplication tensor T_{i,j,k} (where T_{i,j,k}=1 if j is the linking index? No)
    # The tensor is T[i,j,k] = 1 if we are multiplying A[i,j] and B[j,k].
    # So T is a 3-tensor of size n x m x p.
    # T[i,j,k] = 1 if the term A[i,j]B[j,k] is in the sum for C[i,k].
    # Wait, C[i,k] = sum_j A[i,j]B[j,k].
    # So the tensor has non-zero entries at (i, j, k) for all i,j,k?
    # No, the tensor T is such that C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k].
    # So T[i,j,k] = 1 for all i,j,k?
    # No, that would mean C[i,k] = sum_{j} A[i,j] B[j,k].
    # Yes, T[i,j,k] = 1 for all i,j,k.
    # So we are decomposing the tensor T of size 2x2x15 with all ones?
    # No, that's not right.
    # The tensor for matrix multiplication is defined by the structure of the multiplication.
    # T[i,j,k] = 1 if j is the index connecting i and k.
    # In C[i,k] = sum_j A[i,j] B[j,k], the index j runs from 0 to m-1.
    # So for each i, k, the sum is over j.
    # The tensor T is 3-dimensional: indices i (0..n-1), j (0..m-1), k (0..p-1).
    # T[i,j,k] = 1.
    # We want to decompose this tensor into rank-r sum of simple tensors.
    # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # Note: The index j appears in both U and V.
    # This is a "slice" of the tensor.
    
    # Let's verify my code's logic against this.
    # My code fills U, V, W.
    # I will compute C from A, B using the formula:
    # C[i,k] = sum_r W[k*n+i, r] * ( sum_j U[i*m+j, r] * A[i,j] ) * ( sum_j' V[j'*p+k, r] * B[j',k] )
    # Wait, if j and j' are independent, this is not correct.
    # The contraction must be over the same j.
    # So:
    # C[i,k] = sum_r W[k*n+i, r] * ( sum_j U[i*m+j, r] A[i,j] V[j*p+k, r] B[j,k] )
    # This is the correct form.
    
    C_calc = np.zeros((2, 15))
    for i in range(2):
        for k in range(15):
            s = 0
            for r in range(53):
                w_val = W[k*2+i, r]
                if w_val == 0: continue
                inner = 0
                for j in range(2):
                    u_val = U[i*2+j, r]
                    v_val = V[j*15+k, r]
                    inner += u_val * A[i,j] * v_val * B[j,k]
                s += w_val * inner
            C_calc[i,k] = s
            
    err = np.max(np.abs(C_true - C_calc))
    print(f"Max Error: {err}")
    if err > 1e-9:
        print("ERROR: Decomposition failed!")
    else:
        print("Success!")

