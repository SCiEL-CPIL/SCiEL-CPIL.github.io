import numpy as np

def construct():
    n, m, p = 2, 3, 9
    rank = 50
    
    # U: (n*m, rank) -> (6, 50)
    # V: (m*p, rank) -> (27, 50)
    # W: (p*n, rank) -> (18, 50)
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        if col >= rank:
            return
        U[u_idx, col] = u_val
        V[v_idx, col] = v_val
        W[w_idx, col] = w_val
        col += 1

    # Strassen 2x2x2 decomposition
    # Local indices: i in 0..1, j in 0..1, k in 0..1
    # Global mapping: i -> i, j -> j + j_off, k -> k + k_off
    def add_strassen(j_off, k_off):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        # A00 is U[0*1+0], A11 is U[1*1+1]
        # B00 is V[0*p+0], B11 is V[1*p+1]
        # C00 is W[0*n+0], C11 is W[1*n+1] etc.
        
        # Term 1: M1
        # U part: A00, A11. V part: B00, B11.
        # This is not a single rank-1 term in the global sense if we don't expand.
        # Strassen uses 7 rank-1 terms for the whole 2x2x2.
        # The 7 terms are:
        # 1. (A00+A11) * (B00+B11) -> contributes to C00, C01, C10, C11 with coeffs
        # Wait, the standard Strassen decomposition expresses C as sum of 7 terms like (linear comb of A) * (linear comb of B).
        # Each such term is a rank-1 tensor in the multiplication sense?
        # Yes. $C_{ik} = \sum_{r=1}^7 (U_{ir} \cdot A) \cdot (V_{rk} \cdot B)$?
        # No, the decomposition is $T = \sum_{r=1}^7 u^{(r)} \otimes v^{(r)} \otimes w^{(r)}$.
        # Where $u^{(r)}$ is a vector in $A$ space, $v^{(r)}$ in $B$ space, $w^{(r)}$ in $C$ space.
        
        # Strassen vectors:
        # r=1: U: [1,0,0,1] (A00, A11), V: [1,0,0,1] (B00, B11), W: [1, 0, 0, -1] (C00, C11? No)
        # Let's use the explicit coefficients.
        
        # M1 = (A00+A11)(B00+B11)
        # C00 += M1, C11 += -M1? No.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to map these to the tensor decomposition format.
        # Each M is a product of a linear combo of A and a linear combo of B.
        # So for each M, we add 1 column to U, V, W.
        # U column has coeffs for A elements involved.
        # V column has coeffs for B elements involved.
        # W column has coeffs for C elements involved (with signs).
        
        # M1: A: A00, A11. B: B00, B11.
        # U: idx 0 (A00)=1, idx 1 (A11)=1.
        # V: idx 0 (B00)=1, idx 1 (B11)=1.
        # W: C00 += 1, C11 += 1? 
        # From C00 = M1..., C11 = M1...
        # So W: idx 0 (C00)=1, idx 3 (C11)=1.
        
        # Let's define the 7 terms carefully.
        # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        # 1. M1 = (A00+A11)(B00+B11)
        # U: [1, 0, 0, 1]
        # V: [1, 0, 0, 1]
        # W: [1, 0, 0, 1] (C00, C11)
        
        # 2. M2 = (A10+A11)B00
        # U: [0, 0, 1, 1]
        # V: [1, 0, 0, 0]
        # W: [0, 0, 1, -1] (C10, C11)
        
        # 3. M3 = A00(B01-B11)
        # U: [1, 0, 0, 0]
        # V: [0, 1, 0, -1]
        # W: [0, 1, 0, 1] (C01, C11)
        
        # 4. M4 = A11(B10-B00)
        # U: [0, 0, 0, 1]
        # V: [0, 0, 1, -1]
        # W: [1, 0, 1, 0] (C00, C10)
        
        # 5. M5 = (A00+A01)B11
        # U: [1, 1, 0, 0]
        # V: [0, 0, 0, 1]
        # W: [0, 1, -1, 0] (C01, C10)
        
        # 6. M6 = (A10-A00)(B00+B01)
        # U: [-1, 0, 1, 0]
        # V: [1, 1, 0, 0]
        # W: [0, 0, 0, 1] (C11)
        
        # 7. M7 = (A01-A11)(B10+B11)
        # U: [0, 1, 0, -1]
        # V: [0, 0, 1, 1]
        # W: [1, 0, 0, 0] (C00)
        
        terms = [
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
            ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
            ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),
            ([0,0,0,1], [0,0,1,-1], [1,0,1,0]),
            ([1,1,0,0], [0,0,0,1], [0,1,-1,0]),
            ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
            ([0,1,0,-1], [0,0,1,1], [1,0,0,0])
        ]
        
        for u_vec, v_vec, w_vec in terms:
            # Map to global indices
            # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # Global A index: i*m + j.
            # Local j is 0 or 1. Global j is j_off + local_j.
            # Local i is 0 or 1. Global i is i.
            
            # V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # Global B index: j*p + k.
            # Local j is 0 or 1. Global j is j_off + local_j.
            # Local k is 0 or 1. Global k is k_off + local_k.
            
            # W indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # Global C index: k*n + i.
            # Local k is 0 or 1. Global k is k_off + local_k.
            # Local i is 0 or 1. Global i is i.
            
            for idx, val in enumerate(u_vec):
                if val != 0:
                    # idx 0: i=0, j=0
                    # idx 1: i=0, j=1
                    # idx 2: i=1, j=0
                    # idx 3: i=1, j=1
                    local_i = idx // 2
                    local_j = idx % 2
                    global_i = local_i
                    global_j = j_off + local_j
                    global_u_idx = global_i * m + global_j
                    U[global_u_idx, col] += val
            for idx, val in enumerate(v_vec):
                if val != 0:
                    local_j = idx // 2
                    local_k = idx % 2
                    global_j = j_off + local_j
                    global_k = k_off + local_k
                    global_v_idx = global_j * p + global_k
                    V[global_v_idx, col] += val
            for idx, val in enumerate(w_vec):
                if val != 0:
                    local_k = idx // 2
                    local_i = idx % 2
                    global_k = k_off + local_k
                    global_i = local_i
                    global_w_idx = global_k * n + global_i
                    W[global_w_idx, col] += val
            col += 1

    # Schoolbook 2x1x2
    # A: 2x1 (indices 0, 1). B: 1x2 (indices 0, 1). C: 2x2 (indices 0,1,2,3).
    # Terms:
    # A0 * B0 -> C00
    # A0 * B1 -> C01
    # A1 * B0 -> C10
    # A1 * B1 -> C11
    def add_school_212(j_off, k_off):
        nonlocal col
        # A indices: 0:(0,0), 1:(1,0)
        # B indices: 0:(0,0), 1:(0,1)
        # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        # 1. A0 * B0 -> C00
        U[0*m + j_off, col] = 1
        V[j_off*p + k_off, col] = 1
        W[k_off*n + 0, col] = 1
        col += 1
        
        # 2. A0 * B1 -> C01
        U[0*m + j_off, col] = 1
        V[j_off*p + (k_off+1), col] = 1
        W[(k_off+1)*n + 0, col] = 1
        col += 1
        
        # 3. A1 * B0 -> C10
        U[1*m + j_off, col] = 1
        V[j_off*p + k_off, col] = 1
        W[k_off*n + 1, col] = 1
        col += 1
        
        # 4. A1 * B1 -> C11
        U[1*m + j_off, col] = 1
        V[j_off*p + (k_off+1), col] = 1
        W[(k_off+1)*n + 1, col] = 1
        col += 1

    # Schoolbook 2x3x1
    # A: 2x3. B: 3x1. C: 2x1.
    def add_school_231(k_off):
        nonlocal col
        # A indices: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
        # B indices: 0:(0,0), 1:(1,0), 2:(2,0)
        # C indices: 0:(0,0), 1:(1,0)
        
        for i in range(2):
            for j in range(3):
                U[i*m + j, col] = 1
                V[j*p + k_off, col] = 1
                W[k_off*n + i, col] = 1
                col += 1

    # Execute
    # 4 blocks of 2x3x2 (p offsets 0, 2, 4, 6)
    for k_off in [0, 2, 4, 6]:
        # Strassen for j=0,1
        add_strassen(0, k_off)
        # Schoolbook 2x1x2 for j=2
        add_school_212(2, k_off)
        
    # 1 block of 2x3x1 (p offset 8)
    add_school_231(8)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
