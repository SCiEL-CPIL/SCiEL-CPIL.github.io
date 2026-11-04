import numpy as np

def construct():
    # Correct Strassen 2x2 factors
    # U: (4, 7) Rows: A00, A01, A10, A11
    U_S = np.array([
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ])
    
    # V: (4, 7) Rows: B00, B01, B10, B11
    V_S = np.array([
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 0, 1, 0, 1, 0],  # B01
        [0, 0, 1, 0, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ])
    
    # W: (4, 7) Rows: C00, C10, C01, C11 (k*2+i order)
    # C00 (k=0,i=0) -> 0
    # C10 (k=0,i=1) -> 1
    # C01 (k=1,i=0) -> 2
    # C11 (k=1,i=1) -> 3
    W_S = np.array([
        [1, 0, 0, -1, 1, 0, 0], # C00
        [0, 1, 0, 0, 0, 0, 1],  # C10
        [0, 0, 1, 1, 0, 0, 0],  # C01
        [1, 0, 0, 0, 0, 1, 0]   # C11
    ])
    
    # Kronecker products
    U_4 = np.kron(U_S, U_S)
    V_4 = np.kron(V_S, V_S)
    W_4 = np.kron(W_S, W_S)
    
    # Permute rows to match standard indexing
    # U, V: index = i*4 + j
    # W: index = k*4 + i
    
    # Helper to generate permutation indices
    # For U/V: kron index = (2*i1 + j1)*4 + (2*i0 + j0)
    # Target index = (2*i1 + i0)*4 + (2*j1 + j0)
    
    perm_U = []
    for i1 in range(2):
        for i0 in range(2):
            for j1 in range(2):
                for j0 in range(2):
                    target_idx = (2*i1 + i0)*4 + (2*j1 + j0)
                    kron_idx = (2*i1 + j1)*4 + (2*i0 + j0)
                    perm_U.append(kron_idx)
                    
    # perm_U is a list where perm_U[target_idx] = kron_idx?
    # We want to reorder U_4 such that new_U[target_idx] = old_U[kron_idx].
    # So new_U = U_4[perm_U] if perm_U maps target to source.
    # Let's check: new_U[0] should be U_4[kron_idx for i=0,j=0].
    # i=0,j=0 -> i1=0,i0=0,j1=0,j0=0.
    # target=0. kron = 0.
    # perm_U[0] = 0.
    # new_U[0] = U_4[0]. Correct.
    
    # For W: kron index = (2*k1 + i1)*4 + (2*k0 + i0)
    # Target index = (2*k1 + k0)*4 + (2*i1 + i0)
    
    perm_W = []
    for k1 in range(2):
        for k0 in range(2):
            for i1 in range(2):
                for i0 in range(2):
                    target_idx = (2*k1 + k0)*4 + (2*i1 + i0)
                    kron_idx = (2*k1 + i1)*4 + (2*k0 + i0)
                    perm_W.append(kron_idx)
    
    U_final = U_4[perm_U]
    V_final = V_4[perm_U] # V has same structure as U (i,j)
    W_final = W_4[perm_W]
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
