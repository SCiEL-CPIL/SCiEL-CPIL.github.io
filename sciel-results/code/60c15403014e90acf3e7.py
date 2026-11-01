"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 32 solution: 2 * M(2,2,4) + 1 * M(2,2,1)
    # M(2,2,4) uses 2 * Strassen(2,2,2) = 14
    # M(2,2,1) uses 4
    # Total 32
    
    U = []
    V = []
    W = []
    
    # Helper to add a rank-1 tensor
    def add_tensor(u, v, w):
        U.append(u)
        V.append(v)
        W.append(w)

    # Block 1: k=0,1 (Strassen)
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices for k=0,1: 00, 10, 01, 11 -> 0, 1, 2, 3 (relative)
    # W indices for k=0,1: 00, 10, 01, 11 -> 0, 1, 2, 3 (relative)
    
    def add_strassen_block(k_offset):
        # k_offset is 0 for k=0,1; 4 for k=4,5; etc
        # B indices: j*p + k. j in 0,1. k in 0,1 relative.
        # W indices: k*n + i. k in 0,1 relative. i in 0,1.
        
        # M1: (A00+A11)(B00+B11)
        u = [0, 0, 0, 0]; u[0]=1; u[3]=1
        v = [0, 0, 0, 0]; v[0]=1; v[3]=1 # B00, B11
        w = [0, 0, 0, 0]; w[0]=1; w[3]=1 # C00, C11
        # Wait, W index is k*n+i.
        # k=0, i=0 -> 0. k=1, i=1 -> 3.
        # So W00=0, W11=3.
        # But we need to map to global indices.
        
        # Let's build local vectors and then expand.
        
        # Local A: 4 dims
        # Local B: 4 dims (00, 10, 01, 11) -> indices 0, 1, 2, 3
        # Local W: 4 dims (00, 10, 01, 11) -> indices 0, 1, 2, 3
        
        # M1
        u = [1, 0, 0, 1]
        v = [1, 0, 0, 1] # B00, B11
        w = [1, 0, 0, 1] # C00, C11
        add_local(u, v, w, k_offset)
        
        # M2: (A10+A11)B00
        u = [0, 0, 1, 1]
        v = [1, 0, 0, 0]
        w = [0, 1, 0, 0] # C10
        add_local(u, v, w, k_offset)
        
        # M3: A00(B01-B11)
        u = [1, 0, 0, 0]
        v = [0, 0, 1, -1]
        w = [0, 0, 1, 0] # C01
        add_local(u, v, w, k_offset)
        
        # M4: A11(B10-B00)
        u = [0, 0, 0, 1]
        v = [0, 1, 0, -1]
        w = [0, 1, 0, 0] # C10
        add_local(u, v, w, k_offset)
        
        # M5: (A00+A01)B11
        u = [1, 1, 0, 0]
        v = [0, 0, 0, 1]
        w = [0, 0, 1, 1] # C01, C11
        add_local(u, v, w, k_offset)
        
        # M6: (A10-A00)(B00+B01)
        u = [0, 0, 1, -1] # A10-A00
        v = [1, 0, 1, 0] # B00+B01
        w = [0, 0, 1, 0] # C01? No, C10?
        # C10 = A10B00 + A11B10.
        # M6 contributes to C10?
        # Standard Strassen:
        # C00 = M1 - M4 + M6 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M2 - M5
        # So M6 contributes to C00.
        # M7 contributes to C00.
        # So w for M6 should be C00.
        w = [1, 0, 0, 0] # C00
        add_local(u, v, w, k_offset)
        
        # M7: (A01-A11)(B10+B11)
        u = [0, 1, 0, -1]
        v = [0, 1, 0, 1]
        w = [1, 0, 0, 0] # C00
        add_local(u, v, w, k_offset)

    def add_local(u, v, w, k_offset):
        # Expand local vectors to global
        # u is size 4 (A)
        # v is size 4 (B local)
        # w is size 4 (W local)
        
        U_global = [0] * 4
        for i in range(4):
            U_global[i] = u[i]
            
        V_global = [0] * 18
        # v[0] is B00 (k=0, j=0) -> index 0*9+0 = 0
        # v[1] is B10 (k=0, j=1) -> index 1*9+0 = 9
        # v[2] is B01 (k=1, j=0) -> index 0*9+1 = 1
        # v[3] is B11 (k=1, j=1) -> index 1*9+1 = 10
        V_global[k_offset*9 + 0] = v[0]
        V_global[k_offset*9 + 9] = v[1]
        V_global[k_offset*9 + 1] = v[2]
        V_global[k_offset*9 + 10] = v[3]
        
        W_global = [0] * 18
        # w[0] is C00 (k=0, i=0) -> index 0*2+0 = 0
        # w[1] is C10 (k=0, i=1) -> index 0*2+1 = 1
        # w[2] is C01 (k=1, i=0) -> index 1*2+0 = 2
        # w[3] is C11 (k=1, i=1) -> index 1*2+1 = 3
        W_global[k_offset*2 + 0] = w[0]
        W_global[k_offset*2 + 1] = w[1]
        W_global[k_offset*2 + 2] = w[2]
        W_global[k_offset*2 + 3] = w[3]
        
        U.append(U_global)
        V.append(V_global)
        W.append(W_global)

    # Block 1: k=0,1
    add_strassen_block(0)
    # Block 2: k=2,3
    add_strassen_block(2)
    # Block 3: k=4,5
    add_strassen_block(4)
    # Block 4: k=6,7
    add_strassen_block(6)
    
    # Block 5: k=8 (Schoolbook 2x2x1)
    # A00 B08 C08
    U.append([1, 0, 0, 0])
    V.append([0]*18); V[8] = 1
    W.append([0]*18); W[16] = 1 # k=8, i=0 -> 16
    
    # A01 B18 C08
    U.append([0, 1, 0, 0])
    V.append([0]*18); V[17] = 1
    W.append([0]*18); W[16] = 1
    
    # A10 B08 C18
    U.append([0, 0, 1, 0])
    V.append([0]*18); V[8] = 1
    W.append([0]*18); W[17] = 1
    
    # A11 B18 C18
    U.append([0, 0, 0, 1])
    V.append([0]*18); V[17] = 1
    W.append([0]*18); W[17] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
