"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    # Rank 29: Strassen on 2x2x2 sub-block + schoolbook for the rest.
    
    U = []
    V = []
    W = []
    
    def add_term(u, v, w):
        U.append(u)
        V.append(v)
        W.append(w)
    
    # U has 6 entries (i in 0..1, j in 0..2), index = i*3+j
    # V has 15 entries (j in 0..2, k in 0..4), index = j*5+k
    # W has 10 entries (k in 0..4, i in 0..1), index = k*2+i
    
    # Strassen 7 terms for (i=0,1) x (j=0,1) x (k=0,1)
    # W contributions: [C00, C10, C01, C11, C02, C12, C03, C13, C04, C14]
    # W indices: C[i,k] -> k*2+i
    
    # M1 = (A00+A11)(B00+B11): C00+=1, C11+=1
    add_term([1,0,0,0,1,0],
             [1,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
             [1,0,0,1,0,0,0,0,0,0])
    
    # M2 = (A10+A11)B00: C10+=1
    add_term([0,0,0,1,1,0],
             [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
             [0,1,0,0,0,0,0,0,0,0])
    
    # M3 = A00(B10-B11): C01+=1, C11-=1
    add_term([1,0,0,0,0,0],
             [0,0,0,0,0,1,-1,0,0,0,0,0,0,0,0],
             [0,0,1,-1,0,0,0,0,0,0])
    
    # M4 = A11(B01-B00): C00-=1, C11+=1
    add_term([0,0,0,0,1,0],
             [-1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
             [-1,0,0,1,0,0,0,0,0,0])
    
    # M5 = (A00+A01)B11: C00+=1
    add_term([1,1,0,0,0,0],
             [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
             [1,0,0,0,0,0,0,0,0,0])
    
    # M6 = (A10-A00)(B00+B10): C10+=1, C11+=1
    add_term([-1,0,0,1,0,0],
             [1,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
             [0,1,0,1,0,0,0,0,0,0])
    
    # M7 = (A01-A11)(B01+B11): C00-=1, C01+=1, C11-=1
    add_term([0,1,0,0,-1,0],
             [0,1,0,0,0,0,1,0,0,0,0,0,0,0,0],
             [-1,0,1,-1,0,0,0,0,0,0])
    
    # Schoolbook: j=2, all k; and j=0,1 with k=2,3,4
    for i in range(n):
        for k in range(p):
            u = [0]*6; u[i*3+2] = 1
            v = [0]*15; v[2*5+k] = 1
            w = [0]*10; w[k*2+i] = 1
            add_term(u, v, w)
    
    for i in range(n):
        for j in range(2):
            for k in range(2, p):
                u = [0]*6; u[i*3+j] = 1
                v = [0]*15; v[j*5+k] = 1
                w = [0]*10; w[k*2+i] = 1
                add_term(u, v, w)
    
    rank = len(U)
    assert rank == 29, f"Expected 29, got {rank}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
