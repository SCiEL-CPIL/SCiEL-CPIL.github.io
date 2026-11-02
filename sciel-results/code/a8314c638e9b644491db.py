"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-42 decomposition using Strassen's algorithm on each 2x2 block.
    
    A is 2x2, B is 2x12 (split into 6 blocks of 2x2), C is 2x12.
    Each block uses the 7-term Strassen identity.
    
    Strassen for 2x2:
    Given A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B:
    M1 = (a+d)*(e+h)
    M2 = (c+d)*e
    M3 = a*(f-h)
    M4 = d*(g-e)
    M5 = (a+b)*h
    M6 = (c-a)*(e+f)
    M7 = (b-d)*(g+h)
    
    C[0,0] = M1 + M4 - M5 + M7
    C[0,1] = M3 + M5
    C[1,0] = M2 + M4
    C[1,1] = M1 - M2 + M3 + M6
    """
    n, m, p = DIMENSIONS  # 2, 2, 12
    
    # Strassen factors for one 2x2 block
    # U factors: indexed by (i,j) of A, where i in {0,1}, j in {0,1}
    # V factors: indexed by (j,k) of B block, where j in {0,1}, k in {0,1} (within block)
    # W factors: indexed by (k,i) of C block, where k in {0,1} (within block), i in {0,1}
    
    # Strassen: 7 multiplications
    # M1 = (a+d)*(e+h)
    # M2 = (c+d)*e
    # M3 = a*(f-h)
    # M4 = d*(g-e)
    # M5 = (a+b)*h
    # M6 = (c-a)*(e+f)
    # M7 = (b-d)*(g+h)
    
    # U entries for each M (row index = i*m+j = i*2+j for A)
    # V entries for each M (row index = j*p_block + k, but we need global k)
    # W entries for each M (row index = k*n + i for output, with global k)
    
    # For block t (0..5), the B columns are 2*t and 2*t+1
    # B[j, 2*t] corresponds to e, B[j, 2*t+1] corresponds to f for row j
    # Actually: B is 2x12, B[j, k] where j in {0,1}, k in {0..11}
    # For block t: B[0, 2t]=e, B[0, 2t+1]=f, B[1, 2t]=g, B[1, 2t+1]=h
    
    # U factors (4 rows: i*2+j for i,j in {0,1}):
    #   row 0: A[0,0]=a
    #   row 1: A[0,1]=b
    #   row 2: A[1,0]=c
    #   row 3: A[1,1]=d
    
    # V factors (24 rows: j*12+k for j in {0,1}, k in {0..11}):
    #   For block t: rows 2t (B[0,2t]=e), 2t+1 (B[0,2t+1]=f), 12+2t (B[1,2t]=g), 12+2t+1 (B[1,2t+1]=h)
    
    # W factors (24 rows: k*2+i for k in {0..11}, i in {0,1}):
    #   For block t, output C[i, 2t] and C[i, 2t+1]
    #   row 2t: C[0, 2t] (k=2t, i=0)
    #   row 2t+1: C[0, 2t+1] (k=2t+1, i=0)
    #   row 12+2t: C[1, 2t] (k=2t, i=1)
    #   row 12+2t+1: C[1, 2t+1] (k=2t+1, i=1)
    
    rank = 42
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(24)]
    W = [[0] * rank for _ in range(24)]
    
    col = 0
    for t in range(6):
        # Block t: B columns 2t, 2t+1
        # e = B[0, 2t], f = B[0, 2t+1], g = B[1, 2t], h = B[1, 2t+1]
        # V row indices: e->2t, f->2t+1, g->12+2t, h->12+2t+1
        # W row indices: C[0,2t]->2t, C[0,2t+1]->2t+1, C[1,2t]->12+2t, C[1,2t+1]->12+2t+1
        
        # M1 = (a+d)*(e+h): U: a=1 at row0, d=1 at row3; V: e=1 at 2t, h=1 at 12+2t+1
        # W: C[0,2t]+=1, C[0,2t+1]-=1, C[1,2t]+=1, C[1,2t+1]-=1
        U[0][col] += 1; U[3][col] += 1
        V[2*t][col] += 1; V[12+2*t+1][col] += 1
        W[2*t][col] += 1; W[2*t+1][col] -= 1; W[12+2*t][col] += 1; W[12+2*t+1][col] -= 1
        col += 1
        
        # M2 = (c+d)*e: U: c=1 at row2, d=1 at row3; V: e=1 at 2t
        # W: C[1,2t]+=1, C[1,2t+1]-=1
        U[2][col] += 1; U[3][col] += 1
        V[2*t][col] += 1
        W[12+2*t][col] += 1; W[12+2*t+1][col] -= 1
        col += 1
        
        # M3 = a*(f-h): U: a=1 at row0; V: f=1 at 2t+1, h=-1 at 12+2t+1
        # W: C[0,2t+1]+=1, C[1,2t+1]+=1
        U[0][col] += 1
        V[2*t+1][col] += 1; V[12+2*t+1][col] -= 1
        W[2*t+1][col] += 1; W[12+2*t+1][col] += 1
        col += 1
        
        # M4 = d*(g-e): U: d=1 at row3; V: g=1 at 12+2t, e=-1 at 2t
        # W: C[0,2t]+=1, C[1,2t]+=1
        U[3][col] += 1
        V[12+2*t][col] += 1; V[2*t][col] -= 1
        W[2*t][col] += 1; W[12+2*t][col] += 1
        col += 1
        
        # M5 = (a+b)*h: U: a=1 at row0, b=1 at row1; V: h=1 at 12+2t+1
        # W: C[0,2t]-=1, C[0,2t+1]+=1
        U[0][col] += 1; U[1][col] += 1
        V[12+2*t+1][col] += 1
        W[2*t][col] -= 1; W[2*t+1][col] += 1
        col += 1
        
        # M6 = (c-a)*(e+f): U: c=1 at row2, a=-1 at row0; V: e=1 at 2t, f=1 at 2t+1
        # W: C[1,2t+1]+=1
        U[2][col] += 1; U[0][col] -= 1
        V[2*t][col] += 1; V[2*t+1][col] += 1
        W[12+2*t+1][col] += 1
        col += 1
        
        # M7 = (b-d)*(g+h): U: b=1 at row1, d=-1 at row3; V: g=1 at 12+2t, h=1 at 12+2t+1
        # W: C[0,2t]+=1
        U[1][col] += 1; U[3][col] -= 1
        V[12+2*t][col] += 1; V[12+2*t+1][col] += 1
        W[2*t][col] += 1
        col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
