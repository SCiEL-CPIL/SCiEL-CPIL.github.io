import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s margin
    
    n, m, p = DIMENSIONS
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # We will build the matrices as lists of lists, then convert to numpy if needed?
    # The task says "integer/half-integer entries". Lists are fine.
    # But we don't know the rank in advance? 
    # We know it's 84.
    # Let's pre-allocate with size 84.
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define the 6 Strassen blocks
    # I offsets: 0
    # J offsets: 0, 2
    # K offsets: 0, 2, 4
    
    blocks = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 0, 4),
        (0, 2, 0),
        (0, 2, 2),
        (0, 2, 4)
    ]
    
    for io, jo, ko in blocks:
        if time.time() > deadline:
            break
            
        # Local indices 0,1 map to global io, io+1 etc.
        
        # M1
        # U: A00 (io, jo), A11 (io+1, jo+1)
        U[io * m + jo][col] += 1
        U[(io+1) * m + (jo+1)][col] += 1
        # V: B00 (jo, ko), B11 (jo+1, ko+1)
        V[jo * p + ko][col] += 1
        V[(jo+1) * p + (ko+1)][col] += 1
        # W: C00 (ko, io), C11 (ko+1, io+1)
        W[ko * n + io][col] += 1
        W[(ko+1) * n + (io+1)][col] += 1
        col += 1
        
        # M2
        # U: A10 (io+1, jo), A11 (io+1, jo+1)
        U[(io+1) * m + jo][col] += 1
        U[(io+1) * m + (jo+1)][col] += 1
        # V: B00 (jo, ko)
        V[jo * p + ko][col] += 1
        # W: C10 (ko, io+1), C11 (ko+1, io+1) with -1
        W[ko * n + (io+1)][col] += 1
        W[(ko+1) * n + (io+1)][col] += -1
        col += 1
        
        # M3
        # U: A00 (io, jo)
        U[io * m + jo][col] += 1
        # V: B10 (jo+1, ko), B11 (jo+1, ko+1) with -1
        V[(jo+1) * p + ko][col] += 1
        V[(jo+1) * p + (ko+1)][col] += -1
        # W: C01 (ko+1, io), C11 (ko+1, io+1)
        W[(ko+1) * n + io][col] += 1
        W[(ko+1) * n + (io+1)][col] += 1
        col += 1
        
        # M4
        # U: A11 (io+1, jo+1)
        U[(io+1) * m + (jo+1)][col] += 1
        # V: B01 (jo, ko+1), B00 (jo, ko) with -1
        V[jo * p + (ko+1)][col] += 1
        V[jo * p + ko][col] += -1
        # W: C00 (ko, io), C10 (ko, io+1)
        W[ko * n + io][col] += 1
        W[ko * n + (io+1)][col] += 1
        col += 1
        
        # M5
        # U: A00 (io, jo), A01 (io, jo+1)
        U[io * m + jo][col] += 1
        U[io * m + (jo+1)][col] += 1
        # V: B11 (jo+1, ko+1)
        V[(jo+1) * p + (ko+1)][col] += 1
        # W: C00 (ko, io) with -1, C01 (ko+1, io)
        W[ko * n + io][col] += -1
        W[(ko+1) * n + io][col] += 1
        col += 1
        
        # M6
        # U: A10 (io+1, jo), A00 (io, jo) with -1
        U[(io+1) * m + jo][col] += 1
        U[io * m + jo][col] += -1
        # V: B00 (jo, ko), B01 (jo, ko+1)
        V[jo * p + ko][col] += 1
        V[jo * p + (ko+1)][col] += 1
        # W: C11 (ko+1, io+1)
        W[(ko+1) * n + (io+1)][col] += 1
        col += 1
        
        # M7
        # U: A01 (io, jo+1), A11 (io+1, jo+1) with -1
        U[io * m + (jo+1)][col] += 1
        U[(io+1) * m + (jo+1)][col] += -1
        # V: B10 (jo+1, ko), B11 (jo+1, ko+1)
        V[(jo+1) * p + ko][col] += 1
        V[(jo+1) * p + (ko+1)][col] += 1
        # W: C00 (ko, io)
        W[ko * n + io][col] += 1
        col += 1

    # Now handle singletons
    # Indices not covered by blocks:
    # i=2, all j, k
    # i=0,1, j=4, all k
    
    singletons = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if (i,j,k) is in any block
                in_block = False
                for io, jo, ko in blocks:
                    if i in [io, io+1] and j in [jo, jo+1] and k in [ko, ko+1]:
                        in_block = True
                        break
                if not in_block:
                    singletons.append((i, j, k))
    
    for i, j, k in singletons:
        if time.time() > deadline:
            break
        U[i * m + j][col] += 1
        V[j * p + k][col] += 1
        W[k * n + i][col] += 1
        col += 1
        
    # Verify col == rank
    # If col < rank, we have extra zero columns. 
    # The rank is the number of non-zero columns? 
    # The interface returns the matrices. The evaluator calculates the rank or just uses the size?
    # "score is the rank". Usually rank of the matrices provided.
    # If we provide 84 columns, but some are zero, the rank might be lower? 
    # No, the rank of the decomposition is the number of terms. 
    # If I provide 84 columns, the rank is 84.
    # If I provide fewer, it's invalid.
    # My construction uses exactly 6*7 + 42 = 84 columns.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
