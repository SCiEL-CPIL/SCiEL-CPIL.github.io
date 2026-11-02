import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 100 # Safety margin

    n, m, p = 3, 3, 3
    rank = 26
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Helper to set a term
    def set_term(c, u_pairs, v_pairs, w_pairs):
        for i, j, val in u_pairs:
            U[i*m + j][c] += val
        for j, k, val in v_pairs:
            V[j*p + k][c] += val
        for k, i, val in w_pairs:
            W[k*n + i][c] += val

    # Strassen Terms for the 2x2 core (indices 0,1)
    # M1 = (A00+A11)(B00+B11)
    set_term(0, 
             [(0,0,1), (1,1,1)], 
             [(0,0,1), (1,1,1)], 
             [(0,0,1), (1,0,-1), (0,1,1), (1,1,-1)])
             
    # M2 = (A10+A11)B00
    set_term(1, 
             [(1,0,1), (1,1,1)], 
             [(0,0,1)], 
             [(0,0,-1), (1,0,1)])
             
    # M3 = A00(B10-B11)
    set_term(2, 
             [(0,0,1)], 
             [(1,0,1), (1,1,-1)], 
             [(0,1,1), (1,1,1)])
             
    # M4 = A11(B01-B00)
    set_term(3, 
             [(1,1,1)], 
             [(0,1,1), (0,0,-1)], 
             [(0,0,1), (1,0,-1)])
             
    # M5 = (A00+A01)B11
    set_term(4, 
             [(0,0,1), (0,1,1)], 
             [(1,1,1)], 
             [(0,0,-1), (0,1,1), (1,0,-1)])
             
    # M6 = (A10-A00)(B00+B01)
    set_term(5, 
             [(1,0,1), (0,0,-1)], 
             [(0,0,1), (0,1,1)], 
             [(0,1,1), (1,0,1)])
             
    # M7 = (A01-A11)(B10+B11)
    set_term(6, 
             [(0,1,1), (1,1,-1)], 
             [(1,0,1), (1,1,1)], 
             [(0,0,1), (0,1,-1), (1,1,1)])

    # Border Terms (indices involving 2)
    col = 7
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if max(i,j,k) == 2:
                    set_term(col, [(i,j,1)], [(j,k,1)], [(k,i,1)])
                    col += 1
                    
    # Verify
    import random
    random.seed(42)
    valid = True
    for _ in range(5):
        A = [[random.randint(-5,5) for _ in range(3)] for _ in range(3)]
        B = [[random.randint(-5,5) for _ in range(3)] for _ in range(3)]
        AB = [[sum(A[i][j]*B[j][k] for j in range(3)) for k in range(3)] for i in range(3)]
        C = [[0]*3 for _ in range(3)]
        for c in range(rank):
            for i in range(3):
                for k in range(3):
                    # Compute C[i,k] contribution from term c
                    # C[i,k] += sum_j U[i,j,c] * A[i,j] * V[j,k,c] * B[j,k] * W[k,i,c]
                    # Actually, the decomposition is:
                    # C[i,k] = sum_c (sum_j U[i,j,c] A[i,j]) (sum_j V[j,k,c] B[j,k]) W[k,i,c]
                    # Let's compute this way to be safe and fast.
                    sum_U = 0
                    sum_V = 0
                    for j in range(3):
                        sum_U += U[i*3+j][c] * A[i][j]
                        sum_V += V[j*3+k][c] * B[j][k]
                    C[i][k] += sum_U * sum_V * W[k*3+i][c]
        
        if C != AB:
            valid = False
            break
            
    if not valid or time.time() > deadline:
        # Fallback to Schoolbook Rank 27
        rank = 27
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}
