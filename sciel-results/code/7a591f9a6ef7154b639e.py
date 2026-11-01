"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    # Total rank calculation:
    # 2 blocks of j (size 2)
    # Each block has 3 Strassen (2x2x2) and 1 Schoolbook (2x2x1)
    # Strassen rank 7, Schoolbook rank 4
    # Rank per j-block = 3*7 + 4 = 25
    # Total rank = 2 * 25 = 50
    rank = 50
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen block
    def add_strassen(j_off, k_off):
        nonlocal col
        # M1 = (a+d)(e+h)
        # a=A[0,0], d=A[1,1], e=B[0,0], h=B[1,1]
        # U rows: i*m + j
        # V rows: j*p + k
        # W rows: k*n + i
        
        # M1
        U[0*m + j_off][col] = 1   # a
        U[1*m + j_off][col] = 1   # d
        V[j_off*p + k_off][col] = 1   # e
        V[(j_off+1)*p + k_off+1][col] = 1   # h
        # C11 += M1, C22 += M1
        # C11: i=0, k=0 (local) -> global k=k_off, i=0
        W[(k_off)*n + 0][col] = 1
        # C22: i=1, k=1 (local) -> global k=k_off+1, i=1
        W[(k_off+1)*n + 1][col] = 1
        col += 1
        
        # M2 = (c+d)e
        # c=A[1,0], d=A[1,1], e=B[0,0]
        U[1*m + j_off][col] = 1   # c
        U[1*m + j_off+1][col] = 1 # d -- Wait, d is A[1,1] which is row 1, col 1.
        # A is 2x2.
        # A[0,0]=a, A[0,1]=b
        # A[1,0]=c, A[1,1]=d
        # So d is at index 1*m + (j_off+1)
        U[1*m + j_off+1][col] = 1 
        V[j_off*p + k_off][col] = 1   # e
        # C21 += M2 (i=1, k=0 local)
        W[(k_off)*n + 1][col] = 1
        col += 1
        
        # M3 = a(f-h)
        # a=A[0,0], f=B[0,1], h=B[1,1]
        U[0*m + j_off][col] = 1   # a
        V[j_off*p + k_off+1][col] = 1   # f
        V[(j_off+1)*p + k_off+1][col] = -1   # -h
        # C12 += M3 (i=0, k=1 local)
        W[(k_off+1)*n + 0][col] = 1
        col += 1
        
        # M4 = d(g-e)
        # d=A[1,1], g=B[1,0], e=B[0,0]
        U[1*m + j_off+1][col] = 1   # d
        V[(j_off+1)*p + k_off][col] = 1   # g
        V[j_off*p + k_off][col] = -1   # -e
        # C11 += M4, C21 += M4
        W[(k_off)*n + 0][col] = 1
        W[(k_off)*n + 1][col] = 1
        col += 1
        
        # M5 = (a+b)h
        # a=A[0,0], b=A[0,1], h=B[1,1]
        U[0*m + j_off][col] = 1   # a
        U[0*m + j_off+1][col] = 1 # b
        V[(j_off+1)*p + k_off+1][col] = 1   # h
        # C11 -= M5, C12 += M5
        W[(k_off)*n + 0][col] = -1
        W[(k_off+1)*n + 0][col] = 1
        col += 1
        
        # M6 = (c-a)(e+f)
        # c=A[1,0], a=A[0,0], e=B[0,0], f=B[0,1]
        U[1*m + j_off][col] = 1   # c
        U[0*m + j_off][col] = -1  # -a
        V[j_off*p + k_off][col] = 1   # e
        V[j_off*p + k_off+1][col] = 1   # f
        # C22 += M6
        W[(k_off+1)*n + 1][col] = 1
        col += 1
        
        # M7 = (b-d)(g+h)
        # b=A[0,1], d=A[1,1], g=B[1,0], h=B[1,1]
        U[0*m + j_off+1][col] = 1   # b
        U[1*m + j_off+1][col] = -1  # -d
        V[(j_off+1)*p + k_off][col] = 1   # g
        V[(j_off+1)*p + k_off+1][col] = 1   # h
        # C11 += M7, C22 += M7
        W[(k_off)*n + 0][col] = 1
        W[(k_off+1)*n + 1][col] = 1
        col += 1

    # Helper to add Schoolbook 2x2x1 block
    def add_schoolbook_2x2x1(j_off, k_off):
        nonlocal col
        # C11 = a e + b f
        # C21 = c e + d f
        # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
        # e=B[0,0], f=B[1,0]
        
        # Term 1: a e
        U[0*m + j_off][col] = 1
        V[j_off*p + k_off][col] = 1
        W[(k_off)*n + 0][col] = 1
        col += 1
        
        # Term 2: b f
        U[0*m + j_off+1][col] = 1
        V[(j_off+1)*p + k_off][col] = 1
        W[(k_off)*n + 0][col] = 1
        col += 1
        
        # Term 3: c e
        U[1*m + j_off][col] = 1
        V[j_off*p + k_off][col] = 1
        W[(k_off)*n + 1][col] = 1
        col += 1
        
        # Term 4: d f
        U[1*m + j_off+1][col] = 1
        V[(j_off+1)*p + k_off][col] = 1
        W[(k_off)*n + 1][col] = 1
        col += 1

    # Iterate over j blocks
    for j_block in range(2):
        j_off = j_block * 2
        # 3 Strassen blocks for k
        for k_block in range(3):
            k_off = k_block * 2
            if time.time() > deadline:
                break
            add_strassen(j_off, k_off)
        # 1 Schoolbook block for k
        if time.time() <= deadline:
            add_schoolbook_2x2x1(j_off, 6)
        else:
            break
            
    if time.time() > deadline:
        # Return whatever we have, or schoolbook if incomplete?
        # Better to return a valid schoolbook if we fail?
        # But rank is fixed. If we don't fill all columns, it's invalid.
        # We should ensure we complete. 50 is small.
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
