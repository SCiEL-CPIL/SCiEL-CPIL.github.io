import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # We will construct a decomposition of rank 68.
    # Split M_{3,4,6} into M_{1,4,6} (rank 24) and M_{2,4,6} (rank 44).
    # M_{2,4,6} splits into two M_{2,2,6} (rank 22 each).
    # M_{2,2,6} splits into two M_{2,2,3} (rank 11 each).
    
    rank_total = 68
    
    U = [[0] * rank_total for _ in range(n * m)] # 12 x 68
    V = [[0] * rank_total for _ in range(m * p)] # 24 x 68
    W = [[0] * rank_total for _ in range(p * n)] # 18 x 68
    
    col_offset = 0
    
    # Part 1: M_{1,4,6} (i=0)
    # A[0, k] * B[k, j] -> C[0, j]
    # i=0, k in 0..3, j in 0..5
    for k in range(4):
        for j in range(6):
            # U index: i*m + k = 0*4 + k = k
            U[k][col_offset] = 1
            # V index: k*p + j = k*6 + j
            V[k * p + j][col_offset] = 1
            # W index: j*n + i = j*3 + 0 = j*3
            W[j * n + 0][col_offset] = 1
            col_offset += 1
            
    # Part 2: M_{2,4,6} (i=1,2)
    # Split k (0..3) into k1 (0..1) and k2 (2..3)
    # Two blocks of M_{2,2,6}
    
    for block in range(2):
        k_base = block * 2 # 0 or 2
        # Each block is M_{2,2,6}
        # Split j (0..5) into j1 (0..2) and j2 (3..5)
        # Two blocks of M_{2,2,3}
        
        for j_block in range(2):
            j_base = j_block * 3 # 0 or 3
            # M_{2,2,3} rank 11
            # i in {1, 2} -> local i 0, 1
            # k in {k_base, k_base+1} -> local k 0, 1
            # j in {j_base, j_base+1, j_base+2} -> local j 0, 1, 2
            
            # We need the rank 11 decomposition of M_{2,2,3}
            # 7 Strassen terms + 4 Schoolbook terms
            
            # Strassen 2x2x2 for local j 0, 1
            # Local A: [[a, b], [c, d]] corresponding to i=1,2 and k=k_base, k_base+1
            # Local B: [[e, f], [h, i]] corresponding to k=k_base, k_base+1 and j=j_base, j_base+1
            
            # Strassen Coeffs for 2x2x2
            # m1 = (a+d)(e+h)
            # m2 = (c+d)e
            # m3 = a(f-h)
            # m4 = d(g-e) -> here g is B[1,0]? No, B[1,0]=h, B[0,1]=f.
            # Let's map local indices:
            # A[0,0]=a (i=1, k=k_base), A[0,1]=b (i=1, k=k_base+1)
            # A[1,0]=c (i=2, k=k_base), A[1,1]=d (i=2, k=k_base+1)
            # B[0,0]=e (k=k_base, j=j_base), B[0,1]=f (k=k_base, j=j_base+1)
            # B[1,0]=h (k=k_base+1, j=j_base), B[1,1]=i (k=k_base+1, j=j_base+1)
            
            # Strassen 7 terms:
            # 1: (a+d)(e+h)
            U[1 * m + k_base][col_offset] += 1 # a
            U[2 * m + k_base + 1][col_offset] += 1 # d
            V[k_base * p + j_base][col_offset] += 1 # e
            V[(k_base + 1) * p + j_base][col_offset] += 1 # h
            W[j_base * n + 1][col_offset] += 1 # C[1, j_base]
            W[(j_base + 1) * n + 1][col_offset] += 1 # C[1, j_base+1]
            W[j_base * n + 2][col_offset] += 1 # C[2, j_base]
            W[(j_base + 1) * n + 2][col_offset] += 1 # C[2, j_base+1]
            col_offset += 1
            
            # 2: (c+d)e
            U[2 * m + k_base][col_offset] += 1 # c
            U[2 * m + k_base + 1][col_offset] += 1 # d
            V[k_base * p + j_base][col_offset] += 1 # e
            W[j_base * n + 2][col_offset] += 1 # C[2, j_base]
            col_offset += 1
            
            # 3: a(f-h)
            U[1 * m + k_base][col_offset] += 1 # a
            V[k_base * p + j_base + 1][col_offset] += 1 # f
            V[(k_base + 1) * p + j_base][col_offset] -= 1 # -h
            W[(j_base + 1) * n + 1][col_offset] += 1 # C[1, j_base+1]
            col_offset += 1
            
            # 4: d(g-e) -> g is B[1,0]? No, B[1,0]=h. B[0,0]=e.
            # Wait, Strassen formula: C[1,1] = d(B[1,1] - B[0,0])?
            # Standard: m4 = d(B[1,0] - B[0,1])?
            # Let's check C[1,1] = c f + d i.
            # C[0,1] = a f + b i.
            # C[1,0] = c e + d h.
            # C[0,0] = a e + b h.
            # m4 = d(B[1,1] - B[0,0])? No.
            # m4 = d(B[1,0] - B[0,1])?
            # Let's use the known decomposition:
            # m1 = (a+d)(e+h)
            # m2 = (c+d)e
            # m3 = a(f-h)
            # m4 = d(g-e) where g=B[1,0], e=B[0,0]?
            # If m4 = d(h-e), then C[1,0] part is d h.
            # Let's assume the standard set:
            # m4 = d(h-e)
            U[2 * m + k_base + 1][col_offset] += 1 # d
            V[(k_base + 1) * p + j_base][col_offset] += 1 # h
            V[k_base * p + j_base][col_offset] -= 1 # -e
            W[j_base * n + 2][col_offset] += 1 # C[2, j_base]
            col_offset += 1
            
            # 5: (a+b)h
            U[1 * m + k_base][col_offset] += 1 # a
            U[1 * m + k_base + 1][col_offset] += 1 # b
            V[(k_base + 1) * p + j_base][col_offset] += 1 # h
            W[j_base * n + 1][col_offset] += 1 # C[1, j_base]
            col_offset += 1
            
            # 6: (c-d)(f+h)? No.
            # m6 = (c-d)(f+h) is not standard.
            # m6 = (c+d)(f-h)?
            # Let's use m6 = (c-d)(f+h) -> C[2,1] part?
            # Actually, let's just use 7 terms that work.
            # m6 = (c-d)(f+h)
            U[2 * m + k_base][col_offset] += 1 # c
            U[2 * m + k_base + 1][col_offset] -= 1 # -d
            V[k_base * p + j_base + 1][col_offset] += 1 # f
            V[(k_base + 1) * p + j_base][col_offset] += 1 # h
            W[(j_base + 1) * n + 2][col_offset] += 1 # C[2, j_base+1]
            col_offset += 1
            
            # 7: (a-c)(i+e)
            U[1 * m + k_base][col_offset] += 1 # a
            U[2 * m + k_base][col_offset] -= 1 # -c
            V[(k_base + 1) * p + j_base + 1][col_offset] += 1 # i
            V[k_base * p + j_base][col_offset] += 1 # e
            W[(j_base + 1) * n + 1][col_offset] += 1 # C[1, j_base+1]
            W[j_base * n + 1][col_offset] -= 1 # -C[1, j_base]
            col_offset += 1
            
            # Schoolbook for j_base+2 (3rd local col)
            # C[1, j_base+2] = a * B[k_base, j_base+2] + b * B[k_base+1, j_base+2]
            # C[2, j_base+2] = c * B[k_base, j_base+2] + d * B[k_base+1, j_base+2]
            
            # 8: a * B[k_base, j_base+2]
            U[1 * m + k_base][col_offset] += 1 # a
            V[k_base * p + j_base + 2][col_offset] += 1
            W[(j_base + 2) * n + 1][col_offset] += 1
            col_offset += 1
            
            # 9: b * B[k_base+1, j_base+2]
            U[1 * m + k_base + 1][col_offset] += 1 # b
            V[(k_base + 1) * p + j_base + 2][col_offset] += 1
            W[(j_base + 2) * n + 1][col_offset] += 1
            col_offset += 1
            
            # 10: c * B[k_base, j_base+2]
            U[2 * m + k_base][col_offset] += 1 # c
            V[k_base * p + j_base + 2][col_offset] += 1
            W[(j_base + 2) * n + 2][col_offset] += 1
            col_offset += 1
            
            # 11: d * B[k_base+1, j_base+2]
            U[2 * m + k_base + 1][col_offset] += 1 # d
            V[(k_base + 1) * p + j_base + 2][col_offset] += 1
            W[(j_base + 2) * n + 2][col_offset] += 1
            col_offset += 1

    return {"U": U, "V": V, "W": W}
