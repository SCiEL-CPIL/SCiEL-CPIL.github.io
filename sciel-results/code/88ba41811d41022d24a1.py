import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, col):
        for idx in u_indices:
            U[idx][col] += 1
        for idx in v_indices:
            V[idx][col] += 1
        for idx, val in w_indices.items():
            W[idx][col] += val

    # Define Strassen products for 2x2 by 2x2
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    
    # Reconstruction
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6

    # We have two blocks: A1 B1 and A2 B2
    # A1 is A[:, 0:2], A2 is A[:, 2:4]
    # B1 is B[0:2, :], B2 is B[2:4, :]
    
    # For A1 B1:
    # A1 elements: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # B1 elements: e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1] (for Y_L)
    #              i=B[0,2], j=B[0,3], k=B[1,2], l=B[1,3] (for Y_R)
    
    # Indices in U (A is 2x4, row i, col j -> i*4+j)
    # a: 0*4+0=0
    # b: 0*4+1=1
    # c: 1*4+0=4
    # d: 1*4+1=5
    
    # Indices in V (B is 4x4, row j, col k -> j*4+k)
    # e: 0*4+0=0
    # f: 0*4+1=1
    # g: 1*4+0=4
    # h: 1*4+1=5
    # i: 0*4+2=2
    # j: 0*4+3=3
    # k: 1*4+2=6
    # l: 1*4+3=7
    
    # Indices in W (C is 4x2? No, C is 2x4. Output factor W[k*n+i, r] with n=2, p=4.
    # Wait, the task says W[k*n+i, r].
    # C is n x p = 2 x 4.
    # k is row of C (0..1), i is col of C (0..3).
    # Index: k*2 + i.
    # C[0,0]: 0*2+0=0
    # C[0,1]: 0*2+1=1
    # C[0,2]: 0*2+2=2
    # C[0,3]: 0*2+3=3
    # C[1,0]: 1*2+0=4
    # C[1,1]: 1*2+1=5
    # C[1,2]: 1*2+2=6
    # C[1,3]: 1*2+3=7
    
    # Let's process A1 B1 (cols 0,1 of A, rows 0,1 of B)
    # Y_L uses B cols 0,1. Y_R uses B cols 2,3.
    
    # Term 1: P1L = (a+d)(e+h)
    add_term([0, 5], [0, 5], {0: 1, 5: 1}, 0) # C[0,0] and C[1,1]
    
    # Term 2: P2L = (c+d)e
    add_term([4, 5], [0], {4: 1, 5: -1}, 1) # C[1,0] and C[1,1]
    
    # Term 3: P3L = a(f-h)
    add_term([0], [1, 5], {1: 1, 5: 1}, 2) # C[0,1] and C[1,1]
    
    # Term 4: P4L = d(g-e)
    add_term([5], [4, 0], {0: 1, 4: 1}, 3) # C[0,0] and C[1,0]
    
    # Term 5: P5L = (a+b)h
    add_term([0, 1], [5], {0: -1, 1: 1}, 4) # C[0,0] and C[0,1]
    
    # Term 6: P6L = (c-a)(e+f)
    add_term([4, 0], [0, 1], {5: 1}, 5) # C[1,1]
    
    # Term 7: P7L = (b-d)(g+h)
    add_term([1, 5], [4, 5], {0: 1}, 6) # C[0,0]
    
    # Now Y_R (B cols 2,3)
    # e->i, f->j, g->k, h->l
    # i=2, j=3, k=6, l=7
    
    # Term 8: P1R = (a+d)(i+j)
    add_term([0, 5], [2, 3], {2: 1, 7: 1}, 7) # C[0,2] and C[1,3]
    
    # Term 9: P2R = (c+d)i
    add_term([4, 5], [2], {6: 1, 7: -1}, 8) # C[1,2] and C[1,3]
    
    # Term 10: P3R = a(j-l)
    add_term([0], [3, 7], {3: 1, 7: 1}, 9) # C[0,3] and C[1,3]
    
    # Term 11: P4R = d(k-i)
    add_term([5], [6, 2], {2: 1, 6: 1}, 10) # C[0,2] and C[1,2]
    
    # Term 12: P5R = (a+b)l
    add_term([0, 1], [7], {2: -1, 3: 1}, 11) # C[0,2] and C[0,3]
    
    # Term 13: P6R = (c-a)(i+j)
    add_term([4, 0], [2, 3], {7: 1}, 12) # C[1,3]
    
    # Term 14: P7R = (b-d)(k+l)
    add_term([1, 5], [6, 7], {2: 1}, 13) # C[0,2]
    
    # Now A2 B2
    # A2 is A[:, 2:4].
    # a2=A[0,2]=8, b2=A[0,3]=9, c2=A[1,2]=12, d2=A[1,3]=13
    # B2 is B[2:4, :].
    # e2=B[2,0]=8, f2=B[2,1]=9, g2=B[3,0]=12, h2=B[3,1]=13
    # i2=B[2,2]=10, j2=B[2,3]=11, k2=B[3,2]=14, l2=B[3,3]=15
    
    # Term 15: P1L2 = (a2+d2)(e2+h2)
    add_term([8, 13], [8, 13], {0: 1, 5: 1}, 14)
    
    # Term 16: P2L2 = (c2+d2)e2
    add_term([12, 13], [8], {4: 1, 5: -1}, 15)
    
    # Term 17: P3L2 = a2(f2-h2)
    add_term([8], [9, 13], {1: 1, 5: 1}, 16)
    
    # Term 18: P4L2 = d2(g2-e2)
    add_term([13], [12, 8], {0: 1, 4: 1}, 17)
    
    # Term 19: P5L2 = (a2+b2)h2
    add_term([8, 9], [13], {0: -1, 1: 1}, 18)
    
    # Term 20: P6L2 = (c2-a2)(e2+f2)
    add_term([12, 8], [8, 9], {5: 1}, 19)
    
    # Term 21: P7L2 = (b2-d2)(g2+h2)
    add_term([9, 13], [12, 13], {0: 1}, 20)
    
    # Term 22: P1R2 = (a2+d2)(i2+j2)
    add_term([8, 13], [10, 11], {2: 1, 7: 1}, 21)
    
    # Term 23: P2R2 = (c2+d2)i2
    add_term([12, 13], [10], {6: 1, 7: -1}, 22)
    
    # Term 24: P3R2 = a2(j2-l2)
    add_term([8], [11, 15], {3: 1, 7: 1}, 23)
    
    # Term 25: P4R2 = d2(k2-i2)
    add_term([13], [14, 10], {2: 1, 6: 1}, 24)
    
    # Term 26: P5R2 = (a2+b2)l2
    add_term([8, 9], [15], {2: -1, 3: 1}, 25)
    
    # Term 27: P6R2 = (c2-a2)(i2+j2)
    add_term([12, 8], [10, 11], {7: 1}, 26)
    
    # Term 28: P7R2 = (b2-d2)(k2+l2)
    add_term([9, 13], [14, 15], {2: 1}, 27)
    
    return {"U": U, "V": V, "W": W}
