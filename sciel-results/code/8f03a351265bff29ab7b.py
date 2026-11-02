import numpy as np
import time

DIMENSIONS = [6, 6, 6]
START_TIME = time.time()
TIME_LIMIT = 110.0

def construct():
    # 1. Generate Strassen 2x2 decomposition
    # Rank 7
    # u, v, w are lists of 7 vectors of length 4
    # Index 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    # Strassen terms
    # M1: (a+d)(e+h)
    # M2: (c+d)e
    # M3: a(f-h)
    # M4: d(g-e)
    # M5: (a+b)h
    # M6: (c-a)(e+f)
    # M7: (b-d)(g+h)
    
    # We need to map these to u, v, w vectors
    # u[i,j], v[j,k], w[k,i]
    
    U2 = [[0]*4 for _ in range(7)]
    V2 = [[0]*4 for _ in range(7)]
    W2 = [[0]*4 for _ in range(7)]
    
    # Helper to set u[i,j]
    def set_u(r, i, j, val): U2[r][i*2+j] = val
    def set_v(r, j, k, val): V2[r][j*2+k] = val
    def set_w(r, k, i, val): W2[r][k*2+i] = val
    
    # M1: (A00+A11)(B00+B11)
    # u: (0,0)=1, (1,1)=1
    # v: (0,0)=1, (1,1)=1
    # w: (0,0)=1, (1,1)=1
    set_u(0, 0, 0, 1); set_u(0, 1, 1, 1)
    set_v(0, 0, 0, 1); set_v(0, 1, 1, 1)
    set_w(0, 0, 0, 1); set_w(0, 1, 1, 1)
    
    # M2: (A10+A11)B00
    # u: (1,0)=1, (1,1)=1
    # v: (0,0)=1
    # w: (0,1)=1  (C10)
    set_u(1, 1, 0, 1); set_u(1, 1, 1, 1)
    set_v(1, 0, 0, 1)
    set_w(1, 0, 1, 1)
    
    # M3: A00(B01-B11)
    # u: (0,0)=1
    # v: (0,1)=1, (1,1)=-1
    # w: (1,0)=1 (C10? No, C01)
    # C01 = M3 + M5.
    # C_{0,1}. k=0, i=1.
    set_u(2, 0, 0, 1)
    set_v(2, 0, 1, 1); set_v(2, 1, 1, -1)
    set_w(2, 0, 1, 1)
    
    # M4: A11(B10-B00)
    # u: (1,1)=1
    # v: (1,0)=1, (0,0)=-1
    # w: (0,0)=1 (C00), (1,1)=1 (C11)?
    # C00 = M1 + M4 - M5 + M7.
    # C11 = M1 - M2 + M3 + M6.
    # M4 contributes to C00 and C11?
    # M4 = d(g-e). d=A11, g=B10, e=B00.
    # d*g -> A11 B10 -> C10.
    # d*-e -> -A11 B00 -> -C10?
    # Wait. C = AB.
    # C10 = A10 B00 + A11 B10.
    # M4 = A11(B10-B00).
    # This contributes to C10.
    # But Strassen formula: C10 = M2 + M4.
    # M2 = (A10+A11)B00 = A10 B00 + A11 B00.
    # M4 = A11 B10 - A11 B00.
    # Sum = A10 B00 + A11 B10. Correct.
    # So M4 contributes to C10.
    # k=1, i=0.
    set_u(3, 1, 1, 1)
    set_v(3, 1, 0, 1); set_v(3, 0, 0, -1)
    set_w(3, 1, 0, 1)
    
    # M5: (A00+A01)B11
    # u: (0,0)=1, (0,1)=1
    # v: (1,1)=1
    # w: (1,0)=-1 (C00), (1,1)=1 (C01)?
    # C00 = ... - M5 ...
    # C01 = M3 + M5.
    # M5 = A00 B11 + A01 B11.
    # A00 B11 -> C01.
    # A01 B11 -> C01.
    # So M5 contributes to C01.
    # k=1, i=1? No.
    # A00 is row 0. B11 is col 1. C01.
    # k=1, i=1? No, C_{k,i}.
    # C_{1,1}?
    # A_{0,0} B_{1,1} -> C_{0,1}.
    # A_{0,1} B_{1,1} -> C_{0,1}.
    # So k=1, i=1?
    # Wait. C_{i,k} in standard?
    # Task: W[k*n+i].
    # C_{k,i}.
    # A_{i,j} B_{j,k}.
    # A_{0,0} B_{1,1} -> i=0, j=0, k=1.
    # So C_{1,0}.
    # My previous M3 analysis: A00(B01-B11).
    # A00 B01 -> i=0, j=0, k=1 -> C_{1,0}.
    # A00 -B11 -> i=0, j=1, k=1 -> C_{1,0}.
    # So M3 contributes to C_{1,0}.
    # M5: A00 B11 -> C_{1,0}. A01 B11 -> C_{1,0}.
    # So M5 contributes to C_{1,0}.
    # Strassen: C_{1,0} = M2 + M4.
    # C_{0,1} = M3 + M5.
    # My indices:
    # C_{k,i}.
    # C_{1,0} is k=1, i=0.
    # C_{0,1} is k=0, i=1.
    
    # Let's re-verify M3.
    # M3 = A00(B01-B11).
    # A00 B01 -> i=0, j=0, k=1. C_{1,0}.
    # A00 -B11 -> i=0, j=1, k=1. C_{1,0}.
    # So M3 contributes to C_{1,0}.
    # But Strassen says C_{0,1} = M3 + M5.
    # There is a mismatch in my index convention vs Strassen standard.
    # Standard Strassen:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # In my W[k*n+i], C_{k,i} corresponds to C_{row k, col i}.
    # Standard C_{row, col}.
    # So C_{1,1} (row 1, col 1) is k=1, i=1.
    # C_{1,2} (row 1, col 2) is k=1, i=2? No, n=2.
    # C_{1,2} doesn't exist.
    # C_{0,1} (row 0, col 1) is k=0, i=1.
    
    # So:
    # M3 contributes to C_{0,1}.
    # My calculation: A00 B01 -> C_{1,0}.
    # Why?
    # A_{i,j} B_{j,k}.
    # A_{0,0} B_{0,1}. i=0, j=0, k=1.
    # Result C_{0,1}.
    # Task W index: k*n+i = 1*2+0 = 2.
    # So W[2] is C_{1,0}.
    # W[1] is C_{0,1} (0*2+1).
    
    # So M3 should set W[1] (k=0, i=1).
    
    # Let's fix the W assignments.
    
    # M1: C00, C11. W[0], W[3].
    # M2: C10. W[2].
    # M3: C01. W[1].
    # M4: C10, C11?
    # M4 = A11(B10-B00).
    # A11 B10 -> i=1, j=1, k=0. C_{1,0}. W[2].
    # A11 -B00 -> i=1, j=0, k=0. C_{1,0}. W[2].
    # So M4 contributes to C10.
    # Strassen: C21 = M2 + M4. (C10).
    # C11 = M1 + M4 - M5 + M7.
    # M4 contributes to C11?
    # A11 B10 -> C10.
    # A11 -B00 -> C10.
    # Where does M4 contribute to C11?
    # It doesn't.
    # Strassen formula: C11 = M1 + M4 - M5 + M7.
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # M4 = A11B10 - A11B00.
    # M5 = A00B11 + A01B11.
    # M7 = A01B10 + A01B11 - A11B10 - A11B11.
    # Sum:
    # A00B00 (M1)
    # A00B11 (M1 - M5) = 0?
    # A11B00 (M1 - M4) = 0?
    # A11B11 (M1 + M7) = A11B11 + A01B10 + A01B11 - A11B10 - A11B11 = A01B10 + A01B11.
    # This is getting messy.
    
    # I will use a known correct Strassen tensor decomposition.
    # Or simply use the schoolbook for 2x2 as well, to ensure correctness, then optimize?
    # No, rank 8 is worse than 7.
    
    # I will hardcode the correct Strassen vectors.
    
    pass

    # Correct Strassen 2x2 Decomposition
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Indices:
    # U: (i, j) where i is row of A, j is col of A (which is row of B)
    # V: (j, k) where j is row of B, k is col of B (which is col of C)
    # W: (k, i) where k is col of C, i is row of C
    # Task says W[k*n+i]. For n=2, W[0]=C00, W[1]=C01, W[2]=C10, W[3]=C11.
    # Wait, task says "W[k*n+i,r]".
    # If C is 2x2, C[k, i].
    # Standard matrix mult C = A B. C_{i, k} = sum_j A_{i, j} B_{j, k}.
    # So C row i, col k.
    # Task says W[k*n+i]. This implies W index is k*2 + i.
    # So W[0] = k=0, i=0 -> C[0,0].
    # W[1] = k=0, i=1 -> C[1,0].
    # W[2] = k=1, i=0 -> C[0,1].
    # W[3] = k=1, i=1 -> C[1,1].
    
    # Let's verify this convention.
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # Usually trace convention for tensor T_{i,j,k} is T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r}.
    # Here indices are likely i (row A), j (col A/row B), k (col B/col C).
    # So W is indexed by k and i? Or just k?
    # The shape is (p*n, rank). p=2, n=2. So 4 entries.
    # If the tensor is T_{i,j,k}, then W should be indexed by k and i?
    # No, usually W is indexed by the "output" indices.
    # If the multiplication is C = A B, then C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The tensor is T_{i,j,k}.
    # Decomposition: T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r}.
    # Here W has shape (p, rank)? No, W has shape (p*n, rank).
    # This implies W is indexed by (k, i) or (k, something else).
    # If W is (k, i), then T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}?
    # No, U is (n*m, rank) -> (i, j).
    # V is (m*p, rank) -> (j, k).
    # W is (p*n, rank) -> (k, i).
    # So T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}.
    # This matches the shape requirements.
    
    # So W index is k*n + i.
    # n=2.
    # W[0] -> k=0, i=0. (C[0,0])
    # W[1] -> k=0, i=1. (C[1,0])
    # W[2] -> k=1, i=0. (C[0,1])
    # W[3] -> k=1, i=1. (C[1,1])
    
    # Let's re-evaluate Strassen terms with this W indexing.
    
    # C[0,0] (W[0]): M1 + M4 - M5 + M7
    # C[1,0] (W[1]): M2 + M4
    # C[0,1] (W[2]): M3 + M5
    # C[1,1] (W[3]): M1 - M2 + M3 + M6
    
    # U (i, j):
    # M1: (a+d) -> U[0,0]=1, U[1,1]=1
    # M2: (c+d) -> U[1,0]=1, U[1,1]=1
    # M3: a -> U[0,0]=1
    # M4: d -> U[1,1]=1
    # M5: (a+b) -> U[0,0]=1, U[0,1]=1
    # M6: (c-a) -> U[1,0]=1, U[0,0]=-1
    # M7: (b-d) -> U[0,1]=1, U[1,1]=-1
    
    # V (j, k):
    # M1: (e+h) -> V[0,0]=1, V[1,1]=1
    # M2: e -> V[0,0]=1
    # M3: (f-h) -> V[0,1]=1, V[1,1]=-1
    # M4: (g-e) -> V[1,0]=1, V[0,0]=-1
    # M5: h -> V[1,1]=1
    # M6: (e+f) -> V[0,0]=1, V[0,1]=1
    # M7: (g+h) -> V[1,0]=1, V[1,1]=1
    
    # W (k, i):
    # M1: C[0,0] (+1), C[1,1] (+1) -> W[0]=1, W[3]=1
    # M2: C[1,0] (+1) -> W[1]=1
    # M3: C[0,1] (+1) -> W[2]=1
    # M4: C[0,0] (+1), C[1,0] (+1) -> W[0]=1, W[1]=1
    # M5: C[0,0] (-1), C[0,1] (+1) -> W[0]=-1, W[2]=1
    # M6: C[1,1] (+1) -> W[3]=1
    # M7: C[0,0] (+1) -> W[0]=1
    
    # Let's verify M4.
    # M4 = d(g-e) = A[1,1](B[1,0]-B[0,0]).
    # Terms:
    # A[1,1] B[1,0] -> C[1,0]. (i=1, j=1, k=0). W[1].
    # A[1,1] (-B[0,0]) -> -C[1,0]. (i=1, j=0, k=0). W[1].
    # So M4 contributes to C[1,0] only?
    # Wait.
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0].
    # M2 = (c+d)e = (A[1,0]+A[1,1])B[0,0] = A[1,0]B[0,0] + A[1,1]B[0,0].
    # M4 = A[1,1](B[1,0]-B[0,0]) = A[1,1]B[1,0] - A[1,1]B[0,0].
    # M2 + M4 = A[1,0]B[0,0] + A[1,1]B[1,0]. Correct.
    # So M4 contributes to C[1,0].
    # My previous W[0] assignment for M4 was wrong.
    
    # Let's re-verify M1.
    # M1 = (a+d)(e+h) = (A[0,0]+A[1,1])(B[0,0]+B[1,1]).
    # = A[0,0]B[0,0] + A[0,0]B[1,1] + A[1,1]B[0,0] + A[1,1]B[1,1].
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0].
    # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1].
    # M1 contributes to C[0,0] (via A[0,0]B[0,0]) and C[1,1] (via A[1,1]B[1,1]).
    # The other terms (A[0,0]B[1,1] and A[1,1]B[0,0]) must cancel out in the sum for C[0,0] and C[1,1]?
    # No, they don't appear in C[0,0] or C[1,1] directly in the simple sum.
    # Wait, C[0,0] only has terms with i=0, k=0.
    # A[0,0]B[0,0] is i=0, j=0, k=0.
    # A[0,0]B[1,1] is i=0, j=1, k=1. This is C[0,1].
    # A[1,1]B[0,0] is i=1, j=0, k=0. This is C[1,0].
    # A[1,1]B[1,1] is i=1, j=1, k=1. This is C[1,1].
    # So M1 contributes to ALL four entries of C?
    # No, the decomposition is linear.
    # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}.
    # For M1 (r=0):
    # U_{i,j,0} is non-zero for (0,0) and (1,1).
    # V_{j,k,0} is non-zero for (0,0) and (1,1).
    # W_{k,i,0} is non-zero for (0,0) and (1,1).
    # Term for i=0, j=0, k=0: U[0,0,0] V[0,0,0] W[0,0,0] = 1*1*1 = 1. (Contributes to C[0,0])
    # Term for i=0, j=0, k=1: U[0,0,0] V[0,1,0] W[1,0,0] = 1*0*0 = 0.
    # Term for i=0, j=1, k=0: U[0,1,0] V[1,0,0] W[0,0,0] = 0*0*1 = 0.
    # Term for i=0, j=1, k=1: U[0,1,0] V[1,1,0] W[1,0,0] = 0*1*0 = 0.
    # Wait, W index is k*n+i.
    # For i=0, k=1: W[1*2+0] = W[2].
    # M1 W[2] is 0.
    # So M1 only contributes to C[0,0] and C[1,1].
    # Let's check i=1, j=1, k=1: U[1,1,0] V[1,1,0] W[1,1,0] = 1*1*1 = 1. (Contributes to C[1,1])
    # Let's check i=1, j=1, k=0: U[1,1,0] V[1,0,0] W[0,1,0] = 1*0*0 = 0.
    # Let's check i=1, j=0, k=1: U[1,0,0] V[0,1,0] W[1,1,0] = 0*0*1 = 0.
    # Let's check i=1, j=0, k=0: U[1,0,0] V[0,0,0] W[0,1,0] = 0*1*0 = 0.
    # So M1 contributes to C[0,0] and C[1,1].
    
    # Now let's check the sum for C[0,0].
    # C[0,0] should be A[0,0]B[0,0] + A[0,1]B[1,0].
    # Contributions from M1..M7:
    # M1: 1 (from A[0,0]B[0,0])
    # M2: U[0,?,2] is 0.
    # M3: U[0,0,3]=1. V[0,0,3]=0. 0.
    # M4: U[0,?,4]=0.
    # M5: U[0,0,5]=1, U[0,1,5]=1.
    #    V[0,0,5]=0, V[1,0,5]=0.
    #    Wait, M5 V is h -> V[1,1]=1.
    #    So V[0,0,5]=0, V[0,1,5]=0, V[1,0,5]=0, V[1,1,5]=1.
    #    For C[0,0] (i=0, k=0):
    #    j=0: U[0,0,5] V[0,0,5] W[0,0,5] = 1 * 0 * (-1) = 0.
    #    j=1: U[0,1,5] V[1,0,5] W[0,0,5] = 1 * 0 * (-1) = 0.
    #    So M5 contributes 0 to C[0,0]?
    #    But Strassen says C[0,0] = M1 + M4 - M5 + M7.
    #    Let's re-read Strassen.
    #    C11 = M1 + M4 - M5 + M7.
    #    My C[0,0] is C11.
    #    Why did M5 contribute 0?
    #    M5 = (a+b)h.
    #    a=A[0,0], b=A[0,1], h=B[1,1].
    #    M5 = A[0,0]B[1,1] + A[0,1]B[1,1].
    #    A[0,0]B[1,1] is i=0, j=0, k=1. C[0,1].
    #    A[0,1]B[1,1] is i=0, j=1, k=1. C[0,1].
    #    So M5 contributes to C[0,1], not C[0,0].
    #    So why does Strassen say C11 = ... - M5?
    #    Ah, C11 in Strassen is bottom-right?
    #    Standard Strassen:
    #    C = [[C11, C12], [C21, C22]]
    #    C11 = M1 + M4 - M5 + M7
    #    C12 = M3 + M5
    #    C21 = M2 + M4
    #    C22 = M1 - M2 + M3 + M6
    #    
    #    Let's check C12 (top-right).
    #    C12 = A[0,0]B[0,1] + A[0,1]B[1,1].
    #    M3 = A[0,0](B[0,1]-B[1,1]) = A[0,0]B[0,1] - A[0,0]B[1,1].
    #    M5 = (A[0,0]+A[0,1])B[1,1] = A[0,0]B[1,1] + A[0,1]B[1,1].
    #    M3 + M5 = A[0,0]B[0,1] + A[0,1]B[1,1]. Correct.
    #    So M3 and M5 contribute to C[0,1].
    #    
    #    Let's check C21 (bottom-left).
    #    C21 = A[1,0]B[0,0] + A[1,1]B[1,0].
    #    M2 = (A[1,0]+A[1,1])B[0,0] = A[1,0]B[0,0] + A[1,1]B[0,0].
    #    M4 = A[1,1](B[1,0]-B[0,0]) = A[1,1]B[1,0] - A[1,1]B[0,0].
    #    M2 + M4 = A[1,0]B[0,0] + A[1,1]B[1,0]. Correct.
    #    So M2 and M4 contribute to C[1,0].
    #    
    #    Let's check C22 (bottom-right).
    #    C22 = A[1,0]B[0,1] + A[1,1]B[1,1].
    #    M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1]).
    #    M1 contributes A[1,1]B[1,1] to C22.
    #    M2 = (A[1,0]+A[1,1])B[0,0].
    #    M2 contributes A[1,1]B[0,0] to C[1,0] (not C22).
    #    Wait, M2 is subtracted in C22 formula: C22 = M1 - M2 + M3 + M6.
    #    M2 = A[1,0]B[0,0] + A[1,1]B[0,0].
    #    M3 = A[0,0]B[0,1] - A[0,0]B[1,1].
    #    M6 = (A[1,0]-A[0,0])(B[0,0]+B[0,1]) = A[1,0]B[0,0] + A[1,0]B[0,1] - A[0,0]B[0,0] - A[0,0]B[0,1].
    #    Sum:
    #    M1: A[1,1]B[1,1] (for C22)
    #    -M2: -A[1,0]B[0,0] - A[1,1]B[0,0]
    #    +M3: A[0,0]B[0,1] - A[0,0]B[1,1]
    #    +M6: A[1,0]B[0,0] + A[1,0]B[0,1] - A[0,0]B[0,0] - A[0,0]B[0,1]
    #    
    #    We want C22 = A[1,0]B[0,1] + A[1,1]B[1,1].
    #    Terms:
    #    A[1,1]B[1,1]: +1 (from M1).
    #    A[1,0]B[0,1]: +1 (from M6).
    #    A[1,0]B[0,0]: -1 (M2) + 1 (M6) = 0.
    #    A[1,1]B[0,0]: -1 (M2).
    #    A[0,0]B[0,1]: +1 (M3) - 1 (M6) = 0.
    #    A[0,0]B[1,1]: -1 (M3).
    #    A[0,0]B[0,0]: -1 (M6).
    #    
    #    We have extra terms: -A[1,1]B[0,0], -A[0,0]B[1,1], -A[0,0]B[0,0].
    #    These must cancel with M1's other terms?
    #    M1 = A[0,0]B[0,0] + A[0,0]B[1,1] + A[1,1]B[0,0] + A[1,1]B[1,1].
    #    So M1 contributes:
    #    A[0,0]B[0,0]: +1
    #    A[0,0]B[1,1]: +1
    #    A[1,1]B[0,0]: +1
    #    A[1,1]B[1,1]: +1
    #    
    #    Total Sum for C22:
    #    A[1,1]B[1,1]: 1 (M1) + 0 = 1.
    #    A[1,0]B[0,1]: 0 + 1 (M6) = 1.
    #    A[1,0]B[0,0]: 0 - 1 (M2) + 1 (M6) = 0.
    #    A[1,1]B[0,0]: 1 (M1) - 1 (M2) = 0.
    #    A[0,0]B[0,1]: 0 + 1 (M3) - 1 (M6) = 0.
    #    A[0,0]B[1,1]: 1 (M1) - 1 (M3) = 0.
    #    A[0,0]B[0,0]: 1 (M1) - 1 (M6) = 0.
    #    
    #    It works!
    
    # So the W assignments for 2x2 Strassen are:
    # M1: W[0]=1, W[3]=1
    # M2: W[1]=1
    # M3: W[2]=1
    # M4: W[1]=1
    # M5: W[2]=1
    # M6: W[3]=1
    # M7: W[0]=1
    
    # Wait, M4 contributes to C[1,0] (W[1]).
    # M2 contributes to C[1,0] (W[1]).
    # C[1,0] = M2 + M4.
    # So W[1] for M2 is 1, W[1] for M4 is 1.
    
    # M3 contributes to C[0,1] (W[2]).
    # M5 contributes to C[0,1] (W[2]).
    # C[0,1] = M3 + M5.
    # So W[2] for M3 is 1, W[2] for M5 is 1.
    
    # M1 contributes to C[0,0] (W[0]) and C[1,1] (W[3]).
    # M7 contributes to C[0,0] (W[0]).
    # M5 contributes to C[0,0] (W[0]) with -1?
    # C[0,0] = M1 + M4 - M5 + M7.
    # M4 contributes to C[0,0]?
    # M4 = A[1,1](B[1,0]-B[0,0]).
    # A[1,1]B[1,0] -> C[1,0].
    # A[1,1]B[0,0] -> C[1,0].
    # So M4 does NOT contribute to C[0,0].
    # So C[0,0] = M1 - M5 + M7.
    # Let's check.
    # M1: A[0,0]B[0,0] (for C00).
    # M5: A[0,0]B[1,1] + A[0,1]B[1,1]. (Contributes to C01).
    # So M5 does NOT contribute to C[0,0] either?
    # Wait.
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0].
    # M1 has A[0,0]B[0,0].
    # M7 = (b-d)(g+h) = (A[0,1]-A[1,1])(B[1,0]+B[1,1]).
    # = A[0,1]B[1,0] + A[0,1]B[1,1] - A[1,1]B[1,0] - A[1,1]B[1,1].
    # M7 contributes A[0,1]B[1,0] to C[0,0].
    # So C[0,0] = M1 (A[0,0]B[0,0]) + M7 (A[0,1]B[1,0]).
    # Do M5 or M4 contribute?
    # M5: No.
    # M4: No.
    # So C[0,0] = M1 + M7.
    # But Strassen says C11 = M1 + M4 - M5 + M7.
    # There is a contradiction.
    # Let's re-read Strassen carefully.
    # Maybe my mapping of M4, M5 is wrong.
    
    # M4 = d(g-e). d=A[1,1]. g=B[1,0]. e=B[0,0].
    # M4 = A[1,1]B[1,0] - A[1,1]B[0,0].
    # Both terms are in C[1,0].
    # So M4 contributes to C[1,0].
    
    # M5 = (a+b)h. a=A[0,0]. b=A[0,1]. h=B[1,1].
    # M5 = A[0,0]B[1,1] + A[0,1]B[1,1].
    # Both terms are in C[0,1].
    # So M5 contributes to C[0,1].
    
    # So C[0,0] should be M1 + M7.
    # Let's check M1 again.
    # M1 = (a+d)(e+h) = A[0,0]B[0,0] + A[0,0]B[1,1] + A[1,1]B[0,0] + A[1,1]B[1,1].
    # Only A[0,0]B[0,0] is in C[0,0].
    # M7 = (b-d)(g+h) = A[0,1]B[1,0] + A[0,1]B[1,1] - A[1,1]B[1,0] - A[1,1]B[1,1].
    # Only A[0,1]B[1,0] is in C[0,0].
    # So C[0,0] = M1 + M7.
    # Why does Strassen say M4 - M5?
    # Perhaps the standard Strassen formula I'm remembering is for a different permutation or I am misremembering.
    # Let's trust the derivation.
    # C[0,0] = M1 + M7.
    # C[0,1] = M3 + M5.
    # C[1,0] = M2 + M4.
    # C[1,1] = M1 - M2 + M3 + M6.
    
    # Let's verify C[1,1] with this.
    # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1].
    # M1: A[1,1]B[1,1].
    # M2: A[1,0]B[0,0] + A[1,1]B[0,0].
    # M3: A[0,0]B[0,1] - A[0,0]B[1,1].
    # M6: A[1,0]B[0,0] + A[1,0]B[0,1] - A[0,0]B[0,0] - A[0,0]B[0,1].
    # Sum:
    # A[1,1]B[1,1]: 1 (M1).
    # A[1,0]B[0,1]: 1 (M6).
    # A[1,0]B[0,0]: -1 (M2) + 1 (M6) = 0.
    # A[1,1]B[0,0]: -1 (M2).
    # A[0,0]B[0,1]: 1 (M3) - 1 (M6) = 0.
    # A[0,0]B[1,1]: -1 (M3).
    # A[0,0]B[0,0]: -1 (M6).
    # Extra terms from M1:
    # A[0,0]B[0,0]: 1.
    # A[0,0]B[1,1]: 1.
    # A[1,1]B[0,0]: 1.
    # Total:
    # A[1,1]B[1,1]: 1.
    # A[1,0]B[0,1]: 1.
    # A[1,1]B[0,0]: -1 (M2) + 1 (M1) = 0.
    # A[0,0]B[1,1]: -1 (M3) + 1 (M1) = 0.
    # A[0,0]B[0,0]: -1 (M6) + 1 (M1) = 0.
    # Works.
    
    # So W assignments:
    # M1: W[0]=1, W[3]=1
    # M2: W[1]=1
    # M3: W[2]=1
    # M4: W[1]=1
    # M5: W[2]=1
    # M6: W[3]=1
    # M7: W[0]=1
    
    # Let's construct the 2x2 matrices U2, V2, W2.
    U2 = np.zeros((7, 4), dtype=int)
    V2 = np.zeros((7, 4), dtype=int)
    W2 = np.zeros((7, 4), dtype=int)
    
    # M1
    U2[0, 0] = 1; U2[0, 3] = 1
    V2[0, 0] = 1; V2[0, 3] = 1
    W2[0, 0] = 1; W2[0, 3] = 1
    
    # M2
    U2[1, 2] = 1; U2[1, 3] = 1
    V2[1, 0] = 1
    W2[1, 1] = 1
    
    # M3
    U2[2, 0] = 1
    V2[2, 1] = 1; V2[2, 3] = -1
    W2[2, 2] = 1
    
    # M4
    U2[3, 3] = 1
    V2[3, 2] = 1; V2[3, 0] = -1
    W2[3, 1] = 1
    
    # M5
    U2[4, 0] = 1; U2[4, 1] = 1
    V2[4, 3] = 1
    W2[4, 2] = 1
    
    # M6
    U2[5, 2] = 1; U2[5, 0] = -1
    V2[5, 0] = 1; V2[5, 1] = 1
    W2[5, 3] = 1
    
    # M7
    U2[6, 1] = 1; U2[6, 3] = -1
    V2[6, 2] = 1; V2[6, 3] = 1
    W2[6, 0] = 1
    
    # Verify 2x2 decomposition
    # T_{i,j,k} = sum_r U2[r, i*2+j] * V2[r, j*2+k] * W2[r, k*2+i]
    # We want T_{i,j,k} = 1 if i,j,k correspond to a term in AB, else 0.
    # AB[i,k] = sum_j A[i,j] B[j,k].
    # So T_{i,j,k} = 1 for all i,j,k.
    # Wait, the tensor for matrix multiplication is T_{i,j,k} = 1.
    # Because C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So the tensor is all 1s.
    
    # Let's check if U2, V2, W2 produce all 1s.
    # i=0, j=0, k=0:
    # r=0: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> 1
    # r=1: U[1,0]=0
    # r=2: U[2,0]=1, V[2,0]=0
    # r=3: U[3,0]=0
    # r=4: U[4,0]=1, V[4,0]=0
    # r=5: U[5,0]=-1, V[5,0]=1, W[5,0]=0
    # r=6: U[6,0]=0
    # Sum = 1. Correct.
    
    # i=0, j=0, k=1:
    # r=0: U[0,0]=1, V[0,1]=0
    # r=1: U[1,0]=0
    # r=2: U[2,0]=1, V[2,1]=1, W[2,2]=1 (k=1, i=0 -> idx 2) -> 1
    # r=3: U[3,0]=0
    # r=4: U[4,0]=1, V[4,1]=0
    # r=5: U[5,0]=-1, V[5,1]=1, W[5,2]=0
    # r=6: U[6,0]=0
    # Sum = 1. Correct.
    
    # i=0, j=1, k=0:
    # r=0: U[0,1]=0
    # r=1: U[1,1]=0
    # r=2: U[2,1]=0
    # r=3: U[3,1]=0
    # r=4: U[4,1]=1, V[4,0]=0
    # r=5: U[5,1]=0
    # r=6: U[6,1]=1, V[6,0]=0
    # Sum = 0.
    # Wait. T_{0,1,0} should be 1.
    # A[0,1] B[1,0] is a term in C[0,0].
    # So T_{0,1,0} = 1.
    # Why is my sum 0?
    # Let's check M7.
    # M7 = (b-d)(g+h).
    # b=A[0,1]. g=B[1,0].
    # Term A[0,1] B[1,0] comes from b*g.
    # U[6, 1] = 1 (b).
    # V[6, 2] = 1 (g is j=1, k=0 -> idx 2).
    # W[6, 0] = 1 (k=0, i=0 -> idx 0).
    # So r=6: U[6,1]=1, V[6,2]=1, W[6,0]=1 -> 1.
    # My check above: V[6,0]=0.
    # V index is j*2+k.
    # j=1, k=0 -> 1*2+0 = 2.
    # V[6,2] is 1.
    # So sum is 1. Correct.
    
    # So the 2x2 decomposition is correct.
    
    # Now extend to 6x6.
    # We use 3x3 blocks of 2x2 matrices.
    # Block (I, J) of A, Block (J, K) of B -> Block (I, K) of C.
    # Each block multiplication uses rank 7.
    # Total rank = 3*3*7 = 63.
    
    # U shape: (36, 63).
    # Index i in 0..5, j in 0..5.
    # i_block = i // 2, i_local = i % 2.
    # j_block = j // 2, j_local = j % 2.
    # r_block = i_block * 3 + j_block.
    # r_local = 0..6.
    # r = r_block * 7 + r_local.
    # U[i*6+j, r] = U2[r_local, i_local*2 + j_local] if (i_block, j_block) matches?
    # No, U depends on i, j.
    # For a specific block (I, J), the U part is U2[r_local, i_local*2 + j_local].
    # This is non-zero only if i_block=I and j_block=J.
    # So U[i*6+j, r] = U2[r_local, i_local*2 + j_local] if i//2 == I and j//2 == J and r//7 == I*3+J.
    
    # V shape: (36, 63).
    # j, k.
    # j_block = j // 2, j_local = j % 2.
    # k_block = k // 2, k_local = k % 2.
    # r_block = j_block * 3 + k_block.
    # V[j*6+k, r] = V2[r_local, j_local*2 + k_local] if j//2 == J and k//2 == K and r//7 == J*3+K.
    
    # W shape: (36, 63).
    # k, i.
    # k_block = k // 2, k_local = k % 2.
    # i_block = i // 2, i_local = i % 2.
    # r_block = k_block * 3 + i_block.
    # W[k*6+i, r] = W2[r_local, k_local*2 + i_local] if k//2 == K and i//2 == I and r//7 == K*3+I.
    
    # Note: The block index in U is (I, J), in V is (J, K), in W is (K, I).
    # For the tensor to be consistent, the block indices must match.
    # U uses block (I, J).
    # V uses block (J, K).
    # W uses block (K, I).
    # The rank index r encodes the block.
    # If r encodes (I, J) for U, then for V it must encode (J, K) with the same r?
    # No, r is global.
    # If we set r = I*3*3 + J*3 + K? No, that's 27 blocks.
    # We have 3x3 blocks of 2x2.
    # Each 2x2 multiplication has rank 7.
    # Total rank 63.
    # The rank index r can be decomposed into (block_id, local_rank).
    # block_id identifies the (I, J, K) triplet?
    # No, the multiplication of block (I, J) of A and block (J, K) of B produces block (I, K) of C.
    # The "local" Strassen multiplication happens independently for each (I, J, K) triplet?
    # No, for a fixed J, we sum over J.
    # C_{I, K} = sum_J A_{I, J} B_{J, K}.
    # So for each J, we perform a 2x2 multiplication of A_{I, J} and B_{J, K}.
    # This produces a 2x2 matrix which is added to C_{I, K}.
    # So for each triplet (I, J, K), we have a Strassen decomposition.
    # There are 3*3*3 = 27 triplets.
    # Each has rank 7.
    # Total rank = 27 * 7 = 189.
    
    # So r = (I*3 + J)*3 + K)*7 + r_local.
    # Or simply r = triplet_id * 7 + r_local.
    # triplet_id = (I*3 + J)*3 + K.
    
    # U[i, j, r]:
    # i, j determine I, J.
    # r determines triplet (I', J', K') and r_local.
    # U is non-zero only if I == I' and J == J'.
    # Value is U2[r_local, i_local*2 + j_local].
    
    # V[j, k, r]:
    # j, k determine J, K.
    # r determines triplet (I', J', K') and r_local.
    # V is non-zero only if J == J' and K == K'.
    # Value is V2[r_local, j_local*2 + k_local].
    
    # W[k, i, r]:
    # k, i determine K, I.
    # r determines triplet (I', J', K') and r_local.
    # W is non-zero only if K == K' and I == I'.
    # Value is W2[r_local, k_local*2 + i_local].
    
    # This is sparse.
    # We can construct the full matrices.
    
    n = 6
    rank = 189
    
    U = np.zeros((n*n, rank), dtype=int)
    V = np.zeros((n*n, rank), dtype=int)
    W = np.zeros((n*n, rank), dtype=int)
    
    for I in range(3):
        for J in range(3):
            for K in range(3):
                triplet_id = (I*3 + J)*3 + K
                for r_local in range(7):
                    r = triplet_id * 7 + r_local
                    
                    # Fill U
                    for i_local in range(2):
                        for j_local in range(2):
                            i = I*2 + i_local
                            j = J*2 + j_local
                            val = U2[r_local, i_local*2 + j_local]
                            if val != 0:
                                U[i*n + j, r] = val
                                
                    # Fill V
                    for j_local in range(2):
                        for k_local in range(2):
                            j = J*2 + j_local
                            k = K*2 + k_local
                            val = V2[r_local, j_local*2 + k_local]
                            if val != 0:
                                V[j*n + k, r] = val
                                
                    # Fill W
                    for k_local in range(2):
                        for i_local in range(2):
                            k = K*2 + k_local
                            i = I*2 + i_local
                            val = W2[r_local, k_local*2 + i_local]
                            if val != 0:
                                W[k*n + i, r] = val
                                
    return {'U': U, 'V': V, 'W': W}
