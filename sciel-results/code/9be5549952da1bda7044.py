"""Matrix multiplication 3x5x5 tensor decomposition."""
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 5, 5  # A: 3x5, B: 5x5, C: 3x5
    # Split A rows: 2 + 1, A cols: 2 + 3
    # Split B rows: 2 + 3, B cols: 2 + 3
    # C blocks: C11(2x2), C12(2x3), C21(1x2), C22(1x3)
    
    # Strassen 2x2x2 decomposition
    # For A(2x2), B(2x2) -> C(2x2)
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    rank = 0
    U = []  # shape (3*5, rank) = (15, rank)
    V = []  # shape (5*5, rank) = (25, rank)
    W = []  # shape (5*3, rank) = (15, rank)  # W[k*n+i, r] where C[i,j] = sum_k A[i,k]B[k,j]
    
    # Helper to add a rank-1 term
    def add_term(u_row, u_val, v_row, v_val, w_row, w_val):
        nonlocal rank
        U.append([0]*rank + [u_val])
        V.append([0]*rank + [v_val])
        W.append([0]*rank + [w_val])
        rank += 1
        for i in range(rank-1):
            U[i].append(0)
            V[i].append(0)
            W[i].append(0)
    
    # Initialize
    U = []
    V = []
    W = []
    rank = 0
    
    def add_term_full(u_vec, v_vec, w_vec):
        """Add a rank-1 term. u_vec is dict {row: val}, v_vec, w_vec."""
        nonlocal rank
        U.append([0]*rank)
        V.append([0]*rank)
        W.append([0]*rank)
        for row, val in u_vec.items():
            U[-1][rank] = val
        for row, val in v_vec.items():
            V[-1][rank] = val
        for row, val in w_vec.items():
            W[-1][rank] = val
        rank += 1
        for i in range(rank-1):
            U[i].append(0)
            V[i].append(0)
            W[i].append(0)
    
    # C11 (2x2) = A1a(2x2) * B1a(2x2) + A1b(2x3) * B1b(3x2)
    # Use Strassen for A1a * B1a: 7 terms
    # A1a rows: 0,1; cols: 0,1
    # B1a rows: 0,1; cols: 0,1
    # A1a[i,j] = A[i,j] for i in {0,1}, j in {0,1}
    # B1a[i,j] = B[i,j] for i in {0,1}, j in {0,1}
    # C11[i,j] for i in {0,1}, j in {0,1}
    
    # Strassen indices for 2x2 block:
    # A11 = A[0,0], A12 = A[0,1], A21 = A[1,0], A22 = A[1,1]
    # B11 = B[0,0], B12 = B[0,1], B21 = B[1,0], B22 = B[1,1]
    # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
    
    # M1 = (A11+A22)(B11+B22)
    add_term_full({0:1, 3:1}, {0:1, 1:1}, {0:1, 3:1})  # C11[0,0]=M1, C11[1,1]=M1
    # Wait, W index is k*n+i. C[i,j] where i is row of C (0..2), j is col (0..4)
    # W[k*n+i, r] means for output C[i, j], the term contributes to W[j*n + i, r]
    # C11[0,0] -> W[0*3+0, r] = W[0, r]
    # C11[0,1] -> W[1*3+0, r] = W[3, r]
    # C11[1,0] -> W[0*3+1, r] = W[1, r]
    # C11[1,1] -> W[1*3+1, r] = W[4, r]
    
    # Let me redo. Clear and start fresh.
    U = []
    V = []
    W = []
    rank = 0
    
    def add_term_full(u_vec, v_vec, w_vec):
        nonlocal rank
        U.append([0]*rank)
        V.append([0]*rank)
        W.append([0]*rank)
        for row, val in u_vec.items():
            U[-1][rank] = val
        for row, val in v_vec.items():
            V[-1][rank] = val
        for row, val in w_vec.items():
            W[-1][rank] = val
        rank += 1
        for i in range(rank-1):
            U[i].append(0)
            V[i].append(0)
            W[i].append(0)
    
    # U index: i*5 + j for A[i,j] where i in 0..2, j in 0..4
    # V index: k*5 + l for B[k,l] where k in 0..4, l in 0..4
    # W index: l*3 + i for C[i,l] where i in 0..2, l in 0..4
    
    # C11 (2x2): i in {0,1}, l in {0,1}
    # C11[i,l] = sum_{j in 0..4} A[i,j] * B[j,l]
    # = sum_{j in 0..1} A[i,j] B[j,l] + sum_{j in 2..4} A[i,j] B[j,l]
    
    # Strassen for A1a(2x2) * B1a(2x2):
    # A1a[i,j] = A[i,j] for i in {0,1}, j in {0,1}
    # B1a[k,l] = B[k,l] for k in {0,1}, l in {0,1}
    # Contribution to C11[i,l]
    
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    add_term_full({0:1, 6:1}, {0:1, 6:1}, {0:1, 4:1})  # W[0]=C[0,0], W[4]=C[1,1]
    
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    add_term_full({5:1, 6:1}, {0:1}, {1:1})  # C[1,0]
    
    # M3 = A[0,0] * (B[0,1]-B[1,1])
    add_term_full({0:1}, {1:1, 6:-1}, {3:1})  # C[0,1]
    
    # M4 = A[1,1] * (B[1,0]-B[0,0])
    add_term_full({6:1}, {5:1, 0:-1}, {1:1})  # C[1,0]
    
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    add_term_full({0:1, 1:1}, {6:1}, {4:-1, 3:1})  # C[1,1] gets -M5, C[0,1] gets M5
    # Wait: C11[0,1] = M3+M5, C11[1,1] = M1-M2+M3+M6... 
    # Actually: C11[0,0]=M1+M4-M5+M7, C11[0,1]=M3+M5, C11[1,0]=M2+M4, C11[1,1]=M1-M2+M3+M6
    # Let me redo M5: (A[0,0]+A[0,1])*B[1,1]
    # This contributes to C11[0,1] with +1 and C11[1,1] with -1
    # C11[0,1] is W[1*3+0] = W[3], C11[1,1] is W[1*3+1] = W[4]
    # So W[3] gets +1, W[4] gets -1. That's what I have.
    
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    add_term_full({5:1, 0:-1}, {0:1, 1:1}, {4:1})  # C[1,1]
    
    # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    add_term_full({1:1, 6:-1}, {5:1, 6:1}, {0:1})  # C[0,0]
    
    # C11[0,0] = M1+M4-M5+M7: W[0] gets M1(+1), M4(0), -M5(0), M7(+1) ✓
    # C11[0,1] = M3+M5: W[3] gets M3(+1), M5(+1) ✓
    # C11[1,0] = M2+M4: W[1] gets M2(+1), M4(+1) ✓
    # C11[1,1] = M1-M2+M3+M6: W[4] gets M1(+1), -M2(0), M3(0), M6(+1) ✓
    # But wait, I need to check M5's contribution to W[4]. M5 has W[4]=-1.
    # C11[1,1] should have -M5? No, the formula is C11[1,1]=M1-M2+M3+M6. No M5.
    # But M5 contributes to C11[0,1] and... let me recheck Strassen.
    # Standard Strassen:
    # M1=(A11+A22)(B11+B22), M2=(A21+A22)B11, M3=A11(B12-B22), M4=A22(B21-B11)
    # M5=(A11+A12)B22, M6=(A21-A11)(B11+B12), M7=(A12-A22)(B21+B22)
    # C11=M1+M4-M5+M7, C12=M3+M5, C21=M2+M4, C22=M1-M2+M3+M6
    # So C12 (which is C11[0,1]) = M3+M5. M5 has +1 to C[0,1]. ✓
    # C22 (C11[1,1]) = M1-M2+M3+M6. No M5. But M5 was defined to contribute to C[0,1] and C[1,1]?
    # No! M5 = (A11+A12)B22. In the 2x2 product, this term is used in C12 and... 
    # Actually in standard Strassen, M5 only appears in C12. Let me verify:
    # C12 = M3+M5. C22 does NOT have M5. So M5 should only have W[3] (C[0,1]) with +1.
    # I made an error. Let me fix M5.
    
    # I need to clear and restart.
    U = []
    V = []
    W = []
    rank = 0
    
    def add_term_full(u_vec, v_vec, w_vec):
        nonlocal rank
        U.append([0]*rank)
        V.append([0]*rank)
        W.append([0]*rank)
        for row, val in u_vec.items():
            U[-1][rank] = val
        for row, val in v_vec.items():
            V[-1][rank] = val
        for row, val in w_vec.items():
            W[-1][rank] = val
        rank += 1
        for i in range(rank-1):
            U[i].append(0)
            V[i].append(0)
            W[i].append(0)
    
    # Strassen for A1a(2x2)*B1a(2x2) contributing to C[0:2, 0:2]
    # M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1]) -> C[0,0]+=M1, C[1,1]+=M1
    add_term_full({0:1, 6:1}, {0:1, 6:1}, {0:1, 4:1})
    # M2 = (A[1,0]+A[1,1])B[0,0] -> C[1,0]+=M2
    add_term_full({5:1, 6:1}, {0:1}, {1:1})
    # M3 = A[0,0](B[0,1]-B[1,1]) -> C[0,1]+=M3
    add_term_full({0:1}, {1:1, 6:-1}, {3:1})
    # M4 = A[1,1](B[1,0]-B[0,0]) -> C[0,0]+=M4, C[1,0]+=M4
    add_term_full({6:1}, {5:1, 0:-1}, {0:1, 1:1})
    # M5 = (A[0,0]+A[0,1])B[1,1] -> C[0,1]+=M5
    add_term_full({0:1, 1:1}, {6:1}, {3:1})
    # M6 = (A[1,0]-A[0,0])(B[0,0]+B[0,1]) -> C[1,1]+=M6
    add_term_full({5:1, 0:-1}, {0:1, 1:1}, {4:1})
    # M7 = (A[0,1]-A[1,1])(B[1,0]+B[1,1]) -> C[0,0]+=M7
    add_term_full({1:1, 6:-1}, {5:1, 6:1}, {0:1})
    
    # Verify C11:
    # C[0,0] = M1+M4+M7: W[0] has M1(+1), M4(+1), M7(+1) ✓
    # C[0,1] = M3+M5: W[3] has M3(+1), M5(+1) ✓
    # C[1,0] = M2+M4: W[1] has M2(+1), M4(+1) ✓
    # C[1,1] = M1+M6: W[4] has M1(+1), M6(+1) ✓
    
    # Now A1b(2x3)*B1b(3x2) contributing to C[0:2, 0:2]
    # A1b[i,j] = A[i,j] for i in {0,1}, j in {2,3,4}
    # B1b[k,l] = B[k,l] for k in {2,3,4}, l in {0,1}
    # Schoolbook: 2*3*2 = 12 terms
    for i in range(2):
        for j in range(2, 5):
            for l in range(2):
                add_term_full({i*5+j: 1}, {j*5+l: 1}, {l*3+i: 1})
    
    # C12 (2x3): i in {0,1}, l in {2,3,4}
    # C12[i,l] = sum_{j} A[i,j] B[j,l]
    # Split j into {0,1} and {2,3,4}
    # Part 1: A1a(2x2)*B2a(2x3) where B2a[k,l]=B[k,l] for k in {0,1}, l in {2,3,4}
    # Schoolbook: 2*2*3 = 12 terms
    for i in range(2):
        for j in range(2):
            for l in range(2, 5):
                add_term_full({i*5+j: 1}, {j*5+l: 1}, {l*3+i: 1})
    # Part 2: A1b(2x3)*B2b(3x3) where B2b[k,l]=B[k,l] for k in {2,3,4}, l in {2,3,4}
    # Schoolbook: 2*3*3 = 18 terms
    for i in range(2):
        for j in range(2, 5):
            for l in range(2, 5):
                add_term_full({i*5+j: 1}, {j*5+l: 1}, {l*3+i: 1})
    
    # C21 (1x2): i=2, l in {0,1}
    # C21[2,l] = sum_{j} A[2,j] B[j,l]
    # Part 1: A2a(1x2)*B1a(2x2): 1*2*2 = 4 terms
    for j in range(2):
        for l in range(2):
            add_term_full({2*5+j: 1}, {j*5+l: 1}, {l*3+2: 1})
    # Part 2: A2b(1x3)*B1b(3x2): 1*3*2 = 6 terms
    for j in range(2, 5):
        for l in range(2):
            add_term_full({2*5+j: 1}, {j*5+l: 1}, {l*3+2: 1})
    
    # C22 (1x3): i=2, l in {2,3,4}
    # C22[2,l] = sum_{j} A[2,j] B[j,l]
    # Part 1: A2a(1x2)*B2a(2x3): 1*2*3 = 6 terms
    for j in range(2):
        for l in range(2, 5):
            add_term_full({2*5+j: 1}, {j*5+l: 1}, {l*3+2: 1})
    # Part 2: A2b(1x3)*B2b(3x3): 1*3*3 = 9 terms
    for j in range(2, 5):
        for l in range(2, 5):
            add_term_full({2*5+j: 1}, {j*5+l: 1}, {l*3+2: 1})
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
