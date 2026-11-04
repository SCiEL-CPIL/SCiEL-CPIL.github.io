import numpy as np

def construct():
    n, m, p = 2, 3, 3
    
    # Total rank estimate: 7 + 2 + 4 + 2 = 15
    rank = 15
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Term 1: 2x2x2 Strassen
    # i in 0..1, j in 0..1, k in 0..1
    # U rows: i*3+j
    # V rows: j*3+k
    # W rows: k*2+i
    
    # Strassen coefficients
    # M1: (A11+A22)(B11+B22) -> C11+1, C22+1
    # M2: (A21+A22)B11 -> C21+1, C22-1
    # M3: A11(B12-B22) -> C12+1, C22+1
    # M4: A22(B21-B11) -> C11+1, C21+1
    # M5: (A11+A12)B22 -> C11-1, C12+1
    # M6: (A21-A11)(B11+B12) -> C22+1
    # M7: (A12-A22)(B21+B22) -> C11+1
    
    # We need to map these to U, V, W additions
    
    # Helper to add a rank-1 term
    # u_vec: dict (i,j) -> val
    # v_vec: dict (j,k) -> val
    # w_vec: dict (k,i) -> val
    # offset: rank offset
    
    def add_term(offset, u_dict, v_dict, w_dict):
        for (i, j), val in u_dict.items():
            U[i * m + j, offset] += val
        for (j, k), val in v_dict.items():
            V[j * p + k, offset] += val
        for (k, i), val in w_dict.items():
            W[k * n + i, offset] += val
            
    # Term 1: Strassen
    # M1
    add_term(0, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1})
    # M2
    add_term(1, {(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1})
    # M3
    add_term(2, {(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1})
    # M4
    add_term(3, {(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1})
    # M5
    add_term(4, {(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1})
    # M6
    add_term(5, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1})
    # M7
    add_term(6, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})
    
    # Term 2: 2x1x2 (A2 B3)
    # i in 0..1, j=2, k in 0..1
    # U rows: i*3+2
    # V rows: 2*3+k
    # W rows: k*2+i
    # Schoolbook rank 2
    # k=0
    add_term(7, {(0,2):1, (1,2):1}, {(2,0):1}, {(0,0):1, (0,1):1}) 
    # Wait, W index is k*n+i.
    # For k=0, i=0 -> 0. i=1 -> 1.
    # For k=1, i=0 -> 2. i=1 -> 3.
    # Term 2 is C_i,k = A_i2 B_2k.
    # r=7: k=0. A_i2 B_20.
    # U[i,2]=1. V[2,0]=1. W[0,i]=1.
    add_term(7, {(0,2):1, (1,2):1}, {(2,0):1}, {(0,0):1, (0,1):1}) # This adds to W[0,0] and W[1,0]? No, W[k*n+i].
    # W[0*2+0]=W[0,0]. W[0*2+1]=W[0,1].
    # My add_term uses W[k*n+i].
    # So for k=0, i=0: W[0]. i=1: W[1].
    # So w_dict should be {(0,0):1, (0,1):1}? No, keys are (k,i).
    # So {(0,0):1, (0,1):1} is correct for k=0.
    
    # r=8: k=1.
    add_term(8, {(0,2):1, (1,2):1}, {(2,1):1}, {(1,0):1, (1,1):1})
    
    # Term 3: 2x2x1 (A1 B2)
    # i in 0..1, j in 0..1, k=2
    # U rows: i*3+j
    # V rows: j*3+2
    # W rows: 2*2+i = 4+i
    # Schoolbook rank 4
    # j=0, k=2
    add_term(9, {(0,0):1, (1,0):1}, {(0,2):1}, {(2,0):1, (2,1):1})
    # j=1, k=2
    add_term(10, {(0,1):1, (1,1):1}, {(1,2):1}, {(2,0):1, (2,1):1})
    # Wait, C_i,2 = sum_j A_ij B_j2.
    # r=9: j=0. A_i0 B_02.
    # U[i,0]=1. V[0,2]=1. W[2,i]=1.
    # r=10: j=1. A_i1 B_12.
    # U[i,1]=1. V[1,2]=1. W[2,i]=1.
    # This covers i=0,1.
    # But rank 4 means we need 4 terms?
    # A_i0 B_02 and A_i1 B_12.
    # For each i, we have 2 terms?
    # No, U[i,0] is 1 for both i=0 and i=1?
    # If U[i,0]=1 for i=0,1, then it's one vector.
    # The term is sum_i U[i,0] A[i,0] ...
    # Yes, one rank-1 term covers both i=0 and i=1 for a fixed j,k?
    # No, W[2,i] depends on i.
    # So for fixed j=0, k=2:
    # Term: U[i,0] V[0,2] W[2,i].
    # If U[i,0]=1, V[0,2]=1, W[2,i]=1, then C_i2 += A_i0 B_02.
    # This works for i=0 and i=1 simultaneously.
    # So rank 1 for j=0, k=2?
    # And rank 1 for j=1, k=2?
    # Total rank 2?
    # Let's check.
    # C_02 = A_00 B_02 + A_01 B_12.
    # C_12 = A_10 B_02 + A_11 B_12.
    # Term 1: U[0,0]=1, U[1,0]=1. V[0,2]=1. W[2,0]=1, W[2,1]=1.
    # Sum: A_00 B_02 * 1 + A_10 B_02 * 1?
    # No, the sum is over r.
    # For r=9: U[i,0] V[0,2] W[2,i].
    # i=0: 1 * 1 * 1 = 1. Term A_00 B_02.
    # i=1: 1 * 1 * 1 = 1. Term A_10 B_02.
    # So C_02 gets A_00 B_02. C_12 gets A_10 B_02.
    # This is correct.
    # So rank 2 for this block?
    # Wait, B_02 is a scalar.
    # Yes, rank 2.
    
    # Term 4: 2x1x1 (A2 B4)
    # i in 0..1, j=2, k=2
    # U rows: i*3+2
    # V rows: 2*3+2 = 8
    # W rows: 2*2+i = 4+i
    # Rank 1?
    # C_i2 += A_i2 B_22.
    # r=11: U[i,2]=1, V[2,2]=1, W[2,i]=1.
    # i=0: A_02 B_22.
    # i=1: A_12 B_22.
    # Rank 1.
    
    # Total rank: 7 + 2 + 2 + 1 = 12?
    # Why did I think 15?
    # 2x2x1 rank is 4 in schoolbook, but here it's 2?
    # 2x1x2 rank is 2.
    # 2x1x1 rank is 1.
    # 7 + 2 + 2 + 1 = 12.
    
    # Let's verify 2x2x1 rank.
    # C_i = A_i0 B_0 + A_i1 B_1.
    # Can we do it in rank 2?
    # Yes, as shown above.
    # So total rank 12.
    
    # I will return rank 12.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
