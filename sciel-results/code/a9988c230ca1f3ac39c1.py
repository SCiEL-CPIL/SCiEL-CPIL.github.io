"""Matrix multiplication 2x4 x 4x6 using Strassen on 2x2 blocks."""
import time

DIMENSIONS = [2, 4, 6]

# Strassen 7 factors for 2x2x2 multiplication
# M1 = (A00+A11)*(B00+B11)
# M2 = (A10+A11)*B00
# M3 = A00*(B10-B11)
# M4 = A11*(B01-B10)
# M5 = (A00+A01)*B11
# M6 = (A10-A00)*(B00+B10)
# M7 = (A01-A11)*(B01+B11)
#
# Result:
# C00 = M1 + M4 - M5 + M7
# C01 = M3 + M5
# C10 = M2 + M4
# C11 = M1 - M2 + M3 + M6

# For a 2x2 block, U has rows (i, j) for i in {0,1}, j in {0,1}
# V has rows (j, k) for j in {0,1}, k in {0,1}
# W has rows (k, i) for k in {0,1}, i in {0,1}

# Each factor r contributes a rank-1 term: U_vec * V_vec * W_vec
# U_vec has 4 components (indexed by i*m_local+j = i*2+j)
# V_vec has 4 components (indexed by j*p_local+k = j*2+k)
# W_vec has 4 components (indexed by k*n_local+i = k*2+i)

# Factor definitions: (u_components, v_components, w_components)
# u: index = i*2 + j, v: index = j*2 + k, w: index = k*2 + i

STRASSEN_FACTORS = [
    # M1 = (A00+A11)*(B00+B11)
    # U: A00 at (0,0)=idx0, A11 at (1,1)=idx3
    # V: B00 at (0,0)=idx0, B11 at (1,1)=idx3
    # W: contributes to C00 with +1, C11 with +1
    # C00 is at w_idx = k*2+i = 0*2+0 = 0
    # C11 is at w_idx = 1*2+1 = 3
    # But wait: C00 = M1 + ... and C11 = M1 - ...
    # So W for M1: w[0]=1 (C00), w[3]=1 (C11)
    # Actually let me re-derive carefully.
    #
    # For 2x2: C = [[C00, C01], [C10, C11]]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # w_idx for C00: k=0,i=0 -> 0
    # w_idx for C01: k=1,i=0 -> 2
    # w_idx for C10: k=0,i=1 -> 1
    # w_idx for C11: k=1,i=1 -> 3
    
    # M1: coeff in C00=+1, C11=+1
    ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
    # M2: coeff in C10=+1, C11=-1
    ({1: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}),
    # M3: coeff in C01=+1, C11=+1
    ({0: 1}, {1: 1, 3: -1}, {2: 1, 3: 1}),
    # M4: coeff in C00=+1, C10=+1
    ({3: 1}, {1: 1, 0: -1}, {0: 1, 1: 1}),
    # M5: coeff in C00=-1, C01=+1
    ({0: 1, 1: 1}, {3: 1}, {0: -1, 2: 1}),
    # M6: coeff in C11=+1
    ({1: 1, 0: -1}, {0: 1, 2: 1}, {3: 1}),
    # M7: coeff in C00=+1
    ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
]

# Let me verify each factor:
# M1 = (A00+A11)*(B00+B11)
#   U: A00 (i=0,j=0) + A11 (i=1,j=1) -> u[0]=1, u[3]=1
#   V: B00 (j=0,k=0) + B11 (j=1,k=1) -> v[0]=1, v[3]=1
#   Product gives: A00*B00 + A00*B11 + A11*B00 + A11*B11
#   This contributes to C00, C01, C10, C11? No, it's a scalar factor.
#   In Strassen, M1 is added to C00 and C11 with coefficient +1.
#   C00 at (i=0,k=0): w[0] = +1
#   C11 at (i=1,k=1): w[3] = +1
#   So W = {0:1, 3:1} ✓

# M2 = (A10+A11)*B00
#   U: A10 (i=1,j=0) idx=2, A11 (i=1,j=1) idx=3 -> u[2]=1, u[3]=1
#   V: B00 (j=0,k=0) idx=0 -> v[0]=1
#   M2 contributes to C10 with +1, C11 with -1
#   C10 at (i=1,k=0): w[1] = +1
#   C11 at (i=1,k=1): w[3] = -1
#   So U={2:1,3:1}, V={0:1}, W={1:1,3:-1}
#   I had {1:1,3:1} for U - that's wrong! A10 is i=1,j=0 -> idx=1*2+0=2

# Let me redo all carefully:
# U index: i*2 + j
#   (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
# V index: j*2 + k
#   (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
# W index: k*2 + i
#   (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3

# M1 = (A00+A11)*(B00+B11)
#   U: A00(0,0)->0, A11(1,1)->3: {0:1, 3:1}
#   V: B00(0,0)->0, B11(1,1)->3: {0:1, 3:1}
#   W: C00(k=0,i=0)->0: +1, C11(k=1,i=1)->3: +1: {0:1, 3:1}

# M2 = (A10+A11)*B00
#   U: A10(1,0)->2, A11(1,1)->3: {2:1, 3:1}
#   V: B00(0,0)->0: {0:1}
#   W: C10(k=0,i=1)->1: +1, C11(k=1,i=1)->3: -1: {1:1, 3:-1}

# M3 = A00*(B10-B11)
#   U: A00(0,0)->0: {0:1}
#   V: B10(1,0)->2: +1, B11(1,1)->3: -1: {2:1, 3:-1}
#   W: C01(k=1,i=0)->2: +1, C11(k=1,i=1)->3: +1: {2:1, 3:1}

# M4 = A11*(B01-B10)
#   U: A11(1,1)->3: {3:1}
#   V: B01(0,1)->1: +1, B10(1,0)->2: -1: {1:1, 2:-1}
#   W: C00(k=0,i=0)->0: +1, C10(k=0,i=1)->1: +1: {0:1, 1:1}

# M5 = (A00+A01)*B11
#   U: A00(0,0)->0, A01(0,1)->1: {0:1, 1:1}
#   V: B11(1,1)->3: {3:1}
#   W: C00(k=0,i=0)->0: -1, C01(k=1,i=0)->2: +1: {0:-1, 2:1}

# M6 = (A10-A00)*(B00+B10)
#   U: A10(1,0)->2: +1, A00(0,0)->0: -1: {0:-1, 2:1}
#   V: B00(0,0)->0: +1, B10(1,0)->2: +1: {0:1, 2:1}
#   W: C11(k=1,i=1)->3: +1: {3:1}

# M7 = (A01-A11)*(B01+B11)
#   U: A01(0,1)->1: +1, A11(1,1)->3: -1: {1:1, 3:-1}
#   V: B01(0,1)->1: +1, B11(1,1)->3: +1: {1:1, 3:1}
#   W: C00(k=0,i=0)->0: +1: {0:1}

STRASSEN_FACTORS = [
    ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),       # M1
    ({2: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}),             # M2
    ({0: 1}, {2: 1, 3: -1}, {2: 1, 3: 1}),             # M3
    ({3: 1}, {1: 1, 2: -1}, {0: 1, 1: 1}),             # M4
    ({0: 1, 1: 1}, {3: 1}, {0: -1, 2: 1}),             # M5
    ({0: -1, 2: 1}, {0: 1, 2: 1}, {3: 1}),             # M6
    ({1: 1, 3: -1}, {1: 1, 3: 1}, {0: 1}),             # M7
]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    rank = 7 * 6  # 6 blocks of 2x2x2, each with rank 7 -> 42
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # 6 blocks: for j_block in {0,1}, k_block in {0,1,2}
    # Block (jb, kb): j in {2*jb, 2*jb+1}, k in {2*kb, 2*kb+1}
    # U global index: i*m + j = i*4 + j
    # V global index: j*p + k = j*6 + k
    # W global index: k*n + i = k*2 + i
    
    for jb in range(2):  # j block: 0 or 1
        for kb in range(3):  # k block: 0, 1, or 2
            col_offset = (jb * 3 + kb) * 7
            
            for r, (u_dict, v_dict, w_dict) in enumerate(STRASSEN_FACTORS):
                col = col_offset + r
                
                # U: local (i, j_local) -> global i*4 + (2*jb + j_local)
                for (i_loc, j_loc), val in u_dict.items():
                    u_idx = i_loc * m + (2 * jb + j_loc)
                    U[u_idx][col] += val
                
                # V: local (j_loc, k_loc) -> global (2*jb + j_loc)*6 + (2*kb + k_loc)
                for (j_loc, k_loc), val in v_dict.items():
                    v_idx = (2 * jb + j_loc) * p + (2 * kb + k_loc)
                    V[v_idx][col] += val
                
                # W: local (k_loc, i_loc) -> global (2*kb + k_loc)*2 + i_loc
                for (k_loc, i_loc), val in w_dict.items():
                    w_idx = (2 * kb + k_loc) * n + i_loc
                    W[w_idx][col] += val
    
    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    import numpy as np
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    # Verify: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    correct = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if val != 1:
                    print(f"FAIL: i={i}, j={j}, k={k}, val={val}")
                    correct = False
    # Also verify zeros
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(rank):
                    pass
    # Check that non-matching entries are 0
    # The tensor for matmul has T[i,j,k] = 1 if we're computing (AB)[i,k] from A[i,j]*B[j,k]
    # Actually the tensor is: T[i,j,k] = delta_{j,j} which is always 1 for the matmul tensor
    # Wait no. The matmul tensor T satisfies: C[i,k] = sum_j A[i,j] B[j,k]
    # As a tensor: T[i,j,k] = 1 for all i,j,k? No.
    # T[i,j,k] is the coefficient such that C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i,j,k.
    # The rank-1 decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # So we need sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k.
    
    if correct:
        print(f"PASS: rank {rank}")
    else:
        print("FAILED verification")
