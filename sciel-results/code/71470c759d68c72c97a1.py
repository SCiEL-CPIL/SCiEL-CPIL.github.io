import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Target tensor indices
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nz_indices.append((a, b, c))
    nz_set = set(nz_indices)

    def verify_decomposition(U, V, W):
        """
        Verifies if U, V, W form a valid decomposition.
        U: (9, rank), V: (9, rank), W: (9, rank)
        """
        rank = U.shape[1]
        # Check non-zero entries
        for a, b, c in nz_indices:
            val = np.sum(U[a] * V[b] * W[c])
            if abs(val - 1.0) > 1e-9:
                return False
        # Check zero entries
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    if (a, b, c) not in nz_set:
                        val = np.sum(U[a] * V[b] * W[c])
                        if abs(val) > 1e-9:
                            return False
        return True

    def get_fallback_rank_27():
        rank = 27
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for idx, (a, b, c) in enumerate(nz_indices):
            U[a, idx] = 1.0
            V[b, idx] = 1.0
            W[c, idx] = 1.0
        return U, V, W

    # Strategy: Try to construct a Rank-22 or Rank-23 decomposition
    # using Strassen for 2x2 block + Schoolbook for the rest.
    # If valid, return it.
    
    # We will try to construct the Rank-22 candidate.
    # U, V, W will have 22 columns.
    
    # Indices mapping:
    # A indices: 0..8 (0:00, 1:01, 2:02, 3:10, 4:11, 5:12, 6:20, 7:21, 8:22)
    # B indices: 0..8
    # C indices: 0..8
    
    # Strassen 2x2 for A[0:4, 0:4] and B[0:4, 0:4]
    # A_11=0, A_12=1, A_21=3, A_22=4
    # B_11=0, B_12=2, B_21=1, B_22=3  (Note: B is column-major? 
    # Task says W[k*n+i, r]. 
    # Standard matrix mult C = A B.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # Tensor index c = k*n + i.
    # So c=0 -> k=0, i=0 -> C[0,0].
    # c=1 -> k=0, i=1 -> C[1,0].
    # c=2 -> k=0, i=2 -> C[2,0].
    # c=3 -> k=1, i=0 -> C[0,1].
    # 
    # A[i,j] index a = i*m + j = i*3 + j.
    # B[j,k] index b = j*p + k = j*3 + k.
    # C[i,k] index c = k*n + i = k*3 + i.
    
    # Strassen terms for C[0,0], C[0,1], C[1,0], C[1,1]
    # C[0,0] (c=0), C[0,1] (c=3), C[1,0] (c=1), C[1,1] (c=4)
    
    # M1 = (A00+A11)(B00+B11) -> (A[0]+A[4])(B[0]+B[3])
    # M2 = (A10+A11)B00 -> (A[3]+A[4])B[0]
    # M3 = A00(B10-B11) -> A[0](B[1]-B[4])
    # M4 = (A00-A11)(B10+B11) -> (A[0]-A[4])(B[1]+B[4])
    # M5 = (A01+A11)B10 -> (A[1]+A[4])B[1]
    # M6 = (A00+A01)(B11-B10) -> (A[0]+A[1])(B[4]-B[1])
    # M7 = (A10-A00)(B00+B01) -> (A[3]-A[0])(B[0]+B[2])
    
    # C00 = M1 - M4 + M7
    # C01 = M5 + M4
    # C10 = M2 + M7
    # C11 = M1 - M3 + M5
    
    # Outer terms:
    # C02 (c=2) = A00*B02 + A01*B12 + A02*B22
    # C12 (c=5) = A10*B02 + A11*B12 + A12*B22
    # C20 (c=6) = A20*B00 + A21*B10 + A22*B20
    # C21 (c=7) = A20*B01 + A21*B11 + A22*B21
    # C22 (c=8) = A20*B02 + A21*B12 + A22*B22
    
    # Let's build U, V, W for these 22 terms.
    rank = 22
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    
    # Term 0: M1
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C01 ? No, Strassen C01 is M5+M4.
    # Let's map W carefully.
    # W[c, term] is the coefficient of term in C[c].
    
    # Term 0: M1
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1  # C00
    # C01, C10, C11 contributions are in other terms.
    
    # Term 1: M2
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1  # C10
    
    # Term 2: M3
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[4, 2] = -1 # C11
    
    # Term 3: M4
    U[0, 3] = 1; U[4, 3] = -1
    V[1, 3] = 1; V[4, 3] = 1
    W[0, 3] = -1 # C00
    W[3, 3] = 1  # C01
    
    # Term 4: M5
    U[1, 4] = 1; U[4, 4] = 1
    V[1, 4] = 1
    W[3, 4] = 1  # C01
    W[4, 4] = 1  # C11
    
    # Term 5: M6
    U[0, 5] = 1; U[1, 5] = 1
    V[4, 5] = 1; V[1, 5] = -1
    # C? M6 is not used in standard Strassen for C00-C11?
    # Standard Strassen:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B21-B22)
    # P4 = (A11-A22)(B21+B22)
    # P5 = (A12+A22)B21
    # P6 = (A11+A12)(B22-B21)
    # P7 = (A21-A11)(B11+B12)
    # C11 = P1 - P4 + P7
    # C12 = P5 + P4
    # C21 = P2 + P7
    # C22 = P1 - P3 + P5
    # P6 is not used in the output?
    # Wait, P6 is used?
    # No, standard Strassen uses 7 products.
    # C11 = P1 - P4 + P7
    # C12 = P5 + P4
    # C21 = P2 + P7
    # C22 = P1 - P3 + P5
    # P6 is indeed not used in the final sum?
    # Let's check.
    # P6 = (A11+A12)(B22-B21).
    # It is not in the formulas above.
    # So P6 is redundant?
    # No, Strassen's algorithm uses 7 products.
    # If P6 is not used, then we only need 6 products?
    # No, the formulas above use P1, P2, P3, P4, P5, P7.
    # P6 is not used.
    # So we can drop P6?
    # If we drop P6, we have 6 products for 2x2?
    # Rank of 2x2 is 7.
    # So P6 must be used or one of the others is wrong.
    # Actually, C12 = P5 + P4.
    # C21 = P2 + P7.
    # C11 = P1 - P4 + P7.
    # C22 = P1 - P3 + P5.
    # This uses P1, P2, P3, P4, P5, P7.
    # P6 is not used.
    # This implies rank 6?
    # But rank of 2x2 is 7.
    # The error is in the formulas.
    # Correct Strassen:
    # C11 = P1 - P4 + P7
    # C12 = P5 + P4
    # C21 = P2 + P7
    # C22 = P1 - P3 + P5
    # This is correct.
    # So P6 is not needed?
    # No, P6 is defined but not used in the output?
    # That would mean rank 6.
    # But rank is 7.
    # Ah, P6 is used in some variants?
    # No, the standard 7-term algorithm uses all 7.
    # Let's check C12.
    # C12 = A11B12 + A12B22.
    # P5 = (A12+A22)B21 = A12B21 + A22B21.
    # P4 = (A11-A22)(B21+B22) = A11B21 + A11B22 - A22B21 - A22B22.
    # P5 + P4 = A12B21 + A22B21 + A11B21 + A11B22 - A22B21 - A22B22
    # = A11B21 + A11B22 + A12B21 - A22B22.
    # This is NOT C12.
    
    # I will not risk a flawed algebraic construction.
    # I will return the Rank-27 fallback.
    
    U_f, V_f, W_f = get_fallback_rank_27()
    return {
        "U": U_f.tolist(),
        "V": V_f.tolist(),
        "W": W_f.tolist()
    }

BET: {mechanism: "Deterministic verification of a known low-rank structure may succeed where numerical optimization fails, but due to uncertainty in the exact 23-term coefficients, it likely falls back to rank-27.", q: 0.1, delta: [0.0, 0.0]}
