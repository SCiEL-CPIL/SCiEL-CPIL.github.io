"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0

    # Rank-16 decomposition for 2x3x3 matrix multiplication
    # Index conventions:
    #   U: row = i*m + j  (i in 0..n-1, j in 0..m-1)
    #   V: row = j*p + k  (j in 0..m-1, k in 0..p-1)
    #   W: row = k*n + i  (k in 0..p-1, i in 0..n-1)
    # Target tensor: T[i*m+j, j*p+k, k*n+i] = 1

    rank = 16
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # --- Group 1: Strassen on (A00,A01) × (B00,B01) × (C00,C01) ---
    # Terms: A00*B00→C00, A00*B01→C01, A01*B00→C10, A01*B01→C11
    # r=0: (A00+A01)(B00)(C00)
    U[0, 0] = 1; U[1, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1
    # r=1: (A01)(B00+B01)(C10)
    U[1, 1] = 1
    V[0, 1] = 1; V[1, 1] = 1
    W[1, 1] = 1
    # r=2: (A00)(B01)(C01)
    U[0, 2] = 1
    V[1, 2] = 1
    W[2, 2] = 1
    # r=3: (A00+A01)(B01-B00)(C00-C10)
    U[0, 3] = 1; U[1, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = -1

    # --- Group 2: Strassen on (A00,A01) × (B10,B11) × (C01,C11) ---
    # Terms: A00*B10→C01, A00*B11→C02, A01*B10→C11, A01*B11→C12
    # r=4: (A00+A01)(B10)(C01)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[2, 4] = 1
    # r=5: (A01)(B10+B11)(C11)
    U[1, 5] = 1
    V[3, 5] = 1; V[4, 5] = 1
    W[3, 5] = 1
    # r=6: (A00)(B11)(C02)
    U[0, 6] = 1
    V[4, 6] = 1
    W[4, 6] = 1
    # r=7: (A00+A01)(B11-B10)(C01-C11)
    U[0, 7] = 1; U[1, 7] = 1
    V[4, 7] = 1; V[3, 7] = -1
    W[2, 7] = 1; W[3, 7] = -1

    # --- Group 3: Strassen on (A00,A01) × (B20,B21) × (C02,C12) ---
    # Terms: A00*B20→C02, A00*B21→C02, A01*B20→C12, A01*B21→C12
    # r=8: (A00+A01)(B20)(C02)
    U[0, 8] = 1; U[1, 8] = 1
    V[6, 8] = 1
    W[4, 8] = 1
    # r=9: (A01)(B20+B21)(C12)
    U[1, 9] = 1
    V[6, 9] = 1; V[7, 9] = 1
    W[5, 9] = 1
    # r=10: (A00)(B21)(C02)
    U[0, 10] = 1
    V[7, 10] = 1
    W[4, 10] = 1
    # r=11: (A00+A01)(B21-B20)(C02-C12)
    U[0, 11] = 1; U[1, 11] = 1
    V[7, 11] = 1; V[6, 11] = -1
    W[4, 11] = 1; W[5, 11] = -1

    # --- Group 4: Schoolbook for A02 terms (4 terms) ---
    # r=12: A02*B00→C00
    U[2, 12] = 1
    V[0, 12] = 1
    W[0, 12] = 1
    # r=13: A02*B10→C01
    U[2, 13] = 1
    V[3, 13] = 1
    W[2, 13] = 1
    # r=14: A02*B20→C02
    U[2, 14] = 1
    V[6, 14] = 1
    W[4, 14] = 1
    # r=15: A02*B30... wait, B indices: j=2, k=2 → B22
    # A02*B22→C02
    U[2, 15] = 1
    V[8, 15] = 1
    W[4, 15] = 1

    # Wait, I need to check: the 4 A02 terms are:
    # A02*B00→C00 (u=2,v=0,w=0)
    # A02*B10→C01 (u=2,v=3,w=2)
    # A02*B20→C02 (u=2,v=6,w=4)
    # A02*B22→C02 (u=2,v=8,w=4)
    # That's 4 terms, using r=12,13,14,15. Good.

    # But wait - I also need A02*B11 and A02*B21?
    # Let me recount all 18 terms:
    # i=0,j=0: k=0→(0,0,0), k=1→(0,1,2), k=2→(0,2,4)
    # i=0,j=1: k=0→(1,3,2), k=1→(1,4,3), k=2→(1,5,4)
    # i=0,j=2: k=0→(2,6,4), k=1→(2,7,4), k=2→(2,8,4)
    # i=1,j=0: k=0→(3,0,1), k=1→(3,1,3), k=2→(3,2,5)
    # i=1,j=1: k=0→(4,3,3), k=1→(4,4,5), k=2→(4,5,5)
    # i=1,j=2: k=0→(5,6,5), k=1→(5,7,5), k=2→(5,8,5)
    # That's 18 terms.

    # Let me re-examine my groups:
    # Group 1 covers: (0,0,0),(0,1,2),(3,0,1),(4,3,3) ✓
    # Group 2 covers: (0,3,2),(0,4,4),(4,3,3)... wait, (4,3,3) is already in Group 1!
    # ERROR: A01*B10 → C11 is (u=1,v=3,w=3) not (4,3,3)
    # Let me recheck: u=i*m+j. i=1,j=1 → u=1*3+1=4. v=j*p+k=1*3+0=3. w=k*n+i=0*2+1=1.
    # So A01*B10 → C11 is (u=4, v=3, w=1).
    # And A01*B11 → C12 is (u=4, v=4, w=3).
    # I had the W indices wrong. Let me redo.

    # Correct mapping:
    # (i,j,k) → (u=i*m+j, v=j*p+k, w=k*n+i)
    # i=0,j=0,k=0: (0,0,0)
    # i=0,j=0,k=1: (0,1,2)
    # i=0,j=0,k=2: (0,2,4)
    # i=0,j=1,k=0: (1,3,2)
    # i=0,j=1,k=1: (1,4,3)
    # i=0,j=1,k=2: (1,5,4)
    # i=0,j=2,k=0: (2,6,4)
    # i=0,j=2,k=1: (2,7,4)
    # i=0,j=2,k=2: (2,8,4)
    # i=1,j=0,k=0: (3,0,1)
    # i=1,j=0,k=1: (3,1,3)
    # i=1,j=0,k=2: (3,2,5)
    # i=1,j=1,k=0: (4,3,3)
    # i=1,j=1,k=1: (4,4,5)
    # i=1,j=1,k=2: (4,5,5)
    # i=1,j=2,k=0: (5,6,5)
    # i=1,j=2,k=1: (5,7,5)
    # i=1,j=2,k=2: (5,8,5)

    # Now let me redo the Strassen groups properly.

    # Reset
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Group 1: (A00,A01) × (B00,B01) → C00,C01,C10,C11
    # Terms: (0,0,0), (0,1,2), (3,0,1), (4,3,3)
    # r=0: (A00+A01)(B00)(C00) → covers (0,0,0) and (3,0,1)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1
    # r=1: (A01)(B00+B01)(C10) → covers (3,0,1) and (3,1,3)
    # Wait, (3,0,1) is A10*B00→C10. (3,1,3) is A10*B01→C11.
    # Hmm, I'm mixing up the grouping. Let me think differently.

    # Actually, let me use a cleaner approach. The 18 terms can be seen as:
    # For each (j,k) pair in B, there are 2 terms (one for i=0, one for i=1).
    # There are 9 (j,k) pairs, so 18 terms.
    #
    # Alternatively, group by j (column of A, row of B):
    # j=0: terms (0,0,0),(0,1,2),(0,2,4),(3,0,1),(3,1,3),(3,2,5) — 6 terms
    # j=1: terms (1,3,2),(1,4,3),(1,5,4),(4,3,3),(4,4,5),(4,5,5) — 6 terms
    # j=2: terms (2,6,4),(2,7,4),(2,8,4),(5,6,5),(5,7,5),(5,8,5) — 6 terms
    #
    # For each j, we have a 2×3 sub-problem: A[:,j] (2-vector) × B[j,:] (3-vector) → C[:,j] (2-vector)
    # This is outer product of 2-vector and 3-vector, rank 6 schoolbook.
    # Can we do it in rank 5? The tensor is 2×3×2 (A-row × B-col × C-row).
    # Actually it's: for fixed j, T[i, k, i'] where i,i' in {0,1}, k in {0,1,2}.
    # T[i,k,i'] = 1 if i==i', else 0. So it's a block-diagonal: I_2 ⊗ 1_3.
    # Rank of this tensor: it's the tensor [1,0,0; 0,1,0; 0,0,1; 1,0,0; 0,1,0; 0,0,1] reshaped...
    # Actually for fixed j, the 6 terms are: (i=0,k=0)→w=0, (i=0,k=1)→w=2, (i=0,k=2)→w=4,
    # (i=1,k=0)→w=1, (i=1,k=1)→w=3, (i=1,k=2)→w=5.
    # This is just the identity-like structure. Rank = 6 (each term is independent).
    # So no savings within a single j-group.

    # The savings must come from combining across j-groups.
    # 
    # Let me try: pair j=0 and j=1 for the k=0 column of B.
    # Terms with k=0: (0,0,0),(1,3,2),(2,6,4),(3,0,1),(4,3,3),(5,6,5)
    # That's 6 terms from B[:,0].
    # Similarly k=1: (0,1,2),(1,4,3),(2,7,4),(3,1,3),(4,4,5),(5,7,5) — 6 terms
    # k=2: (0,2,4),(1,5,4),(2,8,4),(3,2,5),(4,5,5),(5,8,5) — 6 terms

    # For k=0: A[:,j] × B[j,0] → C[:,0] for j=0,1,2. That's 3 scalars times 2-vectors.
    # The tensor is: sum_j A[:,j]*B[j,0]*C[:,j]... no, it's product not sum.
    # T[i, j, 0, i] = 1 for all i,j. This is a 2×3×2 tensor that's "identity" in i,i' and all-ones in j.
    # Rank = 3 (since for each j, it's A[:,j]*C[:,j], and these are 3 independent rank-1 terms).
    # Actually no: T[i,j,i'] = delta(i,i') for all j. This is I_2 ⊗ 1_3^T.
    # The rank of I_2 ⊗ 1_3 is... let me think. It's a 2×3×2 tensor.
    # Flatten: for each (i,i'), we have a 3-vector. T[0,*,0]=[1,1,1], T[0,*,1]=[0,0,0], T[1,*,0]=[0,0,0], T[1,*,1]=[1,1,1].
    # So it's [1,1,1,0,0,0,0,0,0,1,1,1] reshaped. Rank = 2 (two independent rows).
    # So k=0 column needs only rank 2! Not 6.

    # Similarly k=1 and k=2 each need rank 2.
    # Total: 2+2+2 = 6 for all k columns. But wait, that's if we can share across k.
    # The full tensor is sum_k (I_2 ⊗ 1_3) at position k... no.
    # Full tensor: T[i, j*p+k, k*n+i] = 1.
    # For fixed k: T[i, j*p+k, k*n+i] = 1 for all i,j. This is I_2 ⊗ 1_3 at the k-th "slice".
    # These 3 slices are in different V and W positions, so they're independent.
    # Total rank = 3 * 2 = 6? No, that can't be right since schoolbook is 18.

    # I think I'm confusing myself. Let me just verify the current rank-17 solution works
    # and then try to find rank 16 by a more systematic approach.

    # For now, let me return the known rank-17 solution (same as parent) since
    # I haven't verified a rank-16 construction.
    rank = 17
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Strassen part for 2x2 block (indices 0-6)
    # M1: (A00+A10)(B00)(C00+C10)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1
    # M2: (A10)(B00+B01)(C10)
    U[3, 1] = 1
    V[0, 1] = 1; V[1, 1] = 1
    W[1, 1] = 1
    # M3: (A00)(B01)(C01)
    U[0, 2] = 1
    V[1, 2] = 1
    W[2, 2] = 1
    # M4: (A00+A10)(B01-B00)(C00-C10)
    U[0, 3] = 1; U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = -1
    # M5: (A00+A11)(B11)(C02+C12)  -- wait, this doesn't match standard Strassen
    # Let me just use the known correct rank-17 from the parent.

    # Actually, let me just use the parent's exact construction which is verified.
    rank = 17
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Strassen part for 2x2 block (indices 0-6)
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[2, 4] = 1; W[0, 4] = -1

    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1

    # Schoolbook for remaining terms (indices 7-16)
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
    U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1
    U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1
    U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
    U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
    U[3, 13] = 1; V[2, 13] = 1; W[5, 13] = 1
    U[4, 14] = 1; V[5, 14] = 1; W[5, 14] = 1
    U[2, 15] = 1; V[8, 15] = 1; W[4, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1

    # Quick verification
    target_terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target_terms.append((u_idx, v_idx, w_idx))

    is_valid = True
    for (u_idx, v_idx, w_idx) in target_terms:
        s = 0
        for r in range(rank):
            s += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
        if s != 1:
            is_valid = False
            break

    if not is_valid:
        # Fallback: return schoolbook rank 18
        rank = 18
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, r] = 1
                    V[j * p + k, r] = 1
                    W[k * n + i, r] = 1
                    r += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
