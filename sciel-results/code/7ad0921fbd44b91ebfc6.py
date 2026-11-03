"""Matrix multiplication 4x4x9 rank-minimal exact tensor decomposition.

Combines:
  1. Structural Strassen-based block decomposition (rank 114) as a strong
     starting point and guaranteed valid fallback.
  2. ALS refinement to search for lower ranks.
  3. Exact half-integer/integer polish.
"""
import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DL = t0 + 110  # safety margin before 120s hard kill

    n, m, p = DIMENSIONS
    Nr = n * m          # 16
    Mr = m * p          # 36
    Pr = p * n          # 36

    # Build the set of (a,b,c) triples where T[a,b,c] = 1
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))
    Tset = set(triples)
    ntrip = len(triples)  # 144

    # Precompute index maps
    a_to_bc = {a: [] for a in range(Nr)}
    b_to_ac = {b: [] for b in range(Mr)}
    c_to_ab = {c: [] for c in range(Pr)}
    for (a, b, c) in triples:
        a_to_bc[a].append((b, c))
        b_to_ac[b].append((a, c))
        c_to_ab[c].append((a, b))

    # ---- schoolbook rank-144 baseline ----
    R0 = ntrip
    U0 = np.zeros((Nr, R0))
    V0 = np.zeros((Mr, R0))
    W0 = np.zeros((Pr, R0))
    for idx, (a, b, c) in enumerate(triples):
        U0[a, idx] = 1.0
        V0[b, idx] = 1.0
        W0[c, idx] = 1.0

    def _check_full(U, V, W, tol=1e-9):
        """Exact check over every (a,b,c) entry."""
        for (a, b, c) in triples:
            if abs(np.dot(U[a] * V[b], W[c]) - 1.0) > tol:
                return False
        for a in range(Nr):
            for b in range(Mr):
                for c in range(Pr):
                    if (a, b, c) in Tset:
                        continue
                    if abs(np.dot(U[a] * V[b], W[c])) > tol:
                        return False
        return True

    def _err(U, V, W):
        e = 0.0
        for (a, b, c) in triples:
            d = np.dot(U[a] * V[b], W[c]) - 1.0
            e += d * d
        return e

    def _als(U_init, V_init, W_init, rank, max_it=300, seed=0):
        rng = np.random.RandomState(seed)
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()

        prev_e = float('inf')
        for it in range(max_it):
            if time.time() > DL - 5:
                break

            # --- update U ---
            B = np.stack([np.outer(V[:, r], W[:, r]).ravel() for r in range(rank)], axis=0)
            for a in range(Nr):
                tv = np.zeros(Mr * Pr)
                for (b, c) in a_to_bc[a]:
                    tv[b * Pr + c] = 1.0
                U[a] = np.linalg.lstsq(B.T, tv, rcond=None)[0]

            # --- update V ---
            C = np.stack([np.outer(U[:, r], W[:, r]).ravel() for r in range(rank)], axis=0)
            for b in range(Mr):
                tv = np.zeros(Nr * Pr)
                for (a, c) in b_to_ac[b]:
                    tv[a * Pr + c] = 1.0
                V[b] = np.linalg.lstsq(C.T, tv, rcond=None)[0]

            # --- update W ---
            D = np.stack([np.outer(U[:, r], V[:, r]).ravel() for r in range(rank)], axis=0)
            for c in range(Pr):
                tv = np.zeros(Nr * Mr)
                for (a, b) in c_to_ab[c]:
                    tv[a * Mr + b] = 1.0
                W[c] = np.linalg.lstsq(D.T, tv, rcond=None)[0]

            e = _err(U, V, W)
            if abs(prev_e - e) < 1e-12 * max(1.0, prev_e):
                break
            prev_e = e

        return U, V, W

    def _polish(U, V, W, rank, max_it=50, tol=1e-12):
        """High-precision polish: tighten to exact integer/half-integer if close."""
        for it in range(max_it):
            if time.time() > DL - 3:
                break
            Ur = np.round(U * 2) / 2
            Vr = np.round(V * 2) / 2
            Wr = np.round(W * 2) / 2
            e = _err(Ur, Vr, Wr)
            if e < tol:
                return Ur, Vr, Wr
            Ui = np.round(U)
            Vi = np.round(V)
            Wi = np.round(W)
            e2 = _err(Ui, Vi, Wi)
            if e2 < tol:
                return Ui, Vi, Wi
            U = Ur
            V = Vr
            W = Wr
            e = _err(U, V, W)
            if e < tol:
                break
        return U, V, W

    # ---- Structural Strassen-based block decomposition (rank 114) ----
    def _build_strassen_block():
        """Build rank-114 decomposition: split 9 = 4+4+1.
        Two 4x4*4x4 blocks via Strassen (rank 49 each) + one 4x4*4x1 (rank 16).
        """
        # Strassen for 2x2*2x2: rank 7
        # For 4x4*4x4: view as 2x2 blocks of 2x2, apply Strassen -> rank 7*7 = 49
        #
        # Let's build the 2x2*2x2 Strassen decomposition first.
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-h)
        # m4 = d(g-e)
        # m5 = (a+b)h
        # m6 = (c-a)(e+f)
        # m7 = (b-d)(g+h)
        # c11 = m1 + m4 - m5 + m7
        # c12 = m3 + m5
        # c21 = m2 + m4
        # c22 = m1 - m2 + m3 + m6
        #
        # For 4x4*4x4, A = [[A11,A12],[A21,A22]], B = [[B11,B12],[B21,B22]]
        # where each block is 2x2.
        # Apply Strassen at block level:
        # M1 = (A11+A22)(B11+B22)  -> 2x2*2x2 product, rank 7
        # M2 = (A21+A22)B11        -> rank 7
        # M3 = A11(B12-B22)        -> rank 7
        # M4 = A22(B21-B11)        -> rank 7
        # M5 = (A11+A12)B22        -> rank 7
        # M6 = (A21-A11)(B11+B12)  -> rank 7
        # M7 = (A12-A22)(B21+B22)  -> rank 7
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        #
        # Total rank = 7 * 7 = 49

        # First, build the 2x2*2x2 Strassen decomposition.
        # A is 2x2, B is 2x2, C is 2x2.
        # A indices: a=0 (row0,col0), b=1 (row0,col1), c=2 (row1,col0), d=3 (row1,col1)
        # B indices: e=0 (row0,col0), f=1 (row0,col1), g=2 (row1,col0), h=3 (row1,col1)
        # C indices: c11=0, c12=1, c21=2, c22=3
        #
        # In our tensor convention for 2x2*2x2:
        # a_idx = i*2 + j (i=row of A, j=col of A / row of B)
        # b_idx = j*2 + k (j=row of B, k=col of B)
        # c_idx = k*2 + i (k=col of C, i=row of C)
        #
        # For 2x2*2x2:
        # A element (i,j): a_idx = i*2+j
        # B element (j,k): b_idx = j*2+k
        # C element (i,k): c_idx = k*2+i

        # Strassen 2x2*2x2 rank-7 decomposition
        # Each term m_r = (linear combo of A) * (linear combo of B)
        # We need to express as sum of rank-1 tensors.
        #
        # m1 = (a+d)(e+h)
        #   A part: a + d -> U contribution: U[0] += 1, U[3] += 1
        #   B part: e + h -> V contribution: V[0] += 1, V[3] += 1
        # m2 = (c+d)e
        #   A part: c + d -> U[2] += 1, U[3] += 1
        #   B part: e -> V[0] += 1
        # m3 = a(f-h)
        #   A part: a -> U[0] += 1
        #   B part: f - h -> V[1] += 1, V[3] -= 1
        # m4 = d(g-e)
        #   A part: d -> U[3] += 1
        #   B part: g - e -> V[2] += 1, V[0] -= 1
        # m5 = (a+b)h
        #   A part: a + b -> U[0] += 1, U[1] += 1
        #   B part: h -> V[3] += 1
        # m6 = (c-a)(e+f)
        #   A part: c - a -> U[2] += 1, U[0] -= 1
        #   B part: e + f -> V[0] += 1, V[1] += 1
        # m7 = (b-d)(g+h)
        #   A part: b - d -> U[1] += 1, U[3] -= 1
        #   B part: g + h -> V[2] += 1, V[3] += 1
        #
        # C reconstruction:
        # c11 = m1 + m4 - m5 + m7
        # c12 = m3 + m5
        # c21 = m2 + m4
        # c22 = m1 - m2 + m3 + m6

        # For 2x2*2x2, the Strassen decomposition:
        # U (2x2 -> 4 rows), V (2x2 -> 4 rows), W (2x2 -> 4 rows), rank 7
        # U[row, col] where row = i*2+j, col = term index
        # V[row, col] where row = j*2+k, col = term index
        # W[row, col] where row = k*2+i, col = term index

        # Let's build it explicitly.
        # Term 0: m1 = (a+d)(e+h)
        #   U: a(0,0)=1, d(1,1)=1 -> U[0,0]=1, U[3,0]=1
        #   V: e(0,0)=1, h(1,1)=1 -> V[0,0]=1, V[3,0]=1
        #   W: c11(0,0) gets +1, c22(1,1) gets +1 -> W[0,0]=1, W[3,0]=1
        # Term 1: m2 = (c+d)e
        #   U: c(1,0)=1, d(1,1)=1 -> U[2,1]=1, U[3,1]=1
        #   V: e(0,0)=1 -> V[0,1]=1
        #   W: c21(0,1) gets +1, c22(1,1) gets -1 -> W[2,1]=1, W[3,1]=-1
        # Term 2: m3 = a(f-h)
        #   U: a(0,0)=1 -> U[0,2]=1
        #   V: f(0,1)=1, h(1,1)=-1 -> V[1,2]=1, V[3,2]=-1
        #   W: c12(1,0) gets +1, c22(1,1) gets +1 -> W[1,2]=1, W[3,2]=1
        # Term 3: m4 = d(g-e)
        #   U: d(1,1)=1 -> U[3,3]=1
        #   V: g(1,0)=1, e(0,0)=-1 -> V[2,3]=1, V[0,3]=-1
        #   W: c11(0,0) gets +1, c21(0,1) gets +1 -> W[0,3]=1, W[2,3]=1
        # Term 4: m5 = (a+b)h
        #   U: a(0,0)=1, b(0,1)=1 -> U[0,4]=1, U[1,4]=1
        #   V: h(1,1)=1 -> V[3,4]=1
        #   W: c11(0,0) gets -1, c12(1,0) gets +1 -> W[0,4]=-1, W[1,4]=1
        # Term 5: m6 = (c-a)(e+f)
        #   U: c(1,0)=1, a(0,0)=-1 -> U[2,5]=1, U[0,5]=-1
        #   V: e(0,0)=1, f(0,1)=1 -> V[0,5]=1, V[1,5]=1
        #   W: c22(1,1) gets +1 -> W[3,5]=1
        # Term 6: m7 = (b-d)(g+h)
        #   U: b(0,1)=1, d(1,1)=-1 -> U[1,6]=1, U[3,6]=-1
        #   V: g(1,0)=1, h(1,1)=1 -> V[2,6]=1, V[3,6]=1
        #   W: c11(0,0) gets +1 -> W[0,6]=1

        U22 = np.zeros((4, 7))
        V22 = np.zeros((4, 7))
        W22 = np.zeros((4, 7))
        # m1
        U22[0, 0] = 1; U22[3, 0] = 1
        V22[0, 0] = 1; V22[3, 0] = 1
        W22[0, 0] = 1; W22[3, 0] = 1
        # m2
        U22[2, 1] = 1; U22[3, 1] = 1
        V22[0, 1] = 1
        W22[2, 1] = 1; W22[3, 1] = -1
        # m3
        U22[0, 2] = 1
        V22[1, 2] = 1; V22[3, 2] = -1
        W22[1, 2] = 1; W22[3, 2] = 1
        # m4
        U22[3, 3] = 1
        V22[2, 3] = 1; V22[0, 3] = -1
        W22[0, 3] = 1; W22[2, 3] = 1
        # m5
        U22[0, 4] = 1; U22[1, 4] = 1
        V22[3, 4] = 1
        W22[0, 4] = -1; W22[1, 4] = 1
        # m6
        U22[2, 5] = 1; U22[0, 5] = -1
        V22[0, 5] = 1; V22[1, 5] = 1
        W22[3, 5] = 1
        # m7
        U22[1, 6] = 1; U22[3, 6] = -1
        V22[2, 6] = 1; V22[3, 6] = 1
        W22[0, 6] = 1

        # Verify 2x2*2x2 Strassen
        def _check_22(U, V, W, tol=1e-9):
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        a_idx = i * 2 + j
                        b_idx = j * 2 + k
                        c_idx = k * 2 + i
                        val = np.dot(U[a_idx] * V[b_idx], W[c_idx])
                        if abs(val - 1.0) > tol:
                            return False
            # Check zeros
            all_idx = set()
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        all_idx.add((i*2+j, j*2+k, k*2+i))
            for a in range(4):
                for b in range(4):
                    for c in range(4):
                        if (a, b, c) in all_idx:
                            continue
                        val = np.dot(U[a] * V[b], W[c])
                        if abs(val) > tol:
                            return False
            return True

        if not _check_22(U22, V22, W22):
            # Fallback to schoolbook for 2x2
            U22 = np.zeros((4, 4))
            V22 = np.zeros((4, 4))
            W22 = np.zeros((4, 4))
            for idx, (i, j, k) in enumerate(product(range(2), range(2), range(2))):
                U22[i*2+j, idx] = 1
                V22[j*2+k, idx] = 1
                W22[k*2+i, idx] = 1
            if not _check_22(U22, V22, W22):
                return None  # Should not happen

        # Now build 4x4*4x4 using block Strassen.
        # A is 4x4, viewed as 2x2 blocks of 2x2:
        # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
        # A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
        #
        # In our tensor convention for 4x4*4x4:
        # A element (i,j): a_idx = i*4+j
        # B element (j,k): b_idx = j*4+k
        # C element (i,k): c_idx = k*4+i
        #
        # Block A11: i in {0,1}, j in {0,1} -> a_idx in {0,1,4,5}
        # Block A12: i in {0,1}, j in {2,3} -> a_idx in {2,3,6,7}
        # Block A21: i in {2,3}, j in {0,1} -> a_idx in {8,9,12,13}
        # Block A22: i in {2,3}, j in {2,3} -> a_idx in {10,11,14,15}
        #
        # Block B11: j in {0,1}, k in {0,1} -> b_idx in {0,1,4,5}
        # Block B12: j in {0,1}, k in {2,3} -> b_idx in {2,3,6,7}
        # Block B21: j in {2,3}, k in {0,1} -> b_idx in {8,9,12,13}
        # Block B22: j in {2,3}, k in {2,3} -> b_idx in {10,11,14,15}
        #
        # Block C11: i in {0,1}, k in {0,1} -> c_idx in {0,1,4,5}
        # Block C12: i in {0,1}, k in {2,3} -> c_idx in {2,3,6,7}
        # Block C21: i in {2,3}, k in {0,1} -> c_idx in {8,9,12,13}
        # Block C22: i in {2,3}, k in {2,3} -> c_idx in {10,11,14,15}

        # Strassen at block level:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6

        # For each Mi, we need a 2x2*2x2 decomposition (rank 7).
        # The A-part of Mi is a 2x2 matrix (linear combo of 2x2 blocks),
        # and the B-part is a 2x2 matrix.
        # Each 2x2*2x2 product uses the U22, V22, W22 decomposition.
        #
        # For M1 = (A11+A22)(B11+B22):
        #   A-part: A11 + A22 -> 4x4 matrix with blocks A11, 0; 0, A22
        #   B-part: B11 + B22 -> 4x4 matrix with blocks B11, 0; 0, B22
        #   But wait, this is a 4x4*4x4 product, not 2x2*2x2!
        #
        # Actually, I need to reconsider. The block Strassen gives:
        # M1 is a 4x4 matrix = (A11+A22) * (B11+B22) where each factor is 2x2.
        # So M1 is a 2x2*2x2 product, which has rank 7.
        # But in the full 4x4 tensor, M1 contributes to C11 (a 2x2 block).
        #
        # Let me re-examine. In the block Strassen:
        # A is 4x4, B is 4x4. We partition into 2x2 blocks.
        # M1 = (A11+A22)(B11+B22) is a 2x2 matrix (product of two 2x2 matrices).
        # This 2x2 matrix goes into C11 (which is a 2x2 block of C).
        #
        # So the full decomposition is:
        # For each block-level Strassen term Mi (i=1..7), we have a 2x2*2x2
        # product. Each such product is decomposed using the rank-7 2x2*2x2
        # Strassen decomposition.
        #
        # The A-part of Mi is a 2x2 matrix formed by linear combos of 2x2 blocks
        # of A. In the full 4x4 tensor, this corresponds to specific rows of U.
        # Similarly for B and C.
        #
        # Let's build this carefully.
        #
        # For M1 = (A11+A22)(B11+B22):
        #   A-part: 2x2 matrix where (0,0) block = A11, (0,1) block = 0,
        #           (1,0) block = 0, (1,1) block = A22
        #   In 4x4 A, this is: rows 0,1,2,3 x cols 0,1,2,3
        #           A11 is at rows 0,1 cols 0,1; A22 is at rows 2,3 cols 2,3
        #   So the A-part matrix is:
        #     [[A11, 0], [0, A22]]
        #   In the U tensor, the rows corresponding to this are:
        #     A11: a_idx = 0,1,4,5
        #     A22: a_idx = 10,11,14,15
        #
        #   B-part: 2x2 matrix where (0,0) block = B11, (0,1) block = 0,
        #           (1,0) block = 0, (1,1) block = B22
        #   In 4x4 B, this is:
        #     [[B11, 0], [0, B22]]
        #   V rows: B11: b_idx = 0,1,4,5; B22: b_idx = 10,11,14,15
        #
        #   C-part: M1 contributes +1 to C11.
        #   C11 is at c_idx = 0,1,4,5
        #
        #   So for M1, we use the 2x2*2x2 Strassen decomposition, but with
        #   the A-rows mapped to {0,1,4,5,10,11,14,15} (but actually only
        #   the non-zero parts), B-rows to {0,1,4,5,10,11,14,15},
        #   and C-rows to {0,1,4,5}.
        #
        # Wait, I'm overcomplicating this. Let me think differently.
        #
        # The block Strassen for 4x4*4x4 gives 7 terms, each of which is a
        # 2x2*2x2 product. Each 2x2*2x2 product can be done with rank 7.
        # So total rank = 7*7 = 49.
        #
        # For each block-level term Mi, the 2x2*2x2 product has:
        #   A: a 2x2 matrix (which is a linear combo of 2x2 blocks of the 4x4 A)
        #   B: a 2x2 matrix (linear combo of 2x2 blocks of 4x4 B)
        #   C: contributes to one of C11, C12, C21, C22 (a 2x2 block)
        #
        # The 2x2*2x2 Strassen decomposition (U22, V22, W22) works on
        # 2x2 matrices. We need to "embed" this into the 4x4 tensor.
        #
        # For M1 = (A11+A22)(B11+B22) -> C11:
        #   The 2x2 A-matrix has entries:
        #     A[0,0] = A11[0,0] = A_full[0,0] -> a_idx = 0
        #     A[0,1] = A11[0,1] = A_full[0,1] -> a_idx = 1
        #     A[1,0] = A11[1,0] = A_full[1,0] -> a_idx = 4
        #     A[1,1] = A11[1,1] = A_full[1,1] -> a_idx = 5
        #   Wait, no. A11 is the top-left 2x2 block of A.
        #   A11[i,j] = A_full[i,j] for i,j in {0,1}.
        #   A22[i,j] = A_full[i+2,j+2] for i,j in {0,1}.
        #
        #   The A-part of M1 is A11 + A22, which is a 2x2 matrix where:
        #     (0,0) = A11[0,0] + A22[0,0] = A_full[0,0] + A_full[2,2]
        #     (0,1) = A11[0,1] + A22[0,1] = A_full[0,1] + A_full[2,3]
        #     (1,0) = A11[1,0] + A22[1,0] = A_full[1,0] + A_full[3,2]
        #     (1,1) = A11[1,1] + A22[1,1] = A_full[1,1] + A_full[3,3]
        #
        #   So the 2x2 A-matrix for M1 has:
        #     a_idx for (0,0): 0 (A_full[0,0]) and 10 (A_full[2,2])
        #     a_idx for (0,1): 1 (A_full[0,1]) and 11 (A_full[2,3])
        #     a_idx for (1,0): 4 (A_full[1,0]) and 14 (A_full[3,2])
        #     a_idx for (1,1): 5 (A_full[1,1]) and 15 (A_full[3,3])
        #
        #   Similarly for B-part: B11 + B22
        #     b_idx for (0,0): 0 (B_full[0,0]) and 10 (B_full[2,2])
        #     b_idx for (0,1): 1 (B_full[0,1]) and 11 (B_full[2,3])
        #     b_idx for (1,0): 4 (B_full[1,0]) and 14 (B_full[3,2])
        #     b_idx for (1,1): 5 (B_full[1,1]) and 15 (B_full[3,3])
        #
        #   C-part: contributes to C11, which is a 2x2 block.
        #     c_idx for (0,0): 0 (C_full[0,0])
        #     c_idx for (0,1): 4 (C_full[0,2])... wait, C11 is C[0:2, 0:2]
        #     c_idx for (0,0): k*4+i where i=0,k=0 -> 0
        #     c_idx for (0,1): k*4+i where i=0,k=1 -> 4
        #     c_idx for (1,0): k*4+i where i=1,k=0 -> 1
        #     c_idx for (1,1): k*4+i where i=1,k=1 -> 5
        #
        #   So for M1, the 2x2*2x2 Strassen decomposition is applied with:
        #     U rows: [0, 10] for A[0,0], [1, 11] for A[0,1], [4, 14] for A[1,0], [5, 15] for A[1,1]
        #     V rows: [0, 10] for B[0,0], [1, 11] for B[0,1], [4, 14] for B[1,0], [5, 15] for B[1,1]
        #     W rows: [0] for C[0,0], [4] for C[0,1], [1] for C[1,0], [5] for C[1,1]
        #
        #   The 2x2*2x2 Strassen U22 has entries at rows 0,1,2,3 corresponding
        #   to A[0,0], A[0,1], A[1,0], A[1,1].
        #   So we map:
        #     U22 row 0 -> U row 0 and U row 10 (both with same coefficient)
        #     U22 row 1 -> U row 1 and U row 11
        #     U22 row 2 -> U row 4 and U row 14
        #     U22 row 3 -> U row 5 and U row 15
        #
        #   Similarly for V and W.

        # Let me build the 4x4*4x4 Strassen decomposition.
        # For each of the 7 block-level terms, we have a 2x2*2x2 product.
        # For each 2x2*2x2 product, we use the rank-7 decomposition.
        # Total: 7 * 7 = 49 rank-1 terms.

        # Define the block-level Strassen terms:
        # Each term has: A-blocks (with signs), B-blocks (with signs), C-block (with sign)
        # A-blocks: which of A11, A12, A21, A22 and with what sign
        # B-blocks: which of B11, B12, B21, B22 and with what sign
        # C-block: which of C11, C12, C21, C22 and with what sign

        # M1 = (A11+A22)(B11+B22) -> C11: +1
        # M2 = (A21+A22)B11 -> C21: +1
        # M3 = A11(B12-B22) -> C12: +1, C22: +1
        # M4 = A22(B21-B11) -> C11: +1, C21: +1
        # M5 = (A11+A12)B22 -> C11: -1, C12: +1
        # M6 = (A21-A11)(B11+B12) -> C22: +1
        # M7 = (A12-A22)(B21+B22) -> C11: +1

        # For each term, the A-part is a 2x2 matrix formed by linear combos of
        # 2x2 blocks. The mapping from 2x2 block to 4x4 indices:
        # A11: rows 0,1 cols 0,1 -> a_idx: 0,1,4,5
        # A12: rows 0,1 cols 2,3 -> a_idx: 2,3,6,7
        # A21: rows 2,3 cols 0,1 -> a_idx: 8,9,12,13
        # A22: rows 2,3 cols 2,3 -> a_idx: 10,11,14,15

        # B11: rows 0,1 cols 0,1 -> b_idx: 0,1,4,5
        # B12: rows 0,1 cols 2,3 -> b_idx: 2,3,6,7
        # B21: rows 2,3 cols 0,1 -> b_idx: 8,9,12,13
        # B22: rows 2,3 cols 2,3 -> b_idx: 10,11,14,15

        # C11: rows 0,1 cols 0,1 -> c_idx: 0,1,4,5
        # C12: rows 0,1 cols 2,3 -> c_idx: 2,3,6,7
        # C21: rows 2,3 cols 0,1 -> c_idx: 8,9,12,13
        # C22: rows 2,3 cols 2,3 -> c_idx: 10,11,14,15

        # For a 2x2*2x2 product with A-part = sum of signed blocks,
        # the U rows are the union of the a_idx of the blocks, with coefficients.
        # Similarly for V and W.

        # Let me build the 4x4*4x4 Strassen decomposition by composing
        # the block-level Strassen with the 2x2*2x2 Strassen.

        # For each block-level term t (t=0..6):
        #   A-part: 2x2 matrix = sum of signed 2x2 blocks
        #   B-part: 2x2 matrix = sum of signed 2x2 blocks
        #   C-part: contributes to one or more 2x2 blocks with signs
        #
        # The 2x2*2x2 Strassen decomposition (U22, V22, W22) gives:
        #   For each rank-1 term r (r=0..6):
        #     U22[a, r] * V22[b, r] * W22[c, r] = 1 if (a,b,c) is a valid 2x2*2x2 triple
        #
        # To embed into 4x4:
        #   For each block-level term t and each 2x2 rank-1 term r:
        #     The A-part of the 2x2 product is the A-part of block term t.
        #     The U22 row a corresponds to a specific 2x2 entry (i,j) where a = i*2+j.
        #     This 2x2 entry corresponds to a 2x2 block of the 4x4 A.
        #     So U22[a, r] contributes to U[4x4_a_idx, t*7+r] where 4x4_a_idx
        #     is the a_idx of the 4x4 A corresponding to the 2x2 block entry (i,j).

        # This is getting very complex. Let me just build it programmatically.

        # For each block-level term, define:
        #   A_blocks: list of (block_id, sign) where block_id in {11,12,21,22}
        #   B_blocks: list of (block_id, sign)
        #   C_blocks: list of (block_id, sign)

        A_block_idx = {
            11: [0, 1, 4, 5],
            12: [2, 3, 6, 7],
            21: [8, 9, 12, 13],
            22: [10, 11, 14, 15]
        }
        B_block_idx = {
            11: [0, 1, 4, 5],
            12: [2, 3, 6, 7],
            21: [8, 9, 12, 13],
            22: [10, 11, 14, 15]
        }
        C_block_idx = {
            11: [0, 1, 4, 5],
            12: [2, 3, 6, 7],
            21: [8, 9, 12, 13],
            22: [10, 11, 14, 15]
        }

        # Block-level Strassen terms:
        # M1 = (A11+A22)(B11+B22) -> C11: +1
        # M2 = (A21+A22)B11 -> C21: +1
        # M3 = A11(B12-B22) -> C12: +1, C22: +1
        # M4 = A22(B21-B11) -> C11: +1, C21: +1
        # M5 = (A11+A12)B22 -> C11: -1, C12: +1
        # M6 = (A21-A11)(B11+B12) -> C22: +1
        # M7 = (A12-A22)(B21+B22) -> C11: +1

        block_terms = [
            # (A_blocks, B_blocks, C_blocks)
            ([(11, 1), (22, 1)], [(11, 1), (22, 1)], [(11, 1)]),           # M1
            ([(21, 1), (22, 1)], [(11, 1)], [(21, 1)]),                     # M2
            ([(11, 1)], [(12, 1), (22, -1)], [(12, 1), (22, 1)]),           # M3
            ([(22, 1)], [(21, 1), (11, -1)], [(11, 1), (21, 1)]),           # M4
            ([(11, 1), (12, 1)], [(22, 1)], [(11, -1), (12, 1)]),           # M5
            ([(21, 1), (11, -1)], [(11, 1), (12, 1)], [(22, 1)]),           # M6
            ([(12, 1), (22, -1)], [(21, 1), (22, 1)], [(11, 1)]),           # M7
        ]

        # For each block-level term, the A-part is a 2x2 matrix.
        # The 2x2 entry (i,j) of the A-part corresponds to:
        #   If A11 is in the term: A11[i,j] -> a_idx = A_block_idx[11][i*2+j]
        #   If A12 is in the term: A12[i,j] -> a_idx = A_block_idx[12][i*2+j]
        #   etc.
        #
        # So for a given block-level term, the 2x2 A-matrix has:
        #   Entry (i,j) = sum over blocks in A_blocks of sign * block[i,j]
        #   In the 4x4 tensor, this corresponds to:
        #     For each block with sign s, the 2x2 entry (i,j) maps to
        #     a_idx = A_block_idx[block][i*2+j] with coefficient s.
        #
        # The 2x2*2x2 Strassen U22 has:
        #   U22[a, r] where a = i*2+j is the 2x2 row index.
        #   U22[a, r] is the coefficient for the 2x2 A-entry (i,j) in rank-1 term r.
        #
        # To embed into 4x4:
        #   For block-level term t and 2x2 rank-1 term r:
        #     For each 2x2 row a (a = i*2+j):
        #       For each block in A_blocks with sign s:
        #         U[A_block_idx[block][a], t*7+r] += s * U22[a, r]
        #
        # Similarly for V and W.

        U44 = np.zeros((16, 49))
        V44 = np.zeros((16, 49))
        W44 = np.zeros((16, 49))

        for t, (A_blocks, B_blocks, C_blocks) in enumerate(block_terms):
            for r in range(7):
                col = t * 7 + r
                # U part
                for a in range(4):
                    coeff = U22[a, r]
                    if abs(coeff) < 1e-15:
                        continue
                    for (bid, s) in A_blocks:
                        idx = A_block_idx[bid][a]
                        U44[idx, col] += s * coeff
                # V part
                for b in range(4):
                    coeff = V22[b, r]
                    if abs(coeff) < 1e-15:
                        continue
                    for (bid, s) in B_blocks:
                        idx = B_block_idx[bid][b]
                        V44[idx, col] += s * coeff
                # W part
                for c in range(4):
                    coeff = W22[c, r]
                    if abs(coeff) < 1e-15:
                        continue
                    for (bid, s) in C_blocks:
                        idx = C_block_idx[bid][c]
                        W44[idx, col] += s * coeff

        # Verify 4x4*4x4 Strassen
        def _check_44(U, V, W, tol=1e-9):
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        a_idx = i * 4 + j
                        b_idx = j * 4 + k
                        c_idx = k * 4 + i
                        val = np.dot(U[a_idx] * V[b_idx], W[c_idx])
                        if abs(val - 1.0) > tol:
                            return False
            all_idx = set()
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        all_idx.add((i*4+j, j*4+k, k*4+i))
            for a in range(16):
                for b in range(16):
                    for c in range(16):
                        if (a, b, c) in all_idx:
                            continue
                        val = np.dot(U[a] * V[b], W[c])
                        if abs(val) > tol:
                            return False
            return True

        if not _check_44(U44, V44, W44):
            # Fallback: use schoolbook for 4x4*4x4
            U44 = np.zeros((16, 64))
            V44 = np.zeros((16, 64))
            W44 = np.zeros((16, 64))
            for idx, (i, j, k) in enumerate(product(range(4), range(4), range(4))):
                U44[i*4+j, idx] = 1
                V44[j*4+k, idx] = 1
                W44[k*4+i, idx] = 1
            if not _check_44(U44, V44, W44):
                return None

        # Now build the full 4x4*4x9 decomposition.
        # Split B into B1 (4x4), B2 (4x4), B3 (4x1).
        # C = A*B1 + A*B2 + A*B3
        #
        # A*B1: 4x4*4x4, rank 49 (Strassen) or 64 (schoolbook)
        # A*B2: 4x4*4x4, rank 49 (Strassen) or 64 (schoolbook)
        # A*B3: 4x4*4x1, rank 16 (schoolbook)
        #
        # Total: 49 + 49 + 16 = 114 (Strassen) or 64 + 64 + 16 = 144 (schoolbook)

        # For A*B1: B1 is the first 4 columns of B (cols 0-3).
        #   B1[j,k] = B_full[j,k] for k in {0,1,2,3}
        #   In 4x9 B, b_idx = j*9+k
        #   B1 b_idx: j*9+k for j in {0,1,2,3}, k in {0,1,2,3}
        #   = {0,1,2,3, 9,10,11,12, 18,19,20,21, 27,28,29,30}
        #
        #   C1 = A*B1 is the first 4 columns of C (cols 0-3).
        #   C1[i,k] = C_full[i,k] for k in {0,1,2,3}
        #   In 4x9 C, c_idx = k*4+i
        #   C1 c_idx: k*4+i for k in {0,1,2,3}, i in {0,1,2,3}
        #   = {0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15}
        #
        #   But wait, in the 4x4*4x4 Strassen, the C indices are k*4+i
        #   for k in {0,1,2,3}, i in {0,1,2,3}, which is exactly C1.
        #   And the B indices are j*4+k for j in {0,1,2,3}, k in {0,1,2,3}.
        #   But in the 4x9 B, B1 has b_idx = j*9+k, not j*4+k.
        #
        #   So I need to remap the B indices from the 4x4*4x4 decomposition
        #   to the 4x4*4x9 decomposition.
        #
        #   In the 4x4*4x4, B element (j,k) has b_idx = j*4+k.
        #   In the 4x4*4x9, B1 element (j,k) has b_idx = j*9+k.
        #   So the mapping is: b_idx_44 = j*4+k -> b_idx_49 = j*9+k.

        # For A*B2: B2 is the next 4 columns of B (cols 4-7).
        #   B2[j,k] = B_full[j,k+4] for k in {0,1,2,3}
        #   In 4x9 B, b_idx = j*9+(k+4)
        #   B2 b_idx: j*9+k+4 for j in {0,1,2,3}, k in {0,1,2,3}
        #   = {4,5,6,7, 13,14,15,16, 22,23,24,25, 31,32,33,34}
        #
        #   C2 = A*B2 is the next 4 columns of C (cols 4-7).
        #   C2[i,k] = C_full[i,k+4] for k in {0,1,2,3}
        #   In 4x9 C, c_idx = (k+4)*4+i
        #   C2 c_idx: (k+4)*4+i for k in {0,1,2,3}, i in {0,1,2,3}
        #   = {16,17,18,19, 20,21,22,23, 24,25,26,27, 28,29,30,31}
        #
        #   In the 4x4*4x4, C element (i,k) has c_idx = k*4+i.
        #   In the 4x4*4x9, C2 element (i,k) has c_idx = (k+4)*4+i.
        #   So the mapping is: c_idx_44 = k*4+i -> c_idx_49 = (k+4)*4+i.

        # For A*B3: B3 is the last column of B (col 8).
        #   B3[j,0] = B_full[j,8]
        #   In 4x9 B, b_idx = j*9+8
        #   B3 b_idx: {8, 17, 26, 35}
        #
        #   C3 = A*B3 is the last column of C (col 8).
        #   C3[i,0] = C_full[i,8]
        #   In 4x9 C, c_idx = 8*4+i = 32+i
        #   C3 c_idx: {32, 33, 34, 35}
        #
        #   A*B3 is a 4x4*4x1 product. Schoolbook rank 16.
        #   For each (i,j) in A and k=0 in B3:
        #     a_idx = i*4+j, b_idx = j*9+8, c_idx = 32+i

        # Build the full decomposition.
        rank_total = U44.shape[1] * 2 + 16  # 49*2 + 16 = 114
        Ufull = np.zeros((16, rank_total))
        Vfull = np.zeros((36, rank_total))
        Wfull = np.zeros((36, rank_total))

        # A*B1: first 49 columns
        for col in range(U44.shape[1]):
            Ufull[:, col] = U44[:, col]
            # Remap V indices: b_idx_44 = j*4+k -> b_idx_49 = j*9+k
            for b44 in range(16):
                j = b44 // 4
                k = b44 % 4
                b49 = j * 9 + k
                Vfull[b49, col] = V44[b44, col]
            # Remap W indices: c_idx_44 = k*4+i -> c_idx_49 = k*4+i (same for first 4 cols)
            for c44 in range(16):
                Wfull[c44, col] = W44[c44, col]

        # A*B2: next 49 columns
        offset = U44.shape[1]
        for col in range(U44.shape[1]):
            Ufull[:, offset + col] = U44[:, col]
            # Remap V indices: b_idx_44 = j*4+k -> b_idx_49 = j*9+(k+4)
            for b44 in range(16):
                j = b44 // 4
                k = b44 % 4
                b49 = j * 9 + (k + 4)
                Vfull[b49, offset + col] = V44[b44, col]
            # Remap W indices: c_idx_44 = k*4+i -> c_idx_49 = (k+4)*4+i
            for c44 in range(16):
                k = c44 // 4
                i = c44 % 4
                c49 = (k + 4) * 4 + i
                Wfull[c49, offset + col] = W44[c44, col]

        # A*B3: last 16 columns
        offset2 = 2 * U44.shape[1]
        for idx, (i, j) in enumerate(product(range(4), range(4))):
            a_idx = i * 4 + j
            b_idx = j * 9 + 8
            c_idx = 32 + i
            Ufull[a_idx, offset2 + idx] = 1
            Vfull[b_idx, offset2 + idx] = 1
            Wfull[c_idx, offset2 + idx] = 1

        # Verify the full decomposition
        if _check_full(Ufull, Vfull, Wfull):
            return Ufull, Vfull, Wfull
        else:
            return None

    # Try to build the structural decomposition
    structural = _build_strassen_block()
    if structural is not None:
        U_struct, V_struct, W_struct = structural
        rank_struct = U_struct.shape[1]
        # Verify
        if _check_full(U_struct, V_struct, W_struct):
            # Try to polish it to exact half-integer
            Ur, Vr, Wr = _polish(U_struct, V_struct, W_struct, rank_struct)
            if _check_full(Ur, Vr, Wr):
                # Now try ALS to see if we can go lower
                # But first, this is a valid rank-114 solution
                best_rank = rank_struct
                best = (Ur, Vr, Wr)
            else:
                best_rank = rank_struct
                best = (U_struct, V_struct, W_struct)
        else:
            best_rank = R0
            best = (U0, V0, W0)
    else:
        best_rank = R0
        best = (U0, V0, W0)

    # Try ALS to find lower ranks
    def _try_rank(rank, n_seeds=10):
        """Attempt to find a valid decomposition at a given rank."""
        if time.time() > DL - 15:
            return None

        for s in range(n_seeds):
            if time.time() > DL - 10:
                break

            # Initialization strategies
            if s % 4 == 0 and structural is not None:
                # Use structural as init (padded/truncated to rank)
                U = np.zeros((Nr, rank))
                V = np.zeros((Mr, rank))
                W = np.zeros((Pr, rank))
                r_min = min(rank, U_struct.shape[1])
                U[:, :r_min] = U_struct[:, :r_min]
                V[:, :r_min] = V_struct[:, :r_min]
                W[:, :r_min] = W_struct[:, :r_min]
                if rank > r_min:
                    rng0 = np.random.RandomState(4000 + rank + s)
                    U[:, r_min:] = rng0.randn(Nr, rank - r_min) * 0.1
                    V[:, r_min:] = rng0.randn(Mr, rank - r_min) * 0.1
                    W[:, r_min:] = rng0.randn(Pr, rank - r_min) * 0.1
            elif s % 4 == 1:
                # Structured init: subsample schoolbook
                rng0 = np.random.RandomState(1000 + rank + s)
                idx = rng0.choice(R0, size=min(rank, R0), replace=False)
                U = U0[:, idx].copy()
                V = V0[:, idx].copy()
                W = W0[:, idx].copy()
            elif s % 4 == 2:
                # Random init
                rng0 = np.random.RandomState(2000 + rank + s)
                U = rng0.randn(Nr, rank) * 0.1
                V = rng0.randn(Mr, rank) * 0.1
                W = rng0.randn(Pr, rank) * 0.1
            else:
                # Mixed init: half structured, half random
                rng0 = np.random.RandomState(3000 + rank + s)
                half = rank // 2
                idx = rng0.choice(R0, size=min(half, R0), replace=False)
                U = np.zeros((Nr, rank))
                V = np.zeros((Mr, rank))
                W = np.zeros((Pr, rank))
                U[:, :half] = U0[:, idx]
                V[:, :half] = V0[:, idx]
                W[:, :half] = W0[:, idx]
                U[:, half:] = rng0.randn(Nr, rank - half) * 0.1
                V[:, half:] = rng0.randn(Mr, rank - half) * 0.1
                W[:, half:] = rng0.randn(Pr, rank - half) * 0.1

            U, V, W = _als(U, V, W, rank, max_it=200, seed=s)
            e = _err(U, V, W)

            if e < 1e-8:
                # Polish to exact half-integer
                Ur, Vr, Wr = _polish(U, V, W, rank)
                if _check_full(Ur, Vr, Wr, tol=1e-9):
                    return Ur, Vr, Wr
                # Try integer rounding
                Ui = np.round(U)
                Vi = np.round(V)
                Wi = np.round(W)
                if _check_full(Ui, Vi, Wi, tol=1e-9):
                    return Ui, Vi, Wi
        return None

    # Search ranks below the structural rank
    if best_rank < R0:
        # Try to go below the structural rank
        candidate_ranks = [best_rank - 5, best_rank - 10, best_rank - 15,
                           best_rank - 20, best_rank - 30, best_rank - 40,
                           best_rank - 50]
        candidate_ranks = [rk for rk in candidate_ranks if rk >= 36]

        for rk in candidate_ranks:
            if time.time() > DL - 20:
                break
            res = _try_rank(rk, n_seeds=6)
            if res is not None:
                U, V, W = res
                best_rank = rk
                best = (U, V, W)
                break
    else:
        # No structural solution, try ALS from scratch
        candidate_ranks = [120, 100, 90, 80, 70, 60, 50]
        for rk in candidate_ranks:
            if time.time() > DL - 20:
                break
            res = _try_rank(rk, n_seeds=8)
            if res is not None:
                U, V, W = res
                best_rank = rk
                best = (U, V, W)
                break

    if best_rank < R0:
        Uo, Vo, Wo = best
        return {"U": Uo.tolist(), "V": Vo.tolist(), "W": Wo.tolist()}

    # Fallback: schoolbook
    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
# EVOLVE-BLOCK-END
