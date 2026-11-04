"""
Rank-minimal tensor decomposition for 3x4 x 4x8 matrix multiplication.

Strategy:
  - Schoolbook gives rank 96.
  - Use Strassen's 2x2x2 decomposition (rank 7 instead of 8) on 2x2x2 sub-problems.
  - The 3x4x8 tensor can be decomposed by splitting:
    * i into {0,1} (2 rows) and {2} (1 row)
    * j into two groups of 2
    * k into four groups of 2
  - For the 2x4x8 part (i in {0,1}), we can identify 2x2x2 sub-problems
    by grouping (i, j-group, k-group) appropriately.
  
  Specifically, for i in {0,1}, j in {0,1,2,3}, k in {0,...,7}:
  - Split j into {0,1} and {2,3}
  - Split k into {0,1}, {2,3}, {4,5}, {6,7}
  - For each combination of (j-group, k-group), we have a 2x2x2 sub-problem
    (i in {0,1}, j in 2-element group, k in 2-element group)
  - Each 2x2x2 sub-problem uses Strassen (rank 7 instead of 8)
  - Number of such sub-problems: 2 (j-groups) * 4 (k-groups) = 8
  - Total for 2x4x8 part: 8 * 7 = 56 terms (vs schoolbook 2*4*8=64)
  
  For the 1x4x8 part (i=2): no Strassen savings, just 4*8=32 terms.
  
  Total: 56 + 32 = 88 terms (vs schoolbook 96).

  We verify the decomposition and fall back to schoolbook if needed.
"""

import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 12, 32, 24

    # Construct the matrix multiplication tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1

    def validate(U, V, W):
        """Check if U, V, W correctly decompose T."""
        rank = U.shape[1]
        T_recon = np.zeros((nm, mp, pn), dtype=np.int64)
        for r in range(rank):
            T_recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        return np.array_equal(T_recon, T)

    # Schoolbook baseline
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school), dtype=np.int64)
    V_school = np.zeros((mp, rank_school), dtype=np.int64)
    W_school = np.zeros((pn, rank_school), dtype=np.int64)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1

    # Try Strassen-based decomposition
    # For 2x2x2 Strassen:
    #   m1 = (A11 - A22)(B21 + B22)
    #   m2 = (A11 + A12)B22
    #   m3 = A11(B11 - B21)
    #   m4 = A22(B12 - B11)
    #   m5 = (A11 + A22)(B11 + B22)
    #   m6 = (A12 - A22)(B11 + B21)
    #   m7 = (A11 - A12)(B12 + B22)
    #   C11 = m3 + m5 - m7 - m4
    #   C12 = m1 + m5
    #   C21 = m2 + m4
    #   C22 = m1 - m3 + m6 + m5

    # In tensor form, for 2x2x2 with indices (i,j,k) in {0,1}^3:
    # The 8 schoolbook terms are:
    #   e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i} for each (i,j,k)
    # Strassen gives 7 terms with linear combinations.

    # Let's define the Strassen terms explicitly for 2x2x2:
    # Using 0-indexed: i,j,k in {0,1}
    # A is 2x2: A[0,0], A[0,1], A[1,0], A[1,1]
    # B is 2x2: B[0,0], B[0,1], B[1,0], B[1,1]
    # C is 2x2: C[0,0], C[0,1], C[1,0], C[1,1]
    #
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    #
    # Strassen:
    # m1 = (A[0,0] - A[1,1]) * (B[1,0] + B[1,1])
    # m2 = (A[0,0] + A[0,1]) * B[1,1]
    # m3 = A[0,0] * (B[0,0] - B[1,0])
    # m4 = A[1,1] * (B[0,1] - B[0,0])
    # m5 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # m6 = (A[0,1] - A[1,1]) * (B[0,0] + B[1,0])
    # m7 = (A[0,0] - A[0,1]) * (B[0,1] + B[1,1])
    #
    # C[0,0] = m3 + m5 - m7 - m4
    # C[0,1] = m1 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m3 + m6 + m5

    # In tensor decomposition form, each Strassen term is a rank-1 tensor.
    # For m1: (A[0,0] - A[1,1]) * (B[1,0] + B[1,1])
    #   U component: e_{0,0} - e_{1,1}  (i=0,j=0 and i=1,j=1)
    #   V component: e_{1,0} + e_{1,1}  (j=1,k=0 and j=1,k=1)
    #   W component: e_{0,0} + e_{1,1}  (k=0,i=0 and k=1,i=1)
    #   Wait, this isn't quite right. Let me think more carefully.

    # Actually, for the tensor T_{i,j,k} = e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}:
    # The Strassen decomposition expresses the 2x2x2 tensor as a sum of 7 rank-1 tensors.
    # Each rank-1 tensor corresponds to a "multiplication" term.

    # Let me use a different approach: directly construct the Strassen terms
    # for the 2x2x2 sub-problem and embed them into the full tensor.

    # For a 2x2x2 sub-problem with global indices:
    #   i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
    # The 8 schoolbook terms are:
    #   (i0,j0,k0), (i0,j0,k0+1), (i0,j0+1,k0), (i0,j0+1,k0+1)
    #   (i0+1,j0,k0), (i0+1,j0,k0+1), (i0+1,j0+1,k0), (i0+1,j0+1,k0+1)
    #
    # Strassen replaces these with 7 terms. Let me define them explicitly.

    def strassen_terms_2x2x2(i0, j0, k0):
        """
        Return 7 rank-1 terms (as (U, V, W) sparse dicts) for the 2x2x2
        sub-problem with i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}.
        
        Each term is represented as:
          U: dict mapping (i, j) -> coefficient
          V: dict mapping (j, k) -> coefficient  
          W: dict mapping (k, i) -> coefficient
        """
        # Local indices: il, jl, kl in {0, 1}
        # Global: i = i0+il, j = j0+jl, k = k0+kl
        
        # m1 = (A[0,0] - A[1,1]) * (B[1,0] + B[1,1])
        #   U: {(0,0): 1, (1,1): -1}
        #   V: {(1,0): 1, (1,1): 1}
        #   W: {(0,0): 1, (1,1): 1}
        m1 = ({(0,0): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1})
        
        # m2 = (A[0,0] + A[0,1]) * B[1,1]
        #   U: {(0,0): 1, (0,1): 1}
        #   V: {(1,1): 1}
        #   W: {(1,1): 1}
        m2 = ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(1,1): 1})
        
        # m3 = A[0,0] * (B[0,0] - B[1,0])
        #   U: {(0,0): 1}
        #   V: {(0,0): 1, (1,0): -1}
        #   W: {(0,0): 1, (0,1): 1}
        m3 = ({(0,0): 1}, {(0,0): 1, (1,0): -1}, {(0,0): 1, (0,1): 1})
        
        # m4 = A[1,1] * (B[0,1] - B[0,0])
        #   U: {(1,1): 1}
        #   V: {(0,1): 1, (0,0): -1}
        #   W: {(1,0): 1, (1,1): 1}
        m4 = ({(1,1): 1}, {(0,1): 1, (0,0): -1}, {(1,0): 1, (1,1): 1})
        
        # m5 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
        #   U: {(0,0): 1, (1,1): 1}
        #   V: {(0,0): 1, (1,1): 1}
        #   W: {(0,0): 1, (1,1): 1}
        m5 = ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1})
        
        # m6 = (A[0,1] - A[1,1]) * (B[0,0] + B[1,0])
        #   U: {(0,1): 1, (1,1): -1}
        #   V: {(0,0): 1, (1,0): 1}
        #   W: {(0,0): 1, (1,0): 1}
        m6 = ({(0,1): 1, (1,1): -1}, {(0,0): 1, (1,0): 1}, {(0,0): 1, (1,0): 1})
        
        # m7 = (A[0,0] - A[0,1]) * (B[0,1] + B[1,1])
        #   U: {(0,0): 1, (0,1): -1}
        #   V: {(0,1): 1, (1,1): 1}
        #   W: {(0,0): 1, (1,1): 1}
        m7 = ({(0,0): 1, (0,1): -1}, {(0,1): 1, (1,1): 1}, {(0,0): 1, (1,1): 1})
        
        return [m1, m2, m3, m4, m5, m6, m7]

    # Build the Strassen-based decomposition
    U_full = np.zeros((nm, 200), dtype=np.int64)  # Over-allocate, trim later
    V_full = np.zeros((mp, 200), dtype=np.int64)
    W_full = np.zeros((pn, 200), dtype=np.int64)
    col = 0

    # Process i in {0, 1} with Strassen
    for i0 in [0]:  # Only i0=0 for the 2-row part (i in {0,1})
        for j0 in [0, 2]:  # j groups: {0,1} and {2,3}
            for k0 in [0, 2, 4, 6]:  # k groups: {0,1}, {2,3}, {4,5}, {6,7}
                terms = strassen_terms_2x2x2(i0, j0, k0)
                for u_dict, v_dict, w_dict in terms:
                    for (il, jl), u_val in u_dict.items():
                        i_global = i0 + il
                        j_global = j0 + jl
                        u_global = i_global * m + j_global
                        U_full[u_global, col] += u_val
                    for (jl, kl), v_val in v_dict.items():
                        j_global = j0 + jl
                        k_global = k0 + kl
                        v_global = j_global * p + k_global
                        V_full[v_global, col] += v_val
                    for (kl, il), w_val in w_dict.items():
                        k_global = k0 + kl
                        i_global = i0 + il
                        w_global = k_global * n + i_global
                        W_full[w_global, col] += w_val
                    col += 1

    # Process i=2 with schoolbook
    for j in range(m):
        for k in range(p):
            U_full[2*m+j, col] = 1
            V_full[j*p+k, col] = 1
            W_full[k*n+2, col] = 1
            col += 1

    # Trim and verify
    U_trim = U_full[:, :col]
    V_trim = V_full[:, :col]
    W_trim = W_full[:, :col]

    if validate(U_trim, V_trim, W_trim):
        return {
            "U": U_trim.tolist(),
            "V": V_trim.tolist(),
            "W": W_trim.tolist()
        }
    else:
        # Fallback to schoolbook
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
