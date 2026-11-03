"""
Matrix multiplication 3x5x7 via block decomposition with Strassen for 2x2x2 core.
Rank: 104 (improvement over schoolbook 105).

Approach:
- Partition the middle dimension j into [0,3) and [3,5) (sizes 3 and 2).
- Partition the output dimension k into [0,3) and [3,7) (sizes 3 and 4).
- This gives four blocks:
  Block A: i in [0,3), j in [0,3), k in [0,3) -> 3x3x3 tensor (27 terms)
  Block B: i in [0,3), j in [0,3), k in [3,7) -> 3x3x4 tensor (36 terms)
  Block C: i in [0,3), j in [3,5), k in [0,3) -> 3x2x3 tensor (18 terms)
  Block D: i in [0,3), j in [3,5), k in [3,7) -> 3x2x4 tensor (24 terms)

- For Block A (3x3x3), we further partition:
  - Sub-block A1: i in [0,2), j in [0,2), k in [0,2) -> 2x2x2 tensor (8 terms)
    Use Strassen (rank 7) for this.
  - Sub-block A2: remaining terms in 3x3x3 (19 terms) -> schoolbook (rank 19)
  So Block A rank = 7 + 19 = 26.

- Blocks B, C, D use schoolbook: 36 + 18 + 24 = 78.

Total rank = 26 + 78 = 104.
"""
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=5, p=7
    # U shape: (n*m, rank) = (15, rank)
    # V shape: (m*p, rank) = (35, rank)
    # W shape: (p*n, rank) = (21, rank)
    
    # We'll build up the decomposition column by column.
    # Each column corresponds to a rank-1 term: u * v * w
    # where u is in R^(n*m), v in R^(m*p), w in R^(p*n)
    # and the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Start with empty lists for each column's U, V, W entries
    columns = []  # each entry is (u_dict, v_dict, w_dict) where dicts map index -> coefficient
    
    def add_column(u_entries, v_entries, w_entries):
        """Add a rank-1 term. u_entries, v_entries, w_entries are dicts {index: coeff}."""
        columns.append((u_entries, v_entries, w_entries))
    
    # --- Block A: i in [0,3), j in [0,3), k in [0,3) ---
    # Sub-block A1: i in [0,2), j in [0,2), k in [0,2) -> Strassen 2x2x2
    # Strassen for 2x2x2:
    # Let A = [[a00, a01], [a10, a11]] (2x2), B = [[b00, b01], [b10, b11]] (2x2)
    # C = A*B (2x2)
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b10-b11)
    # M4 = a11*(b01-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b10)
    # M7 = (a01-a11)*(b01+b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # In our tensor notation:
    # T[i,j,k] = A[i,j] * B[j,k]
    # For the 2x2x2 sub-block (i in {0,1}, j in {0,1}, k in {0,1}):
    # A indices: (i,j) with i in {0,1}, j in {0,1}
    #   a00 = A[0,0] -> U index 0*5+0 = 0
    #   a01 = A[0,1] -> U index 0*5+1 = 1
    #   a10 = A[1,0] -> U index 1*5+0 = 5
    #   a11 = A[1,1] -> U index 1*5+1 = 6
    # B indices: (j,k) with j in {0,1}, k in {0,1}
    #   b00 = B[0,0] -> V index 0*7+0 = 0
    #   b01 = B[0,1] -> V index 0*7+1 = 1
    #   b10 = B[1,0] -> V index 1*7+0 = 7
    #   b11 = B[1,1] -> V index 1*7+1 = 8
    # C indices: (k,i) with k in {0,1}, i in {0,1}
    #   C[0,0] = C00 -> W index 0*3+0 = 0
    #   C[0,1] = C01 -> W index 0*3+1 = 1
    #   C[1,0] = C10 -> W index 1*3+0 = 3
    #   C[1,1] = C11 -> W index 1*3+1 = 4
    
    # Strassen terms:
    # M1 = (a00+a11)*(b00+b11)
    add_column({0: 1, 6: 1}, {0: 1, 8: 1}, {0: 1, 1: 1, 3: 1, 4: -1})
    # Wait, I need to be more careful. Each M_r contributes to multiple C entries.
    # Let me re-derive.
    
    # Actually, let me just list each of the 7 Strassen products as a separate rank-1 term,
    # but each product like M1 = (a00+a11)*(b00+b11) is itself a product of two linear forms,
    # so it contributes to the tensor as:
    # For M1: the contribution to T[i,j,k] is (a00+a11)*(b00+b11) when it appears in C entries.
    # But M1 appears in C00 with coeff +1 and C11 with coeff +1.
    # So M1 contributes:
    #   +1 * (a00+a11)*(b00+b11) to C00 = T[0,0,0]... wait no.
    
    # Let me think differently. The tensor T[i,j,k] for the 2x2x2 block is:
    # T[i,j,k] = A[i,j]*B[j,k]
    # This is a trilinear form. Strassen decomposes the multiplication into 7 bilinear forms M1..M7,
    # each of which is a product of a linear form in A and a linear form in B.
    # Then C is a linear combination of M1..M7.
    
    # So the tensor T[i,j,k] = sum_r (coeff_r(i,k) * M_r)
    # where M_r = u_r(A) * v_r(B)
    
    # For the full tensor, we need:
    # T[i,j,k] = sum over all rank-1 terms of U[i*m+j] * V[j*p+k] * W[k*n+i]
    
    # For the Strassen part, each M_r gives a rank-1 contribution where:
    # U part: the linear form u_r in A (only non-zero for j in {0,1}, i in {0,1})
    # V part: the linear form v_r in B (only non-zero for j in {0,1}, k in {0,1})
    # W part: the linear form in C, which depends on (i,k)
    
    # Let me define the 7 Strassen terms properly.
    
    # Reset and redo.
    columns = []
    
    def add_column(u_entries, v_entries, w_entries):
        columns.append((u_entries, v_entries, w_entries))
    
    # Strassen 2x2x2 for i in {0,1}, j in {0,1}, k in {0,1}
    # U indices: i*m+j = i*5+j for i in {0,1}, j in {0,1}
    #   (0,0)->0, (0,1)->1, (1,0)->5, (1,1)->6
    # V indices: j*p+k = j*7+k for j in {0,1}, k in {0,1}
    #   (0,0)->0, (0,1)->1, (1,0)->7, (1,1)->8
    # W indices: k*n+i = k*3+i for k in {0,1}, i in {0,1}
    #   (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4
    
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # Contributes to C[0,0] with +1 and C[1,1] with +1
    add_column({0: 1, 6: 1}, {0: 1, 8: 1}, {0: 1, 4: 1})
    
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # Contributes to C[1,0] with +1 and C[1,1] with -1
    add_column({5: 1, 6: 1}, {0: 1}, {3: 1, 4: -1})
    
    # M3 = A[0,0] * (B[1,0]-B[1,1])
    # Contributes to C[0,1] with +1 and C[1,1] with +1
    add_column({0: 1}, {7: 1, 8: -1}, {1: 1, 4: 1})
    
    # M4 = A[1,1] * (B[0,1]-B[0,0])
    # Contributes to C[0,0] with +1 and C[1,0] with +1
    add_column({6: 1}, {1: 1, 0: -1}, {0: 1, 3: 1})
    
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    # Contributes to C[0,1] with +1 and C[0,0] with -1
    add_column({0: 1, 1: 1}, {8: 1}, {1: 1, 0: -1})
    
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[1,0])
    # Contributes to C[1,1] with +1
    add_column({5: 1, 0: -1}, {0: 1, 7: 1}, {4: 1})
    
    # M7 = (A[0,1]-A[1,1]) * (B[0,1]+B[1,1])
    # Contributes to C[0,0] with +1
    add_column({1: 1, 6: -1}, {1: 1, 8: 1}, {0: 1})
    
    # Now the remaining 19 terms in the 3x3x3 block (i in [0,3), j in [0,3), k in [0,3))
    # These are all (i,j,k) not in {0,1}x{0,1}x{0,1}
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i < 2 and j < 2 and k < 2:
                    continue  # already handled by Strassen
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                add_column({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    # Block B: i in [0,3), j in [0,3), k in [3,7) -> schoolbook
    for i in range(3):
        for j in range(3):
            for k in range(3, 7):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                add_column({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    # Block C: i in [0,3), j in [3,5), k in [0,3) -> schoolbook
    for i in range(3):
        for j in range(3, 5):
            for k in range(3):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                add_column({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    # Block D: i in [0,3), j in [3,5), k in [3,7) -> schoolbook
    for i in range(3):
        for j in range(3, 5):
            for k in range(3, 7):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                add_column({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    rank = len(columns)
    
    # Build U, V, W matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r, (u_entries, v_entries, w_entries) in enumerate(columns):
        for idx, coeff in u_entries.items():
            U[idx, r] = coeff
        for idx, coeff in v_entries.items():
            V[idx, r] = coeff
        for idx, coeff in w_entries.items():
            W[idx, r] = coeff
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
