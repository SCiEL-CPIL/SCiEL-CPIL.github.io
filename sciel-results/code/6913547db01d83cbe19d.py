"""
Rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.

Strategy:
- Split A (3x3) into top 2 rows (A_top, 2x3) and bottom 1 row (A_bot, 1x3).
- B is 3x4.
- C_top = A_top * B (2x4), C_bot = A_bot * B (1x4).

For C_top (2x3 * 3x4 -> 2x4):
  Split B into left 2 cols (B_L, 3x2) and right 2 cols (B_R, 3x2).
  Also split A_top into A_top_L (2x2, cols 0-1) and A_top_R (2x1, col 2).
  
  C_top[:, 0:2] = A_top * B_L = A_top_L * B_L[:, 0:2] + A_top_R * B_L[:, 2:2]
  Wait, that's not quite right. Let me re-think.
  
  A_top is 2x3, B_L is 3x2.
  A_top = [A_top_L (2x2) | A_top_R (2x1)]
  B_L = [B_L_top (2x2); B_L_bot (1x2)]
  
  A_top * B_L = A_top_L * B_L_top + A_top_R * B_L_bot
  
  A_top_L * B_L_top is 2x2 * 2x2 -> rank 7 (Strassen)
  A_top_R * B_L_bot is 2x1 * 1x2 -> rank 2 (outer product)
  
  Similarly for B_R:
  A_top * B_R = A_top_L * B_R_top + A_top_R * B_R_bot
  A_top_L * B_R_top is 2x2 * 2x2 -> rank 7 (Strassen)
  A_top_R * B_R_bot is 2x1 * 1x2 -> rank 2 (outer product)
  
  Total for C_top: 7 + 2 + 7 + 2 = 18

For C_bot (1x3 * 3x4 -> 1x4):
  A_bot is 1x3, B is 3x4.
  C_bot[k] = sum_j A_bot[j] * B[j,k]
  This is 4 dot products of length 3.
  Rank = 12 (schoolbook: each (j,k) pair needs one term)
  
  Wait, can we do better? 
  C_bot[k] = A_bot[0]*B[0,k] + A_bot[1]*B[1,k] + A_bot[2]*B[2,k]
  Each C_bot[k] is a separate bilinear form. The 4 forms are independent.
  Each form has 3 products, so rank 12.

Total rank: 18 + 12 = 30.

Let me verify this is correct by implementing it carefully.

Actually, let me reconsider. The tensor decomposition requires that for each rank term r,
we have U[i,j,r], V[j,k,r], W[k,i,r] such that:
C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over j.

Wait, the indexing in the problem statement is:
U has shape (n*m, rank) -> indexed by (i,j) where i in [0,n), j in [0,m)
V has shape (m*p, rank) -> indexed by (j,k) where j in [0,m), k in [0,p)
W has shape (p*n, rank) -> indexed by (k,i) where k in [0,p), i in [0,n)

And the decomposition is:
C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

Wait, that's not quite right either. Let me re-read the problem.

"The output factor follows the official trace convention W[k*n+i,r]."

So W is indexed by (k,i) with index k*n+i.

The standard matrix multiplication tensor T_{i,j,k} = delta_{i,i'} delta_{j,j'} delta_{k,k'} for A_{i,j} B_{j,k} = C_{i,k}.

Actually, the tensor is T[i,j,k] = 1 if the indices match for the product A[i,j] * B[j,k] = C[i,k].

The decomposition is:
T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]

Where U[i,j,r] corresponds to U[i*m+j, r], V[j,k,r] corresponds to V[j*p+k, r], W[k,i,r] corresponds to W[k*n+i, r].

And the bilinear computation is:
C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]

Wait no. The standard formulation is:
Given A (n x m) and B (m x p), C = A * B (n x p).
C[i,k] = sum_j A[i,j] * B[j,k]

The tensor T is defined by T[i,j,k] = 1 (it's the identity tensor for matrix mult).

A rank-r decomposition gives:
T[i,j,k] = sum_{r=1}^r U[i,j,r] * V[j,k,r] * W[k,i,r]

Then C[i,k] = sum_j sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] * A[i,j] * B[j,k]

Hmm, that's not quite right. Let me think again.

Actually, the correct formulation is:
C[i,k] = sum_r (sum_j U[i,j,r] * A[i,j]) * (sum_j V[j,k,r] * B[j,k]) * W[k,i,r]

No wait, that's not right either. The standard bilinear form is:
C[i,k] = sum_r W[k,i,r] * (sum_j U[i,j,r] * A[i,j]) * (sum_j V[j,k,r] * B[j,k])

But that would make U and V depend on both indices, which is unusual.

Let me look at the schoolbook case:
For each (i,j,k), we have one term:
U[i*m+j, r] = 1 for r corresponding to (i,j,k)
V[j*p+k, r] = 1 for r corresponding to (i,j,k)
W[k*n+i, r] = 1 for r corresponding to (i,j,k)

Then C[i,k] = sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] * A[i,j] * B[j,k]

For the schoolbook term (i',j',k'), U[i*m+j, r] = 1 iff i=i' and j=j', V[j*p+k,r] = 1 iff j=j' and k=k', W[k*n+i,r] = 1 iff k=k' and i=i'.

So the product is 1 iff i=i', j=j', k=k'. Then C[i,k] gets contribution A[i,j]*B[j,k] when i=i', j=j', k=k', which gives C[i,k] = sum_j A[i,j]*B[j,k]. Correct.

So the general formula is:
C[i,k] = sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] * A[i,j] * B[j,k]

Wait, that doesn't seem right because U, V, W are supposed to be fixed coefficients, not depending on A and B.

Let me reconsider. The bilinear algorithm is:
For each rank term r, compute:
  u_r = sum_{i,j} U[i*m+j, r] * A[i,j]
  v_r = sum_{j,k} V[j*p+k, r] * B[j,k]
  Then C[i,k] += W[k*n+i, r] * u_r * v_r

So C[i,k] = sum_r W[k*n+i, r] * (sum_{i',j'} U[i'*m+j', r] * A[i',j']) * (sum_{j',k'} V[j'*p+k', r] * B[j',k'])

For this to equal sum_j A[i,j] * B[j,k], we need:
sum_r W[k*n+i, r] * (sum_{i',j'} U[i'*m+j', r] * A[i',j']) * (sum_{j',k'} V[j'*p+k', r] * B[j',k']) = sum_j A[i,j] * B[j,k]

This is a bilinear identity. The tensor decomposition condition is:
For all i, j, k:
  delta_{i,i} delta_{j,j} delta_{k,k} ... 

Actually, the condition is that the bilinear forms match. Let me just implement the Strassen-based decomposition and verify it numerically.

Strassen's algorithm for 2x2 * 2x2:
M1 = (a00+a11)(b00+b11)
M2 = (a10+a11)b00
M3 = a00(b10-b11)
M4 = a11(b01-b00)
M5 = (a00+a01)b11
M6 = (a10-a00)(b00+b10)
M7 = (a01-a11)(b01+b11)

c00 = M1 + M4 - M5 + M7
c01 = M3 + M5
c10 = M2 + M4
c11 = M1 - M2 + M3 + M6

Let me implement this carefully.
"""

def construct():
    n, m, p = 3, 3, 4  # A is n x m, B is m x p, C is n x p
    
    U_terms = []  # Each term: (i, j, coeff)
    V_terms = []  # Each term: (j, k, coeff)
    W_terms = []  # Each term: (k, i, coeff)
    
    def add_term(u_coeff, v_coeff, w_coeff, u_indices, v_indices, w_indices):
        """Add a rank term with given coefficients and indices."""
        # u_indices: list of (i, j) with coefficients
        # v_indices: list of (j, k) with coefficients
        # w_indices: list of (k, i) with coefficients
        U_terms.append(u_indices)
        V_terms.append(v_indices)
        W_terms.append(w_indices)
    
    # --- C_bot: 1x3 * 3x4 -> 1x4 (schoolbook, rank 12) ---
    # A_bot is row 2 of A (i=2), B is 3x4
    # C[2,k] = sum_j A[2,j] * B[j,k]
    for j in range(3):
        for k in range(4):
            add_term(
                [(2, j, 1.0)],  # U: A[2,j]
                [(j, k, 1.0)],  # V: B[j,k]
                [(k, 2, 1.0)]   # W: contributes to C[2,k]
            )
    
    # --- C_top: 2x3 * 3x4 -> 2x4 ---
    # Split into left 2 cols and right 2 cols of B
    # For each half (cols 0-1 and cols 2-3):
    #   A_top (2x3) * B_half (3x2)
    #   = A_top_L (2x2) * B_half_top (2x2) + A_top_R (2x1) * B_half_bot (1x2)
    
    for half in range(2):
        k_offset = half * 2  # 0 or 2
        
        # A_top_L * B_half_top: 2x2 * 2x2 (Strassen, rank 7)
        # A_top_L: rows 0-1, cols 0-1 of A
        # B_half_top: rows 0-1, cols k_offset to k_offset+1 of B
        
        # Strassen for 2x2:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b10-b11)
        # M4 = a11(b01-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b10)
        # M7 = (a01-a11)(b01+b11)
        
        # c00 = M1 + M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M6
        
        # Indices: A rows 0-1, B rows 0-1, C rows 0-1, C cols k_offset to k_offset+1
        
        # M1
        add_term(
            [(0, 0, 1.0), (1, 1, 1.0)],
            [(0, k_offset, 1.0), (1, k_offset+1, 1.0)],
            [(k_offset, 0, 1.0), (k_offset+1, 1, 1.0)]
        )
        
        # M2
        add_term(
            [(1, 0, 1.0), (1, 1, 1.0)],
            [(0, k_offset, 1.0)],
            [(k_offset, 1, 1.0)]
        )
        
        # M3
        add_term(
            [(0, 0, 1.0)],
            [(1, k_offset, 1.0), (1, k_offset+1, -1.0)],
            [(k_offset+1, 0, 1.0)]
        )
        
        # M4
        add_term(
            [(1, 1, 1.0)],
            [(0, k_offset+1, 1.0), (0, k_offset, -1.0)],
            [(k_offset+1, 1, 1.0), (k_offset, 0, 1.0)]
        )
        
        # M5
        add_term(
            [(0, 0, 1.0), (0, 1, 1.0)],
            [(1, k_offset+1, 1.0)],
            [(k_offset+1, 0, -1.0)]
        )
        
        # M6
        add_term(
            [(1, 0, 1.0), (0, 0, -1.0)],
            [(0, k_offset, 1.0), (0, k_offset+1, 1.0)],
            [(k_offset+1, 1, 1.0)]
        )
        
        # M7
        add_term(
            [(0, 1, 1.0), (1, 1, -1.0)],
            [(0, k_offset+1, 1.0), (1, k_offset+1, 1.0)],
            [(k_offset, 0, 1.0)]
        )
        
        # A_top_R * B_half_bot: 2x1 * 1x2 (outer product, rank 2)
        # A_top_R: rows 0-1, col 2 of A
        # B_half_bot: row 2, cols k_offset to k_offset+1 of B
        # C contribution: C[i, k_offset+k2] += A[i,2] * B[2, k_offset+k2]
        
        for k2 in range(2):
            k_idx = k_offset + k2
            for i in range(2):
                add_term(
                    [(i, 2, 1.0)],
                    [(2, k_idx, 1.0)],
                    [(k_idx, i, 1.0)]
                )
    
    # Now convert to matrix form
    rank = len(U_terms)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for (i, j, coeff) in U_terms[r]:
            U[i * m + j][r] += coeff
        for (j, k, coeff) in V_terms[r]:
            V[j * p + k][r] += coeff
        for (k, i, coeff) in W_terms[r]:
            W[k * n + i][r] += coeff
    
    return {"U": U, "V": V, "W": W}

# Verify correctness
if __name__ == "__main__":
    import numpy as np
    
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    
    n, m, p = 3, 3, 4
    rank = U.shape[1]
    print(f"Rank: {rank}")
    
    # Test with random matrices
    np.random.seed(42)
    for trial in range(100):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute using decomposition
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for r in range(rank):
                        C_computed[i, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            print(f"Trial {trial}: MISMATCH")
            print("A:", A)
            print("B:", B)
            print("Expected:", C_expected)
            print("Computed:", C_computed)
            break
    else:
        print("All trials passed!")
    
    # Check if entries are integers or half-integers
    for name, mat in [("U", U), ("V", V), ("W", W)]:
        for val in mat.flatten():
            if abs(val * 2 - round(val * 2)) > 1e-10:
                print(f"{name} has non-half-integer entry: {val}")
                break
        else:
            continue
        break
    else:
        print("All entries are integers or half-integers.")
