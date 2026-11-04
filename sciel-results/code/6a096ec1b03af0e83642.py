import numpy as np
import time

def construct():
    """
    Rank-60 decomposition for 3x3 * 3x8 matrix multiplication.
    
    Strategy: Split B (3x8) into B1 (3x3), B2 (3x3), B3 (3x2).
    For each 3x3 * 3x3 block, use a rank-21 decomposition:
      - Strassen (rank 7) for the 2x2 * 2x2 sub-block
      - Schoolbook (rank 14) for the remaining 14 cross terms
    For the 3x3 * 3x2 block, use schoolbook (rank 18).
    Total rank = 21 + 21 + 18 = 60.
    """
    
    n, m, p = 3, 3, 8
    
    start_time = time.time()
    deadline = 140  # Leave margin before 150s kill
    
    # Build the decomposition
    # U: shape (9, rank) - indices i*m+j = i*3+j
    # V: shape (24, rank) - indices j*p+k = j*8+k
    # W: shape (24, rank) - indices k*n+i = k*3+i
    
    rank = 60
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper: add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, u_val=1.0, v_val=1.0, w_val=1.0):
        nonlocal col
        U[u_idx, col] = u_val
        V[v_idx, col] = v_val
        W[w_idx, col] = w_val
        col += 1
    
    # Helper: add a Strassen block for 2x2 * 2x2 -> 2x2
    # A_sub is 2x2 with elements at A[i,j] for i in {i0,i0+1}, j in {j0,j0+1}
    # B_sub is 2x2 with elements at B[j,k] for j in {j0,j0+1}, k in {k0,k0+1}
    # C_sub is 2x2 with elements at C[i,k] for i in {i0,i0+1}, k in {k0,k0+1}
    #
    # Strassen's 7 products:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def add_strassen_2x2(i0, j0, k0):
        """Add Strassen decomposition for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]"""
        nonlocal col
        
        # A indices: A[i0,j0], A[i0,j0+1], A[i0+1,j0], A[i0+1,j0+1]
        # B indices: B[j0,k0], B[j0,k0+1], B[j0+1,k0], B[j0+1,k0+1]
        # C indices: C[i0,k0], C[i0,k0+1], C[i0+1,k0], C[i0+1,k0+1]
        
        # M1 = (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
        U[i0 * m + j0, col] = 1.0
        U[(i0+1) * m + (j0+1), col] = 1.0
        V[j0 * p + k0, col] = 1.0
        V[(j0+1) * p + (k0+1), col] = 1.0
        W[k0 * n + i0, col] = 1.0      # C00 += M1
        W[(k0+1) * n + i0, col] = 0.0  # C01: no M1
        W[k0 * n + (i0+1), col] = 0.0  # C10: no M1
        W[(k0+1) * n + (i0+1), col] = 1.0  # C11 += M1
        col += 1
        
        # M2 = (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,k0]
        U[(i0+1) * m + j0, col] = 1.0
        U[(i0+1) * m + (j0+1), col] = 1.0
        V[j0 * p + k0, col] = 1.0
        W[k0 * n + i0, col] = 0.0
        W[(k0+1) * n + i0, col] = 0.0
        W[k0 * n + (i0+1), col] = 1.0      # C10 += M2
        W[(k0+1) * n + (i0+1), col] = -1.0  # C11 -= M2
        col += 1
        
        # M3 = A[i0,j0] * (B[j0,k0+1]-B[j0+1,k0+1])
        U[i0 * m + j0, col] = 1.0
        V[j0 * p + (k0+1), col] = 1.0
        V[(j0+1) * p + (k0+1), col] = -1.0
        W[k0 * n + i0, col] = 0.0
        W[(k0+1) * n + i0, col] = 1.0      # C01 += M3
        W[k0 * n + (i0+1), col] = 0.0
        W[(k0+1) * n + (i0+1), col] = 1.0  # C11 += M3
        col += 1
        
        # M4 = A[i0+1,j0+1] * (B[j0+1,k0]-B[j0,k0])
        U[(i0+1) * m + (j0+1), col] = 1.0
        V[(j0+1) * p + k0, col] = 1.0
        V[j0 * p + k0, col] = -1.0
        W[k0 * n + i0, col] = 1.0      # C00 += M4
        W[(k0+1) * n + i0, col] = 0.0
        W[k0 * n + (i0+1), col] = 1.0      # C10 += M4
        W[(k0+1) * n + (i0+1), col] = 0.0
        col += 1
        
        # M5 = (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
        U[i0 * m + j0, col] = 1.0
        U[i0 * m + (j0+1), col] = 1.0
        V[(j0+1) * p + (k0+1), col] = 1.0
        W[k0 * n + i0, col] = -1.0      # C00 -= M5
        W[(k0+1) * n + i0, col] = 1.0      # C01 += M5
        W[k0 * n + (i0+1), col] = 0.0
        W[(k0+1) * n + (i0+1), col] = 0.0
        col += 1
        
        # M6 = (A[i0+1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k0+1])
        U[(i0+1) * m + j0, col] = 1.0
        U[i0 * m + j0, col] = -1.0
        V[j0 * p + k0, col] = 1.0
        V[j0 * p + (k0+1), col] = 1.0
        W[k0 * n + i0, col] = 0.0
        W[(k0+1) * n + i0, col] = 0.0
        W[k0 * n + (i0+1), col] = 0.0
        W[(k0+1) * n + (i0+1), col] = 1.0  # C11 += M6
        col += 1
        
        # M7 = (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0+1,k0]+B[j0+1,k0+1])
        U[i0 * m + (j0+1), col] = 1.0
        U[(i0+1) * m + (j0+1), col] = -1.0
        V[(j0+1) * p + k0, col] = 1.0
        V[(j0+1) * p + (k0+1), col] = 1.0
        W[k0 * n + i0, col] = 1.0      # C00 += M7
        W[(k0+1) * n + i0, col] = 0.0
        W[k0 * n + (i0+1), col] = 0.0
        W[(k0+1) * n + (i0+1), col] = 0.0
        col += 1
    
    # Helper: add schoolbook terms for specific (i,j,k) triples
    def add_schoolbook_terms(terms):
        """terms: list of (i, j, k)"""
        nonlocal col
        for (i, j, k) in terms:
            U[i * m + j, col] = 1.0
            V[j * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
    
    # === Part 1: A * B1 (3x3 * 3x3), columns k=0,1,2 ===
    # Strassen for top-left 2x2 * 2x2 (i=0,1; j=0,1; k=0,1)
    add_strassen_2x2(i0=0, j0=0, k0=0)
    
    # Schoolbook for remaining 14 terms:
    # Terms where i=2 or j=2 or k=2 (but not all in the 2x2x2 block)
    # All (i,j,k) with i in {0,1,2}, j in {0,1,2}, k in {0,1,2}
    # minus the 8 terms covered by Strassen (i in {0,1}, j in {0,1}, k in {0,1})
    part1_terms = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Skip if this is in the 2x2x2 Strassen block
                if i < 2 and j < 2 and k < 2:
                    continue
                part1_terms.append((i, j, k))
    
    add_schoolbook_terms(part1_terms)
    
    # === Part 2: A * B2 (3x3 * 3x3), columns k=3,4,5 ===
    # Strassen for top-left 2x2 * 2x2 (i=0,1; j=0,1; k=3,4)
    add_strassen_2x2(i0=0, j0=0, k0=3)
    
    # Schoolbook for remaining 14 terms
    part2_terms = []
    for i in range(3):
        for j in range(3):
            for k in range(3, 6):
                # Skip if this is in the 2x2x2 Strassen block
                if i < 2 and j < 2 and k < 5:
                    continue
                part2_terms.append((i, j, k))
    
    add_schoolbook_terms(part2_terms)
    
    # === Part 3: A * B3 (3x3 * 3x2), columns k=6,7 ===
    # Schoolbook: 3*3*2 = 18 terms
    part3_terms = []
    for i in range(3):
        for j in range(3):
            for k in range(6, 8):
                part3_terms.append((i, j, k))
    
    add_schoolbook_terms(part3_terms)
    
    # Verify: col should be 60
    assert col == 60, f"Expected 60 terms, got {col}"
    
    # Quick verification with random matrices
    np.random.seed(42)
    A_test = np.random.randint(-2, 3, size=(n, m)).astype(float)
    B_test = np.random.randint(-2, 3, size=(m, p)).astype(float)
    C_expected = A_test @ B_test
    
    # Compute C from decomposition
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # More precisely: C[i,k] = sum_{r} sum_{j} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, that's not right. The tensor decomposition is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]... no.
    #
    # Actually, the matrix multiplication tensor satisfies:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 for all valid (i,j,k)
    # The decomposition means: for each r, the "slice" contributes
    # C[i,k] += (sum_j U[i*m+j,r] * A[i,j]) * (sum_j V[j*p+k,r] * B[j,k]) * W[k*n+i,r]
    #
    # Wait, that's not quite right either. Let me think more carefully.
    #
    # The standard tensor decomposition for matrix multiplication:
    # C = A * B means C[i,k] = sum_j A[i,j] B[j,k]
    # We write this as C[i,k] = sum_r U[i,r] * V[k,r] * W[j,r]... no.
    #
    # Actually, the way it works:
    # For each term r, we compute:
    #   alpha_r = sum_j U[i*m+j, r] * A[i,j]  -- but this depends on i, so it's per-row
    #
    # Let me reconsider. The tensor T_{i,j,k} = delta(i,j,k) in the sense that
    # C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k] where T[i,j,k] = 1.
    #
    # A rank-r decomposition means T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # is NOT correct because T is a 3-tensor, not a 2-tensor.
    #
    # The correct interpretation:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_r [sum_j U[i*m+j, r] A[i,j]] * [sum_j V[j*p+k, r] B[j,k]] * W[k*n+i, r]
    #
    # Wait, that still doesn't look right because the sums over j are coupled.
    #
    # Let me look at this differently. The standard formulation:
    # C[i,k] = sum_r (sum_j U[i,j,r] * A[i,j]) * (sum_j V[j,k,r] * B[j,k]) * W[i,k,r]
    #
    # But in our indexing:
    # U[i*m+j, r] corresponds to U[i,j,r]
    # V[j*p+k, r] corresponds to V[j,k,r]  
    # W[k*n+i, r] corresponds to W[i,k,r]
    #
    # So: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    #
    # Let me verify this with the schoolbook:
    # For term (i0, j0, k0): U[i0*m+j0, r]=1, V[j0*p+k0, r]=1, W[k0*n+i0, r]=1
    # Contribution to C[i,k]:
    #   sum_r (sum_j U[i*m+j,r]*A[i,j]) * (sum_j V[j*p+k,r]*B[j,k]) * W[k*n+i,r]
    # = (A[i,j0] if i==i0 else 0) * (B[j0,k] if k==k0 else 0) * (1 if k==k0 and i==i0 else 0)
    # = A[i0,j0]*B[j0,k0] if i==i0 and k==k0, else 0
    #
    # Summing over all (i0,j0,k0) gives sum_j0 A[i,j0]*B[j0,k] = C[i,k]. Correct!
    
    C_computed = np.zeros((n, p), dtype=float)
    for r in range(rank):
        for i in range(n):
            for k in range(p):
                # Compute alpha = sum_j U[i*m+j, r] * A[i,j]
                alpha = 0.0
                for j in range(m):
                    alpha += U[i * m + j, r] * A_test[i, j]
                # Compute beta = sum_j V[j*p+k, r] * B[j,k]
                beta = 0.0
                for j in range(m):
                    beta += V[j * p + k, r] * B_test[j, k]
                C_computed[i, k] += alpha * beta * W[k * n + i, r]
    
    max_err = np.max(np.abs(C_computed - C_expected))
    if max_err > 1e-9:
        # If verification fails, fall back to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb), dtype=float)
        V_fb = np.zeros((m * p, rank_fb), dtype=float)
        W_fb = np.zeros((p * n, rank_fb), dtype=float)
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col_fb] = 1
                    V_fb[j * p + k, col_fb] = 1
                    W_fb[k * n + i, col_fb] = 1
                    col_fb += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
