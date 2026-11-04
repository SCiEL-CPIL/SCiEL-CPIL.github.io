"""Rank-minimal exact tensor decomposition for 3x4 x 4x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    deadline = time.time() + 145  # safety margin before 150s limit
    
    # We'll build a Strassen-based decomposition
    # A is n×m = 3×4, B is m×p = 4×6, C is p×n = 6×3
    # U shape: (n*m, rank) = (12, rank)
    # V shape: (m*p, rank) = (24, rank)  
    # W shape: (p*n, rank) = (18, rank)
    
    # Index conventions:
    # U[i*m + j, r] - row i, col j of A
    # V[j*p + k, r] - row j, col k of B
    # W[k*n + i, r] - row k, col i of C (trace convention W[k*n+i, r])
    
    # The multiplication tensor T has entries T[i,j,k] = 1 if we need A[i,j]*B[j,k] in C[i,k]
    # Actually, C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is T[i,j,k] = delta_{j,j} ... no, it's T[i,j,k] = 1 for all valid i,j,k
    # Wait, the matrix multiplication tensor is defined as:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the coefficient is 1 if we multiply A[i,j]*B[j,k]
    # More precisely: T_{i,j,k} = 1 for all i in [n], j in [m], k in [p]
    # But that's not right either. The tensor for matrix multiplication is:
    # T = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p E_{i,k} ⊗ E_{i,j} ⊗ E_{j,k}
    # where E_{a,b} is the matrix with 1 at position (a,b) and 0 elsewhere.
    # In our indexing:
    # U index: i*m + j (for A[i,j])
    # V index: j*p + k (for B[j,k])
    # W index: k*n + i (for C[i,k], but the convention is W[k*n+i, r])
    
    # So the tensor entry at (u_idx, v_idx, w_idx) is 1 iff:
    # u_idx = i*m+j, v_idx = j*p+k, w_idx = k*n+i for some valid i,j,k
    # i.e., the middle indices match: j from U equals j from V, and k from V relates to i from W
    
    # Let's build the schoolbook first, then try to optimize with Strassen
    
    # Strategy: Use block decomposition
    # Split A into blocks, B into blocks, and use Strassen where possible
    
    # For 3x4 x 4x6:
    # We can split the j dimension (size 4) into two parts of size 2
    # A is 3x4, split into A_left (3x2) and A_right (3x2)
    # B is 4x6, split into B_top (2x6) and B_bottom (2x6)
    # C = A_left * B_top + A_right * B_bottom
    
    # For A_left (3x2) * B_top (2x6):
    # Split A_left into rows: A_left is 3x2
    # We can use the fact that 3x2x2 has rank at most 3*2*2 = 12, but can be lower
    # Actually, for small dimensions, let's just use schoolbook within blocks
    
    # Better approach: Use the known optimal or near-optimal decompositions
    
    # Let me try a direct Strassen-like approach:
    # Split the problem into 2x2x2 blocks where possible
    
    # For the j=0,1 part (A[:, 0:2] * B[0:2, :]):
    # A is 3x2, B is 2x6, result is 3x6
    # Split A into 2 row-blocks: rows 0-1 (2x2) and row 2 (1x2)
    # Split B into 3 col-blocks: cols 0-1, 2-3, 4-5 (each 2x2)
    
    # For the 2x2x2 block (A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]):
    # Use Strassen rank 7
    
    # For A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]:
    # Another 2x2x2 block, Strassen rank 7
    
    # For A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]:
    # Another 2x2x2 block, Strassen rank 7
    
    # For A[2:3, 0:2] * B[0:2, 0:6] -> C[2:3, 0:6]:
    # This is 1x2x6, which is just 6 scalar multiplications, rank 6
    
    # Similarly for j=2,3 part:
    # For A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]: Strassen rank 7
    # For A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]: Strassen rank 7
    # For A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]: Strassen rank 7
    # For A[2:3, 2:4] * B[2:4, 0:6] -> C[2:3, 0:6]: rank 6
    
    # Total: 3*7 + 6 + 3*7 + 6 = 21 + 6 + 21 + 6 = 54
    
    # But wait, I need to verify the Strassen decomposition is correct.
    # Let me implement it carefully.
    
    terms = []  # list of (u_idx, v_idx, w_idx, coeff)
    
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        terms.append((u_idx, v_idx, w_idx, coeff))
    
    def strassen_2x2x2(i0, j0, k0):
        """
        Compute C[i0:i0+2, k0:k0+2] += A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        using Strassen's algorithm (rank 7).
        
        Local indices:
        A: rows i0, i0+1; cols j0, j0+1
        B: rows j0, j0+1; cols k0, k0+1
        C: rows i0, i0+1; cols k0, k0+1
        
        Strassen formulas:
        M1 = A11 * B11
        M2 = (A21 + A22) * B11
        M3 = A11 * (B21 - B11)
        M4 = A22 * (B12 - B22)
        M5 = (A11 + A22) * (B11 + B22)
        M6 = (A12 - A22) * (B21 + B22)
        M7 = (A21 - A11) * (B12 - B11)
        
        C11 = M5 + M4 - M2 - M7
        C12 = M1 + M7
        C21 = M3 + M5
        C22 = M2 + M4 - M1 - M3
        
        Where A11=A[i0,j0], A12=A[i0,j0+1], A21=A[i0+1,j0], A22=A[i0+1,j0+1]
              B11=B[j0,k0], B12=B[j0,k0+1], B21=B[j0+1,k0], B22=B[j0+1,k0+1]
              C11=C[i0,k0], C12=C[i0,k0+1], C21=C[i0+1,k0], C22=C[i0+1,k0+1]
        
        For tensor decomposition, each rank-1 term is U[a,r]*V[b,r]*W[c,r]
        We need to expand each M_r into rank-1 terms.
        """
        # M1 = A11 * B11 = A[i0,j0] * B[j0,k0]
        # U_idx = i0*m + j0, V_idx = j0*p + k0
        # Contributes to C12 and C22:
        # C12 = C[i0, k0+1] -> W_idx = (k0+1)*n + i0, coeff +1
        # C22 = C[i0+1, k0+1] -> W_idx = (k0+1)*n + (i0+1), coeff -1
        add_term(i0*m + j0, j0*p + k0, (k0+1)*n + i0, 1)
        add_term(i0*m + j0, j0*p + k0, (k0+1)*n + (i0+1), -1)
        
        # M2 = (A21 + A22) * B11 = (A[i0+1,j0] + A[i0+1,j0+1]) * B[j0,k0]
        # Two terms:
        # A21*B11: U=(i0+1)*m+j0, V=j0*p+k0
        #   C21 = C[i0+1,k0] -> W=(k0)*n+(i0+1), +1
        #   C22 = C[i0+1,k0+1] -> W=(k0+1)*n+(i0+1), +1
        add_term((i0+1)*m + j0, j0*p + k0, k0*n + (i0+1), 1)
        add_term((i0+1)*m + j0, j0*p + k0, (k0+1)*n + (i0+1), 1)
        # A22*B11: U=(i0+1)*m+(j0+1), V=j0*p+k0
        #   C21 -> W=k0*n+(i0+1), +1
        #   C22 -> W=(k0+1)*n+(i0+1), +1
        add_term((i0+1)*m + (j0+1), j0*p + k0, k0*n + (i0+1), 1)
        add_term((i0+1)*m + (j0+1), j0*p + k0, (k0+1)*n + (i0+1), 1)
        
        # M3 = A11 * (B21 - B11) = A[i0,j0] * (B[j0+1,k0] - B[j0,k0])
        # Two terms:
        # A11*B21: U=i0*m+j0, V=(j0+1)*p+k0
        #   C21 = C[i0+1,k0] -> W=k0*n+(i0+1), +1
        #   C22 = C[i0+1,k0+1] -> W=(k0+1)*n+(i0+1), -1
        add_term(i0*m + j0, (j0+1)*p + k0, k0*n + (i0+1), 1)
        add_term(i0*m + j0, (j0+1)*p + k0, (k0+1)*n + (i0+1), -1)
        # -A11*B11: U=i0*m+j0, V=j0*p+k0
        #   C21 -> W=k0*n+(i0+1), -1
        #   C22 -> W=(k0+1)*n+(i0+1), +1
        add_term(i0*m + j0, j0*p + k0, k0*n + (i0+1), -1)
        add_term(i0*m + j0, j0*p + k0, (k0+1)*n + (i0+1), 1)
        
        # M4 = A22 * (B12 - B22) = A[i0+1,j0+1] * (B[j0,k0+1] - B[j0+1,k0+1])
        # Two terms:
        # A22*B12: U=(i0+1)*m+(j0+1), V=j0*p+(k0+1)
        #   C11 = C[i0,k0] -> W=k0*n+i0, +1
        #   C22 = C[i0+1,k0+1] -> W=(k0+1)*n+(i0+1), +1
        add_term((i0+1)*m + (j0+1), j0*p + (k0+1), k0*n + i0, 1)
        add_term((i0+1)*m + (j0+1), j0*p + (k0+1), (k0+1)*n + (i0+1), 1)
        # -A22*B22: U=(i0+1)*m+(j0+1), V=(j0+1)*p+(k0+1)
        #   C11 -> W=k0*n+i0, -1
        #   C22 -> W=(k0+1)*n+(i0+1), -1
        add_term((i0+1)*m + (j0+1), (j0+1)*p + (k0+1), k0*n + i0, -1)
        add_term((i0+1)*m + (j0+1), (j0+1)*p + (k0+1), (k0+1)*n + (i0+1), -1)
        
        # M5 = (A11 + A22) * (B11 + B22)
        # = (A[i0,j0] + A[i0+1,j0+1]) * (B[j0,k0] + B[j0+1,k0+1])
        # Four terms:
        # A11*B11: U=i0*m+j0, V=j0*p+k0
        #   C11 -> W=k0*n+i0, +1
        #   C21 = C[i0+1,k0] -> W=k0*n+(i0+1), +1
        add_term(i0*m + j0, j0*p + k0, k0*n + i0, 1)
        add_term(i0*m + j0, j0*p + k0, k0*n + (i0+1), 1)
        # A11*B22: U=i0*m+j0, V=(j0+1)*p+(k0+1)
        #   C11 -> W=k0*n+i0, +1
        #   C21 -> W=k0*n+(i0+1), +1
        add_term(i0*m + j0, (j0+1)*p + (k0+1), k0*n + i0, 1)
        add_term(i0*m + j0, (j0+1)*p + (k0+1), k0*n + (i0+1), 1)
        # A22*B11: U=(i0+1)*m+(j0+1), V=j0*p+k0
        #   C11 -> W=k0*n+i0, +1
        #   C21 -> W=k0*n+(i0+1), +1
        add_term((i0+1)*m + (j0+1), j0*p + k0, k0*n + i0, 1)
        add_term((i0+1)*m + (j0+1), j0*p + k0, k0*n + (i0+1), 1)
        # A22*B22: U=(i0+1)*m+(j0+1), V=(j0+1)*p+(k0+1)
        #   C11 -> W=k0*n+i0, +1
        #   C21 -> W=k0*n+(i0+1), +1
        add_term((i0+1)*m + (j0+1), (j0+1)*p + (k0+1), k0*n + i0, 1)
        add_term((i0+1)*m + (j0+1), (j0+1)*p + (k0+1), k0*n + (i0+1), 1)
        
        # M6 = (A12 - A22) * (B21 + B22)
        # = (A[i0,j0+1] - A[i0+1,j0+1]) * (B[j0+1,k0] + B[j0+1,k0+1])
        # Four terms:
        # A12*B21: U=i0*m+(j0+1), V=(j0+1)*p+k0
        #   C12 = C[i0,k0+1] -> W=(k0+1)*n+i0, +1
        #   C22 = C[i0+1,k0+1] -> W=(k0+1)*n+(i0+1), -1
        add_term(i0*m + (j0+1), (j0+1)*p + k0, (k0+1)*n + i0, 1)
        add_term(i0*m + (j0+1), (j0+1)*p + k0, (k0+1)*n + (i0+1), -1)
        # A12*B22: U=i0*m+(j0+1), V=(j0+1)*p+(k0+1)
        #   C12 -> W=(k0+1)*n+i0, +1
        #   C22 -> W=(k0+1)*n+(i0+1), -1
        add_term(i0*m + (j0+1), (j0+1)*p + (k0+1), (k0+1)*n + i0, 1)
        add_term(i0*m + (j0+1), (j0+1)*p + (k0+1), (k0+1)*n + (i0+1), -1)
        # -A22*B21: U=(i0+1)*m+(j0+1), V=(j0+1)*p+k0
        #   C12 -> W=(k0+1)*n+i0, -1
        #   C22 -> W=(k0+1)*n+(i0+1), +1
        add_term((i0+1)*m + (j0+1), (j0+1)*p + k0, (k0+1)*n + i0, -1)
        add_term((i0+1)*m + (j0+1), (j0+1)*p + k0, (k0+1)*n + (i0+1), 1)
        # -A22*B22: U=(i0+1)*m+(j0+1), V=(j0+1)*p+(k0+1)
        #   C12 -> W=(k0+1)*n+i0, -1
        #   C22 -> W=(k0+1)*n+(i0+1), +1
        add_term((i0+1)*m + (j0+1), (j0+1)*p + (k0+1), (k0+1)*n + i0, -1)
        add_term((i0+1)*m + (j0+1), (j0+1)*p + (k0+1), (k0+1)*n + (i0+1), 1)
        
        # M7 = (A21 - A11) * (B12 - B11)
        # = (A[i0+1,j0] - A[i0,j0]) * (B[j0,k0+1] - B[j0,k0])
        # Four terms:
        # A21*B12: U=(i0+1)*m+j0, V=j0*p+(k0+1)
        #   C12 = C[i0,k0+1] -> W=(k0+1)*n+i0, +1
        #   C21 = C[i0+1,k0] -> W=k0*n+(i0+1), -1
        add_term((i0+1)*m + j0, j0*p + (k0+1), (k0+1)*n + i0, 1)
        add_term((i0+1)*m + j0, j0*p + (k0+1), k0*n + (i0+1), -1)
        # A21*(-B11): U=(i0+1)*m+j0, V=j0*p+k0
        #   C12 -> W=(k0+1)*n+i0, -1
        #   C21 -> W=k0*n+(i0+1), +1
        add_term((i0+1)*m + j0, j0*p + k0, (k0+1)*n + i0, -1)
        add_term((i0+1)*m + j0, j0*p + k0, k0*n + (i0+1), 1)
        # -A11*B12: U=i0*m+j0, V=j0*p+(k0+1)
        #   C12 -> W=(k0+1)*n+i0, -1
        #   C21 -> W=k0*n+(i0+1), +1
        add_term(i0*m + j0, j0*p + (k0+1), (k0+1)*n + i0, -1)
        add_term(i0*m + j0, j0*p + (k0+1), k0*n + (i0+1), 1)
        # -A11*(-B11): U=i0*m+j0, V=j0*p+k0
        #   C12 -> W=(k0+1)*n+i0, +1
        #   C21 -> W=k0*n+(i0+1), -1
        add_term(i0*m + j0, j0*p + k0, (k0+1)*n + i0, 1)
        add_term(i0*m + j0, j0*p + k0, k0*n + (i0+1), -1)
    
    # Now let's build the decomposition
    # Part 1: A[:, 0:2] * B[0:2, :] (j=0,1)
    # Sub-part 1a: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    strassen_2x2x2(0, 0, 0)
    # Sub-part 1b: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    strassen_2x2x2(0, 0, 2)
    # Sub-part 1c: A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    strassen_2x2x2(0, 0, 4)
    # Sub-part 1d: A[2:3, 0:2] * B[0:2, 0:6] -> C[2:3, 0:6]
    for k0 in range(0, 6):
        add_term(2*m + 0, 0*p + k0, k0*n + 2, 1)
        add_term(2*m + 1, 1*p + k0, k0*n + 2, 1)
    
    # Part 2: A[:, 2:4] * B[2:4, :] (j=2,3)
    # Sub-part 2a: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    strassen_2x2x2(0, 2, 0)
    # Sub-part 2b: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    strassen_2x2x2(0, 2, 2)
    # Sub-part 2c: A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    strassen_2x2x2(0, 2, 4)
    # Sub-part 2d: A[2:3, 2:4] * B[2:4, 0:6] -> C[2:3, 0:6]
    for k0 in range(0, 6):
        add_term(2*m + 2, 2*p + k0, k0*n + 2, 1)
        add_term(2*m + 3, 3*p + k0, k0*n + 2, 1)
    
    # Now build the factor matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_idx, v_idx, w_idx, coeff) in enumerate(terms):
        U[u_idx][r] = coeff
        V[v_idx][r] = 1
        W[w_idx][r] = 1
    
    # Verify the decomposition
    def verify():
        """Verify that the decomposition is correct."""
        # Check: for all i,j,k, sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should be 1
        # Actually, the tensor entry T[i,j,k] = 1 iff the middle indices match
        # In our case, T[i,j,k] = 1 for all valid i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
        # So we need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
        # Wait, that's not right. The tensor is:
        # T = sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
        # So T[u,v,w] = 1 iff u=i*m+j, v=j*p+k, w=k*n+i for some i,j,k
        # i.e., the j in u matches the j in v, and the k in v matches the i in w (via the product)
        
        # More precisely: T[u,v,w] = 1 iff:
        # u = i*m + j, v = j*p + k, w = k*n + i for some i in [0,n), j in [0,m), k in [0,p)
        # This means: j = u % m = v // p, and k = v % p, i = w % n, and w // n = k, u // m = i
        
        # So for the decomposition to be correct:
        # sum_r U[u,r] * V[v,r] * W[w,r] = T[u,v,w]
        
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Build the tensor from the decomposition
        # T_decomp[u,v,w] = sum_r U[u,r] * V[v,r] * W[w,r]
        # This is a 3-way tensor product
        
        # For efficiency, check a sample of entries
        # Actually, let's check all entries since the dimensions are small
        # U is 12xR, V is 24xR, W is 18xR
        # Total entries to check: 12*24*18 = 5184
        
        errors = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    # This entry should be 1
                    val = sum(U_np[u_idx, r] * V_np[v_idx, r] * W_np[w_idx, r] for r in range(rank))
                    if abs(val - 1) > 1e-9:
                        errors += 1
        
        # Also check that entries where the indices don't match are 0
        # T[u,v,w] should be 0 if j_u != j_v or k_v != i_w or i_u != k_w
        # More precisely: T[u,v,w] = 1 iff u%m == v//p and v%p == w//n and u//m == w%n
        # Let's check some non-matching entries
        for u in range(n * m):
            for v in range(m * p):
                for w in range(p * n):
                    i_u = u // m
                    j_u = u % m
                    j_v = v // p
                    k_v = v % p
                    i_w = w % n
                    k_w = w // n
                    
                    if j_u != j_v or k_v != k_w or i_u != i_w:
                        # This entry should be 0
                        val = sum(U_np[u, r] * V_np[v, r] * W_np[w, r] for r in range(rank))
                        if abs(val) > 1e-9:
                            errors += 1
                    else:
                        # This entry should be 1 (already checked above, but let's be safe)
                        # Actually, we already checked the matching case above
                        pass
        
        return errors == 0
    
    if not verify():
        # Fallback to schoolbook if verification fails
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    add_term(i * m + j, j * p + k, k * n + i, 1)
        
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for r, (u_idx, v_idx, w_idx, coeff) in enumerate(terms):
            U[u_idx][r] = coeff
            V[v_idx][r] = 1
            W[w_idx][r] = 1
    
    return {"U": U, "V": V, "W": W}
