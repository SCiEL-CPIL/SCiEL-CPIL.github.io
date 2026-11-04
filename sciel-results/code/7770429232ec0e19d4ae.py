"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    n, m, p = DIMENSIONS

    # Build the target tensor: T[i,j,k] = 1 if (i,j,k) is a valid multiplication term
    # The multiplication tensor M_{n,m,p} has entries:
    # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]

    # Strategy: Block decomposition using Strassen for 2x2x2 sub-blocks
    # Partition the 3x5x5 multiplication into blocks
    # For the 3x5x5 case, we can use:
    # - Split the 3 rows into [1, 2] or [2, 1]
    # - Split the 5 cols into [2, 2, 1] or [1, 2, 2]
    # - Split the 5 output cols into [2, 2, 1] or [1, 2, 2]
    # Where 2x2x2 blocks use Strassen (rank 7), 1x2x2 or similar use schoolbook

    def build_strassen_2x2x2(offset_i, offset_j, offset_k):
        """Build Strassen decomposition for a 2x2x2 block.
        Returns (U_entries, V_entries, W_entries) where each is list of (row, col, val)
        for the 7 Strassen terms.
        """
        # Strassen for 2x2: C = A*B
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[p,q],[r,s]]
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-h)
        # m4 = d(g-e)
        # m5 = (a+b)h
        # m6 = (c-a)(e+f)
        # m7 = (b-d)(g+h)
        # C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
        #
        # In tensor form for 2x2x2:
        # U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
        #
        # Let's index locally: i in {0,1}, j in {0,1}, k in {0,1}
        # A[i,j], B[j,k], C[i,k]
        #
        # Term 1: (a+d)(e+h): U has 1s at (0,0),(1,1); V has 1s at (0,0),(0,1),(1,0),(1,1)
        #   Actually V[j,k]: we need sum over j of A[i,j]*B[j,k]
        #   (a+d)(e+h) = (A[0,0]+A[1,1])(B[0,0]+B[0,1]+B[1,0]+B[1,1])... no
        #   Wait, (a+d)(e+h) where a=A[0,0], d=A[1,1], e=B[0,0], h=B[1,1]
        #   So U: (0,0)=1, (1,1)=1; V: (0,0)=1, (1,1)=1; W: all 1s
        #   Contribution to C[i,k]: for (i,k) in {(0,0),(0,1),(1,0),(1,1)}
        #   C[0,0] += 1*1*1 = 1 (from U[0,0]*V[0,0]*W[0,0])... 
        #   Hmm, this is getting complex. Let me use the standard formulation.
        
        # Standard Strassen in tensor decomposition form:
        # The 7 rank-1 terms for 2x2x2 multiplication:
        # Each term r has U_r (2x2), V_r (2x2), W_r (2x2) such that
        # C[i,k] = sum_r sum_j U_r[i,j] * V_r[j,k] * W_r[k,i]
        
        # Term 1: m1 = (a+d)(e+h)
        # U1 = [[1,0],[0,1]], V1 = [[1,0],[0,1]], W1 = [[1,1],[1,1]]
        # Check: C[0,0] += U1[0,0]*V1[0,0]*W1[0,0] + U1[0,1]*V1[1,0]*W1[0,0] = 1*1*1 + 0 = 1
        # C[0,1] += U1[0,0]*V1[0,1]*W1[1,0] + U1[0,1]*V1[1,1]*W1[1,0] = 1*0*1 + 0 = 0
        # Hmm that's not right either.
        
        # Let me think more carefully. The tensor is T[i,j,k] where the multiplication
        # C[i,k] = sum_j A[i,j] B[j,k].
        # A rank-1 decomposition term contributes U[i,j]*V[j,k]*W[k,i] to C[i,k]
        # summed over j... no wait.
        
        # Actually the standard formulation: C[i,k] = sum_j A[i,j] * B[j,k]
        # A rank-1 term in the decomposition of the multiplication tensor:
        # contributes U[i,j] * V[j,k] * W[k,i] -- but this doesn't sum over j.
        
        # I think the correct interpretation is:
        # sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = delta (the multiplication tensor)
        # where the multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
        # and the "output" is implicit -- actually no.
        
        # Let me re-read the task: "exactly decompose the matrix multiplication tensor"
        # The matrix multiplication tensor M_{n,m,p} is a 3-tensor of size n x m x p
        # where M[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in this indexing)
        # Wait no. The multiplication tensor for multiplying n x m by m x p:
        # It's indexed by (i, j, k) where i indexes rows of A (and C), 
        # j indexes columns of A (and rows of B), k indexes columns of B (and C).
        # M[i,j,k] = 1 for all valid indices.
        # A rank-r decomposition: M = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r
        # where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
        # M[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # So for 2x2x2: M[i,j,k] = 1 for all i,j,k in {0,1}
        # Strassen gives rank 7:
        # The 7 terms are (using 0-indexed, A[i,j], B[j,k], C[i,k]):
        # m1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])  -> not quite standard Strassen
        
        # Let me just use the known rank-7 decomposition for 2x2x2:
        # It's well known that M_{2,2,2} has rank 7.
        # One explicit decomposition:
        # Term r: U_r, V_r, W_r (each 2x2 matrices flattened)
        
        # From the literature, one 7-term decomposition:
        # (I'll use the standard one)
        # m1 = (a+d)(e+h): U=[[1,0],[0,1]], V=[[1,0],[0,1]], W=[[1,1],[1,1]]
        # m2 = (c+d)e: U=[[0,0],[1,1]], V=[[1,0],[0,0]], W=[[1,0],[0,0]]
        # m3 = a(f-h): U=[[1,0],[0,0]], V=[[0,1],[0,-1]], W=[[1,0],[0,0]]
        # m4 = d(g-e): U=[[0,0],[0,1]], V=[[0,-1],[1,0]], W=[[1,0],[0,0]]
        # m5 = (a+b)h: U=[[1,1],[0,0]], V=[[0,0],[0,1]], W=[[1,0],[0,0]]
        # m6 = (c-a)(e+f): U=[[0,0],[1,-1]], V=[[1,1],[0,0]], W=[[0,0],[0,1]]
        # m7 = (b-d)(g+h): U=[[0,1],[0,-1]], V=[[0,0],[1,1]], W=[[0,0],[0,1]]
        
        # Verify m1: C[i,k] += sum_j U[i,j]*V[j,k]*W[k,i]
        # C[0,0]: j=0: U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1; j=1: U[0,1]*V[1,0]*W[0,0] = 0*0*1 = 0. Total = 1
        # C[0,1]: j=0: U[0,0]*V[0,1]*W[1,0] = 1*0*1 = 0; j=1: U[0,1]*V[1,1]*W[1,0] = 0*1*1 = 0. Total = 0
        # C[1,0]: j=0: U[1,0]*V[0,0]*W[0,1] = 0*1*1 = 0; j=1: U[1,1]*V[1,0]*W[0,1] = 1*0*1 = 0. Total = 0
        # C[1,1]: j=0: U[1,0]*V[0,1]*W[1,1] = 0*0*1 = 0; j=1: U[1,1]*V[1,1]*W[1,1] = 1*1*1 = 1. Total = 1
        # So m1 contributes to C[0,0] and C[1,1] each by 1. Good, that's (a+d)(e+h) pattern.
        
        # This is getting very complex to verify by hand. Let me just implement it
        # and verify numerically.
        
        U_terms = [
            [[1,0],[0,1]],  # m1
            [[0,0],[1,1]],  # m2
            [[1,0],[0,0]],  # m3
            [[0,0],[0,1]],  # m4
            [[1,1],[0,0]],  # m5
            [[0,0],[1,-1]], # m6
            [[0,1],[0,-1]], # m7
        ]
        V_terms = [
            [[1,0],[0,1]],  # m1
            [[1,0],[0,0]],  # m2
            [[0,1],[0,-1]], # m3
            [[0,-1],[1,0]], # m4
            [[0,0],[0,1]],  # m5
            [[1,1],[0,0]],  # m6
            [[0,0],[1,1]],  # m7
        ]
        W_terms = [
            [[1,1],[1,1]],  # m1
            [[1,0],[0,0]],  # m2
            [[1,0],[0,0]],  # m3
            [[1,0],[0,0]],  # m4
            [[1,0],[0,0]],  # m5
            [[0,0],[0,1]],  # m6
            [[0,0],[0,1]],  # m7
        ]
        
        # Verify this decomposition
        C = np.zeros((2,2))
        for r in range(7):
            Ur = np.array(U_terms[r], dtype=float)
            Vr = np.array(V_terms[r], dtype=float)
            Wr = np.array(W_terms[r], dtype=float)
            for i in range(2):
                for k in range(2):
                    for j in range(2):
                        C[i,k] += Ur[i,j] * Vr[j,k] * Wr[k,i]
        
        # C should be... well, it should represent the multiplication
        # Actually for the tensor decomposition, we need:
        # sum_r U[i,j]*V[j,k]*W[k,i] = 1 for all (i,j,k)
        # Let me check that instead
        T = np.zeros((2,2,2))
        for r in range(7):
            Ur = np.array(U_terms[r], dtype=float)
            Vr = np.array(V_terms[r], dtype=float)
            Wr = np.array(W_terms[r], dtype=float)
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        T[i,j,k] += Ur[i,j] * Vr[j,k] * Wr[k,i]
        
        # T should be all ones
        if not np.allclose(T, 1.0, atol=1e-10):
            # Fallback: use schoolbook for this block
            pass
        
        return U_terms, V_terms, W_terms

    def build_block_decomposition():
        """Build a block-based decomposition of the 3x5x5 multiplication.
        
        Partition:
        - Rows of A (n=3): split as [2, 1] -> blocks at row offsets 0, 2
        - Cols of A / Rows of B (m=5): split as [2, 2, 1] -> offsets 0, 2, 4
        - Cols of B (p=5): split as [2, 2, 1] -> offsets 0, 2, 4
        
        For each combination of (row_block, col_block, out_block):
        - If all sizes are 2: use Strassen (rank 7)
        - If sizes are (2,2,1), (2,1,2), (1,2,2), etc.: use schoolbook (rank = product of sizes)
        - If any size is 1: schoolbook
        """
        # Row partition: [2, 1]
        row_splits = [(0, 2), (2, 3)]  # (start, end)
        # Col partition: [2, 2, 1]
        col_splits = [(0, 2), (2, 4), (4, 5)]
        # Out partition: [2, 2, 1]
        out_splits = [(0, 2), (2, 4), (4, 5)]
        
        # Accumulate U, V, W contributions
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # We'll build lists of (row, value) per column
        
        u_cols = []  # list of dicts: {row: value}
        v_cols = []
        w_cols = []
        
        for (r_start, r_end) in row_splits:
            r_size = r_end - r_start
            for (c_start, c_end) in col_splits:
                c_size = c_end - c_start
                for (o_start, o_end) in out_splits:
                    o_size = o_end - o_start
                    
                    if r_size == 2 and c_size == 2 and o_size == 2:
                        # Use Strassen
                        U_terms, V_terms, W_terms = build_strassen_2x2x2(r_start, c_start, o_start)
                        for r in range(7):
                            u_dict = {}
                            v_dict = {}
                            w_dict = {}
                            for i in range(2):
                                for j in range(2):
                                    row = (r_start + i) * m + (c_start + j)
                                    u_dict[row] = U_terms[r][i][j]
                            for j in range(2):
                                for k in range(2):
                                    row = (c_start + j) * p + (o_start + k)
                                    v_dict[row] = V_terms[r][j][k]
                            for k in range(2):
                                for i in range(2):
                                    row = (o_start + k) * n + (r_start + i)
                                    w_dict[row] = W_terms[r][k][i]
                            u_cols.append(u_dict)
                            v_cols.append(v_dict)
                            w_cols.append(w_dict)
                    else:
                        # Schoolbook for this block
                        for i in range(r_size):
                            for j in range(c_size):
                                for k in range(o_size):
                                    u_dict = {}
                                    v_dict = {}
                                    w_dict = {}
                                    u_row = (r_start + i) * m + (c_start + j)
                                    v_row = (c_start + j) * p + (o_start + k)
                                    w_row = (o_start + k) * n + (r_start + i)
                                    u_dict[u_row] = 1
                                    v_dict[v_row] = 1
                                    w_dict[w_row] = 1
                                    u_cols.append(u_dict)
                                    v_cols.append(v_dict)
                                    w_cols.append(w_dict)
        
        rank = len(u_cols)
        
        # Convert to matrices
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for c in range(rank):
            for row, val in u_cols[c].items():
                U[row, c] = val
            for row, val in v_cols[c].items():
                V[row, c] = val
            for row, val in w_cols[c].items():
                W[row, c] = val
        
        return U, V, W, rank

    # Build the block decomposition
    U_block, V_block, W_block, rank_block = build_block_decomposition()
    
    # Verify the block decomposition
    def verify_decomposition(U, V, W):
        """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all (i,j,k)."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(U.shape[1]):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    valid_block = verify_decomposition(U_block, V_block, W_block)
    
    if valid_block:
        best_U = U_block
        best_V = V_block
        best_W = W_block
        best_rank = rank_block
    else:
        # Fallback to schoolbook
        best_rank = n * m * p
        best_U = np.zeros((n * m, best_rank))
        best_V = np.zeros((m * p, best_rank))
        best_W = np.zeros((p * n, best_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i*m+j, col] = 1
                    best_V[j*p+k, col] = 1
                    best_W[k*n+i, col] = 1
                    col += 1
    
    # Try to improve with linear system approach for lower ranks
    # Given random U, V, solve for W exactly, then check if W is half-integer
    
    def try_linear_solve(R, max_attempts=30):
        """Try to find a rank-R decomposition by random U,V and solving for W."""
        if time.time() > deadline - 20:
            return None
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
            
            # Generate random integer U, V
            if attempt % 3 == 0:
                U_rand = np.random.randint(-1, 2, (n*m, R)).astype(float)
                V_rand = np.random.randint(-1, 2, (m*p, R)).astype(float)
            elif attempt % 3 == 1:
                U_rand = np.random.choice([-1.0, 1.0], size=(n*m, R))
                V_rand = np.random.choice([-1.0, 1.0], size=(m*p, R))
            else:
                U_rand = np.random.randint(-2, 3, (n*m, R)).astype(float)
                V_rand = np.random.randint(-2, 3, (m*p, R)).astype(float)
            
            # Build linear system: for each (i,j,k):
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # Unknowns: W flattened, size p*n*R
            # Equations: n*m*p = 75
            
            num_eqs = n * m * p
            num_vars = p * n * R
            
            A = np.zeros((num_eqs, num_vars))
            b = np.ones(num_eqs)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    u_row = i * m + j
                    for k in range(p):
                        v_row = j * p + k
                        w_row = k * n + i
                        for r in range(R):
                            A[idx, w_row * R + r] = U_rand[u_row, r] * V_rand[v_row, r]
                        idx += 1
            
            try:
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                W_rand = W_flat.reshape(p * n, R)
                
                # Check if W is close to half-integers
                W_check = W_rand * 2
                dist = np.abs(W_check - np.round(W_check))
                max_dist = np.max(dist)
                
                if max_dist < 1e-4:
                    W_int = np.round(W_check) / 2.0
                    
                    # Verify exact solution
                    if verify_decomposition(U_rand, V_rand, W_int):
                        return U_rand, V_rand, W_int
                        
            except Exception:
                pass
        
        return None
    
    # Try to find lower rank decompositions
    # Start from the block rank and try to go lower
    target_ranks = list(range(max(60, best_rank - 5), best_rank))
    target_ranks.sort()  # Try lower ranks first
    
    for R in target_ranks:
        if time.time() > deadline - 25:
            break
        
        result = try_linear_solve(R, max_attempts=20)
        if result is not None:
            U_found, V_found, W_found = result
            if R < best_rank:
                best_U = U_found
                best_V = V_found
                best_W = W_found
                best_rank = R
                break  # Found a valid lower rank, stop
    
    # Final verification
    if not verify_decomposition(best_U, best_V, best_W):
        # Emergency fallback to schoolbook
        best_rank = n * m * p
        best_U = np.zeros((n * m, best_rank))
        best_V = np.zeros((m * p, best_rank))
        best_W = np.zeros((p * n, best_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i*m+j, col] = 1
                    best_V[j*p+k, col] = 1
                    best_W[k*n+i, col] = 1
                    col += 1
    
    result = {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
    
    return result
# EVOLVE-BLOCK-END
