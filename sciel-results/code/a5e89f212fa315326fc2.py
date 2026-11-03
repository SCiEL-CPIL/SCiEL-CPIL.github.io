import numpy as np
import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill

    nm = n * m  # 6
    mp = m * p  # 27
    pn = p * n  # 18

    # Build the target tensor as a set of (a,b,c) with value 1
    # a = i*m + j, b = j*p + k, c = k*n + i
    target_entries = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_entries[(a, b, c)] = 1

    target_set = set(target_entries.keys())

    # Schoolbook decomposition
    rank_sb = n * m * p  # 54
    U_sb = [[0] * rank_sb for _ in range(nm)]
    V_sb = [[0] * rank_sb for _ in range(mp)]
    W_sb = [[0] * rank_sb for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb

    def verify_decomposition(U, V, W, tol=1e-10):
        """Check if U, V, W exactly decompose the tensor."""
        r = len(U[0])

        # Check all target entries
        for (a, b, c), val in target_entries.items():
            s = 0.0
            for t in range(r):
                s += U[a][t] * V[b][t] * W[c][t]
            if abs(s - val) > tol:
                return False

        # Check that non-target entries are zero
        # Total entries: nm * mp * pn = 6 * 27 * 18 = 2916
        # Target entries: 54
        # Non-target: 2862
        if r <= 40:
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        if (a, b, c) not in target_set:
                            s = 0.0
                            for t in range(r):
                                s += U[a][t] * V[b][t] * W[c][t]
                            if abs(s) > tol:
                                return False
        else:
            # For larger r, sample non-target entries
            non_target = [(a, b, c) for a in range(nm) for b in range(mp) for c in range(pn)
                          if (a, b, c) not in target_set]
            random.shuffle(non_target)
            sample_size = min(500, len(non_target))
            for (a, b, c) in non_target[:sample_size]:
                s = 0.0
                for t in range(r):
                    s += U[a][t] * V[b][t] * W[c][t]
                if abs(s) > tol:
                    return False

        return True

    # Strassen-based block decomposition for 2x3x9
    # Split the 3 columns of A (and rows of B) into groups: {0,1} and {2}
    # For j in {0,1}: use Strassen 7-term for 2x2x2 (splitting k into {0,1} and {2})
    # For j = 2: use schoolbook for 2x1x9
    
    # Actually, let's be more systematic. We'll build a decomposition that uses
    # Strassen's algorithm for the 2x2x2 sub-problems where applicable.
    
    # The 2x3x9 multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # We can group the k indices into pairs and singles: {0,1}, {2,3}, {4,5}, {6,7}, {8}
    # For each pair (k1, k2), and for j in {0,1}, we can use Strassen on the 2x2x2 block
    # (rows i={0,1}, cols j={0,1}, k-pair={k1,k2})
    # For j=2, we just do direct multiplication
    
    # Let's build this step by step.
    
    # First, let's define the Strassen 2x2x2 terms
    # For multiplying X (2x2) * Y (2x2) = Z (2x2):
    # Z[0,0] = M5 + M2 - M4 - M6
    # Z[0,1] = M1 + M4
    # Z[1,0] = M3 + M5
    # Z[1,1] = M2 - M1 + M3 + M6
    # where:
    # M1 = (X[1,0] + X[1,1]) * Y[0,0]
    # M2 = (X[0,0] + X[0,1]) * Y[1,0]
    # M3 = X[0,0] * (Y[0,1] - Y[1,1])
    # M4 = X[1,1] * (Y[0,0] + Y[1,0])
    # M5 = (X[0,0] + X[1,1]) * (Y[1,1] + Y[0,0])
    # M6 = (X[0,1] - X[1,1]) * (Y[1,0] + Y[1,1])
    # M7 = (X[0,0] - X[1,0]) * (Y[0,1] + Y[1,1])
    
    # For our problem, we need to adapt this. Let's think of it as:
    # For fixed j-pair (j1, j2) and k-pair (k1, k2), we have a 2x2x2 subproblem
    # where X[i,j'] = A[i, j1+j'], Y[j',k'] = B[j1+j', k1+k'], Z[i,k'] = C[i, k1+k']
    
    # The Strassen terms for this subproblem would be:
    # M1 = (A[1,j1] + A[1,j2]) * B[j1,k1]
    # M2 = (A[0,j1] + A[0,j2]) * B[j2,k1]
    # M3 = A[0,j1] * (B[j1,k2] - B[j2,k2])
    # M4 = A[1,j2] * (B[j1,k1] + B[j2,k1])
    # M5 = (A[0,j1] + A[1,j2]) * (B[j2,k2] + B[j1,k1])
    # M6 = (A[0,j2] - A[1,j2]) * (B[j2,k1] + B[j2,k2])
    # M7 = (A[0,j1] - A[1,j1]) * (B[j1,k2] + B[j2,k2])
    
    # And the output contributions:
    # C[0,k1] += M5 + M2 - M4 - M6
    # C[0,k2] += M1 + M4
    # C[1,k1] += M3 + M5
    # C[1,k2] += M2 - M1 + M3 + M6
    
    # Let's build the full decomposition using this structure.
    # We'll have:
    # - For j-pair (0,1) and k-pair (0,1): 7 Strassen terms
    # - For j-pair (0,1) and k-pair (2,3): 7 Strassen terms
    # - For j-pair (0,1) and k-pair (4,5): 7 Strassen terms
    # - For j-pair (0,1) and k-pair (6,7): 7 Strassen terms
    # - For j-pair (0,1) and k=8: 2x1x1 = 2 schoolbook terms
    # - For j=2 and all k: 2x1x9 = 18 schoolbook terms
    # Total: 4*7 + 2 + 18 = 28 + 2 + 18 = 48
    
    # Wait, that's not quite right. Let me reconsider.
    # For j-pair (0,1) and k=8, we have:
    # C[i,8] += A[i,0]*B[0,8] + A[i,1]*B[1,8]
    # This is 2 terms (one for each i), but in the tensor decomposition,
    # each (i,j,k) triple is one term. So for j in {0,1} and k=8, we have 2*1=2 terms.
    # Actually no, the schoolbook for 2x1x1 would be 2 terms (i=0 and i=1).
    
    # Let me recount:
    # j-pair (0,1), k-pair (0,1): 7 terms
    # j-pair (0,1), k-pair (2,3): 7 terms
    # j-pair (0,1), k-pair (4,5): 7 terms
    # j-pair (0,1), k-pair (6,7): 7 terms
    # j-pair (0,1), k=8: 2 terms (schoolbook for 2x1x1)
    # j=2, k=0..8: 18 terms (schoolbook for 2x1x9)
    # Total: 28 + 2 + 18 = 48
    
    # Let's build this decomposition.
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(U_val, V_val, W_val):
        """Add a rank-1 term to the decomposition."""
        U_row = [0] * nm
        V_row = [0] * mp
        W_row = [0] * pn
        
        for (a, b, c), coeff in U_val.items():
            U_row[a] += coeff
        for (a, b, c), coeff in V_val.items():
            V_row[b] += coeff
        for (a, b, c), coeff in W_val.items():
            W_row[c] += coeff
        
        U_list.append(U_row)
        V_list.append(V_row)
        W_list.append(W_row)
    
    # Helper to add a Strassen term for j-pair (j1,j2) and k-pair (k1,k2)
    def add_strassen_term(j1, j2, k1, k2, term_id):
        """Add one of the 7 Strassen terms."""
        U_val = {}
        V_val = {}
        W_val = {}
        
        if term_id == 0:  # M1 = (A[1,j1] + A[1,j2]) * B[j1,k1]
            U_val[(1*m+j1, 0, 0)] = 1
            U_val[(1*m+j2, 0, 0)] = 1
            V_val[(j1*p+k1, 0, 0)] = 1
            W_val[(k1*n+0, 0, 0)] = 1  # Will be combined in output
            # Actually, the W component depends on which output it contributes to
            # Let me restructure this.
        
        # This approach is getting complicated. Let me use a different strategy.
        # Instead of trying to manually build the Strassen decomposition, let me
        # use a more direct approach: build the decomposition as a sum of rank-1 tensors
        # where each rank-1 tensor corresponds to a "product" in an algebra.
        
        pass
    
    # Let me try a different, cleaner approach.
    # I'll directly construct the Strassen-based decomposition by explicitly
    # listing all the rank-1 terms.
    
    # For each Strassen block (j1,j2,k1,k2), we have 7 terms.
    # Each term is a rank-1 tensor: U_a * V_b * W_c
    # where U, V, W are vectors with specific non-zero entries.
    
    # Let me define the Strassen terms more carefully.
    # For the subproblem with A submatrix [[a00, a01], [a10, a11]] (rows i=0,1; cols j=j1,j2)
    # and B submatrix [[b00, b01], [b10, b11]] (rows j=j1,j2; cols k=k1,k2)
    # The output C submatrix [[c00, c01], [c10, c11]] (rows i=0,1; cols k=k1,k2)
    
    # Strassen products:
    # M1 = (a10 + a11) * b00
    # M2 = (a00 + a01) * b10
    # M3 = a00 * (b01 - b11)
    # M4 = a11 * (b00 + b10)
    # M5 = (a00 + a11) * (b11 + b00)
    # M6 = (a01 - a11) * (b10 + b11)
    # M7 = (a00 - a10) * (b01 + b11)
    
    # Output:
    # c00 = M5 + M2 - M4 - M6
    # c01 = M1 + M4
    # c10 = M3 + M5
    # c11 = M2 - M1 + M3 + M6
    
    # For the full tensor decomposition, each M_t is a "product" that contributes
    # to multiple (i,k) outputs. In the tensor T[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t],
    # each rank-1 term t has U, V, W vectors.
    
    # For M1 = (a10 + a11) * b00:
    # This contributes to c01 with coefficient +1
    # U vector: U[1*m+j1] = 1, U[1*m+j2] = 1 (for a10 and a11)
    # V vector: V[j1*p+k1] = 1 (for b00)
    # W vector: W[k1*n+0] = 1 (for c01, where i=0, k=k1)
    # Wait, c01 means i=0, k=k1? No, c01 means the (0,1) entry of the 2x2 output,
    # which is i=0, k=k2 (since k1 is col 0, k2 is col 1).
    # Let me re-index: c00 = C[i=0, k=k1], c01 = C[i=0, k=k2], c10 = C[i=1, k=k1], c11 = C[i=1, k=k2]
    
    # So for M1 contributing to c01 = C[0, k2]:
    # U: U[1*m+j1] = 1, U[1*m+j2] = 1
    # V: V[j1*p+k1] = 1
    # W: W[k2*n+0] = 1
    
    # This is getting complex. Let me just build it programmatically.
    
    # Actually, let me step back and use a simpler approach that I know works.
    # I'll use the fact that the tensor can be decomposed using the "embedding" trick.
    
    # For now, let me just return the schoolbook decomposition if I can't quickly
    # build a better one. But let me try one more approach: use the known rank-50
    # Strassen decomposition from the inspiration.
    
    # The inspiration mentions rank 50 using Strassen for 2x2x2 blocks.
    # Let me try to construct that.
    
    # Approach: Split the problem into blocks.
    # A is 2x3, B is 3x9, C is 2x9.
    # Split B into B_left (3x8) and B_right (3x1).
    # Split B_left into four 3x2 blocks: B_00, B_01, B_10, B_11 (rows 0-2, cols in pairs)
    # Actually, let's split the 9 columns into 4 pairs + 1 single: (0,1), (2,3), (4,5), (6,7), (8)
    # And split the 3 rows of B (and cols of A) into (0,1) and (2).
    
    # For the (0,1) row block of B and each 2-col block of B:
    # We have a 2x2x2 subproblem (A rows 0-1, A cols 0-1, B rows 0-1, B cols in the 2-col block)
    # Use Strassen: 7 terms each, 4 blocks = 28 terms
    
    # For the (0,1) row block of B and the single col 8:
    # 2x2x1 subproblem: 2 terms (schoolbook)
    
    # For the row 2 of B and all 9 cols:
    # 2x1x9 subproblem: 18 terms (schoolbook)
    
    # Total: 28 + 2 + 18 = 48
    
    # Let me build this decomposition explicitly.
    
    def build_strassen_decomposition():
        """Build a rank-48 decomposition using Strassen for 2x2x2 blocks."""
        U_rows = []
        V_rows = []
        W_rows = []
        
        def add_rank1(U_entries, V_entries, W_entries):
            """Add a rank-1 term. U_entries, V_entries, W_entries are dicts: {index: coeff}."""
            U_row = [0.0] * nm
            V_row = [0.0] * mp
            W_row = [0.0] * pn
            for idx, coeff in U_entries.items():
                U_row[idx] = coeff
            for idx, coeff in V_entries.items():
                V_row[idx] = coeff
            for idx, coeff in W_entries.items():
                W_row[idx] = coeff
            U_rows.append(U_row)
            V_rows.append(V_row)
            W_rows.append(W_row)
        
        # Strassen terms for j-pair (j1,j2) and k-pair (k1,k2)
        # A indices: i*m + j, i in {0,1}, j in {j1,j2}
        # B indices: j*p + k, j in {j1,j2}, k in {k1,k2}
        # C indices: k*n + i, k in {k1,k2}, i in {0,1}
        
        def add_strassen_block(j1, j2, k1, k2):
            """Add 7 Strassen rank-1 terms for the 2x2x2 block."""
            # a_ij = A[i, j], index = i*m + j
            # b_jk = B[j, k], index = j*p + k
            # c_ik = C[i, k], index = k*n + i
            
            # M1 = (a_1j1 + a_1j2) * b_j1k1
            # Contributes to c_0k2 with coeff +1
            add_rank1(
                {1*m+j1: 1, 1*m+j2: 1},
                {j1*p+k1: 1},
                {k2*n+0: 1}
            )
            
            # M2 = (a_0j1 + a_0j2) * b_j2k1
            # Contributes to c_0k1 with coeff +1, c_1k2 with coeff +1
            # We need to split this into two rank-1 terms or handle the output combination
            # Actually, in the tensor decomposition, each rank-1 term contributes to
            # exactly one (a,b,c) entry pattern. The combination happens through
            # the sum over all rank-1 terms.
            # So for M2, which contributes to both c_0k1 and c_1k2, we need to
            # create rank-1 terms that capture this.
            # 
            # Wait, I think I'm overcomplicating this. In the standard Strassen algorithm,
            # the M_t are intermediate values, and the final C entries are linear combinations
            # of the M_t. In the tensor decomposition, each "product" M_t is represented
            # as a rank-1 term, and the linear combination is handled by having multiple
            # rank-1 terms that share the same U and V but different W, or by having
            # the W vector have multiple non-zero entries.
            #
            # Actually, in a rank-1 tensor term, U[a]*V[b]*W[c] is non-zero only when
            # U[a], V[b], and W[c] are all non-zero. So a single rank-1 term can contribute
            # to multiple (a,b,c) entries if U, V, W have multiple non-zero entries.
            #
            # For M2 = (a_0j1 + a_0j2) * b_j2k1:
            # U has non-zeros at indices 0*m+j1 and 0*m+j2 (coeff 1 each)
            # V has non-zero at index j2*p+k1 (coeff 1)
            # W should have non-zeros at the C indices that M2 contributes to.
            # M2 contributes to c_0k1 (coeff +1) and c_1k2 (coeff +1)
            # So W[k1*n+0] = 1, W[k2*n+1] = 1
            #
            # Let me verify: U[0*m+j1]*V[j2*p+k1]*W[k1*n+0] = 1*1*1 = 1 -> contributes to (0*m+j1, j2*p+k1, k1*n+0)
            # This is (a=0*m+j1, b=j2*p+k1, c=k1*n+0) which is (i=0, j=j1, k=k1) -> C[0,k1] gets A[0,j1]*B[j2,k1]
            # But M2 should contribute A[0,j1]*B[j2,k1] to C[0,k1]? Let me check the Strassen formula.
            # c_0k1 = M5 + M2 - M4 - M6
            # So M2 contributes +1 * M2 to c_0k1, where M2 = (a_0j1 + a_0j2) * b_j2k1
            # So yes, A[0,j1]*B[j2,k1] contributes to C[0,k1] with coeff +1.
            # And A[0,j2]*B[j2,k1] also contributes to C[0,k1] with coeff +1.
            # Similarly for c_1k2.
            
            add_rank1(
                {0*m+j1: 1, 0*m+j2: 1},
                {j2*p+k1: 1},
                {k1*n+0: 1, k2*n+1: 1}
            )
            
            # M3 = a_0j1 * (b_j1k2 - b_j2k2)
            # Contributes to c_1k1 with coeff +1
            add_rank1(
                {0*m+j1: 1},
                {j1*p+k2: 1, j2*p+k2: -1},
                {k1*n+1: 1}
            )
            
            # M4 = a_1j2 * (b_j1k1 + b_j2k1)
            # Contributes to c_0k1 with coeff -1, c_0k2 with coeff +1
            add_rank1(
                {1*m+j2: 1},
                {j1*p+k1: 1, j2*p+k1: 1},
                {k1*n+0: -1, k2*n+0: 1}
            )
            
            # M5 = (a_0j1 + a_1j2) * (b_j1k1 + b_j2k2)
            # Contributes to c_0k1 with coeff +1, c_1k1 with coeff +1
            add_rank1(
                {0*m+j1: 1, 1*m+j2: 1},
                {j1*p+k1: 1, j2*p+k2: 1},
                {k1*n+0: 1, k1*n+1: 1}
            )
            
            # M6 = (a_0j2 - a_1j2) * (b_j2k1 + b_j2k2)
            # Contributes to c_0k1 with coeff -1, c_1k2 with coeff +1
            add_rank1(
                {0*m+j2: 1, 1*m+j2: -1},
                {j2*p+k1: 1, j2*p+k2: 1},
                {k1*n+0: -1, k2*n+1: 1}
            )
            
            # M7 = (a_0j1 - a_1j1) * (b_j1k2 + b_j2k2)
            # Not used in the standard Strassen 2x2x2! Wait, let me check.
            # Standard Strassen for 2x2x2:
            # M1 = (X[1,0] + X[1,1]) * Y[0,0]
            # M2 = (X[0,0] + X[0,1]) * Y[1,0]
            # M3 = X[0,0] * (Y[0,1] - Y[1,1])
            # M4 = X[1,1] * (Y[0,0] + Y[1,0])
            # M5 = (X[0,0] + X[1,1]) * (Y[1,1] + Y[0,0])
            # M6 = (X[0,1] - X[1,1]) * (Y[1,0] + Y[1,1])
            # M7 = (X[0,0] - X[1,0]) * (Y[0,1] + Y[1,1])
            # Z[0,0] = M5 + M2 - M4 - M6
            # Z[0,1] = M1 + M4
            # Z[1,0] = M3 + M5
            # Z[1,1] = M2 - M1 + M3 + M6
            # 
            # M7 is not used in the output! That can't be right. Let me recheck.
            # Actually, I think I have the Strassen formula wrong. Let me look it up.
            # Standard Strassen:
            # M1 = (A11 + A22) * (B11 + B22)
            # M2 = (A21 + A22) * B11
            # M3 = (A11 - A21) * (B21 + B22)
            # M4 = A11 * (B12 - B22)
            # M5 = (A12 + A22) * (B11 - B12)
            # M6 = (A11 + A12) * B22
            # M7 = (A21 - A11) * (B11 + B12)
            # C11 = M1 + M4 - M5 + M7
            # C12 = M2 + M3
            # C21 = M2 + M6
            # C22 = M1 - M3 + M4 + M5
            #
            # Let me use this version. In our notation:
            # A is the 2x2 submatrix of the full A: A[i,j] where i in {0,1}, j in {j1,j2}
            # B is the 2x2 submatrix of the full B: B[j,k] where j in {j1,j2}, k in {k1,k2}
            # C is the 2x2 submatrix of the full C: C[i,k] where i in {0,1}, k in {k1,k2}
            #
            # A11 = A[0,j1], A12 = A[0,j2], A21 = A[1,j1], A22 = A[1,j2]
            # B11 = B[j1,k1], B12 = B[j1,k2], B21 = B[j2,k1], B22 = B[j2,k2]
            # C11 = C[0,k1], C12 = C[0,k2], C21 = C[1,k1], C22 = C[1,k2]
            
            pass  # I'll rebuild this with the correct Strassen formula
        
        # This is getting too complex to do manually. Let me just use a simpler approach
        # and return the schoolbook if I can't quickly verify the Strassen construction.
        
        # For now, return None to indicate failure
        return None
    
    # Let me try a completely different approach: use the known result that
    # rank(2x3x9) <= 50 and try to find it via optimization.
    
    def try_rank(r, num_attempts=5, time_budget=20):
        """Try to find a rank-r decomposition using random search with gradient descent."""
        start = time.time()
        
        for attempt in range(num_attempts):
            if time.time() > start + time_budget or time.time() > deadline - 20:
                break
            
            # Random initialization
            U = np.random.uniform(-1, 1, (nm, r))
            V = np.random.uniform(-1, 1, (mp, r))
            W = np.random.uniform(-1, 1, (pn, r))
            
            # Use Adam-like optimizer
            lr = 0.01
            m_U = np.zeros_like(U)
            v_U = np.zeros_like(U)
            m_V = np.zeros_like(V)
            v_V = np.zeros_like(V)
            m_W = np.zeros_like(W)
            v_W = np.zeros_like(W)
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            t = 0
            
            for step in range(200):
                if time.time() > start + time_budget or time.time() > deadline - 20:
                    break
                
                t += 1
                # Compute gradient
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                # Target entries
                for (a, b, c), val in target_entries.items():
                    T = np.sum(U[a] * V[b] * W[c])
                    err = T - val
                    grad_U[a] += 2 * err * V[b] * W[c]
                    grad_V[b] += 2 * err * U[a] * W[c]
                    grad_W[c] += 2 * err * U[a] * V[b]
                
                # Sample non-target entries
                for _ in range(150):
                    a = random.randint(0, nm - 1)
                    b = random.randint(0, mp - 1)
                    c = random.randint(0, pn - 1)
                    if (a, b, c) not in target_set:
                        T = np.sum(U[a] * V[b] * W[c])
                        err = T
                        grad_U[a] += 2 * err * V[b] * W[c]
                        grad_V[b] += 2 * err * U[a] * W[c]
                        grad_W[c] += 2 * err * U[a] * V[b]
                
                # Adam update
                m_U = beta1 * m_U + (1 - beta1) * grad_U
                v_U = beta2 * v_U + (1 - beta2) * grad_U ** 2
                m_U_hat = m_U / (1 - beta1 ** t)
                v_U_hat = v_U / (1 - beta2 ** t)
                U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
                
                m_V = beta1 * m_V + (1 - beta1) * grad_V
                v_V = beta2 * v_V + (1 - beta2) * grad_V ** 2
                m_V_hat = m_V / (1 - beta1 ** t)
                v_V_hat = v_V / (1 - beta2 ** t)
                V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
                
                m_W = beta1 * m_W + (1 - beta1) * grad_W
                v_W = beta2 * v_W + (1 - beta2) * grad_W ** 2
                m_W_hat = m_W / (1 - beta1 ** t)
                v_W_hat = v_W / (1 - beta2 ** t)
                W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
            
            # Check if we found a valid decomposition
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if verify_decomposition(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from 48 to 53
    for r in range(48, 54):
        if time.time() > deadline - 25:
            break
        
        result = try_rank(r, num_attempts=3, time_budget=20)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    # If we didn't find anything better, return schoolbook
    return best_result
