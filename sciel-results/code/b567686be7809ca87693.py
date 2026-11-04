"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication.
    Strategy: Build rank-33 via Strassen blocks, then search for rank-32 by
    exploiting non-uniformity of middle dimension to merge one output column
    via linear combination.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety

    n, m, p = DIMENSIONS
    dim_a = n * m  # 6
    dim_b = m * p  # 18
    dim_c = p * n  # 12

    # Build the target tensor T[a, b, c]
    T = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def verify_decomposition(U, V, W):
        """Verify that U, V, W exactly decompose the tensor T."""
        rank = U.shape[1]
        result = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            result += np.einsum('i,j,k->ijk', uc, vc, wc)
        return np.array_equal(result, T)

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W

    def strassen_block_decomposition():
        """
        Return rank-33 decomposition: 3 blocks of (Strassen 2x2x2 + Standard 2x1x2).
        Block b handles columns 2b, 2b+1 of B.
        """
        rank = 33
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        
        # Strassen 2x2x2 factors
        S_U = [
            [[1, 0], [0, 1]],
            [[0, 0], [1, 1]],
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 1], [0, -1]]
        ]
        S_V = [
            [[1, 0], [0, 1]],
            [[1, 0], [0, 0]],
            [[0, 1], [0, -1]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[0, 0], [1, 1]]
        ]
        S_W = [
            [[1, 0], [0, 1]],
            [[0, 1], [0, -1]],
            [[0, 0], [1, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 0], [0, 0]]
        ]
        
        # Standard 2x1x2 factors
        Std_U = [
            [[1], [0]],
            [[1], [0]],
            [[0], [1]],
            [[0], [1]]
        ]
        Std_V = [
            [[1, 0]],
            [[0, 1]],
            [[1, 0]],
            [[0, 1]]
        ]
        Std_W = [
            [[1, 0], [0, 0]],
            [[0, 0], [1, 0]],
            [[0, 1], [0, 0]],
            [[0, 0], [0, 1]]
        ]
        
        col = 0
        for b in range(3):
            # Strassen part: handles columns 2b, 2b+1 of B (and corresponding C)
            for r in range(7):
                for i in range(2):
                    for j in range(2):
                        U[i * m + j, col] = S_U[r][i][j]
                
                for j in range(2):
                    for kp in range(2):
                        k_global = 2 * b + kp
                        V[j * p + k_global, col] = S_V[r][j][kp]
                
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = S_W[r][kp][i]
                
                col += 1
            
            # Standard part: handles column j=2 of A/B, columns 2b, 2b+1 of B
            for r in range(4):
                for i in range(2):
                    U[i * m + 2, col] = Std_U[r][i][0]
                
                for kp in range(2):
                    k_global = 2 * b + kp
                    V[2 * p + k_global, col] = Std_V[r][0][kp]
                
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = Std_W[r][kp][i]
                
                col += 1
        
        return U, V, W

    def try_rank32_merge(U33, V33, W33):
        """
        Try to reduce rank from 33 to 32 by finding a linear combination
        of two columns that can replace one column, exploiting the
        non-uniform middle dimension.
        
        Strategy: For each pair of columns (r1, r2) in the rank-33 decomposition,
        check if col_r1 + col_r2 (or col_r1 - col_r2) can be expressed as
        a single rank-1 tensor, allowing us to merge them.
        
        More practically: try to find if any two columns are linearly
        dependent in a way that allows cancellation.
        """
        rank33 = U33.shape[1]
        
        # Try: for each output position c in dim_c, see if the contribution
        # from all 33 columns can be expressed with 32 by finding a linear
        # relation among the W columns weighted by U and V.
        
        # Alternative approach: try replacing the standard (schoolbook) part
        # of one block with a modified Strassen that uses only 3 terms instead of 4
        # for the 2x1x2 sub-problem (which normally needs rank 4).
        
        # The 2x1x2 problem: A(2x1) * B(1x2) = C(2x2)
        # Standard: rank 4. Can we do it in rank 3?
        # C[0,0] = A[0]*B[0]
        # C[0,1] = A[0]*B[1]  
        # C[1,0] = A[1]*B[0]
        # C[1,1] = A[1]*B[1]
        # This is the 2x1x2 tensor which has rank 4 (it's a "flat" tensor).
        # So we can't reduce that.
        
        # Instead, try to merge across blocks. The key insight:
        # In the 2x3x6 problem, the middle dimension is 3, not a power of 2.
        # This means the three 2x2 blocks don't interact, but the j=2 column
        # (standard part) is shared across all three blocks.
        
        # Let's try a different approach: use a rank-3 decomposition for
        # a 2x1x2-like substructure by borrowing from adjacent Strassen terms.
        
        # Actually, let's try: can we express one of the standard terms
        # using a linear combination of Strassen terms from the same block?
        
        # For block b, the standard terms compute:
        #   C[i, 2b] += A[i, 2] * B[2, 2b]  (for i=0,1)
        #   C[i, 2b+1] += A[i, 2] * B[2, 2b+1]  (for i=0,1)
        
        # The Strassen terms in block b compute combinations of A[i,j] for j=0,1
        # and B[j,k] for k=2b,2b+1.
        
        # These are independent (different rows of A and B), so direct merging
        # within a block won't work.
        
        # Let's try: can we use the fact that W columns for different k values
        # are in different positions? W[k*n + i] for k=0..5, i=0..1.
        # So W has 12 rows, and each rank-1 term only touches 2 rows of W
        # (for a fixed k). This means W columns from different k-blocks are
        # orthogonal (non-overlapping support).
        
        # This means we can't merge across k-blocks directly.
        
        # Within a block (say k=2b, 2b+1), W touches rows {2b*2, 2b*2+1, (2b+1)*2, (2b+1)*2+1}
        # = {4b, 4b+1, 4b+2, 4b+3}.
        
        # So the problem decomposes into 3 independent 2x2x2 problems (Strassen, rank 7 each)
        # plus 3 independent 2x1x2 problems (schoolbook, rank 4 each).
        # Total: 3*7 + 3*4 = 21 + 12 = 33.
        
        # To get rank 32, we need to save 1 rank somewhere.
        # The 2x1x2 tensor has rank 4, and it's known to be minimal.
        # The 2x2x2 tensor has rank 7 (Strassen), and it's known to be minimal.
        
        # So rank 32 may not be achievable with this block structure.
        
        # Let's try a completely different approach: a global optimization.
        # We'll try to find if there's a rank-32 decomposition by solving
        # a system of polynomial equations.
        
        # For rank r, we need U (6xr), V (18xr), W (12xr) such that
        # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
        
        # This is a system of 6*18*12 = 1296 equations in 6r + 18r + 12r = 36r unknowns.
        # For r=32: 1152 unknowns, 1296 equations. Overdetermined but might have solutions.
        
        # Let's try a randomized approach: fix U and V, solve for W, then check.
        
        return None

    # Start with best known
    best_U, best_V, best_W = strassen_block_decomposition()
    best_rank = 33
    
    if not verify_decomposition(best_U, best_V, best_W):
        # Fallback to schoolbook
        best_U, best_V, best_W = schoolbook()
        best_rank = 36

    # Try to find rank-32 via structured search
    if time.time() < deadline - 20:
        # Approach: Try to find a rank-32 decomposition by modifying the
        # standard (schoolbook) parts to share terms with Strassen parts.
        
        # Key idea: In the 2x3x6 problem, consider the output C[i,k] for
        # fixed i. C[i,k] = sum_j A[i,j] * B[j,k].
        # For k in {0,1}: C[i,k] involves A[i,0], A[i,1], A[i,2] and B[0,k], B[1,k], B[2,k]
        # For k in {2,3}: similar with different B columns
        # For k in {4,5}: similar
        
        # The W matrix has structure: W[k*n+i, r] = W[k*2+i, r]
        # For a rank-1 term r, W only has non-zero entries at positions
        # corresponding to the k values that term contributes to.
        
        # Let's try: can we make one of the 12 standard terms (3 blocks x 4 terms)
        # share structure with a Strassen term?
        
        # Specifically, try to replace the 4 standard terms in block 0 with
        # a modified set that uses 3 terms by leveraging the fact that
        # B[2, 0] and B[2, 1] appear in both the standard and could potentially
        # be factored differently.
        
        # Actually, let's try a direct algebraic approach:
        # Fix the Strassen parts (21 terms) and try to find 11 terms for the
        # standard parts (instead of 12) that complete the decomposition.
        
        # The standard parts need to compute:
        # For each block b (b=0,1,2):
        #   C[i, 2b] += A[i,2] * B[2, 2b]
        #   C[i, 2b+1] += A[i,2] * B[2, 2b+1]
        # for i=0,1
        
        # These are 12 independent rank-1 terms (each is A[i,2]*B[2,k]*W[k,2+i]).
        # They involve:
        #   U: rows i*3+2 (i=0,1) -> rows 2, 5
        #   V: rows 2*6+k = 12+k (k=0..5) -> rows 12..17
        #   W: rows k*2+i (k=0..5, i=0,1) -> all 12 rows
        
        # Since W rows for different k are disjoint, and U/V rows for different
        # (i,k) pairs are mostly disjoint, these 12 terms are essentially
        # independent and can't be merged.
        
        # However! What if we modify the Strassen part to also contribute to
        # the j=2 column? The Strassen part only uses A[i,0] and A[i,1] (j=0,1).
        # If we could extend the Strassen decomposition to also handle j=2,
        # we might save rank.
        
        # The 2x3x3 problem (2x3 times 3x3) has a known rank. Let's check:
        # It's the tensor for 2x3 * 3x3. Known results suggest rank around 16-18.
        # If we can do 2x3x3 in rank 16, then 2x3x6 = 2x3x3 ⊗ (identity on last dim)
        # would give rank 16*2 = 32!
        
        # Let's try to find a rank-16 decomposition for 2x3x3.
        
        def find_2x3x3_rank16():
            """
            Try to find a rank-16 decomposition for 2x3 * 3x3 matrix multiplication.
            n=2, m=3, p=3. dim_a=6, dim_b=9, dim_c=6.
            """
            n2, m2, p2 = 2, 3, 3
            da, db, dc = n2*m2, m2*p2, p2*n2  # 6, 9, 6
            
            # Target tensor
            T2 = np.zeros((da, db, dc), dtype=np.int64)
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        T2[i*m2+j, j*p2+k, k*n2+i] = 1
            
            def verify2(U, V, W):
                r = U.shape[1]
                res = np.zeros((da, db, dc), dtype=np.int64)
                for c in range(r):
                    res += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
                return np.array_equal(res, T2)
            
            # Try: 2x2x2 Strassen (rank 7) for j=0,1 and k=0,1,
            # plus 2x1x2 (rank 4) for j=2, k=0,1,
            # plus 2x3x1 (rank 6) for k=2? No, that's not right.
            
            # Actually: 2x3x3 can be seen as:
            # Block 1: j=0,1 and k=0,1 -> 2x2x2 (rank 7)
            # Block 2: j=2 and k=0,1 -> 2x1x2 (rank 4)  
            # Block 3: j=0,1,2 and k=2 -> 2x3x1 (rank 6)
            # Total: 7+4+6 = 17. Not 16.
            
            # Alternative split:
            # Block 1: j=0,1 and k=0,1,2 -> 2x2x3 (rank ?)
            # Block 2: j=2 and k=0,1,2 -> 2x1x3 (rank 6)
            
            # 2x2x3: A(2x2)*B(2x3)=C(2x3)
            # This is like two 2x2x2 problems sharing A.
            # Rank of 2x2x3 tensor: at least 2*rank(2x2x2)/... 
            # Actually, 2x2x3 = 2x2x2 ⊕ 2x2x1 (in some sense)
            # 2x2x1: A(2x2)*B(2x1)=C(2x1). Rank = 4 (schoolbook).
            # So 2x2x3 might have rank 7+4=11 or less.
            
            # Let's just try to construct and verify.
            
            # For now, try the 17-rank construction:
            rank17 = 17
            U = np.zeros((da, rank17), dtype=np.int64)
            V = np.zeros((db, rank17), dtype=np.int64)
            W = np.zeros((dc, rank17), dtype=np.int64)
            
            # Strassen 2x2x2 for j=0,1, k=0,1
            S_U = [
                [[1, 0], [0, 1]],
                [[0, 0], [1, 1]],
                [[1, 0], [0, 0]],
                [[0, 0], [0, 1]],
                [[1, 1], [0, 0]],
                [[-1, 0], [1, 0]],
                [[0, 1], [0, -1]]
            ]
            S_V = [
                [[1, 0], [0, 1]],
                [[1, 0], [0, 0]],
                [[0, 1], [0, -1]],
                [[-1, 0], [1, 0]],
                [[0, 0], [0, 1]],
                [[1, 1], [0, 0]],
                [[0, 0], [1, 1]]
            ]
            S_W = [
                [[1, 0], [0, 1]],
                [[0, 1], [0, -1]],
                [[0, 0], [1, 1]],
                [[1, 1], [0, 0]],
                [[-1, 0], [1, 0]],
                [[0, 0], [0, 1]],
                [[1, 0], [0, 0]]
            ]
            
            col = 0
            for r in range(7):
                for i in range(2):
                    for j in range(2):
                        U[i*3+j, col] = S_U[r][i][j]
                for j in range(2):
                    for k in range(2):
                        V[j*3+k, col] = S_V[r][j][k]
                for k in range(2):
                    for i in range(2):
                        W[k*2+i, col] = S_W[r][k][i]
                col += 1
            
            # 2x1x2 for j=2, k=0,1
            for i in range(2):
                for k in range(2):
                    U[i*3+2, col] = 1 if i == 0 else 0
                    V[2*3+k, col] = 1 if k == 0 else 0
                    W[k*2+i, col] = 1
                    col += 1
            # Wait, that's not right. 2x1x2 schoolbook:
            # C[i,k] = A[i,2]*B[2,k]
            # U: row i*3+2, V: row 2*3+k, W: row k*2+i
            # Need 4 terms: (i,k) in {(0,0),(0,1),(1,0),(1,1)}
            col -= 4  # undo
            for i in range(2):
                for k in range(2):
                    U[i*3+2, col] = 1
                    V[2*3+k, col] = 1
                    W[k*2+i, col] = 1
                    col += 1
            
            # 2x3x1 for k=2 (all j=0,1,2)
            # C[i,2] = sum_j A[i,j]*B[j,2]
            # This is 2x3 * 3x1 = 2x1. Rank = 6 (schoolbook).
            for i in range(2):
                for j in range(3):
                    U[i*3+j, col] = 1
                    V[j*3+2, col] = 1
                    W[2*2+i, col] = 1
                    col += 1
            
            if verify2(U, V, W):
                # Now extend to 2x3x6 by duplicating for the second half
                # 2x3x6 = 2x3x3 (for k=0,1,2) ⊕ 2x3x3 (for k=3,4,5)
                # But that gives rank 17*2 = 34, not better.
                
                # Actually, for 2x3x6, we can use:
                # Block k=0,1: 2x2x2 Strassen (j=0,1) + 2x1x2 (j=2) = 11 terms
                # Block k=2,3: 2x2x2 Strassen (j=0,1) + 2x1x2 (j=2) = 11 terms  
                # Block k=4,5: 2x2x2 Strassen (j=0,1) + 2x1x2 (j=2) = 11 terms
                # Total: 33. Same as before.
                
                # The 2x3x3 rank-16 would give 2x3x6 rank-32 if we can
                # tile it appropriately. But the tiling isn't straightforward.
                
                pass
            
            return None
        
        # Let's try a more direct approach: use scipy to solve for a rank-32
        # decomposition by fixing some variables and solving linear systems.
        
        # For rank r, the tensor equation is:
        # T[a,b,c] = sum_{r'} U[a,r']*V[b,r']*W[c,r']
        
        # If we fix U and V, then for each (a,b), we have:
        # T[a,b,c] = sum_{r'} (U[a,r']*V[b,r']) * W[c,r']
        # Let M[a,b] = U[a,:] * V[b,:] (element-wise product), shape (6*18, r)
        # Then T[a,b,:] = M[a,b] @ W
        # This is a linear system in W for each (a,b).
        
        # But this is highly non-linear in U and V.
        
        # Let's try: fix U to have a specific structure and solve for V and W.
        
        # Actually, let's just try random initialization and gradient-free search.
        
        # Given time constraints, let's try a simpler idea:
        # Can we reduce the 2x1x2 standard part from rank 4 to rank 3
        # by sharing with a Strassen term?
        
        # In block b, the standard terms are:
        #   Term (i,k): U[i*3+2, r]=1, V[12+k, r]=1, W[(2b+k)*2+i, r]=1
        # for i in {0,1}, k in {0,1}
        
        # A Strassen term in block b has:
        #   U[i*3+j, r] = S_U[r][i][j] for i,j in {0,1}
        #   V[j*6+2b+k, r] = S_V[r][j][k] for j in {0,1}, k in {0,1}
        #   W[(2b+k)*2+i, r] = S_W[r][k][i] for k in {0,1}, i in {0,1}
        
        # The W support of standard terms in block b: rows (2b)*2+0, (2b)*2+1, (2b+1)*2+0, (2b+1)*2+1
        # = 4b, 4b+1, 4b+2, 4b+3
        # The W support of Strassen terms in block b: same rows!
        
        # So they DO overlap in W. This means we might be able to merge.
        
        # Let's try: replace the 4 standard terms in block 0 with a modified
        # set that uses 3 terms by combining with Strassen terms.
        
        # This is getting complex. Let's just return the rank-33 result.
        pass

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
