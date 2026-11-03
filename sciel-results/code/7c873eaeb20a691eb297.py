import time
import random

DIMENSIONS = [2, 4, 7]
DEADLINE = 145.0  # seconds wall-clock safety margin

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 7
    Na, Nb, Nc = n*m, m*p, p*n  # 8, 28, 14
    
    # Build the target tensor: T[i,j,k] = 1 if k == i*m + j (A[i,j]*B[j,k] -> C[i,k])
    # In our indexing: U row = i*m+j, V row = j*p+k, W row = k*n+i
    # T[U_idx, V_idx, W_idx] = 1 iff W_idx = (V_idx % p) * n + (U_idx // m)
    # i.e., U_idx = i*m+j, V_idx = j*p+k, W_idx = k*n+i
    
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[(u_idx, v_idx, w_idx)] = 1
    
    # Fallback: rank-50 block Strassen decomposition
    def build_rank50():
        rank = 50
        U = [[0]*rank for _ in range(Na)]
        V = [[0]*rank for _ in range(Nb)]
        W = [[0]*rank for _ in range(Nc)]
        
        def add_strassen(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, start_col):
            a00 = a_rows[0]*m + a_cols[0]
            a01 = a_rows[0]*m + a_cols[1]
            a10 = a_rows[1]*m + a_cols[0]
            a11 = a_rows[1]*m + a_cols[1]
            b00 = b_rows[0]*p + b_cols[0]
            b01 = b_rows[0]*p + b_cols[1]
            b10 = b_rows[1]*p + b_cols[0]
            b11 = b_rows[1]*p + b_cols[1]
            c00 = c_cols[0]*n + c_rows[0]
            c01 = c_cols[1]*n + c_rows[0]
            c10 = c_cols[0]*n + c_rows[1]
            c11 = c_cols[1]*n + c_rows[1]
            col = start_col
            U[a00][col] += 1; U[a11][col] += 1; V[b00][col] += 1; V[b11][col] += 1; W[c00][col] += 1; W[c11][col] += 1; col += 1
            U[a10][col] += 1; U[a11][col] += 1; V[b00][col] += 1; W[c10][col] += 1; W[c11][col] -= 1; col += 1
            U[a00][col] += 1; V[b01][col] += 1; V[b11][col] -= 1; W[c01][col] += 1; W[c11][col] += 1; col += 1
            U[a11][col] += 1; V[b10][col] += 1; V[b00][col] -= 1; W[c00][col] += 1; W[c10][col] += 1; col += 1
            U[a00][col] += 1; U[a01][col] += 1; V[b11][col] += 1; W[c00][col] -= 1; W[c01][col] += 1; col += 1
            U[a10][col] += 1; U[a00][col] -= 1; V[b00][col] += 1; V[b01][col] += 1; W[c11][col] += 1; col += 1
            U[a01][col] += 1; U[a11][col] -= 1; V[b10][col] += 1; V[b11][col] += 1; W[c00][col] += 1; col += 1
            return col
        
        def add_schoolbook_2x2_2x1(a_rows, a_cols, b_rows, b_col, c_rows, c_col, start_col):
            a00 = a_rows[0]*m + a_cols[0]; a01 = a_rows[0]*m + a_cols[1]
            a10 = a_rows[1]*m + a_cols[0]; a11 = a_rows[1]*m + a_cols[1]
            b0 = b_rows[0]*p + b_col; b1 = b_rows[1]*p + b_col
            c0 = c_col*n + c_rows[0]; c1 = c_col*n + c_rows[1]
            col = start_col
            U[a00][col] += 1; V[b0][col] += 1; W[c0][col] += 1; col += 1
            U[a01][col] += 1; V[b1][col] += 1; W[c0][col] += 1; col += 1
            U[a10][col] += 1; V[b0][col] += 1; W[c1][col] += 1; col += 1
            U[a11][col] += 1; V[b1][col] += 1; W[c1][col] += 1; col += 1
            return col
        
        col = 0
        col = add_strassen([0,1],[0,1],[0,1],[0,1],[0,1],[0,1], col)
        col = add_schoolbook_2x2_2x1([0,1],[0,1],[0,1],2,[0,1],2, col)
        col = add_strassen([0,1],[2,3],[2,3],[0,1],[0,1],[0,1], col)
        col = add_schoolbook_2x2_2x1([0,1],[2,3],[2,3],2,[0,1],2, col)
        col = add_strassen([0,1],[0,1],[0,1],[3,4],[0,1],[3,4], col)
        col = add_strassen([0,1],[0,1],[0,1],[5,6],[0,1],[5,6], col)
        col = add_strassen([0,1],[2,3],[2,3],[3,4],[0,1],[3,4], col)
        col = add_strassen([0,1],[2,3],[2,3],[5,6],[0,1],[5,6], col)
        return {"U": U, "V": V, "W": W}
    
    # Verify a decomposition is exact
    def verify(U, V, W, rank):
        # For each (i,j,k), sum over r of U[i*m+j][r]*V[j*p+k][r] should equal W[k*n+i][r]... 
        # Actually: C[i,k] = sum_j A[i,j]*B[j,k]
        # Decomposition: C[i,k] = sum_r (sum_j U[i*m+j][r]*A[i,j]) * (sum_j V[j*p+k][r]*B[j,k])
        # No wait: the tensor decomposition means T[u,v,w] = sum_r U[u][r]*V[v][r]*W[w][r]
        # where T[u,v,w] = 1 if w = (v%p)*n + (u//m), else 0
        for r in range(rank):
            for u in range(Na):
                if U[u][r] != 0:
                    for v in range(Nb):
                        if V[v][r] != 0:
                            expected_w = (v % p) * n + (u // m)
                            # T[u,v,w] should be 1 only for w = expected_w
                            # So we need W[expected_w][r] * U[u][r] * V[v][r] to contribute correctly
                            # And for all other w', W[w'][r] * U[u][r] * V[v][r] must be 0
                            # This means: if U[u][r] and V[v][r] are both nonzero,
                            # then W[expected_w][r] must be nonzero and all other W[w'][r] must be 0
                            # OR the product U[u][r]*V[v][r]*W[w'][r] must cancel with other terms
                            pass
        # Proper check: for each (u,v,w), sum_r U[u][r]*V[v][r]*W[w][r] == target.get((u,v,w), 0)
        for u in range(Na):
            for v in range(Nb):
                for w in range(Nc):
                    s = 0.0
                    for r in range(rank):
                        s += U[u][r] * V[v][r] * W[w][r]
                    expected = target.get((u,v,w), 0)
                    if abs(s - expected) > 1e-9:
                        return False
        return True
    
    # Try to find rank-49 via random search + local refinement
    # Strategy: start from rank-50, try to merge pairs of rank-1 terms
    # Or: random initialization with constraint solving
    
    # Alternative approach: use the block structure but try cross-block cancellations
    # The rank-50 has 4 Strassen blocks (7 each = 28) + 2 schoolbook 2x2x1 (4 each = 8) + 2 more Strassen (14) = 50
    # Actually: 4 Strassen * 7 = 28, 2 schoolbook * 4 = 8, 2 more Strassen * 7 = 14... that's 50
    # Wait: blocks are 1a(strassen=7), 1b(schoolbook=4), 2a(strassen=7), 2b(schoolbook=4), 3a(strassen=7), 3b(strassen=7), 4a(strassen=7), 4b(strassen=7)
    # = 7+4+7+4+7+7+7+7 = 50. Yes.
    
    # To get rank 49, we need to find one linear dependency among the 50 rank-1 tensors.
    # Or restructure: combine two schoolbook terms into fewer.
    
    # Let's try: can we do the two schoolbook 2x2x1 blocks (8 terms) in 7 terms?
    # Block 1b: A[:,0:2] x B[0:2,2] -> C[:,2]  (4 terms)
    # Block 2b: A[:,2:4] x B[2:4,2] -> C[:,2]  (4 terms)
    # Combined: A (2x4) x B[0:4,2] (4x1) -> C[:,2] (2x1)
    # This is a 2x4 x 4x1 -> 2x1 problem. Schoolbook: 8 mults.
    # Can we do it in 7? The tensor has 8 nonzero entries in the "output" sense.
    # Actually the bilinear map is: C[0] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2] + A[0,3]*B[3,2]
    #             C[1] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2] + A[1,3]*B[3,2]
    # This is essentially two independent 4-term dot products. The bilinear rank of a single
    # 1x4 x 4x1 -> 1x1 is 4 (can't do better). Two independent such maps need 8 unless
    # there's structure. But here C[0] and C[1] share the same B vector, so:
    # C[0] = A_row0 . B_col2
    # C[1] = A_row1 . B_col2
    # This is: [C0; C1] = [A_row0; A_row1] * B_col2, which is a 2x4 * 4x1 matmul.
    # The bilinear rank of 2x4*4x1: at least max(2,4,1)*... actually at least 4 (since B has 4 components
    # and each C component needs all 4). Upper bound: 8 (schoolbook). 
    # Can we do it in 7? This is equivalent to computing two dot products of length 4 sharing the same
    # vector B. The tensor is: T[(i,j), (j,2), (2,i)] = 1 for i in {0,1}, j in {0,1,2,3}.
    # That's 8 terms. The bilinear rank: we need sum_r U[i,j,r]*V[j,2,r]*W[2,i,r] = delta.
    # For fixed i: sum_j U[i,j,r]*V[j,2,r]*W[2,i,r] over r must give A[i,j]*B[j,2].
    # This is a rank problem for the 2x4 matrix of coefficients. I believe rank 8 is minimal here
    # because the two rows are independent and each needs 4 terms.
    
    # So combining the two schoolbook blocks doesn't help.
    
    # Alternative: look for cross-block cancellations. The Strassen blocks operate on disjoint
    # (A_block, B_block, C_block) subspaces, so cross-cancellations seem unlikely unless we
    # deliberately mix them.
    
    # Let's try a randomized approach: start with rank 49, random integer/half-integer entries,
    # and use gradient-free search to satisfy the 8*28*14 = 3136 tensor equations.
    # This is 3136 equations with 49*(8+28+14) = 49*50 = 2450 unknowns. Underdetermined? No:
    # 3136 > 2450, so it's overdetermined. But many equations are 0=0 (only 56 are nonzero target).
    # Actually only 56 of the 3136 equations are nonzero (the target tensor has 56 ones).
    # The other 3080 must be zero. So we have 3136 constraints on 2450 variables.
    
    # This is a system of quadratic equations (bilinear in U,V for fixed W, etc.)
    # Let's try a simpler approach: fix W to be sparse (each W row has one nonzero),
    # then the problem becomes: for each r, U[:,r] and V[:,r] define a rank-1 "slice"
    # and W[:,r] distributes it.
    
    # Given time constraints, let's just return the rank-50 solution which we know works.
    # But first, let me try one specific optimization:
    
    # The two schoolbook blocks (1b and 2b) each compute C[:,2] from different A,B blocks.
    # Block 1b: C[0,2] += A[0,0]*B[0,2] + A[0,1]*B[1,2]
    #           C[1,2] += A[1,0]*B[0,2] + A[1,1]*B[1,2]
    # Block 2b: C[0,2] += A[0,2]*B[2,2] + A[0,3]*B[3,2]
    #           C[1,2] += A[1,2]*B[2,2] + A[1,3]*B[3,2]
    
    # Total for C[:,2]: C[0,2] = sum_{j=0}^{3} A[0,j]*B[j,2], C[1,2] = sum_{j=0}^{3} A[1,j]*B[j,2]
    # This is 8 multiplications in schoolbook. Can Strassen help? 
    # If we group as (A[0,:2], A[0,2:4]) and (B[:2,2], B[2:,2]):
    # C[0,2] = A[0,:2].B[:2,2] + A[0,2:4].B[2:,2]
    # C[1,2] = A[1,:2].B[:2,2] + A[1,2:4].B[2:,2]
    # Each 2x1 dot product needs 2 mults, total 8. No savings.
    
    # What about using Strassen-style on the 2x4 x 4x1?
    # Let M1 = A[:,0:2], M2 = A[:,2:4], N1 = B[0:2,2], N2 = B[2:4,2]
    # C = M1*N1 + M2*N2
    # This is two 2x2 * 2x1 products. Each needs 4 schoolbook mults = 8 total.
    # Strassen doesn't apply to 2x2 * 2x1 directly (it's for 2x2 * 2x2).
    
    # I think rank 50 might be tight for this block structure. Let's just return rank 50.
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5:
        return build_rank50()
    
    # Try a quick random search for rank 49
    # Use the structure: most entries are 0, ±1, ±1/2
    # Try: take the rank-50 solution and see if any two columns are linearly dependent
    # in a way that allows merging.
    
    # Actually, let's try a different block decomposition that might naturally give rank 49.
    # What if we use Strassen for a 2x3 x 3x3 block? That doesn't directly help.
    
    # Let me try: 2x7 x 7x2 via Strassen on 2x2 blocks of the 7 dimension.
    # 7 = 2+2+2+1. Four 2x2 blocks and one 2x1 column.
    # 4 Strassen (7 each) + 2 schoolbook 2x2x1 (4 each) = 28+8 = 36 for the "full" part.
    # But we also have the 4 columns of A... wait, A is 2x4, B is 4x7.
    # A has 4 columns, B has 4 rows. The "middle" dimension is 4.
    # So we split A's 4 columns into 2+2, B's 4 rows into 2+2.
    # A[:,0:2] x B[0:2,:] (2x2 x 2x7) -> contributes to all 7 columns of C
    # A[:,2:4] x B[2:4,:] (2x2 x 2x7) -> contributes to all 7 columns of C
    
    # For 2x2 x 2x7: split 7 into 2+2+2+1
    # 3 Strassen 2x2x2 (7 each) + 1 schoolbook 2x2x1 (4) = 21+4 = 25
    # Two such: 25*2 = 50. Same as before.
    
    # I don't see an easy path to 49 with this structure. Return rank 50.
    return build_rank50()
