"""Matrix multiplication 3,5,7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 130  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for c in range(rank):
            i = c // (m * p)
            j = (c // p) % m
            k = c % p
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        return U, V, W, rank

    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    
    # Verify schoolbook is correct
    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = 1.0
                    rhs = 0.0
                    for c in range(U.shape[1]):
                        rhs += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(lhs - rhs) > 1e-6:
                        return False
        return True

    # Use the 6 Strassen blocks (Rank 99) and then try to remove redundant terms
    # via local search. The key insight: some of the 105 tensor entries may be
    # linearly dependent given the Strassen structure, allowing rank reduction.
    
    # Let's try to search for a 7th block programmatically.
    
    blocks = [
        ({0, 1}, {0, 1}, {0, 1}),
        ({1, 2}, {0, 1}, {2, 3}),
        ({0, 2}, {0, 1}, {4, 5}),
        ({0, 1}, {2, 3}, {0, 1}),
        ({1, 2}, {2, 3}, {2, 3}),
        ({0, 2}, {2, 3}, {4, 5}),
    ]
    
    # Try alternative block partitions to see if we can find a better set
    # that allows more Strassen blocks or fewer schoolbook terms
    alternative_blocks_sets = [
        # Original 6 blocks
        [
            ({0, 1}, {0, 1}, {0, 1}),
            ({1, 2}, {0, 1}, {2, 3}),
            ({0, 2}, {0, 1}, {4, 5}),
            ({0, 1}, {2, 3}, {0, 1}),
            ({1, 2}, {2, 3}, {2, 3}),
            ({0, 2}, {2, 3}, {4, 5}),
        ],
        # Try different pairing strategies
        [
            ({0, 1}, {0, 1}, {0, 1}),
            ({0, 1}, {2, 3}, {2, 3}),
            ({1, 2}, {0, 1}, {0, 1}),
            ({1, 2}, {2, 3}, {4, 5}),
            ({0, 2}, {0, 1}, {2, 3}),
            ({0, 2}, {2, 3}, {0, 1}),
        ],
        [
            ({0, 1}, {0, 1}, {0, 1}),
            ({1, 2}, {2, 3}, {0, 1}),
            ({0, 2}, {0, 1}, {2, 3}),
            ({0, 1}, {2, 3}, {4, 5}),
            ({1, 2}, {0, 1}, {2, 3}),
            ({0, 2}, {2, 3}, {0, 1}),
        ],
    ]
    
    best_blocks = blocks
    best_rank = float('inf')
    
    for block_set in alternative_blocks_sets:
        # Check if blocks are pairwise disjoint
        valid = True
        for idx1 in range(len(block_set)):
            for idx2 in range(idx1 + 1, len(block_set)):
                I1, J1, K1 = block_set[idx1]
                I2, J2, K2 = block_set[idx2]
                for i in I1:
                    for j in J1:
                        for k in K1:
                            if i in I2 and j in J2 and k in K2:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if not valid: break
            if not valid: break
        
        if not valid:
            continue
        
        # Count how many terms this would use
        covered_count = 0
        for I, J, K in block_set:
            covered_count += len(I) * len(J) * len(K)
        
        schoolbook_terms = n * m * p - covered_count
        strassen_terms = 7 * len(block_set)
        total_rank = schoolbook_terms + strassen_terms
        
        if total_rank < best_rank:
            best_rank = total_rank
            best_blocks = block_set
    
    blocks = best_blocks
    
    covered = set()
    for I, J, K in blocks:
        for i in I:
            for j in J:
                for k in K:
                    covered.add((i, j, k))
                    
    # Search for a 7th block
    found_7th = None
    for I in [ {0,1}, {0,2}, {1,2} ]:
        for J in [ {0,1}, {0,2}, {0,3}, {0,4}, {1,2}, {1,3}, {1,4}, {2,3}, {2,4}, {3,4} ]:
            for K in [ {0,1}, {0,2}, {0,3}, {0,4}, {0,5}, {0,6}, {1,2}, {1,3}, {1,4}, {1,5}, {1,6}, {2,3}, {2,4}, {2,5}, {2,6}, {3,4}, {3,5}, {3,6}, {4,5}, {4,6}, {5,6} ]:
                new_block = (I, J, K)
                is_disjoint = True
                for i in I:
                    for j in J:
                        for k in K:
                            if (i, j, k) in covered:
                                is_disjoint = False
                                break
                        if not is_disjoint: break
                    if not is_disjoint: break
                if is_disjoint:
                    found_7th = new_block
                    break
            if found_7th: break
        if found_7th: break
        
    if found_7th:
        blocks.append(found_7th)
        for i in found_7th[0]:
            for j in found_7th[1]:
                for k in found_7th[2]:
                    covered.add((i, j, k))
                    
    # Construct the decomposition
    U = []
    V = []
    W = []
    
    def add_strassen_block(I, J, K):
        """Add 7 Strassen terms for a 2x2x2 block.
        
        Standard Strassen for C = A*B where:
        A = [[a00, a01], [a10, a11]]
        B = [[b00, b01], [b10, b11]]
        C = [[c00, c01], [c10, c11]]
        
        M1 = (a00+a11)(b00+b11)
        M2 = (a10+a11)b00
        M3 = a00(b10-b00)
        M4 = a11(b11-b01)
        M5 = (a00+a01)b11
        M6 = (a10-a00)(b00+b01)
        M7 = (a01-a11)(b10+b11)
        
        c00 = M1 + M4 - M5 + M7
        c01 = M3 + M5
        c10 = M2 + M4
        c11 = M1 - M2 + M3 + M6
        """
        i0, i1 = sorted(I)
        j0, j1 = sorted(J)
        k0, k1 = sorted(K)
        
        # M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
        u1 = np.zeros(n*m); v1 = np.zeros(m*p); w1 = np.zeros(p*n)
        u1[i0*m + j0] = 1.0; u1[i1*m + j1] = 1.0
        v1[j0*p + k0] = 1.0; v1[j1*p + k1] = 1.0
        # Contributes to c[i0,k0], c[i0,k1], c[i1,k0], c[i1,k1]
        w1[k0*n + i0] = 1.0   # c00 += M1
        w1[k1*n + i0] = 0.0   # c01: no direct M1
        w1[k0*n + i1] = 0.0   # c10: no direct M1
        w1[k1*n + i1] = 1.0   # c11 += M1
        U.append(u1); V.append(v1); W.append(w1)
        
        # M2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0]
        u2 = np.zeros(n*m); v2 = np.zeros(m*p); w2 = np.zeros(p*n)
        u2[i1*m + j0] = 1.0; u2[i1*m + j1] = 1.0
        v2[j0*p + k0] = 1.0
        # Contributes to c[i1,k0] and c[i1,k1]
        w2[k0*n + i1] = 1.0   # c10 += M2
        w2[k1*n + i1] = -1.0  # c11 -= M2
        U.append(u2); V.append(v2); W.append(w2)
        
        # M3 = A[i0,j0] * (B[j1,k0]-B[j0,k0])
        u3 = np.zeros(n*m); v3 = np.zeros(m*p); w3 = np.zeros(p*n)
        u3[i0*m + j0] = 1.0
        v3[j1*p + k0] = 1.0; v3[j0*p + k0] = -1.0
        # Contributes to c[i0,k0] and c[i0,k1]
        w3[k0*n + i0] = 1.0   # c00 += M3
        w3[k1*n + i0] = 1.0   # c01 += M3
        U.append(u3); V.append(v3); W.append(w3)
        
        # M4 = A[i1,j1] * (B[j1,k1]-B[j0,k1])
        u4 = np.zeros(n*m); v4 = np.zeros(m*p); w4 = np.zeros(p*n)
        u4[i1*m + j1] = 1.0
        v4[j1*p + k1] = 1.0; v4[j0*p + k1] = -1.0
        # Contributes to c[i0,k0] and c[i1,k0]
        w4[k0*n + i0] = 1.0   # c00 += M4
        w4[k0*n + i1] = 1.0   # c10 += M4
        U.append(u4); V.append(v4); W.append(w4)
        
        # M5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
        u5 = np.zeros(n*m); v5 = np.zeros(m*p); w5 = np.zeros(p*n)
        u5[i0*m + j0] = 1.0; u5[i0*m + j1] = 1.0
        v5[j1*p + k1] = 1.0
        # Contributes to c[i0,k0] and c[i0,k1]
        w5[k0*n + i0] = -1.0  # c00 -= M5
        w5[k1*n + i0] = 1.0   # c01 += M5
        U.append(u5); V.append(v5); W.append(w5)
        
        # M6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
        u6 = np.zeros(n*m); v6 = np.zeros(m*p); w6 = np.zeros(p*n)
        u6[i1*m + j0] = 1.0; u6[i0*m + j0] = -1.0
        v6[j0*p + k0] = 1.0; v6[j0*p + k1] = 1.0
        # Contributes to c[i1,k1]
        w6[k1*n + i1] = 1.0   # c11 += M6
        U.append(u6); V.append(v6); W.append(w6)
        
        # M7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
        u7 = np.zeros(n*m); v7 = np.zeros(m*p); w7 = np.zeros(p*n)
        u7[i0*m + j1] = 1.0; u7[i1*m + j1] = -1.0
        v7[j1*p + k0] = 1.0; v7[j1*p + k1] = 1.0
        # Contributes to c[i0,k0]
        w7[k0*n + i0] = 1.0   # c00 += M7
        U.append(u7); V.append(v7); W.append(w7)

    for I, J, K in blocks:
        add_strassen_block(I, J, K)
        
    # Add remaining schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    u = np.zeros(n*m)
                    v = np.zeros(m*p)
                    w = np.zeros(p*n)
                    u[i*m + j] = 1.0
                    v[j*p + k] = 1.0
                    w[k*n + i] = 1.0
                    U.append(u)
                    V.append(v)
                    W.append(w)
                    
    U_mat = np.array(U).T
    V_mat = np.array(V).T
    W_mat = np.array(W).T
    
    # Verify
    if verify(U_mat, V_mat, W_mat):
        # Try to remove redundant terms via local search
        # For each term, check if it can be expressed as a combination of others
        current_rank = U_mat.shape[1]
        
        # Build the tensor coefficient matrix
        # Each term c has vectors U[:,c], V[:,c], W[:,c]
        # The tensor entry (i,j,k) = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        
        # For each term, try to see if we can express it using others
        # This is equivalent to checking if the tensor is in the span of others
        
        removed = 0
        max_removal_attempts = 200
        attempts = 0
        
        while attempts < max_removal_attempts and time.time() < deadline - 10:
            attempts += 1
            rank = U_mat.shape[1]
            if rank <= 1:
                break
            
            # Try removing each term (in random order for diversity)
            np.random.seed(attempts)
            order = np.random.permutation(rank)
            
            found_removal = False
            for c in order:
                if time.time() > deadline - 15:
                    break
                
                # Try to express term c as linear combination of others
                # We need: U[:,c] * V[:,c] * W[:,c] = sum_{c' != c} alpha_{c'} * U[:,c'] * V[:,c'] * W[:,c']
                
                # This is a non-linear system, but we can try a greedy approach:
                # Check if removing term c and adjusting others still works
                
                # Save current state
                U_save = U_mat.copy()
                V_save = V_mat.copy()
                W_save = W_mat.copy()
                
                # Remove term c
                mask = np.ones(rank, dtype=bool)
                mask[c] = False
                U_red = U_mat[:, mask]
                V_red = V_mat[:, mask]
                W_red = W_mat[:, mask]
                
                # Check if the tensor without term c still matches (it won't, but we can try to fix)
                # Actually, let's just verify if term c is "redundant" by checking
                # if all entries it contributes to are already covered by others
                
                # For each (i,j,k) that term c contributes to, check if others also contribute
                u_c = U_mat[i_c * m + j_c, c] if False else None  # placeholder
                
                # Simpler: just try removing and verifying
                U_test = np.delete(U_mat, c, axis=1)
                V_test = np.delete(V_mat, c, axis=1)
                W_test = np.delete(W_mat, c, axis=1)
                
                if verify(U_test, V_test, W_test):
                    U_mat = U_test
                    V_mat = V_test
                    W_mat = W_test
                    removed += 1
                    found_removal = True
                    break
            
            if not found_removal:
                break
        
        if verify(U_mat, V_mat, W_mat):
            return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
        else:
            return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END
