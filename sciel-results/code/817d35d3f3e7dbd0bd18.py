"""Matrix multiplication 3,5,7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    nm, mp, pn = n*m, m*p, p*n

    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        for c in range(rank):
            i = c // (m * p)
            j = (c // p) % m
            k = c % p
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        return U, V, W, rank

    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    
    # Verify decomposition
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

    # Generate all possible 2x2x2 Strassen blocks
    # A block is defined by (I, J, K) where I is a pair of i indices, J is a pair of j indices, K is a pair of k indices
    all_blocks = []
    for i_pair in combinations(range(n), 2):
        for j_pair in combinations(range(m), 2):
            for k_pair in combinations(range(p), 2):
                all_blocks.append((set(i_pair), set(j_pair), set(k_pair)))
    
    # Greedy search: find maximum number of disjoint blocks
    # Two blocks are disjoint if they don't share any (i,j,k) triple
    def blocks_disjoint(block1, block2):
        I1, J1, K1 = block1
        I2, J2, K2 = block2
        # Check if any (i,j,k) is in both blocks
        for i in I1:
            for j in J1:
                for k in K1:
                    if i in I2 and j in J2 and k in K2:
                        return False
        return True
    
    def find_max_disjoint_blocks(blocks, max_blocks=None):
        """Greedy search for maximum number of mutually disjoint blocks."""
        if max_blocks is None:
            max_blocks = len(blocks)
        
        best = []
        remaining = list(blocks)
        
        # Try all possible starting blocks
        for start_block in remaining:
            current = [start_block]
            current_remaining = [b for b in remaining if b != start_block]
            
            # Greedily add blocks that are disjoint from all current blocks
            while current_remaining:
                if time.time() > deadline:
                    break
                added = False
                for idx, block in enumerate(current_remaining):
                    if all(blocks_disjoint(block, cb) for cb in current):
                        current.append(block)
                        current_remaining.pop(idx)
                        added = True
                        break
                if not added:
                    break
            
            if len(current) > len(best):
                best = current
                if len(best) >= max_blocks:
                    break
        
        return best
    
    # Find maximum number of disjoint Strassen blocks
    max_disjoint = find_max_disjoint_blocks(all_blocks)
    num_blocks = len(max_disjoint)
    
    # If we didn't find enough blocks, use a fallback set
    if num_blocks < 6:
        max_disjoint = [
            ({0, 1}, {0, 1}, {0, 1}),
            ({1, 2}, {0, 1}, {2, 3}),
            ({0, 2}, {0, 1}, {4, 5}),
            ({0, 1}, {2, 3}, {0, 1}),
            ({1, 2}, {2, 3}, {2, 3}),
            ({0, 2}, {2, 3}, {4, 5}),
        ]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_strassen_block(I, J, K):
        i0, i1 = sorted(I)
        j0, j1 = sorted(J)
        k0, k1 = sorted(K)
        
        terms = []
        
        # M1 = (A00+A11)(B00+B11)
        u1 = np.zeros(nm)
        v1 = np.zeros(mp)
        w1 = np.zeros(pn)
        u1[i0*m + j0] = 1.0
        u1[i1*m + j1] = 1.0
        v1[j0*p + k0] = 1.0
        v1[j1*p + k1] = 1.0
        w1[k0*n + i0] = 1.0
        w1[k1*n + i1] = 1.0
        terms.append((u1, v1, w1))
        
        # M2 = (A10+A11)B00
        u2 = np.zeros(nm)
        v2 = np.zeros(mp)
        w2 = np.zeros(pn)
        u2[i1*m + j0] = 1.0
        u2[i1*m + j1] = 1.0
        v2[j0*p + k0] = 1.0
        w2[k0*n + i1] = 1.0
        w2[k1*n + i1] = -1.0
        terms.append((u2, v2, w2))
        
        # M3 = A00(B01-B11)
        u3 = np.zeros(nm)
        v3 = np.zeros(mp)
        w3 = np.zeros(pn)
        u3[i0*m + j0] = 1.0
        v3[j0*p + k1] = 1.0
        v3[j1*p + k1] = -1.0
        w3[k1*n + i0] = 1.0
        w3[k1*n + i1] = 1.0
        terms.append((u3, v3, w3))
        
        # M4 = A11(B10-B00)
        u4 = np.zeros(nm)
        v4 = np.zeros(mp)
        w4 = np.zeros(pn)
        u4[i1*m + j1] = 1.0
        v4[j1*p + k0] = 1.0
        v4[j0*p + k0] = -1.0
        w4[k0*n + i0] = 1.0
        w4[k0*n + i1] = 1.0
        terms.append((u4, v4, w4))
        
        # M5 = (A00+A01)B11
        u5 = np.zeros(nm)
        v5 = np.zeros(mp)
        w5 = np.zeros(pn)
        u5[i0*m + j0] = 1.0
        u5[i0*m + j1] = 1.0
        v5[j1*p + k1] = 1.0
        w5[k0*n + i0] = -1.0
        w5[k1*n + i0] = 1.0
        terms.append((u5, v5, w5))
        
        # M6 = (A10-A00)(B00+B01)
        u6 = np.zeros(nm)
        v6 = np.zeros(mp)
        w6 = np.zeros(pn)
        u6[i1*m + j0] = 1.0
        u6[i0*m + j0] = -1.0
        v6[j0*p + k0] = 1.0
        v6[j0*p + k1] = 1.0
        w6[k1*n + i1] = 1.0
        terms.append((u6, v6, w6))
        
        # M7 = (A01-A11)(B10+B11)
        u7 = np.zeros(nm)
        v7 = np.zeros(mp)
        w7 = np.zeros(pn)
        u7[i0*m + j1] = 1.0
        u7[i1*m + j1] = -1.0
        v7[j1*p + k0] = 1.0
        v7[j1*p + k1] = 1.0
        w7[k0*n + i0] = 1.0
        terms.append((u7, v7, w7))
        
        for u, v, w in terms:
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)

    # Add Strassen blocks
    for I, J, K in max_disjoint:
        add_strassen_block(I, J, K)
        
    # Add remaining schoolbook terms
    covered = set()
    for I, J, K in max_disjoint:
        for i in I:
            for j in J:
                for k in K:
                    covered.add((i, j, k))
                    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    u = np.zeros(nm)
                    v = np.zeros(mp)
                    w = np.zeros(pn)
                    u[i*m + j] = 1.0
                    v[j*p + k] = 1.0
                    w[k*n + i] = 1.0
                    U_cols.append(u)
                    V_cols.append(v)
                    W_cols.append(w)
                    
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    
    if verify(U_mat, V_mat, W_mat):
        return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
    else:
        # Fallback to schoolbook
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END
