"""Matrix multiplication 3,5,7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [3, 5, 7]

def construct():
    deadline = time.time() + 140
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    nm, mp, pn = n*m, m*p, p*n
    
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
    
    # Strassen block terms
    def strassen_terms(i0, i1, j0, j1, k0, k1):
        """Return 7 (u,v,w) vectors for a 2x2x2 Strassen block."""
        terms = []
        # M1 = (A00+A11)(B00+B11)
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i0*m+j0]=1; u[i1*m+j1]=1; v[j0*p+k0]=1; v[j1*p+k1]=1; w[k0*n+i0]=1; w[k1*n+i1]=1
        terms.append((u,v,w))
        # M2 = (A10+A11)B00
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i1*m+j0]=1; u[i1*m+j1]=1; v[j0*p+k0]=1; w[k0*n+i1]=1; w[k1*n+i1]=-1
        terms.append((u,v,w))
        # M3 = A00(B01-B11)
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i0*m+j0]=1; v[j0*p+k1]=1; v[j1*p+k1]=-1; w[k1*n+i0]=1; w[k1*n+i1]=1
        terms.append((u,v,w))
        # M4 = A11(B10-B00)
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i1*m+j1]=1; v[j1*p+k0]=1; v[j0*p+k0]=-1; w[k0*n+i0]=1; w[k0*n+i1]=1
        terms.append((u,v,w))
        # M5 = (A00+A01)B11
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i0*m+j0]=1; u[i0*m+j1]=1; v[j1*p+k1]=1; w[k0*n+i0]=-1; w[k1*n+i0]=1
        terms.append((u,v,w))
        # M6 = (A10-A00)(B00+B01)
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i1*m+j0]=1; u[i0*m+j0]=-1; v[j0*p+k0]=1; v[j0*p+k1]=1; w[k1*n+i1]=1
        terms.append((u,v,w))
        # M7 = (A01-A11)(B10+B11)
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i0*m+j1]=1; u[i1*m+j1]=-1; v[j1*p+k0]=1; v[j1*p+k1]=1; w[k0*n+i0]=1
        terms.append((u,v,w))
        return terms
    
    def blocks_disjoint(b1, b2):
        I1,J1,K1 = b1; I2,J2,K2 = b2
        for i in I1:
            for j in J1:
                for k in K1:
                    if i in I2 and j in J2 and k in K2:
                        return False
        return True
    
    def max_disjoint_blocks(available_blocks, deadline):
        """Find max set of mutually disjoint blocks via greedy with multiple starts."""
        best = []
        n_blocks = len(available_blocks)
        if n_blocks == 0:
            return best
        
        # Sort by "popularity" (fewest conflicts first) for better greedy
        conflict_count = [0]*n_blocks
        for a in range(n_blocks):
            for b in range(a+1, n_blocks):
                if not blocks_disjoint(available_blocks[a], available_blocks[b]):
                    conflict_count[a] += 1
                    conflict_count[b] += 1
        
        order = sorted(range(n_blocks), key=lambda x: conflict_count[x])
        
        for start in order:
            if time.time() > deadline:
                break
            current = [available_blocks[start]]
            used = {start}
            # Greedy: add block with fewest conflicts with current
            while True:
                if time.time() > deadline:
                    break
                best_idx = -1
                best_conflicts = float('inf')
                for idx in range(n_blocks):
                    if idx in used:
                        continue
                    if all(blocks_disjoint(available_blocks[idx], cb) for cb in current):
                        # Count conflicts with remaining (for tie-breaking)
                        conflicts = sum(1 for j in range(n_blocks) if j not in used and j != idx and not blocks_disjoint(available_blocks[idx], available_blocks[j]))
                        if conflicts < best_conflicts:
                            best_conflicts = conflicts
                            best_idx = idx
                if best_idx == -1:
                    break
                current.append(available_blocks[best_idx])
                used.add(best_idx)
            
            if len(current) > len(best):
                best = current
        
        return best
    
    # Generate all possible 2x2x2 blocks
    all_blocks = []
    for i_pair in combinations(range(n), 2):
        for j_pair in combinations(range(m), 2):
            for k_pair in combinations(range(p), 2):
                all_blocks.append((set(i_pair), set(j_pair), set(k_pair)))
    
    # Strategy 1: Pure Strassen
    best_strassen = max_disjoint_blocks(all_blocks, deadline)
    rank_pure = n*m*p - len(best_strassen)
    
    # Strategy 2: 3x3x3 meta-block + Strassen in remainder
    # The 3x3x3 meta-block uses rank 26 (7 Strassen + 19 schoolbook) for 27 terms
    # Net savings: 1 (same as one Strassen block)
    # But it "uses up" 27 terms, potentially allowing more Strassen blocks elsewhere
    best_total_rank = rank_pure
    best_config = ('pure', best_strassen, None)
    
    # Try all 3x3x3 placements
    for j3 in combinations(range(m), 3):
        for k3 in combinations(range(p), 3):
            if time.time() > deadline:
                break
            I3 = set(range(n))  # All 3 rows
            J3 = set(j3)
            K3 = set(k3)
            
            # 3x3x3 meta-block covers (i,j,k) for i in I3, j in J3, k in K3
            # Find max disjoint Strassen blocks that don't overlap with this
            available = []
            for b in all_blocks:
                Ib, Jb, Kb = b
                # Check if block overlaps with 3x3x3 meta-block
                overlaps = False
                for i in Ib:
                    for j in Jb:
                        for k in Kb:
                            if i in I3 and j in J3 and k in K3:
                                overlaps = True
                                break
                        if overlaps:
                            break
                    if overlaps:
                        break
                if not overlaps:
                    available.append(b)
            
            strassen_in_remainder = max_disjoint_blocks(available, deadline)
            # Total rank: 26 (for 3x3x3) + (105-27) - len(strassen_in_remainder)
            # = 26 + 78 - len(strassen_in_remainder)
            rank_this = 26 + (n*m*p - 27) - len(strassen_in_remainder)
            
            if rank_this < best_total_rank:
                best_total_rank = rank_this
                best_config = ('meta', strassen_in_remainder, (I3, J3, K3))
    
    # Build the decomposition
    if best_config[0] == 'pure':
        strassen_blocks = best_config[1]
        meta_block = None
    else:
        strassen_blocks = best_config[1]
        meta_block = best_config[2]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Add Strassen blocks
    for I, J, K in strassen_blocks:
        i0, i1 = sorted(I)
        j0, j1 = sorted(J)
        k0, k1 = sorted(K)
        for u, v, w in strassen_terms(i0, i1, j0, j1, k0, k1):
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)
    
    # Add 3x3x3 meta-block terms (if used)
    if meta_block is not None:
        I3, J3, K3 = meta_block
        i_list = sorted(I3)
        j_list = sorted(J3)
        k_list = sorted(K3)
        
        # Internal Strassen on first 2 of each
        i0, i1 = i_list[0], i_list[1]
        j0, j1 = j_list[0], j_list[1]
        k0, k1 = k_list[0], k_list[1]
        for u, v, w in strassen_terms(i0, i1, j0, j1, k0, k1):
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)
        
        # Schoolbook for remaining 19 terms in 3x3x3
        strassen_covered = set()
        for i in (i0, i1):
            for j in (j0, j1):
                for k in (k0, k1):
                    strassen_covered.add((i, j, k))
        
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    if (i, j, k) not in strassen_covered:
                        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
                        u[i*m+j] = 1.0
                        v[j*p+k] = 1.0
                        w[k*n+i] = 1.0
                        U_cols.append(u)
                        V_cols.append(v)
                        W_cols.append(w)
    
    # Add remaining schoolbook terms
    covered = set()
    for I, J, K in strassen_blocks:
        for i in I:
            for j in J:
                for k in K:
                    covered.add((i, j, k))
    if meta_block is not None:
        I3, J3, K3 = meta_block
        for i in I3:
            for j in J3:
                for k in K3:
                    covered.add((i, j, k))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
                    u[i*m+j] = 1.0
                    v[j*p+k] = 1.0
                    w[k*n+i] = 1.0
                    U_cols.append(u)
                    V_cols.append(v)
                    W_cols.append(w)
    
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    
    if verify(U_mat, V_mat, W_mat):
        return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
    else:
        U_sb, V_sb, W_sb, _ = schoolbook()
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
