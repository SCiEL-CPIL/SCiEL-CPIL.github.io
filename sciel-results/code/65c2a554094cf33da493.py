import time
import sys
import random

def construct():
    n, m, p = 2, 4, 4
    deadline = time.time() + 115  # Safety margin
    
    # Build rank-28 Strassen solution
    rank28 = 28
    U = [[0]*rank28 for _ in range(n*m)]
    V = [[0]*rank28 for _ in range(m*p)]
    W = [[0]*rank28 for _ in range(p*n)]
    
    def add_strassen(Ar0, Ac0, Br0, Bc0, Cr0, Cc0, r):
        a11 = Ar0*m + Ac0
        a12 = Ar0*m + Ac0 + 1
        a21 = (Ar0+1)*m + Ac0
        a22 = (Ar0+1)*m + Ac0 + 1
        b11 = Br0*p + Bc0
        b12 = Br0*p + Bc0 + 1
        b21 = (Br0+1)*p + Bc0
        b22 = (Br0+1)*p + Bc0 + 1
        c11 = Cc0*n + Cr0
        c12 = (Cc0+1)*n + Cr0
        c21 = Cc0*n + (Cr0+1)
        c22 = (Cc0+1)*n + (Cr0+1)
        
        # M1 = (A11+A22)(B11+B22) -> C11, C22
        U[a11][r] += 1; U[a22][r] += 1
        V[b11][r] += 1; V[b22][r] += 1
        W[c11][r] += 1; W[c22][r] += 1
        r += 1
        # M2 = (A21+A22)B11 -> C21, C22
        U[a21][r] += 1; U[a22][r] += 1
        V[b11][r] += 1
        W[c21][r] += 1; W[c22][r] += -1
        r += 1
        # M3 = A11(B12-B22) -> C12, C22
        U[a11][r] += 1
        V[b12][r] += 1; V[b22][r] += -1
        W[c12][r] += 1; W[c22][r] += 1
        r += 1
        # M4 = A22(B21-B11) -> C11, C21
        U[a22][r] += 1
        V[b21][r] += 1; V[b11][r] += -1
        W[c11][r] += 1; W[c21][r] += 1
        r += 1
        # M5 = (A11+A12)B22 -> C11, C12
        U[a11][r] += 1; U[a12][r] += 1
        V[b22][r] += 1
        W[c11][r] += -1; W[c12][r] += 1
        r += 1
        # M6 = (A21-A11)(B11+B12) -> C22
        U[a21][r] += 1; U[a11][r] += -1
        V[b11][r] += 1; V[b12][r] += 1
        W[c22][r] += 1
        r += 1
        # M7 = (A12-A22)(B21+B22) -> C11
        U[a12][r] += 1; U[a22][r] += -1
        V[b21][r] += 1; V[b22][r] += 1
        W[c11][r] += 1
        r += 1
        return r
    
    r = add_strassen(0, 0, 0, 0, 0, 0, 0)    # P1: A_L @ B_LL -> C_L
    r = add_strassen(0, 2, 2, 0, 0, 0, r)    # P2: A_R @ B_RL -> C_L
    r = add_strassen(0, 0, 0, 2, 0, 2, r)    # P3: A_L @ B_LR -> C_R
    r = add_strassen(0, 2, 2, 2, 0, 2, r)    # P4: A_R @ B_RR -> C_R
    assert r == 28
    
    # Verify
    def verify(Umat, Vmat, Wmat, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ai = i*m + j
                    bi = j*p + k
                    ci = k*n + i
                    s = sum(Umat[ai][rr]*Vmat[bi][rr]*Wmat[ci][rr] for rr in range(rank))
                    if abs(s - 1) > 1e-9:
                        return False
                for j2 in range(m):
                    if j2 == j: continue
                    for k2 in range(p):
                        ai = i*m + j
                        bi = j2*p + k2
                        ci = k2*n + i
                        s = sum(Umat[ai][rr]*Vmat[bi][rr]*Wmat[ci][rr] for rr in range(rank))
                        if abs(s) > 1e-9:
                            return False
        return True
    
    if not verify(U, V, W, 28):
        # Fallback: return schoolbook
        U28 = [[0]*32 for _ in range(n*m)]
        V28 = [[0]*32 for _ in range(m*p)]
        W28 = [[0]*32 for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i*m*p + j*p + k
                    U28[i*m+j][idx] = 1
                    V28[j*p+k][idx] = 1
                    W28[k*n+i][idx] = 1
        return {"U": U28, "V": V28, "W": W28}
    
    # Try to find rank 27
    # Get individual Strassen terms for each block
    def get_terms(Ar0, Ac0, Br0, Bc0, Cr0, Cc0):
        a11 = Ar0*m + Ac0
        a12 = Ar0*m + Ac0 + 1
        a21 = (Ar0+1)*m + Ac0
        a22 = (Ar0+1)*m + Ac0 + 1
        b11 = Br0*p + Bc0
        b12 = Br0*p + Bc0 + 1
        b21 = (Br0+1)*p + Bc0
        b22 = (Br0+1)*p + Bc0 + 1
        c11 = Cc0*n + Cr0
        c12 = (Cc0+1)*n + Cr0
        c21 = Cc0*n + (Cr0+1)
        c22 = (Cc0+1)*n + (Cr0+1)
        
        terms = []
        terms.append(({a11:1, a22:1}, {b11:1, b22:1}, {c11:1, c22:1}))
        terms.append(({a21:1, a22:1}, {b11:1}, {c21:1, c22:-1}))
        terms.append(({a11:1}, {b12:1, b22:-1}, {c12:1, c22:1}))
        terms.append(({a22:1}, {b21:1, b11:-1}, {c11:1, c21:1}))
        terms.append(({a11:1, a12:1}, {b22:1}, {c11:-1, c12:1}))
        terms.append(({a21:1, a11:-1}, {b11:1, b12:1}, {c22:1}))
        terms.append(({a12:1, a22:-1}, {b21:1, b22:1}, {c11:1}))
        return terms
    
    P1 = get_terms(0, 0, 0, 0, 0, 0)
    P2 = get_terms(0, 2, 2, 0, 0, 0)
    P3 = get_terms(0, 0, 0, 2, 0, 2)
    P4 = get_terms(0, 2, 2, 2, 0, 2)
    blocks = [P1, P2, P3, P4]
    
    # All valid (a,b,c) triples
    valid_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid_triples.append((i*m+j, j*p+k, k*n+i))
    
    a_vals = sorted(set(t[0] for t in valid_triples))
    b_vals = sorted(set(t[1] for t in valid_triples))
    c_vals = sorted(set(t[2] for t in valid_triples))
    
    # Improved try_merge: try all non-zero pivots
    def try_merge(terms_list):
        # terms_list is a list of (U_dict, V_dict, W_dict)
        # Sum them up
        R = {}
        for U1, V1, W1 in terms_list:
            for (a, b, c) in valid_triples:
                val = U1.get(a,0)*V1.get(b,0)*W1.get(c,0)
                if val != 0:
                    R[(a,b,c)] = R.get((a,b,c), 0) + val
        
        if not R:
            return ({},{},{})
        
        # Try each non-zero entry as a pivot
        for pivot, R0 in R.items():
            pa, pb, pc = pivot
            if R0 == 0: continue
            
            SU = {}
            for a in a_vals:
                if (a, pb, pc) in R:
                    SU[a] = R[(a,pb,pc)]/R0
                else:
                    SU[a] = 0
            SU = {k:v for k,v in SU.items() if abs(v) > 1e-12}
            
            SV = {}
            for b in b_vals:
                if (pa, b, pc) in R:
                    SV[b] = R[(pa,b,pc)]/R0
                else:
                    SV[b] = 0
            SV = {k:v for k,v in SV.items() if abs(v) > 1e-12}
            
            SW = {}
            for c in c_vals:
                if (pa, pb, c) in R:
                    SW[c] = R[(pa,pb,c)]/R0
                else:
                    SW[c] = 0
            SW = {k:v for k,v in SW.items() if abs(v) > 1e-12}
            
            if not SU or not SV or not SW:
                continue
            
            # Verify
            valid = True
            for (a,b,c) in valid_triples:
                expected = R.get((a,b,c), 0)
                actual = SU.get(a,0)*SV.get(b,0)*SW.get(c,0)
                if abs(expected - actual) > 1e-9:
                    valid = False
                    break
            
            if valid:
                return (SU, SV, SW)
        
        return None

    # Build solution helper
    def build_solution(exclusions, merged_term, target_rank):
        U_sol = [[0]*target_rank for _ in range(n*m)]
        V_sol = [[0]*target_rank for _ in range(m*p)]
        W_sol = [[0]*target_rank for _ in range(p*n)]
        
        r = 0
        for blk in range(4):
            for t in range(7):
                if (blk, t) in exclusions:
                    continue
                ud, vd, wd = blocks[blk][t]
                for a,val in ud.items(): U_sol[a][r] += val
                for b,val in vd.items(): V_sol[b][r] += val
                for c,val in wd.items(): W_sol[c][r] += val
                r += 1
        
        if merged_term:
            SU, SV, SW = merged_term
            for a,val in SU.items(): U_sol[a][r] += val
            for b,val in SV.items(): V_sol[b][r] += val
            for c,val in SW.items(): W_sol[c][r] += val
            r += 1
        
        if r != target_rank:
            return None
            
        return (U_sol, V_sol, W_sol)

    # Try rank 27: Merge 2 terms into 1
    best = None
    for bi in range(4):
        for bj in range(bi, 4): # Allow same block
            for ii in range(7):
                for jj in range(7):
                    if bi == bj and ii == jj:
                        continue
                    if time.time() > deadline:
                        break
                    merged = try_merge([blocks[bi][ii], blocks[bj][jj]])
                    if merged is None:
                        continue
                    SU, SV, SW = merged
                    if not SU or not SV or not SW:
                        continue
                    
                    exclusions = {(bi, ii), (bj, jj)}
                    sol = build_solution(exclusions, merged, 27)
                    if sol:
                        U27, V27, W27 = sol
                        if verify(U27, V27, W27, 27):
                            best = (U27, V27, W27)
                            break
                if best: break
            if best: break
        if best: break
        if time.time() > deadline:
            break
    
    if best:
        return {"U": best[0], "V": best[1], "W": best[2]}
    
    # Try rank 26: Merge 3 terms into 1
    if time.time() < deadline - 10:
        # Sample some triplets to avoid combinatorial explosion
        # There are 28 terms. Choosing 3 is ~3276.
        # We can just iterate through all unique triplets of terms from different blocks first?
        # Or just random sampling.
        rng = random.Random(42)
        found = False
        # Generate all pairs of blocks and terms, then try to add a third
        # Actually, let's just iterate through all combinations of 3 terms
        # 28 terms, C(28,3) = 3276. This is small enough.
        term_indices = []
        for b in range(4):
            for t in range(7):
                term_indices.append((b, t))
                
        for idx1 in range(len(term_indices)):
            for idx2 in range(idx1+1, len(term_indices)):
                for idx3 in range(idx2+1, len(term_indices)):
                    if time.time() > deadline:
                        break
                    b1, t1 = term_indices[idx1]
                    b2, t2 = term_indices[idx2]
                    b3, t3 = term_indices[idx3]
                    
                    # Optional: skip if all from same block? No, allow it.
                    merged = try_merge([blocks[b1][t1], blocks[b2][t2], blocks[b3][t3]])
                    if merged is None:
                        continue
                    SU, SV, SW = merged
                    if not SU or not SV or not SW:
                        continue
                    
                    exclusions = {(b1, t1), (b2, t2), (b3, t3)}
                    sol = build_solution(exclusions, merged, 26)
                    if sol:
                        U26, V26, W26 = sol
                        if verify(U26, V26, W26, 26):
                            return {"U": U26, "V": V26, "W": W26}
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}
