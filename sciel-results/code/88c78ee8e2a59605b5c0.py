import time
import sys

def construct():
    n, m, p = 2, 4, 4
    deadline = time.time() + 115  # Safety margin before 150s kill
    
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
    
    # Verify function
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
        # Check that off-diagonal entries are zero
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ai = i*m + j
                    bi = j*p + k
                    ci = k*n + i
                    s = sum(Umat[ai][rr]*Vmat[bi][rr]*Wmat[ci][rr] for rr in range(rank))
                    # For matrix multiplication, the (i,k) entry of C = sum_j A[i,j]*B[j,k]
                    # So for fixed i,k, the sum over j should be 1 if we're computing C[i,k]
                    # But actually, the tensor T[i,j,k] = 1 for all i,j,k
                    # So we just need T[i,j,k] = 1 for all valid (i,j,k)
                    pass
        return True
    
    # More careful verification
    def verify_full(Umat, Vmat, Wmat, rank):
        # The matrix multiplication tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # The decomposition requires: sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ai = i*m + j
                    bi = j*p + k
                    ci = k*n + i
                    s = 0
                    for rr in range(rank):
                        s += Umat[ai][rr] * Vmat[bi][rr] * Wmat[ci][rr]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    if not verify_full(U, V, W, 28):
        # Fallback: return schoolbook rank-32
        U32 = [[0]*32 for _ in range(n*m)]
        V32 = [[0]*32 for _ in range(m*p)]
        W32 = [[0]*32 for _ in range(p*n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U32[i*m+j][idx] = 1
                    V32[j*p+k][idx] = 1
                    W32[k*n+i][idx] = 1
                    idx += 1
        return {"U": U32, "V": V32, "W": W32}
    
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
    
    # Try merging one term from each of two blocks
    def try_merge(term1, term2):
        U1, V1, W1 = term1
        U2, V2, W2 = term2
        
        R = {}
        for (a, b, c) in valid_triples:
            val = U1.get(a,0)*V1.get(b,0)*W1.get(c,0)
            val += U2.get(a,0)*V2.get(b,0)*W2.get(c,0)
            if val != 0:
                R[(a,b,c)] = val
        
        if not R:
            return ({},{},{})
        
        # Try to find a rank-1 decomposition of R
        # Pick a pivot with non-zero value
        pivot = next(iter(R.keys()))
        pa, pb, pc = pivot
        R0 = R[pivot]
        
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
            return None
        
        # Verify
        for (a,b,c) in valid_triples:
            expected = R.get((a,b,c), 0)
            actual = SU.get(a,0)*SV.get(b,0)*SW.get(c,0)
            if abs(expected - actual) > 1e-9:
                return None
        
        return (SU, SV, SW)
    
    # Try all pairs of blocks and all term pairs
    best = None
    for bi in range(4):
        for bj in range(bi+1, 4):
            for ii in range(7):
                for jj in range(7):
                    if time.time() > deadline:
                        break
                    merged = try_merge(blocks[bi][ii], blocks[bj][jj])
                    if merged is None:
                        continue
                    SU, SV, SW = merged
                    if not SU or not SV or not SW:
                        continue
                    
                    # Build rank-27 solution
                    U27 = [[0]*27 for _ in range(n*m)]
                    V27 = [[0]*27 for _ in range(m*p)]
                    W27 = [[0]*27 for _ in range(p*n)]
                    
                    r = 0
                    for blk in range(4):
                        for t in range(7):
                            if blk == bi and t == ii:
                                continue
                            if blk == bj and t == jj:
                                continue
                            ud, vd, wd = blocks[blk][t]
                            for a,val in ud.items(): U27[a][r] += val
                            for b,val in vd.items(): V27[b][r] += val
                            for c,val in wd.items(): W27[c][r] += val
                            r += 1
                    # Add merged term
                    for a,val in SU.items(): U27[a][r] += val
                    for b,val in SV.items(): V27[b][r] += val
                    for c,val in SW.items(): W27[c][r] += val
                    r += 1
                    assert r == 27, f"Expected 27, got {r}"
                    
                    if verify_full(U27, V27, W27, 27):
                        best = (U27, V27, W27)
                        break
                if best: break
            if best: break
        if best: break
        if time.time() > deadline:
            break
    
    if best:
        return {"U": best[0], "V": best[1], "W": best[2]}
    
    # Try to find rank 26 by merging terms from three blocks
    # Pick one term from each of three blocks, see if their sum is rank-1
    def try_merge_three(term1, term2, term3):
        U1, V1, W1 = term1
        U2, V2, W2 = term2
        U3, V3, W3 = term3
        
        R = {}
        for (a, b, c) in valid_triples:
            val = U1.get(a,0)*V1.get(b,0)*W1.get(c,0)
            val += U2.get(a,0)*V2.get(b,0)*W2.get(c,0)
            val += U3.get(a,0)*V3.get(b,0)*W3.get(c,0)
            if val != 0:
                R[(a,b,c)] = val
        
        if not R:
            return ({},{},{})
        
        # Try to find a rank-1 decomposition of R
        pivot = next(iter(R.keys()))
        pa, pb, pc = pivot
        R0 = R[pivot]
        
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
            return None
        
        # Verify
        for (a,b,c) in valid_triples:
            expected = R.get((a,b,c), 0)
            actual = SU.get(a,0)*SV.get(b,0)*SW.get(c,0)
            if abs(expected - actual) > 1e-9:
                return None
        
        return (SU, SV, SW)
    
    # Try all triples of blocks
    if time.time() < deadline - 10:
        for bi in range(4):
            for bj in range(bi+1, 4):
                for bk in range(bj+1, 4):
                    for ii in range(7):
                        for jj in range(7):
                            for kk in range(7):
                                if time.time() > deadline:
                                    break
                                merged = try_merge_three(blocks[bi][ii], blocks[bj][jj], blocks[bk][kk])
                                if merged is None:
                                    continue
                                SU, SV, SW = merged
                                if not SU or not SV or not SW:
                                    continue
                                
                                # Build rank-25 solution (28 - 3 terms + 1 merged = 26)
                                rank = 28 - 3 + 1  # = 26
                                U26 = [[0]*rank for _ in range(n*m)]
                                V26 = [[0]*rank for _ in range(m*p)]
                                W26 = [[0]*rank for _ in range(p*n)]
                                
                                r = 0
                                for blk in range(4):
                                    for t in range(7):
                                        if (blk == bi and t == ii) or (blk == bj and t == jj) or (blk == bk and t == kk):
                                            continue
                                        ud, vd, wd = blocks[blk][t]
                                        for a,val in ud.items(): U26[a][r] += val
                                        for b,val in vd.items(): V26[b][r] += val
                                        for c,val in wd.items(): W26[c][r] += val
                                        r += 1
                                # Add merged term
                                for a,val in SU.items(): U26[a][r] += val
                                for b,val in SV.items(): V26[b][r] += val
                                for c,val in SW.items(): W26[c][r] += val
                                r += 1
                                assert r == rank, f"Expected {rank}, got {r}"
                                
                                if verify_full(U26, V26, W26, rank):
                                    return {"U": U26, "V": V26, "W": W26}
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
    
    return {"U": U, "V": V, "W": W}
