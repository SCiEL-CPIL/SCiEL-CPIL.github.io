import numpy as np
import time

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Try 3+2 / 3+3 split
    # j = {0,1,2} + {3,4}, k = {0,1,2} + {3,4,5}
    # Blocks: 2x3x3, 2x3x3, 2x2x3, 2x2x3
    
    # 2x2x3: Strassen(7) + schoolbook(4) = 11
    # 2x3x3: try rank 9, fallback to schoolbook(18)
    
    def strassen_222(ja, jb, ka, kb):
        # Returns list of (u_coeffs, v_coeffs, w_coeffs)
        terms = []
        ua, ub = ja, 5 + ja
        va, vb = jb, 5 + jb
        wka0, wka1 = 2*ka, 2*ka+1
        wkb0, wkb1 = 2*kb, 2*kb+1
        vka = ja*p + ka
        vkb = ja*p + kb
        vjb_ka = jb*p + ka
        vjb_kb = jb*p + kb
        
        terms.append(([(ua,1),(vb,1)], [(vka,1),(vjb_kb,1)], [(wka0,1),(wkb1,1)]))
        terms.append(([(ub,1),(vb,1)], [(vka,1)], [(wka1,1),(wkb1,-1)]))
        terms.append(([(ua,1)], [(vkb,1),(vjb_kb,-1)], [(wkb0,1),(wkb1,1)]))
        terms.append(([(vb,1)], [(vjb_ka,1),(vka,-1)], [(wka0,1),(wka1,1)]))
        terms.append(([(ua,1),(va,1)], [(vjb_kb,1)], [(wka0,-1),(wkb0,1)]))
        terms.append(([(ub,1),(ua,-1)], [(vka,1),(vkb,1)], [(wkb1,1)]))
        terms.append(([(va,1),(vb,-1)], [(vjb_ka,1),(vjb_kb,1)], [(wka0,1)]))
        return terms
    
    def schoolbook_221(ja, jb, ka):
        terms = []
        for i in range(n):
            for j in [ja, jb]:
                u_idx = i*m + j
                v_idx = j*p + ka
                w_idx = ka*n + i
                terms.append(([(u_idx,1)], [(v_idx,1)], [(w_idx,1)]))
        return terms
    
    def block_223(ja, jb, ka, kb, kc):
        terms = strassen_222(ja, jb, ka, kb)
        terms += schoolbook_221(ja, jb, kc)
        return terms
    
    # For 2x3x3, try to find rank-9 decomposition
    # A is 2x3, B is 3x3, C is 2x3
    # C[i,k] = sum_j A[i,j]*B[j,k]
    
    # Let me try a specific construction
    # Using the approach from the literature
    def block_233_rank9(ja, jb, jc, ka, kb, kc):
        # j indices: ja, jb, jc
        # k indices: ka, kb, kc
        # i indices: 0, 1
        u = lambda i, j: i*m + j
        v = lambda j, k: j*p + k
        w = lambda k, i: k*n + i
        
        terms = []
        # M1: (a00+a01+a02)*(b00+b10+b20) -> C[0,0]
        terms.append(([(u(0,ja),1),(u(0,jb),1),(u(0,jc),1)],
                     [(v(ja,ka),1),(v(jb,ka),1),(v(jc,ka),1)],
                     [(w(ka,0),1)]))
        # M2: (a10+a11+a12)*(b00+b10+b20) -> C[1,0]
        terms.append(([(u(1,ja),1),(u(1,jb),1),(u(1,jc),1)],
                     [(v(ja,ka),1),(v(jb,ka),1),(v(jc,ka),1)],
                     [(w(ka,1),1)]))
        # M3: (a00+a01)*(b01+b11) -> C[0,1]
        terms.append(([(u(0,ja),1),(u(0,jb),1)],
                     [(v(ja,kb),1),(v(jb,kb),1)],
                     [(w(kb,0),1)]))
        # M4: (a10+a11)*(b01+b11) -> C[1,1]
        terms.append(([(u(1,ja),1),(u(1,jb),1)],
                     [(v(ja,kb),1),(v(jb,kb),1)],
                     [(w(kb,1),1)]))
        # M5: (a00+a02)*(b02+b22) -> C[0,2]
        terms.append(([(u(0,ja),1),(u(0,jc),1)],
                     [(v(ja,kc),1),(v(jc,kc),1)],
                     [(w(kc,0),1)]))
        # M6: (a10+a12)*(b02+b22) -> C[1,2]
        terms.append(([(u(1,ja),1),(u(1,jc),1)],
                     [(v(ja,kc),1),(v(jc,kc),1)],
                     [(w(kc,1),1)]))
        # M7: (a01-a02)*(b12-b21) -> C[0,1], C[0,2]
        terms.append(([(u(0,jb),1),(u(0,jc),-1)],
                     [(v(jb,kc),1),(v(jc,kb),-1)],
                     [(w(kb,0),1),(w(kc,0),-1)]))
        # M8: (a11-a12)*(b12-b21) -> C[1,1], C[1,2]
        terms.append(([(u(1,jb),1),(u(1,jc),-1)],
                     [(v(jb,kc),1),(v(jc,kb),-1)],
                     [(w(kb,1),1),(w(kc,1),-1)]))
        # M9: (a00-a01+a02)*(b00-b10+b20) -> C[0,0]
        terms.append(([(u(0,ja),1),(u(0,jb),-1),(u(0,jc),1)],
                     [(v(ja,ka),1),(v(jb,ka),-1),(v(jc,ka),1)],
                     [(w(ka,0),1)]))
        return terms
    
    # Build the full decomposition
    all_terms = []
    # Block 2x3x3: j={0,1,2}, k={0,1,2}
    all_terms += block_233_rank9(0, 1, 2, 0, 1, 2)
    # Block 2x3x3: j={0,1,2}, k={3,4,5}
    all_terms += block_233_rank9(0, 1, 2, 3, 4, 5)
    # Block 2x2x3: j={3,4}, k={0,1,2}
    all_terms += block_223(3, 4, 0, 1, 2)
    # Block 2x2x3: j={3,4}, k={3,4,5}
    all_terms += block_223(3, 4, 3, 4, 5)
    
    rank = len(all_terms)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for r, (u_coeffs, v_coeffs, w_coeffs) in enumerate(all_terms):
        for idx, coeff in u_coeffs:
            U[idx, r] += coeff
        for idx, coeff in v_coeffs:
            V[idx, r] += coeff
        for idx, coeff in w_coeffs:
            W[idx, r] += coeff
    
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                result = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(result - 1) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook rank 60
        U = np.zeros((n*m, 60))
        V = np.zeros((m*p, 60))
        W = np.zeros((p*n, 60))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
