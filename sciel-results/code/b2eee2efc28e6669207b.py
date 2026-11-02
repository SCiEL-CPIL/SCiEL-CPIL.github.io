import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 6]

def _verify(U, V, W, n, m, p):
    """
    Verify tensor decomposition identity:
    sum_l U[a*m+b, l] * V[c*p+d, l] * W[k*n+i, l] == 1 if (a==i and c==k and b==d) else 0
    """
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    T = np.einsum('abl,cdl,kil->abcdki', U3, V3, W3)
    
    target = np.zeros((n, m, m, p, p, n), dtype=float)
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    if b == d and c < p:
                        target[a, b, c, d, c, a] = 1.0
    
    return np.allclose(T, target, atol=1e-9)

def _build_target(n, m, p):
    T = np.zeros((n, m, m, p, p, n), dtype=float)
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    if b == d and c < p:
                        T[a, b, c, d, c, a] = 1.0
    return T

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook decomposition as guaranteed fallback
    schoolbook_rank = n * m * p  # 36
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = schoolbook_rank
    
    # Strategy: Construct a rank-22 decomposition for 2x3x6.
    # The 2x3x6 tensor T[a,b,c,d,k,i] = delta(a,i)*delta(c,k)*delta(b,d)
    # 
    # Key insight: We can write this as a sum over "intermediate" structures.
    # For 2x3x3, the rank is 11. For 2x3x6, we can't just double it to 22
    # because the U factor is shared. But we CAN use a structured construction.
    #
    # Let me try a direct algebraic construction for rank 22.
    # 
    # Approach: Use the fact that C = A*B where A is 2x3, B is 3x6, C is 2x6.
    # C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # We need: sum_l U[a,m+b,l] * V[c,p+d,l] * W[k*n+i,l] = delta(a,i)*delta(c,k)*delta(b,d)
    #
    # Let's try to build this with rank 22 using a specific pattern.
    # 
    # One known approach: For 2x3x6, use the "block" structure.
    # Split B into two 3x3 blocks: B = [B1 | B2]
    # C = A*B1 + A*B2 (block-wise)
    # C1 = A*B1 (2x3), C2 = A*B2 (2x3)
    #
    # If we had rank-11 for 2x3x3, we could use 22 total.
    # Let me try to find a rank-11 2x3x3 decomposition using a smarter search.
    
    # Build target for 2x3x3
    n2, m2, p2 = 2, 3, 3
    target_233 = _build_target(n2, m2, p2)
    
    # Try to find rank-11 decomposition for 2x3x3 using structured search
    # The 18 schoolbook terms: (a,b,c) for a in {0,1}, b in {0,1,2}, c in {0,1,2}
    # U index: a*3+b, V index: c*3+b, W index: c*2+a
    #
    # To reduce rank, we look for linear combinations where:
    # A term can have U with multiple nonzeros, V with multiple nonzeros, W with multiple nonzeros
    # as long as all (u,v,w) triples are in the target support.
    
    # Target support for 2x3x3:
    # (a*3+b, c*3+b, c*2+a) for all a,b,c
    # a=0,b=0: (0, 0, 0), (0, 3, 2), (0, 6, 4)  -> wait
    # V index = c*p2 + b = c*3 + b
    # W index = c*n2 + a = c*2 + a
    # 
    # Let me list all 18 support points:
    support_233 = set()
    for a in range(2):
        for b in range(3):
            for c in range(3):
                u_idx = a * 3 + b
                v_idx = c * 3 + b
                w_idx = c * 2 + a
                support_233.add((u_idx, v_idx, w_idx))
    
    # Now, a rank-11 decomposition needs 11 atoms (u_l, v_l, w_l) such that
    # sum_l u_l ⊗ v_l ⊗ w_l = target
    # Each atom's support (set of (u,v,w) where all three are nonzero) must be subset of support_233.
    #
    # Let me try a systematic search with small integer coefficients.
    # Each atom: u has support in {0..5}, v in {0..8}, w in {0..5}
    # For the atom to be compatible, every (u_i, v_j, w_k) in the product of supports must be in support_233.
    
    # Precompute compatible atoms
    def _get_compatible_atoms(n, m, p, max_nnz=2):
        support = set()
        for a in range(n):
            for b in range(m):
                for c in range(m):
                    for d in range(p):
                        if b == d and c < p:
                            u_idx = a * m + b
                            v_idx = c * p + d
                            w_idx = c * n + a
                            support.add((u_idx, v_idx, w_idx))
        
        U_size = n * m
        V_size = m * p
        W_size = p * n
        
        atoms = []
        for u_nnz in range(1, max_nnz + 1):
            for u_pos in itertools.combinations(range(U_size), u_nnz):
                for u_signs in itertools.product([-1, 1], repeat=u_nnz):
                    u = np.zeros(U_size)
                    for pos, s in zip(u_pos, u_signs):
                        u[pos] = s
                    u_nz_set = set(u_pos)
                    
                    for v_nnz in range(1, max_nnz + 1):
                        for v_pos in itertools.combinations(range(V_size), v_nnz):
                            for v_signs in itertools.product([-1, 1], repeat=v_nnz):
                                v = np.zeros(V_size)
                                for pos, s in zip(v_pos, v_signs):
                                    v[pos] = s
                                v_nz_set = set(v_pos)
                                
                                # Check u-v compatibility
                                ok_uv = True
                                for ui in u_nz_set:
                                    for vi in v_nz_set:
                                        # Check if there exists any w such that (ui, vi, w) in support
                                        # Actually we need: for all w in w_nz_set, (ui, vi, w) in support
                                        # But we haven't picked w yet. Let's just check if (ui,vi,*) has any support.
                                        # Better: just check full triple later.
                                        pass
                                
                                for w_nnz in range(1, max_nnz + 1):
                                    for w_pos in itertools.combinations(range(W_size), w_nnz):
                                        for w_signs in itertools.product([-1, 1], repeat=w_nnz):
                                            w = np.zeros(W_size)
                                            for pos, s in zip(w_pos, w_signs):
                                                w[pos] = s
                                            w_nz_set = set(w_pos)
                                            
                                            # Full compatibility check
                                            ok = True
                                            for ui in u_nz_set:
                                                for vi in v_nz_set:
                                                    for wi in w_nz_set:
                                                        if (ui, vi, wi) not in support:
                                                            ok = False
                                                            break
                                                    if not ok:
                                                        break
                                                if not ok:
                                                    break
                                            if ok:
                                                atoms.append((u, v, w))
        return atoms
    
    # This might be too slow for the full 2x3x6. Let's do it for 2x3x3 first.
    if time.time() < deadline - 20:
        atoms_233 = _get_compatible_atoms(2, 3, 3, max_nnz=2)
        
        # Now search for 11 atoms whose sum equals target
        # Target flat: 324 entries, 18 nonzero
        target_flat_233 = target_233.flatten()
        
        # This is a hard combinatorial problem. Let me try a different approach:
        # Use a known algebraic construction.
        # 
        # For 2x3x3, one rank-11 construction:
        # Terms 1-3: handle b=0 column of A
        # Terms 4-6: handle b=1 column of A  
        # Terms 7-9: handle b=2 column of A
        # Terms 10-11: correction terms
        #
        # Actually, let me just try to verify if a specific known construction works.
        # 
        # Alternative: use rank 12 for 2x3x3 (schoolbook minus one dependency)
        # and see if we can get to rank 24 for 2x3x6, which is still better than 36.
        #
        # Let me try rank 24: two 2x3x3 blocks each with rank 12.
        # For 2x3x3, can we get rank 12?
        # Schoolbook is 18. We need 6 savings.
        
        # Let me try a direct construction for 2x3x6 with rank 24.
        # Use the "pairing" trick: pair output columns k and k+3.
        # For k=0,3: C[i,0] and C[i,3]
        # C[i,0] = sum_j A[i,j]*B[j,0]
        # C[i,3] = sum_j A[i,j]*B[j,3]
        # 
        # A term with U[a*3+b,l]=1, V[c*6+0,l]=1, V[c*6+3,l]=1 (wait, V can't have two nonzeros in the same term easily)
        # 
        # Actually, let me try a cleaner approach:
        # For each b (column of A / row of B), we have 6 products: A[i,b]*B[b,k] for i in {0,1}, k in {0..5}
        # That's 12 products per b, 36 total.
        # 
        # A single term can cover: U has nonzeros at (i1,b) and (i2,b) [2 rows], V has nonzeros at (b,k1) and (b,k2) [2 cols], W has nonzeros at (k1,i1),(k1,i2),(k2,i1),(k2,i2) [4 outputs]
        # This covers 4 products: A[i1,b]*B[b,k1], A[i1,b]*B[b,k2], A[i2,b]*B[b,k1], A[i2,b]*B[b,k2]
        # 
        # With 9 such terms (3 per b), we cover all 36 products with rank 9? No, that's too optimistic.
        # Each term covers 4 products, 9 terms cover 36. But we need the coefficients to work out.
        # 
        # Let me try: for each b, use 3 terms to cover the 12 products (2 rows x 6 cols).
        # Term 1: U[(0,b)]=1, U[(1,b)]=1, V[(b,0)]=1, V[(b,1)]=1, W[0*2+0]=1, W[0*2+1]=1, W[1*2+0]=1, W[1*2+1]=1
        # This gives: A[0,b]*B[b,0]*1 + A[0,b]*B[b,1]*1 + A[1,b]*B[b,0]*1 + A[1,b]*B[b,1]*1
        # We need: A[0,b]*B[b,0] -> C[0,0], A[0,b]*B[b,1] -> C[0,1], A[1,b]*B[b,0] -> C[1,0], A[1,b]*B[b,1] -> C[1,1]
        # W indices: k*n+i = k*2+i
        # C[0,0]: k=0,i=0 -> W[0]
        # C[0,1]: k=1,i=0 -> W[2]
        # C[1,0]: k=0,i=1 -> W[1]
        # C[1,1]: k=1,i=1 -> W[3]
        # So W should have nonzeros at 0,2,1,3 with values 1,1,1,1. That's W[0]=1,W[1]=1,W[2]=1,W[3]=1.
        # 
        # Term 2: U[(0,b)]=1, U[(1,b)]=1, V[(b,2)]=1, V[(b,3)]=1, W[4]=1,W[5]=1,W[6]=1,W[7]=1
        # C[0,2]: k=2,i=0 -> W[4], C[0,3]: k=3,i=0 -> W[6], C[1,2]: k=2,i=1 -> W[5], C[1,3]: k=3,i=1 -> W[7]
        # 
        # Term 3: U[(0,b)]=1, U[(1,b)]=1, V[(b,4)]=1, V[(b,5)]=1, W[8]=1,W[9]=1,W[10]=1,W[11]=1
        # C[0,4]: k=4,i=0 -> W[8], C[0,5]: k=5,i=0 -> W[10], C[1,4]: k=4,i=1 -> W[9], C[1,5]: k=5,i=1 -> W[11]
        # 
        # So for each b, 3 terms cover all 12 products. Total 9 terms for all 36 products!
        # But wait, this gives rank 9, which is way too low. Let me check.
        # 
        # The issue: all 9 terms have the SAME U pattern (both rows nonzero) and the W patterns don't interfere.
        # Let me verify: the sum of all 9 terms gives:
        # For each (a,b,c,d,k,i): sum over terms where U[a*3+b]≠0, V[c*6+d]≠0, W[k*2+i]≠0
        # 
        # For a=0, b=0, c=0, d=0, k=0, i=0:
        # Term b=0, pair(0,1): U[0*3+0]=1, V[0*6+0]=1, W[0*2+0]=1 -> contributes 1
        # Any other terms? No. So T[0,0,0,0,0,0] = 1. Correct!
        # 
        # For a=0, b=0, c=0, d=0, k=1, i=0:
        # Term b=0, pair(0,1): U[0]=1, V[0*6+0]=1, W[1*2+0]=W[2]=1 -> contributes 1
        # But we need T[0,0,0,0,1,0] = 0 (since d=0≠k=1... wait, the target is delta(b,d)*delta(c,k)*delta(a,i))
        # b=0, d=0: delta=1. c=0, k=1: delta=0. So target=0. But we got 1. WRONG!
        # 
        # Ah, I see the issue. The V factor has V[c*p+d] where c is the row of B (which must equal k for the delta) and d is the column of B (which must equal b).
        # In my construction, V[(b,0)]=1 means V[c=0, d=0] when b=0. But V is indexed by (c,d) where c is row of B, d is col of B.
        # For the product A[a,b]*B[c,d] to contribute to C[k,i], we need c=k and d=b and a=i.
        # So V should have nonzeros at (c=k, d=b) positions.
        # 
        # In my term for b=0, pair(0,1): V[(0,0)]=1 and V[(0,1)]=1. This means c=0,d=0 and c=0,d=1.
        # But we need d=b=0 for both! So V should only have V[(0,0)]=1, not V[(0,1)]=1.
        # 
        # I made an error. V index is c*p+d where c is row of B, d is col of B.
        # For a given b (col of A = row of B in the contraction), we need V[c*p+b] to be nonzero.
        # So V nonzeros must be at positions c*6+b for various c.
        # 
        # Let me redo: For b=0:
        # V nonzeros at: 0*6+0=0, 1*6+0=6, 2*6+0=12
        # 
        # Term 1: U[0*3+0]=1, U[1*3+0]=1, V[0*6+0]=1, V[1*6+0]=1, W[0*2+0]=1, W[1*2+0]=1, W[0*2+1]=1, W[1*2+1]=1
        # Wait, W[k*n+i]: for k=0: W[0*2+0]=W[0], W[0*2+1]=W[1]; for k=1: W[1*2+0]=W[2], W[1*2+1]=W[3]
        # 
        # This term contributes to:
        # (a=0,b=0,c=0,d=0,k=0,i=0): U[0]=1,V[0]=1,W[0]=1 -> 1. Target: delta(0,0)*delta(0,0)*delta(0,0)=1. OK
        # (a=0,b=0,c=0,d=0,k=0,i=1): U[0]=1,V[0]=1,W[1]=1 -> 1. Target: delta(0,0)*delta(0,0)*delta(0,1)=0. WRONG!
        # 
        # So this doesn't work either. The W factor couples k and i, so we can't just set all W entries to 1.
        # 
        # Let me go back to basics. The schoolbook has 36 terms, one for each (a,b,c) with U[a*3+b]=1, V[c*6+b]=1, W[c*2+a]=1.
        # 
        # To reduce rank, we need to find linear dependencies among these 36 terms.
        # 
        # Let me try a different approach: use the known result that 2x3x6 has rank 22,
        # and construct it via a verified algebraic method.
        # 
        # Since finding it from scratch is hard, let me try rank 24 (two blocks of rank 12 for 2x3x3).
        # For 2x3x3, can we get rank 12?
        
        # Let me try to verify a rank-12 construction for 2x3x3.
        # 18 schoolbook terms. Find 6 linear dependencies.
        
        # Actually, let me just use the schoolbook rank 36 and try to improve by a different method.
        # 
        # New idea: use the "transpose" trick. 
        # C = A*B, A is 2x3, B is 3x6, C is 2x6.
        # C^T = B^T * A^T, B^T is 6x3, A^T is 3x2, C^T is 6x2.
        # This is a 6x3 * 3x2 multiplication.
        # 
        # The tensor for 6x3*3x2: T'[a',b',c',d',k',i'] where a',b' index B^T (6x3), c',d' index A^T (3x2), k',i' index C^T (6x2).
        # T'[a',b',c',d',k',i'] = delta(a',k')*delta(b',c')*delta(d',i')
        # 
        # This is the same tensor, just with different index conventions. The rank is the same.
        # 
        # Let me try yet another approach: use a pre-computed rank-22 decomposition.
        # I'll try to construct it using the following ansatz:
        # 22 terms, where some terms have 2 nonzeros in U, V, or W.
        
        # Given the difficulty, let me just return the schoolbook rank 36.
        # But the inspirations show rank 34 is achievable. Let me try to find a rank-34 construction.
        # 
        # For rank 34, we need to save 2 ranks from the 36 schoolbook terms.
        # 
        # One way: find 2 linear dependencies among the 36 terms.
        # 
        # Let me check if there's a simple dependency.
        # Consider terms with the same (a,b) but different c.
        # For fixed a,b: terms are (a,b,c) for c=0,1,2,3,4,5.
        # U[a*3+b] is the same for all 6 terms.
        # V[c*6+b] differs.
        # W[c*2+a] differs.
        # 
        # No obvious linear dependency here.
        
        # Let me just return schoolbook for now.
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
