import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    t0 = time.time()
    deadline = t0 + 110  # safety margin before 120s limit

    # M_{4,4,6} = M_{2,2,3} ⊗ M_{2,2,2}
    # Kronecker product of decompositions:
    # If (U1,V1,W1) is rank-r1 decomp of M_{2,2,3} and (U2,V2,W2) is rank-r2 decomp of M_{2,2,2}
    # Then U = U1⊗U2, V = V1⊗V2, W = W1⊗W2 gives rank r1*r2 decomp of M_{4,4,6}
    
    # For M_{2,2,2} (n=2,m=2,p=2), schoolbook rank-4:
    # U2[i*2+j, r] = 1 if r==i*2+j else 0
    # V2[j*2+k, r] = 1 if r==j*2+k else 0
    # W2[k*2+i, r] = 1 if r==k*2+i else 0
    # Check: sum_r U2[i*2+j,r]*V2[j*2+k,r]*W2[k*2+i,r] = 1 iff i*2+j == j*2+k == k*2+i
    # That's only true when i=j=k=0 or i=j=k=1... No, that's wrong.
    # Schoolbook: each term r corresponds to a specific (i,j,k) triple? No.
    # Actually schoolbook for 2x2x2x2: C[i,k] = sum_j A[i,j]B[j,k]
    # Term r picks one j: U picks A[i,j], V picks B[j,k], W picks C[i,k]
    # So U2[i*2+j, r] = 1 if r==j, V2[j*2+k, r] = 1 if r==j, W2[k*2+i, r] = 1 if r==j
    # Wait, that's only 2 terms for 2x2. But rank of 2x2x2x2 is 4.
    # Let me reconsider. The tensor T[i,j,k] = 1 for all i,j,k in [0,2).
    # We need sum_r U[i*2+j,r]*V[j*2+k,r]*W[k*2+i,r] = 1 for all i,j,k.
    # Schoolbook: 4 terms, one for each (i,k) pair? No...
    # Actually: C[i,k] = sum_j A[i,j]B[j,k]. Each j gives a rank-1 term.
    # Term j: U[i*2+j, r_j] = 1 for all i (i.e., U[0*2+j]=1, U[1*2+j]=1), 
    #          V[j*2+k, r_j] = 1 for all k, W[k*2+i, r_j] = 1 for all i,k
    # But that's not rank-1 in the tensor sense. 
    # 
    # Let me think again. The multiplication tensor M_{n,m,p} has entries T[i,j,k]=1.
    # A rank-1 decomposition term is u⊗v⊗w where u∈R^{nm}, v∈R^{mp}, w∈R^{pn}.
    # The product u[i*m+j]*v[j*p+k]*w[k*n+i] should sum to 1 over r.
    #
    # For schoolbook M_{2,2,2}: 4 terms.
    # Term 0: u=[1,0,0,0], v=[1,0,0,0], w=[1,0,0,0] → contributes 1 at (i=0,j=0,k=0)
    # Term 1: u=[0,1,0,0], v=[0,0,1,0], w=[0,1,0,0] → contributes 1 at (i=0,j=1,k=1)? 
    #   u[0*2+1]=1, v[1*2+1]=1, w[1*2+0]=1 → (i=0,j=1,k=1) ✓
    # Term 2: u=[0,0,1,0], v=[0,1,0,0], w=[0,0,1,0] → u[1*2+0]=1,v[0*2+1]=1,w[1*2+1]=1 → (i=1,j=0,k=1)
    #   But we need (i=1,j=0,k=0) and (i=1,j=0,k=1) both covered.
    # 
    # This approach isn't working cleanly. Let me just use the known fact that
    # the schoolbook algorithm uses n*m*p terms, but for 2x2x2 that's 8, not 4.
    # The rank of M_{2,2,2} is actually 4 (Strassen shows <8, and it's known to be 4).
    #
    # Let me use a verified rank-4 decomposition for M_{2,2,2}:
    # Based on the structure, I'll construct it numerically and verify.
    
    # Actually, let me take a different approach. I'll construct the full M_{4,4,6}
    # decomposition directly using a Kronecker product of two explicitly verified small decompositions.
    
    # Step 1: Build rank-4 decomposition of M_{2,2,2}
    # n2=2, m2=2, p2=2
    n2, m2, p2 = 2, 2, 2
    R2 = 4
    U2 = np.zeros((n2*m2, R2))
    V2 = np.zeros((m2*p2, R2))
    W2 = np.zeros((p2*n2, R2))
    
    # Schoolbook for 2x2x2x2: C[i,k] = sum_j A[i,j]*B[j,k]
    # 4 multiplications: (i,j) pairs for j=0,1 and... actually we need 4 terms.
    # Let me use: for each j in {0,1}, for each... no, that's 2 terms.
    # 
    # The tensor T[i,j,k]=1. We need 4 rank-1 terms.
    # Let me try:
    # Term 0: u=e_{0,0}, v=e_{0,0}, w=e_{0,0} → T[0,0,0] += 1
    # Term 1: u=e_{0,1}, v=e_{1,1}, w=e_{1,0} → T[0,1,1] += 1
    # Term 2: u=e_{1,0}, v=e_{0,1}, w=e_{1,1} → T[1,0,1] += 1
    # Term 3: u=e_{1,1}, v=e_{1,0}, w=e_{0,1} → T[1,1,0] += 1
    # But this only covers 4 of the 8 entries. Not correct.
    #
    # I think I'm confusing myself. Let me just verify numerically what works.
    # For now, let me use a simpler but valid construction: the "naive" Kronecker
    # product where I use schoolbook for both factors.
    
    # Schoolbook M_{2,2,2} has rank 8 (n*m*p = 2*2*2 = 8 terms, one per (i,j,k)? No...)
    # Actually the standard schoolbook for multiplying 2x2 by 2x2 uses 4 scalar multiplications.
    # The tensor rank is 4. Let me find it.
    #
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    #
    # 4 scalar mults: m1=A[0,0]*B[0,0], m2=A[0,1]*B[1,0], m3=A[1,0]*B[0,1], m4=A[1,1]*B[1,1]
    # But C[0,1] needs A[0,0]*B[0,1] which isn't one of these. So schoolbook is 8 mults for 2x2.
    # Wait no, schoolbook for n x n by n x n is n^3 mults. For 2x2 that's 8.
    # Strassen reduces to 7. The rank of M_{2,2,2} is 7? No, it's known to be 4.
    # 
    # Hmm, I'm getting confused. Let me just verify: the rank of the 2x2x2 matrix multiplication
    # tensor is 4. This is a well-known result. Let me construct it.
    #
    # T[i,j,k] = 1 for i,j,k in {0,1}. 8 entries.
    # We need 4 terms (u_r, v_r, w_r) with u_r in R^4, v_r in R^4, w_r in R^4.
    #
    # Let me try:
    # u0 = [1,0,0,0], v0 = [1,1,0,0], w0 = [1,1,0,0]
    #   Contributes: u0[i*2+j]*v0[j*2+k]*w0[k*2+i]
    #   (0,0,0): 1*1*1=1, (0,0,1): 1*0*0=0, (0,1,0): 0*1*1=0, (0,1,1): 0*0*0=0
    #   (1,0,0): 0*1*1=0, (1,0,1): 0*0*0=0, (1,1,0): 0*1*1=0, (1,1,1): 0*0*0=0
    #   Only covers (0,0,0). Not enough.
    #
    # I'll use a numerical solver approach within the time budget.
    
    # For M_{2,2,2}, let me just use the 8-term schoolbook (rank 8) to be safe,
    # and for M_{2,2,3} use a reasonable decomposition.
    # Then total rank = 8 * R_{2,2,3}. If R_{2,2,3} <= 9, total <= 72.
    # Better: use rank 4 for M_{2,2,2} if I can find it.
    
    # Let me try to find rank-4 for M_{2,2,2} by solving.
    # 8 equations, 4*(4+4+4)=48 unknowns.
    
    def find_rank_decomp(n, m, p, R, max_time=30):
        """Find rank-R decomposition of M_{n,m,p} using iterative approach."""
        t_start = time.time()
        nm, mp, pn = n*m, m*p, p*n
        U = np.random.randn(nm, R) * 0.1
        V = np.random.randn(mp, R) * 0.1
        W = np.random.randn(pn, R) * 0.1
        
        # Build tensor: T[i,j,k] = 1
        # Reshape into a 3D array indexed by (U_idx, V_idx, W_idx)
        # U_idx = i*m+j, V_idx = j*p+k, W_idx = k*n+i
        # For each (i,j,k): U_idx, V_idx, W_idx are determined.
        
        # Collect all (u_idx, v_idx, w_idx) triples
        triples = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    triples.append((u_idx, v_idx, w_idx))
        
        # We need sum_r U[u,r]*V[v,r]*W[w,r] = 1 for each triple
        # Use alternating minimization: fix two, solve for third.
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(2000):
            if time.time() - t_start > max_time:
                break
            
            # Update U: for each u_idx, solve linear system
            # sum_r U[u,r]*V[v,r]*W[w,r] = 1
            # Group by u_idx: for fixed u, we have equations over (v,w) pairs
            # sum_r U[u,r] * (V[v,r]*W[w,r]) = 1
            # This is: U[u,:] dot (V[v,:]*W[w,:]) = 1 for each (v,w) with that u
            
            for u_idx in range(nm):
                # Find all (v,w) that pair with this u
                eqs = []
                for (uu, vv, ww) in triples:
                    if uu == u_idx:
                        eqs.append((vv, ww))
                if not eqs:
                    continue
                # Build matrix: each row is V[v,:]*W[w,:]
                A_mat = np.array([V[vv,:]*W[ww,:] for vv, ww in eqs])
                b_vec = np.ones(len(eqs))
                # Solve A_mat @ U[u,:] = b_vec (least squares)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    U[u_idx, :] = sol
                except:
                    pass
            
            # Update V
            for v_idx in range(mp):
                eqs = []
                for (uu, vv, ww) in triples:
                    if vv == v_idx:
                        eqs.append((uu, ww))
                if not eqs:
                    continue
                A_mat = np.array([U[uu,:]*W[ww,:] for uu, ww in eqs])
                b_vec = np.ones(len(eqs))
                try:
                    sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    V[v_idx, :] = sol
                except:
                    pass
            
            # Update W
            for w_idx in range(pn):
                eqs = []
                for (uu, vv, ww) in triples:
                    if ww == w_idx:
                        eqs.append((uu, vv))
                if not eqs:
                    continue
                A_mat = np.array([U[uu,:]*V[vv,:] for uu, vv in eqs])
                b_vec = np.ones(len(eqs))
                try:
                    sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    W[w_idx, :] = sol
                except:
                    pass
            
            # Check error
            err = 0
            for (uu, vv, ww) in triples:
                val = np.dot(U[uu,:]*V[vv,:], W[ww,:])
                err += (val - 1)**2
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-10:
                break
        
        return best_U, best_V, best_W, best_err
    
    # Try to find rank-4 for M_{2,2,2}
    U2, V2, W2, err2 = find_rank_decomp(2, 2, 2, 4, max_time=20)
    print(f"M_{{2,2,2}} rank-4 error: {err2}")
    
    if err2 > 1e-6:
        # Fall back to rank 8 (schoolbook-like)
        R2 = 8
        U2, V2, W2, err2 = find_rank_decomp(2, 2, 2, 8, max_time=20)
        print(f"M_{{2,2,2}} rank-8 error: {err2}")
    
    # Find decomposition for M_{2,2,3}
    # Try ranks 6, 7, 8, 9
    R3 = None
    U3 = V3 = W3 = None
    for try_R in [6, 7, 8, 9, 10]:
        if time.time() > deadline - 40:
            break
        U3_try, V3_try, W3_try, err3 = find_rank_decomp(2, 2, 3, try_R, max_time=15)
        print(f"M_{{2,2,3}} rank-{try_R} error: {err3}")
        if err3 < 1e-6:
            R3 = try_R
            U3, V3, W3 = U3_try, V3_try, W3_try
            break
    
    if R3 is None:
        # Use the best one found
        U3, V3, W3, err3 = find_rank_decomp(2, 2, 3, 12, max_time=15)
        R3 = 12
        print(f"M_{{2,2,3}} rank-{R3} error: {err3}")
    
    # Now Kronecker product
    R_total = R2 * R3
    print(f"Total rank: {R2} x {R3} = {R_total}")
    
    # Kronecker product of decompositions:
    # U[i4*m4+j4, r1*R2+r2] = U2[i2*m2+j2, r2] * U2[i2',m2+j2',r2]... 
    # Wait, I need to be careful with the Kronecker product structure.
    #
    # M_{4,4,6} = M_{2,2,3} ⊗ M_{2,2,2}
    # A is 4x4 = (2x2) ⊗ (2x2), B is 4x6 = (2x2) ⊗ (2x3), C is 4x6 = (2x2) ⊗ (2x3)
    #
    # A[i1*2+i2, j1*2+j2] = A1[i1,j1] * A2[i2,j2]  (Kronecker)
    # B[j1*2+j2, k1*3+k2] = B1[j1,k1] * B2[j2,k2]
    # C[i1*2+i2, k1*3+k2] = C1[i1,k1] * C2[i2,k2]
    #
    # C1[i1,k1] = sum_{j1} A1[i1,j1]*B1[j1,k1]  (2x2 by 2x3 mult)
    # C2[i2,k2] = sum_{j2} A2[i2,j2]*B2[j2,k2]  (2x2 by 2x2 mult)
    #
    # C[i1*2+i2, k1*3+k2] = C1[i1,k1]*C2[i2,k2]
    #                        = (sum_j1 A1[i1,j1]*B1[j1,k1]) * (sum_j2 A2[i2,j2]*B2[j2,k2])
    #                        = sum_{j1,j2} A1[i1,j1]*A2[i2,j2] * B1[j1,k1]*B2[j2,k2] * 1
    #
    # So the tensor decomposes as:
    # T[i1,i2,j1,j2,k1,k2] = sum_{j1,j2} A1[i1,j1]*A2[i2,j2] * B1[j1,k1]*B2[j2,k2] * C1[i1,k1]*C2[i2,k2]
    #
    # Using rank-r1 decomp of M_{2,2,3}: sum_{r1} U1[i1*2+j1,r1]*V1[j1*3+k1,r1]*W1[k1*2+i1,r1] = 1
    # Using rank-r2 decomp of M_{2,2,2}: sum_{r2} U2[i2*2+j2,r2]*V2[j2*2+k2,r2]*W2[k2*2+i2,r2] = 1
    #
    # Combined: sum_{r1,r2} U1[i1*2+j1,r1]*U2[i2*2+j2,r2] * V1[j1*3+k1,r1]*V2[j2*2+k2,r2] * W1[k1*2+i1,r1]*W2[k2*2+i2,r2] = 1
    #
    # Now map to the 4x4x6 indices:
    # i = i1*2+i2, j = j1*2+j2, k = k1*3+k2
    # U_idx = i*4+j = (i1*2+i2)*4 + (j1*2+j2) = i1*8 + i2*4 + j1*2 + j2
    # V_idx = j*6+k = (j1*2+j2)*6 + (k1*3+k2) = j1*12 + j2*6 + k1*3 + k2
    # W_idx = k*4+i = (k1*3+k2)*4 + (i1*2+i2) = k1*12 + k2*4 + i1*2 + i2
    #
    # So: U[i1*8+i2*4+j1*2+j2, r1*R2+r2] = U1[i1*2+j1, r1] * U2[i2*2+j2, r2]
    #     V[j1*12+j2*6+k1*3+k2, r1*R2+r2] = V1[j1*3+k1, r1] * V2[j2*2+k2, r2]
    #     W[k1*12+k2*4+i1*2+i2, r1*R2+r2] = W1[k1*2+i1, r1] * W2[k2*2+i2, r2]
    
    U = np.zeros((16, R_total))
    V = np.zeros((24, R_total))
    W = np.zeros((24, R_total))
    
    for r1 in range(R3):
        for r2 in range(R2):
            col = r1 * R2 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            u_idx = i1*8 + i2*4 + j1*2 + j2
                            U[u_idx, col] = U1[i1*2+j1, r1] * U2[i2*2+j2, r2]
            for j1 in range(2):
                for k1 in range(3):
                    for j2 in range(2):
                        for k2 in range(2):
                            v_idx = j1*12 + j2*6 + k1*3 + k2
                            V[v_idx, col] = V1[j1*3+k1, r1] * V2[j2*2+k2, r2]
            for k1 in range(3):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            w_idx = k1*12 + k2*4 + i1*2 + i2
                            W[w_idx, col] = W1[k1*2+i1, r1] * W2[k2*2+i2, r2]
    
    # Verify
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i*m + j
                v_idx = j*p + k
                w_idx = k*n + i
                val = np.dot(U[u_idx,:]*V[v_idx,:], W[w_idx,:])
                err = abs(val - 1)
                if err > max_err:
                    max_err = err
    
    print(f"Verification max error: {max_err}")
    
    def to_list(mat):
        return [[round(x, 10) for x in row] for row in mat]
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
