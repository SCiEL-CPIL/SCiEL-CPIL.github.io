import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 2, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Target tensor T
    # U shape (n*m, r), V shape (m*p, r), W shape (p*n, r)
    # T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i)
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    
    def build_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    T[u, v, w] = 1
        return T

    T = build_tensor()
    
    best_result = None
    best_rank = n * m * p  # 20
    
    # Fallback: Rank 18 Construction
    def construct_rank_18():
        rank = 18
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Block 1: Cols 0,1 of B (V rows 0-3), Cols 0,1 of C (W rows 0-3)
        # Strassen
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1  # C10 uses M2? C10 = M2+M4. C10 is W[1].
        
        # M3 = A00(B10-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[2, 2] = -1 # B10 is V[1], B11 is V[2]
        W[2, 2] = 1  # C01 uses M3. C01 is W[2].
        
        # M4 = A11(B01-B00)
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1 # B01 is V[1], B00 is V[0]
        W[0, 3] = 1; W[1, 3] = 1 # C00 uses M4, C10 uses M4
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[2, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1 # C00 uses -M5, C01 uses M5
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = -1 # C11 uses -M6. C11 is W[3].
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[1, 6] = 1; V[2, 6] = 1
        W[0, 6] = 1; W[2, 6] = 1 # C00 uses M7, C01 uses M7
        
        # Check W indices for Block 1:
        # C00 (W0) = M1 + M4 - M5 + M7 -> 1, 1, -1, 1. Correct.
        # C10 (W1) = M2 + M4 -> 1, 1. Correct.
        # C01 (W2) = M3 + M5 -> 1, 1. Correct.
        # C11 (W3) = M1 - M2 + M3 - M6 -> 1, -1, 1, -1. Correct.
        
        # Block 2: Cols 2,3 of B (V rows 4-7), Cols 2,3 of C (W rows 4-7)
        # Same Strassen structure, offset by 7 in rank, 4 in V, 4 in W.
        for c in range(7):
            U[:, 7+c] = U[:, c]
            V[4:8, 7+c] = V[0:4, c]
            W[4:8, 7+c] = W[0:4, c]
            
        # Block 3: Col 4 of B (V rows 8-9), Col 4 of C (W rows 8-9)
        # Schoolbook
        # C04 = A00 B04 + A01 B14
        # C14 = A10 B04 + A11 B14
        # 4 terms
        # T1: A00 B04
        U[0, 14] = 1
        V[8, 14] = 1
        W[8, 14] = 1 # C04 is W[8] (k=4, i=0 -> 4*2+0=8)
        
        # T2: A01 B14
        U[1, 15] = 1
        V[9, 15] = 1
        W[8, 15] = 1
        
        # T3: A10 B04
        U[2, 16] = 1
        V[8, 16] = 1
        W[9, 16] = 1 # C14 is W[9] (k=4, i=1 -> 4*2+1=9)
        
        # T4: A11 B14
        U[3, 17] = 1
        V[9, 17] = 1
        W[9, 17] = 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    result_18 = construct_rank_18()
    best_result = result_18
    best_rank = 18
    
    # Search for lower ranks
    for r in range(14, 18):
        if time.time() > deadline - 20:
            break
            
        found = False
        for restart in range(10):
            if time.time() > deadline - 10:
                break
                
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Normalize
            for c in range(r):
                norm = np.linalg.norm(U[:, c])
                if norm > 0:
                    U[:, c] /= norm
                    V[:, c] *= norm
            
            for it in range(50):
                if time.time() > deadline - 10:
                    break
                
                # Update W
                for c in range(r):
                    # W[:, c] = argmin || T - sum U V W ||_F
                    # Residual R = T - sum_{c'!=c} U[:, c'] V[:, c'] W[:, c']
                    # W[:, c] = R contracted with U[:, c] and V[:, c]
                    # R[u, v, w] = T[u, v, w] - sum_{c'!=c} U[u, c'] V[v, c'] W[w, c']
                    # W[w, c] = sum_{u, v} R[u, v, w] U[u, c] V[v, c]
                    
                    # Efficient update:
                    # Let M_w = sum_{u, v} R[u, v, w] U[u, c] V[v, c]
                    # This is equivalent to (U^T R V) flattened?
                    # R is 3D.
                    # temp = np.einsum('uc,uvw,vc->w', U[:, c], R, V[:, c])
                    
                    # Calculate R without c
                    R = T.copy()
                    for c2 in range(r):
                        if c2 == c: continue
                        R -= np.einsum('uc,vc,wc->uvw', U[:, c2], V[:, c2], W[:, c2])
                    
                    W[:, c] = np.einsum('uc,uvw,vc->w', U[:, c], R, V[:, c])
                
                # Update V
                for c in range(r):
                    R = T.copy()
                    for c2 in range(r):
                        if c2 == c: continue
                        R -= np.einsum('uc,vc,wc->uvw', U[:, c2], V[:, c2], W[:, c2])
                    V[:, c] = np.einsum('uc,uvw,wc->v', U[:, c], R, W[:, c])
                    
                # Update U
                for c in range(r):
                    R = T.copy()
                    for c2 in range(r):
                        if c2 == c: continue
                        R -= np.einsum('uc,vc,wc->uvw', U[:, c2], V[:, c2], W[:, c2])
                    U[:, c] = np.einsum('vc,uvw,wc->u', V[:, c], R, W[:, c])
                
                # Check residual
                R = T.copy()
                for c in range(r):
                    R -= np.einsum('uc,vc,wc->uvw', U[:, c], V[:, c], W[:, c])
                res = np.linalg.norm(R)
                
                if res < 1e-9:
                    # Try to rationalize
                    U_rat = []
                    V_rat = []
                    W_rat = []
                    valid = True
                    for i in range(len(U)):
                        row = []
                        for j in range(r):
                            frac = Fraction(U[i, j]).limit_denominator(100)
                            if abs(float(frac) - U[i, j]) > 1e-6:
                                valid = False
                                break
                            row.append(frac)
                        if not valid: break
                        U_rat.append(row)
                    if valid:
                        for i in range(len(V)):
                            row = []
                            for j in range(r):
                                frac = Fraction(V[i, j]).limit_denominator(100)
                                if abs(float(frac) - V[i, j]) > 1e-6:
                                    valid = False
                                    break
                                row.append(frac)
                            if not valid: break
                            V_rat.append(row)
                    if valid:
                        for i in range(len(W)):
                            row = []
                            for j in range(r):
                                frac = Fraction(W[i, j]).limit_denominator(100)
                                if abs(float(frac) - W[i, j]) > 1e-6:
                                    valid = False
                                    break
                                row.append(frac)
                            if not valid: break
                            W_rat.append(row)
                            
                    if valid:
                        # Verify exact
                        U_np = np.array([[float(x) for x in row] for row in U_rat])
                        V_np = np.array([[float(x) for x in row] for row in V_rat])
                        W_np = np.array([[float(x) for x in row] for row in W_rat])
                        R_exact = T.copy()
                        for c in range(r):
                            R_exact -= np.einsum('uc,vc,wc->uvw', U_np[:, c], V_np[:, c], W_np[:, c])
                        if np.linalg.norm(R_exact) < 1e-10:
                            best_result = {"U": U_rat, "V": V_rat, "W": W_rat}
                            best_rank = r
                            found = True
                            break
            if found:
                break
                
    return best_result
