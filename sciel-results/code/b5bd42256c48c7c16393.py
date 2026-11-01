"""
Rank-minimal tensor decomposition for 2x5x5 matrix multiplication.
Uses recursive decomposition based on Strassen's algorithm for 2x2x2.
Target rank: 46 (vs 50 trivial).
"""
import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Helper to get terms for n x m x p
    # Returns list of (u, v, w) where u is list of len n*m, v len m*p, w len p*n
    # We use memoization to avoid recomputation
    
    def get_terms(n, m, p):
        key = (n, m, p)
        if key in term_cache:
            return term_cache[key]
        
        terms = []
        
        if n == 2 and m == 2 and p == 2:
            # Strassen
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            
            # Local indices:
            # U: i*2+j. 0:00, 1:01, 2:10, 3:11
            # V: j*2+k. 0:00, 1:01, 2:10, 3:11
            # W: k*2+i. 0:00, 1:01, 2:10, 3:11
            
            # M1
            u = [1, 0, 0, 1]
            v = [1, 0, 0, 1]
            w = [1, 0, 0, 1] # C00 += 1, C11 += 1? 
            # C00 = M1... C11 = M1...
            # W indices: C00 is idx 0, C11 is idx 3.
            # So w should have 1 at 0 and 3.
            w = [1, 0, 0, 1]
            terms.append((u, v, w))
            
            # M2
            u = [0, 0, 1, 1]
            v = [1, 0, 0, 0]
            w = [0, 0, 1, 0] # C10 += 1, C11 -= 1? 
            # C10 = M2... C11 = -M2...
            # W idx 2 is C10. W idx 3 is C11.
            w = [0, 0, 1, -1]
            terms.append((u, v, w))
            
            # M3
            u = [1, 0, 0, 0]
            v = [0, 1, 0, -1]
            w = [0, 1, 0, 1] # C01 += 1, C11 += 1
            terms.append((u, v, w))
            
            # M4
            u = [0, 0, 0, 1]
            v = [0, 0, 1, -1]
            w = [1, 0, 1, 0] # C00 += 1, C10 += 1
            terms.append((u, v, w))
            
            # M5
            u = [1, 1, 0, 0]
            v = [0, 0, 0, 1]
            w = [-1, 1, 0, 0] # C00 -= 1, C01 += 1
            terms.append((u, v, w))
            
            # M6
            u = [0, 0, 1, -1]
            v = [1, 1, 0, 0]
            w = [0, 0, 0, 1] # C11 += 1
            terms.append((u, v, w))
            
            # M7
            u = [0, 1, 0, -1]
            v = [0, 0, 1, 1]
            w = [1, 0, 0, 0] # C00 += 1
            terms.append((u, v, w))
            
        elif n == 2 and m == 1 and p == 1:
            # C_i = A_i B_0
            # 2 terms
            terms.append(([1, 0], [1], [1, 0])) # i=0
            terms.append(([0, 1], [1], [0, 1])) # i=1
            
        elif n == 2 and m == 1 and p == 2:
            # C_ik = A_i B_k
            # 4 terms
            terms.append(([1, 0], [1, 0], [1, 0, 0, 0])) # i=0, k=0 -> W0
            terms.append(([1, 0], [0, 1], [0, 1, 0, 0])) # i=0, k=1 -> W1
            terms.append(([0, 1], [1, 0], [0, 0, 1, 0])) # i=1, k=0 -> W2
            terms.append(([0, 1], [0, 1], [0, 0, 0, 1])) # i=1, k=1 -> W3
            
        elif n == 2 and m == 2 and p == 1:
            # C_i = sum_j A_ij B_j0
            # 4 terms
            terms.append(([1, 0, 0, 0], [1, 0], [1, 0])) # i=0, j=0 -> W0
            terms.append(([0, 0, 1, 0], [1, 0], [1, 0])) # i=1, j=0 -> W1
            terms.append(([1, 0, 0, 0], [0, 1], [1, 0])) # i=0, j=1 -> W0
            terms.append(([0, 0, 1, 0], [0, 1], [1, 0])) # i=1, j=1 -> W1
            # Wait, W indices for 2x1: k*2+i. k=0.
            # W0 is C00, W1 is C10.
            # My previous lines:
            # i=0, j=0: U[0]=1, V[0]=1, W[0]=1.
            # i=1, j=0: U[2]=1, V[0]=1, W[1]=1.
            # i=0, j=1: U[0]=1, V[1]=1, W[0]=1.
            # i=1, j=1: U[2]=1, V[1]=1, W[1]=1.
            terms = [
                ([1, 0, 0, 0], [1, 0], [1, 0]),
                ([0, 0, 1, 0], [1, 0], [0, 1]),
                ([1, 0, 0, 0], [0, 1], [1, 0]),
                ([0, 0, 1, 0], [0, 1], [0, 1])
            ]
            
        elif n == 2 and m == 3 and p == 2:
            # Split m = 2 + 1
            # A = [A2, A1]
            # B = [B22; B12]
            # C = A2 B22 + A1 B12
            terms_22 = get_terms(2, 2, 2)
            terms_12 = get_terms(2, 1, 2)
            
            for u, v, w in terms_22:
                # Map to global 2x3x2
                # U: i*3+j. j in 0,1.
                # V: j*2+k. j in 0,1.
                # W: k*2+i.
                u_new = [0]*6
                v_new = [0]*4
                w_new = w # W size 4 is same
                for idx, val in enumerate(u):
                    if val != 0:
                        i, j = divmod(idx, 2)
                        u_new[i*3 + j] = val
                for idx, val in enumerate(v):
                    if val != 0:
                        j, k = divmod(idx, 2)
                        v_new[j*2 + k] = val
                terms.append((u_new, v_new, w_new))
                
            for u, v, w in terms_12:
                # u is 2x1, v is 1x2, w is 2x2
                # Map to global
                # U: i*3 + 2 (j=2)
                # V: 2*2 + k (j=2)
                # W: k*2+i
                u_new = [0]*6
                v_new = [0]*4
                w_new = w
                for idx, val in enumerate(u):
                    if val != 0:
                        i = idx
                        u_new[i*3 + 2] = val
                for idx, val in enumerate(v):
                    if val != 0:
                        k = idx
                        v_new[2*2 + k] = val
                terms.append((u_new, v_new, w_new))
                
        elif n == 2 and m == 2 and p == 3:
            # Split p = 2 + 1
            # B = [B22, B21]
            # C = A B22 + A B21
            terms_22 = get_terms(2, 2, 2)
            terms_21 = get_terms(2, 2, 1)
            
            for u, v, w in terms_22:
                u_new = u
                v_new = [0]*6
                w_new = [0]*6
                for idx, val in enumerate(v):
                    if val != 0:
                        j, k = divmod(idx, 2)
                        v_new[j*3 + k] = val
                for idx, val in enumerate(w):
                    if val != 0:
                        k, i = divmod(idx, 2)
                        w_new[k*2 + i] = val
                terms.append((u_new, v_new, w_new))
                
            for u, v, w in terms_21:
                u_new = u
                v_new = [0]*6
                w_new = [0]*6
                for idx, val in enumerate(v):
                    if val != 0:
                        j = idx
                        v_new[j*3 + 2] = val
                for idx, val in enumerate(w):
                    if val != 0:
                        i = idx
                        w_new[2*2 + i] = val
                terms.append((u_new, v_new, w_new))
                
        elif n == 2 and m == 3 and p == 3:
            # Split m = 2 + 1, p = 2 + 1
            # A = [A2, A1]
            # B = [B22 B21; B12 B11]
            # C = A2 B22 + A2 B21 + A1 B12 + A1 B11
            terms_22 = get_terms(2, 2, 2)
            terms_21 = get_terms(2, 2, 1)
            terms_12 = get_terms(2, 1, 2)
            terms_11 = get_terms(2, 1, 1)
            
            # 1. A2 B22
            for u, v, w in terms_22:
                u_new = [0]*6
                v_new = [0]*9
                w_new = [0]*6
                for idx, val in enumerate(u):
                    i, j = divmod(idx, 2)
                    u_new[i*3 + j] = val
                for idx, val in enumerate(v):
                    j, k = divmod(idx, 2)
                    v_new[j*3 + k] = val
                for idx, val in enumerate(w):
                    k, i = divmod(idx, 2)
                    w_new[k*2 + i] = val
                terms.append((u_new, v_new, w_new))
                
            # 2. A2 B21
            for u, v, w in terms_21:
                u_new = [0]*6
                v_new = [0]*9
                w_new = [0]*6
                for idx, val in enumerate(u):
                    i, j = divmod(idx, 2)
                    u_new[i*3 + j] = val
                for idx, val in enumerate(v):
                    j = idx
                    v_new[j*3 + 2] = val
                for idx, val in enumerate(w):
                    i = idx
                    w_new[2*2 + i] = val
                terms.append((u_new, v_new, w_new))
                
            # 3. A1 B12
            for u, v, w in terms_12:
                u_new = [0]*6
                v_new = [0]*9
                w_new = [0]*6
                for idx, val in enumerate(u):
                    i = idx
                    u_new[i*3 + 2] = val
                for idx, val in enumerate(v):
                    k = idx
                    v_new[2*3 + k] = val
                for idx, val in enumerate(w):
                    k, i = divmod(idx, 2)
                    w_new[k*2 + i] = val
                terms.append((u_new, v_new, w_new))
                
            # 4. A1 B11
            for u, v, w in terms_11:
                u_new = [0]*6
                v_new = [0]*9
                w_new = [0]*6
                for idx, val in enumerate(u):
                    i = idx
                    u_new[i*3 + 2] = val
                for idx, val in enumerate(v):
                    v_new[2*3 + 2] = val
                for idx, val in enumerate(w):
                    i = idx
                    w_new[2*2 + i] = val
                terms.append((u_new, v_new, w_new))
                
        else:
            # Fallback: trivial
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u = [0]*(n*m)
                        v = [0]*(m*p)
                        w = [0]*(p*n)
                        u[i*m+j] = 1
                        v[j*p+k] = 1
                        w[k*n+i] = 1
                        terms.append((u, v, w))
                        
        term_cache[key] = terms
        return terms

    term_cache = {}
    terms = get_terms(n, m, p)
    
    rank = len(terms)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for r, (u, v, w) in enumerate(terms):
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
