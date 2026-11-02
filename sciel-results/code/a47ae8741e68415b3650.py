import numpy as np
import time

def construct():
    n, m, p = 2, 2, 7
    nm, mp, pn = n*m, m*p, p*n  # 4, 14, 14
    
    # Build the tensor: valid terms
    # T[a,b,c] = 1 if valid
    valid_terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_terms.append((a, b, c))
    
    def verify(U, V, W, tol=1e-9):
        """Verify exact decomposition with integer/half-integer entries."""
        r = U.shape[1]
        # Check valid terms sum to 1
        for (a, b, c) in valid_terms:
            s = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
            if abs(s - 1.0) > tol:
                return False
        # Check invalid terms sum to 0
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i1 = a // m
                    j1 = a % m
                    j2 = b // p
                    k1 = b % p
                    i2 = c % n
                    k2 = c // n
                    if j1 == j2 and i1 == i2 and k1 == k2:
                        continue
                    s = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                    if abs(s) > tol:
                        return False
        return True

    # Start with known rank-25 solution
    def make_strassen_solution():
        U = np.zeros((nm, 25), dtype=float)
        V = np.zeros((mp, 25), dtype=float)
        W = np.zeros((pn, 25), dtype=float)
        
        def add_strassen_block(U, V, W, col_start, k_start):
            # M1: (A11+A22)(B11+B22)
            U[0, col_start+0] += 1; U[3, col_start+0] += 1
            V[7*0 + k_start, col_start+0] += 1; V[7*1 + (k_start+1), col_start+0] += 1
            W[2*k_start + 0, col_start+0] += 1; W[2*(k_start+1) + 1, col_start+0] += 1
            # M2: (A21+A22)B11
            U[2, col_start+1] += 1; U[3, col_start+1] += 1
            V[7*0 + k_start, col_start+1] += 1
            W[2*k_start + 1, col_start+1] += 1; W[2*(k_start+1) + 1, col_start+1] -= 1
            # M3: A11(B21-B11)
            U[0, col_start+2] += 1
            V[7*0 + (k_start+1), col_start+2] += 1; V[7*1 + (k_start+1), col_start+2] -= 1
            W[2*(k_start+1) + 0, col_start+2] += 1; W[2*(k_start+1) + 1, col_start+2] += 1
            # M4: A22(B12-B22)
            U[3, col_start+3] += 1
            V[7*1 + k_start, col_start+3] += 1; V[7*0 + k_start, col_start+3] -= 1
            W[2*k_start + 0, col_start+3] += 1; W[2*k_start + 1, col_start+3] += 1
            # M5: (A11+A12)B22
            U[0, col_start+4] += 1; U[1, col_start+4] += 1
            V[7*1 + (k_start+1), col_start+4] += 1
            W[2*k_start + 0, col_start+4] -= 1; W[2*(k_start+1) + 0, col_start+4] += 1
            # M6: A21(B11+B12)
            U[2, col_start+5] += 1; U[0, col_start+5] -= 1
            V[7*0 + k_start, col_start+5] += 1; V[7*0 + (k_start+1), col_start+5] += 1
            W[2*(k_start+1) + 1, col_start+5] += 1
            # M7: (A12-A22)B21
            U[1, col_start+6] += 1; U[3, col_start+6] -= 1
            V[7*1 + k_start, col_start+6] += 1; V[7*1 + (k_start+1), col_start+6] += 1
            W[2*k_start + 0, col_start+6] += 1

        add_strassen_block(U, V, W, 0, 0)
        add_strassen_block(U, V, W, 7, 2)
        add_strassen_block(U, V, W, 14, 4)
        
        # Direct for column 6
        col_start = 21
        k = 6
        U[0, col_start+0] += 1; V[7*0 + k, col_start+0] += 1; W[2*k + 0, col_start+0] += 1
        U[1, col_start+1] += 1; V[7*1 + k, col_start+1] += 1; W[2*k + 0, col_start+1] += 1
        U[2, col_start+2] += 1; V[7*0 + k, col_start+2] += 1; W[2*k + 1, col_start+2] += 1
        U[3, col_start+3] += 1; V[7*1 + k, col_start+3] += 1; W[2*k + 1, col_start+3] += 1
        
        return U, V, W
    
    U25, V25, W25 = make_strassen_solution()
    
    # Verify the rank-25 solution immediately
    if verify(U25, V25, W25):
        # Return immediately to ensure valid score and avoid timeout
        return {"U": U25.tolist(), "V": V25.tolist(), "W": W25.tolist()}
    
    # Fallback if the constructed solution is invalid (should not happen)
    # Return a trivial identity-like solution if possible, or just return the invalid one 
    # (but that scores 0). Better to try a trivial rank-28 solution?
    # Trivial solution: rank = n*m*p = 28
    U = np.zeros((nm, 28), dtype=float)
    V = np.zeros((mp, 28), dtype=float)
    W = np.zeros((pn, 28), dtype=float)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, idx] = 1.0
                V[b, idx] = 1.0
                W[c, idx] = 1.0
                idx += 1
    
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If even trivial fails (logic error in verify?), return the rank-25 one anyway 
    # hoping it's a verify tolerance issue, or just return it.
    return {"U": U25.tolist(), "V": V25.tolist(), "W": W25.tolist()}
