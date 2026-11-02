import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = 4, 4, 6
    
    # Strassen coefficients for 2x2x2
    strassen_terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):-1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    def build_kronecker(n1, m1, p1, terms1, n2, m2, p2, terms2):
        """Build Kronecker product of two tensor decompositions."""
        rank = len(terms1) * len(terms2)
        U = np.zeros((n1*n2, m1*m2, rank))
        V = np.zeros((m1*m2, p1*p2, rank))
        W = np.zeros((p1*p2, n1*n2, rank))
        
        col = 0
        for t1 in terms1:
            a1, b1, c1 = t1
            for t2 in terms2:
                a2, b2, c2 = t2
                for (i1, j1), c_a1 in a1.items():
                    for (i2, j2), c_a2 in a2.items():
                        U[i1*n2+i2, j1*m2+j2, col] += c_a1 * c_a2
                for (j1, k1), c_b1 in b1.items():
                    for (j2, k2), c_b2 in b2.items():
                        V[j1*m2+j2, k1*p2+k2, col] += c_b1 * c_b2
                for (k1, i1), c_c1 in c1.items():
                    for (k2, i2), c_c2 in c2.items():
                        W[k1*p2+k2, i1*n2+i2, col] += c_c1 * c_c2
                col += 1
        return U, V, W

    def verify(U, V, W, n, m, p):
        """Verify tensor decomposition."""
        rank = U.shape[2]
        U = U.reshape(-1, rank)
        V = V.reshape(-1, rank)
        W = W.reshape(-1, rank)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(total - 1) > 1e-9:
                        return False
        return True

    def to_list(mat):
        """Convert numpy array to list of lists."""
        return mat.tolist()

    # Search for a 2x2x3 decomposition with rank < 12
    def search_223(max_rank, time_limit=30):
        """Search for a decomposition of 2x2x3 matrix multiplication."""
        n1, m1, p1 = 2, 2, 3
        start = time.time()
        
        candidates = [-1, -0.5, 0, 0.5, 1]
        nonzero_candidates = [c for c in candidates if c != 0]
        
        best_result = None
        
        for rank in range(9, min(max_rank+1, 12)):
            if time.time() - start > time_limit:
                break
            
            for trial in range(20000):
                if time.time() - start > time_limit:
                    break
                
                U = np.zeros((4, rank))
                V = np.zeros((6, rank))
                W = np.zeros((6, rank))
                
                for r in range(rank):
                    u_pos = np.random.choice(4, size=2, replace=False)
                    for pos in u_pos:
                        U[pos, r] = np.random.choice(nonzero_candidates)
                    
                    v_pos = np.random.choice(6, size=2, replace=False)
                    for pos in v_pos:
                        V[pos, r] = np.random.choice(nonzero_candidates)
                    
                    w_pos = np.random.choice(6, size=2, replace=False)
                    for pos in w_pos:
                        W[pos, r] = np.random.choice(nonzero_candidates)
                
                valid = True
                for i in range(2):
                    for j in range(2):
                        for k in range(3):
                            total = np.sum(U[i*2+j, :] * V[j*3+k, :] * W[k*2+i, :])
                            if abs(total - 1) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    terms = []
                    for r in range(rank):
                        a_coefs = {}
                        for i in range(2):
                            for j in range(2):
                                if U[i*2+j, r] != 0:
                                    a_coefs[(i,j)] = U[i*2+j, r]
                        b_coefs = {}
                        for j in range(2):
                            for k in range(3):
                                if V[j*3+k, r] != 0:
                                    b_coefs[(j,k)] = V[j*3+k, r]
                        c_coefs = {}
                        for k in range(3):
                            for i in range(2):
                                if W[k*2+i, r] != 0:
                                    c_coefs[(k,i)] = W[k*2+i, r]
                        terms.append((a_coefs, b_coefs, c_coefs))
                    best_result = (rank, terms)
                    return best_result
        
        return best_result

    # Try to find a better 2x2x3 decomposition
    result_223 = search_223(max_rank=11, time_limit=25)
    
    if result_223 is not None:
        rank_223, terms_223 = result_223
        U_k, V_k, W_k = build_kronecker(
            2, 2, 2, strassen_terms,
            2, 2, 3, terms_223
        )
        
        U_flat = U_k.reshape(-1, U_k.shape[2])
        V_flat = V_k.reshape(-1, V_k.shape[2])
        W_flat = W_k.reshape(-1, W_k.shape[2])
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.sum(U_flat[i*m+j, :] * V_flat[j*p+k, :] * W_flat[k*n+i, :])
                    if abs(total - 1) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return {
                "U": to_list(U_flat),
                "V": to_list(V_flat),
                "W": to_list(W_flat)
            }
    
    # Fallback: Kronecker product with 3 copies of 2x2x2
    rank = 12 * 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for a in range(2):
        for c in range(2):
            for b in range(3):
                for a_coefs, b_coefs, c_coefs in strassen_terms:
                    for (ar, ac), coef in a_coefs.items():
                        gi = 2*a + ar
                        gj = 2*c + ac
                        U[gi * m + gj, col] += coef
                    for (br, bc), coef in b_coefs.items():
                        gi = 2*c + br
                        gj = 2*b + bc
                        V[gi * p + gj, col] += coef
                    for (cr, cc), coef in c_coefs.items():
                        gi = 2*b + cr
                        gj = 2*a + cc
                        W[gi * n + gj, col] += coef
                    col += 1
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(total - 1) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    else:
        rank_school = n * m * p
        U_s = np.zeros((n*m, rank_school))
        V_s = np.zeros((m*p, rank_school))
        W_s = np.zeros((p*n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j, col] = 1
                    V_s[j * p + k, col] = 1
                    W_s[k * n + i, col] = 1
                    col += 1
        return {"U": to_list(U_s), "V": to_list(V_s), "W": to_list(W_s)}
