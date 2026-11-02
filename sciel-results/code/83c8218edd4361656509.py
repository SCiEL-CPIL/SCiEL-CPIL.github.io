import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    Rank 7.
    U: 4x7, V: 4x7, W: 4x7
    Indexing:
    U row (i, j) -> i*2+j
    V row (j, k) -> j*2+k
    W row (k, i) -> k*2+i
    """
    # Strassen 7 products:
    # P1 = (A11 + A22) * (B11 + B22)
    # P2 = (A21 + A22) * B11
    # P3 = A11 * (B12 - B22)
    # P4 = A22 * (B21 - B11)
    # P5 = (A11 + A12) * B22
    # P6 = (A21 - A11) * (B11 + B12)
    # P7 = (A12 - A22) * (B21 + B22)
    
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Helper to add to U, V, W
    def add_term(i, j, u_val, j_v, k, v_val, k_w, i_w, w_val):
        # U index i*2+j
        # V index j_v*2+k
        # W index k_w*2+i_w
        U[i*2+j, 0] += u_val # Placeholder, we need distinct columns
        # This approach is hard to do with a single add_term if we don't manage columns.
        pass
    
    # Let's define the 7 rank-1 terms explicitly.
    # Term 1: (A11 + A22) * (B11 + B22) -> C11, C22
    # U: A11(1,0), A22(1,0) -> idx 0, 3
    # V: B11(1,0), B22(1,0) -> idx 0, 3
    # W: C11(1,0), C22(1,0) -> idx 0, 3 (Note: W index is k*2+i. C11 is k=0,i=0 -> 0. C22 is k=1,i=1 -> 3)
    
    terms = []
    # 1
    terms.append(([(0,0,1), (3,0,1)], [(0,0,1), (3,0,1)], [(0,0,1), (3,0,1)]))
    # 2: (A21 + A22) * B11 -> C21, C22? No, P2 = (A21+A22)B11.
    # C21 = P2 + P4. C22 = P1 - P2 + P3 + P6.
    # So P2 contributes to C21 (sign +) and C22 (sign -).
    # U: A21(idx 2), A22(idx 3)
    # V: B11(idx 0)
    # W: C21(idx 2, k=1,i=0 -> 2), C22(idx 3)
    terms.append(([(2,0,1), (3,0,1)], [(0,0,1)], [(2,0,1), (3,0,-1)]))
    
    # 3: A11 * (B12 - B22) -> C12 (sign +)
    # U: A11(idx 0)
    # V: B12(idx 1), B22(idx 3)
    # W: C12(idx 1, k=0,i=1 -> 1)
    terms.append(([(0,0,1)], [(1,0,1), (3,0,-1)], [(1,0,1)]))
    
    # 4: A22 * (B21 - B11) -> C11 (sign +), C21 (sign +)
    # U: A22(idx 3)
    # V: B21(idx 2), B11(idx 0)
    # W: C11(idx 0), C21(idx 2)
    terms.append(([(3,0,1)], [(2,0,1), (0,0,-1)], [(0,0,1), (2,0,1)]))
    
    # 5: (A11 + A12) * B22 -> C11 (sign -), C12 (sign +)
    # U: A11(idx 0), A12(idx 1)
    # V: B22(idx 3)
    # W: C11(idx 0), C12(idx 1)
    terms.append(([(0,0,1), (1,0,1)], [(3,0,1)], [(0,0,-1), (1,0,1)]))
    
    # 6: (A21 - A11) * (B11 + B12) -> C22 (sign +)
    # U: A21(idx 2), A11(idx 0)
    # V: B11(idx 0), B12(idx 1)
    # W: C22(idx 3)
    terms.append(([(2,0,1), (0,0,-1)], [(0,0,1), (1,0,1)], [(3,0,1)]))
    
    # 7: (A12 - A22) * (B21 + B22) -> C11 (sign +), C22 (sign +)
    # U: A12(idx 1), A22(idx 3)
    # V: B21(idx 2), B22(idx 3)
    # W: C11(idx 0), C22(idx 3)
    terms.append(([(1,0,1), (3,0,-1)], [(2,0,1), (3,0,1)], [(0,0,1), (3,0,1)]))
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    for c, (u_terms, v_terms, w_terms) in enumerate(terms):
        for (idx, _, val) in u_terms:
            U[idx, c] = val
        for (idx, _, val) in v_terms:
            V[idx, c] = val
        for (idx, _, val) in w_terms:
            W[idx, c] = val
            
    return U, V, W

def _schoolbook_1mp(m, p):
    """
    Returns U, V, W for 1xm xp multiplication.
    Rank mp.
    U: m x mp
    V: mp x mp
    W: p x mp
    """
    rank = m * p
    U = np.zeros((m, rank))
    V = np.zeros((mp, rank)) # Wait, V size is m*p.
    V = np.zeros((m * p, rank))
    W = np.zeros((p, rank))
    
    col = 0
    for j in range(p):
        for k in range(m):
            # Term: A_k * B_{k,j} -> C_j
            # U index: k
            # V index: k*p + j
            # W index: j
            U[k, col] = 1
            V[k * p + j, col] = 1
            W[j, col] = 1
            col += 1
    return U, V, W

def _combine_mid(U1, V1, W1, U2, V2, W2, n, m1, m2, p):
    """
    Combine M_{n, m1, p} and M_{n, m2, p} into M_{n, m1+m2, p}.
    U: vstack
    V: vstack
    W: hstack
    """
    U = np.vstack([U1, U2])
    V = np.vstack([V1, V2])
    W = np.hstack([W1, W2])
    return U, V, W

def _combine_n(U1, V1, W1, U2, V2, W2, n1, m, p):
    """
    Combine M_{n1, m, p} and M_{n2, m, p} into M_{n1+n2, m, p}.
    U: vstack
    V: hstack
    W: vstack
    """
    U = np.vstack([U1, U2])
    V = np.hstack([V1, V2])
    W = np.vstack([W1, W2])
    return U, V, W

def _combine_p(U1, V1, W1, U2, V2, W2, n, m, p1):
    """
    Combine M_{n, m, p1} and M_{n, m, p2} into M_{n, m, p1+p2}.
    U: hstack
    V: vstack
    W: vstack
    """
    U = np.hstack([U1, U2])
    V = np.vstack([V1, V2])
    W = np.vstack([W1, W2])
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS # 3, 5, 6
    
    # 1. Get base components
    # M_{2,2,2}
    U_222, V_222, W_222 = _strassen_2x2x2()
    
    # M_{2,2,6} = 3 * M_{2,2,2}
    # Split p=6 into 2+2+2
    U_226, V_226, W_226 = U_222.copy(), V_222.copy(), W_222.copy()
    # Add second block
    U_226, V_226, W_226 = _combine_p(U_226, V_226, W_226, U_222, V_222, W_222, 2, 2, 2)
    # Add third block
    U_226, V_226, W_226 = _combine_p(U_226, V_226, W_226, U_222, V_222, W_222, 2, 2, 2)
    
    # M_{2,1,6}
    U_216, V_216, W_216 = _schoolbook_1mp(1, 6)
    # Note: _schoolbook_1mp returns U(m, rank), V(mp, rank), W(p, rank).
    # For m=1, p=6. U(1, 6), V(6, 6), W(6, 6).
    # But we need M_{2,1,6}.
    # M_{2,1,6} is not 1xm xp. It is n=2, m=1, p=6.
    # My _schoolbook_1mp was for n=1.
    # We need a general schoolbook or specific for M_{2,1,6}.
    # M_{2,1,6}: C_{2x6} = A_{2x1} B_{1x6}.
    # Rank 12.
    # U (2x1=2 rows), V (1x6=6 rows), W (6x2=12 rows).
    # Let's build M_{2,1,6} manually.
    U_216 = np.zeros((2, 12))
    V_216 = np.zeros((6, 12))
    W_216 = np.zeros((12, 12))
    col = 0
    for i in range(2):
        for k in range(6): # k is col of B, row of W?
            # B is 1x6. Index j in 0..5.
            # A is 2x1. Index i in 0..1.
            # C is 2x6. Index i, j.
            # Term A_i * B_j -> C_ij.
            # U index i.
            # V index j.
            # W index j*2 + i.
            U_216[i, col] = 1
            V_216[j, col] = 1
            W_216[j * 2 + i, col] = 1
            col += 1

    # M_{2,3,6} = M_{2,2,6} + M_{2,1,6}
    # Split m=3 into 2+1.
    U_236, V_236, W_236 = _combine_mid(U_226, V_226, W_226, U_216, V_216, W_216, 2, 2, 1, 6)
    
    # M_{2,5,6} = M_{2,2,6} + M_{2,3,6}
    # Split m=5 into 2+3.
    U_256, V_256, W_256 = _combine_mid(U_226, V_226, W_226, U_236, V_236, W_236, 2, 2, 3, 6)
    
    # M_{1,5,6}
    # n=1, m=5, p=6.
    # Rank 30.
    U_156 = np.zeros((5, 30))
    V_156 = np.zeros((30, 30))
    W_156 = np.zeros((6, 30))
    col = 0
    for j in range(6):
        for k in range(5):
            U_156[k, col] = 1
            V_156[k * 6 + j, col] = 1
            W_156[j, col] = 1
            col += 1

    # M_{3,5,6} = M_{2,5,6} + M_{1,5,6}
    # Split n=3 into 2+1.
    U_356, V_356, W_356 = _combine_n(U_256, V_256, W_256, U_156, V_156, W_156, 2, 5, 6)
    
    # Convert to lists
    return {"U": U_356.tolist(), "V": V_356.tolist(), "W": W_356.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
