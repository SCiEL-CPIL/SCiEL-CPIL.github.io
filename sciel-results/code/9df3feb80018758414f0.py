DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2x2_terms():
    """
    Return the 7 rank-1 terms for Strassen's M_{2,2,2} decomposition.
    Each term is (U_dict, V_dict, W_dict) with keys (i,j), (j,k), (k,i) in local 2x2 coords.
    """
    terms = []
    # M1 = (A00+A11)(B00+B11)
    terms.append(({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}))
    # M2 = (A10+A11)B00
    terms.append(({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}))
    # M3 = A00(B10-B00)
    terms.append(({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}))
    # M4 = A11(B01-B10)
    terms.append(({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}))
    # M5 = (A00+A01)B11
    terms.append(({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}))
    # M6 = (A10-A00)(B00+B01)
    terms.append(({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}))
    # M7 = (A01-A11)(B10+B11)
    terms.append(({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}))
    return terms

def _construct_2x3x9():
    """
    Construct decomposition for 2x3 * 3x9.
    Strategy:
      - Use Strassen (rank 7) for the 2x2 sub-block: A[0:2, 0:2] x B[0:2, :]
      - Use direct multiplication (rank 18) for A[i,2] * B[2,k]
    Total rank: 25
    """
    n, m, p = 2, 3, 9
    U_terms = []
    V_terms = []
    W_terms = []
    
    # Strassen terms for A[0:2, 0:2] x B[0:2, :]
    # Local coords: li in {0,1} -> row i, lj in {0,1} -> col j of A / row of B
    # lk in {0,1} -> output col (0 or 1 of the 2-col block)
    strassen_terms = _strassen_2x2x2_terms()
    
    for (u_dict, v_dict, w_dict) in strassen_terms:
        U_term = {}
        for (li, lj), coeff in u_dict.items():
            U_term[(li, lj)] = coeff  # li=row, lj=col of A (0 or 1)
        
        V_term = {}
        for (lj, lk), coeff in v_dict.items():
            # lj is row of B (0 or 1), lk is col of B within the 2-col block
            # But B has 9 columns! We need B[0:2, :] which is 2x9
            # The Strassen decomposition works on B[0:2, 0:2] only
            # For the remaining columns B[0:2, 2:9], we need direct terms
            V_term[(lj, lk)] = coeff
        
        W_term = {}
        for (li, lk), coeff in w_dict.items():
            W_term[(li, lk)] = coeff
        
        U_terms.append(U_term)
        V_terms.append(V_term)
        W_terms.append(W_term)
    
    # Direct terms for A[i,2] * B[2,k] for i in {0,1}, k in 0..8
    for i in range(2):
        for k in range(9):
            U_terms.append({(i, 2): 1})
            V_terms.append({(2, k): 1})
            W_terms.append({(i, k): 1})
    
    # But wait - the Strassen terms only handle B[0:2, 0:2], not B[0:2, 2:9]
    # I need to handle B[0:2, 2:9] separately
    # For B[0:2, k] where k >= 2: A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is a 2x2 x 2x7 problem (2 rows of A, 2 rows of B, 7 cols)
    # Rank for 2x2 x 2x1 is 4 (or 3 with Strassen on the 2x2 part)
    # Actually for each column k>=2: we need A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # That's 2*2 = 4 terms per column, or we can use Strassen-like: rank 3 per column?
    # No, for a single column output with 2 inputs, it's just 2 terms: A[i,0]*B[0,k] and A[i,1]*B[1,k]
    # Wait no, C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for each i,k
    # So for fixed k, we need 2 terms (one for each i)
    # For 7 columns (k=2..8): 2*7 = 14 terms
    
    # Let me redo this properly
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    # Part 1: Strassen for A[0:2, 0:2] x B[0:2, 0:2] -> C[0:2, 0:2]
    for (u_dict, v_dict, w_dict) in strassen_terms:
        U_term = {}
        for (li, lj), coeff in u_dict.items():
            U_term[(li, lj)] = coeff
        
        V_term = {}
        for (lj, lk), coeff in v_dict.items():
            V_term[(lj, lk)] = coeff  # lk is 0 or 1
        
        W_term = {}
        for (li, lk), coeff in w_dict.items():
            W_term[(li, lk)] = coeff
        
        U_terms.append(U_term)
        V_terms.append(V_term)
        W_terms.append(W_term)
    
    # Part 2: A[0:2, 0:2] x B[0:2, 2:9] -> C[0:2, 2:9]
    # For each column k in 2..8: C[i,k] += A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This requires 2 terms per (i,k) pair, but we can share:
    # Term: A[0,0]*B[0,k] -> C[0,k], A[1,0]*B[0,k] -> C[1,k]
    # Actually each rank-1 term is U[i,j]*V[j,k]*W[k,i]
    # So for A[i,0]*B[0,k]: U[(i,0)]=1, V[(0,k)]=1, W[(k,i)]=1
    # That's 2*7 = 14 terms for i in {0,1}, k in {2..8}
    # Plus A[i,1]*B[1,k]: another 2*7 = 14 terms
    # Total: 28 terms for this part
    
    for i in range(2):
        for k in range(2, 9):
            U_terms.append({(i, 0): 1})
            V_terms.append({(0, k): 1})
            W_terms.append({(i, k): 1})
    
    for i in range(2):
        for k in range(2, 9):
            U_terms.append({(i, 1): 1})
            V_terms.append({(1, k): 1})
            W_terms.append({(i, k): 1})
    
    # Part 3: A[0:2, 2] x B[2, 0:9] -> C[0:2, 0:9]
    for i in range(2):
        for k in range(9):
            U_terms.append({(i, 2): 1})
            V_terms.append({(2, k): 1})
            W_terms.append({(i, k): 1})
    
    rank = len(U_terms)
    # rank = 7 + 28 + 18 = 53... that's worse than 50!
    
    # Let me think again. The issue is that Strassen only helps when we have a full 2x2x2 block.
    # For the cross terms, we're back to direct multiplication.
    
    # Better approach: Use Strassen on the full 2x3 x 3x9 by padding?
    # Or: split differently.
    
    # Actually, the original Strassen approach in the seed got rank 50.
    # Let me try: Strassen on 2x2 sub-blocks of columns.
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    # Split 9 columns into 4 pairs + 1 single: (0,1), (2,3), (4,5), (6,7), (8)
    # For each pair (k1,k2): use Strassen 2x2x2 on A[0:2, 0:2] x B[0:2, {k1,k2}]
    # But B[0:2, {k1,k2}] is not contiguous in the 3x9 matrix...
    # Actually it is: B[0,k1], B[0,k2], B[1,k1], B[1,k2]
    # We can treat these as a 2x2 block with columns k1,k2
    
    k_pairs = [(0,1), (2,3), (4,5), (6,7)]
    k_single = [8]
    
    for (k1, k2) in k_pairs:
        for (u_dict, v_dict, w_dict) in strassen_terms:
            U_term = {}
            for (li, lj), coeff in u_dict.items():
                U_term[(li, lj)] = coeff
            
            V_term = {}
            for (lj, lk), coeff in v_dict.items():
                # lk=0 -> k1, lk=1 -> k2
                actual_k = k1 if lk == 0 else k2
                V_term[(lj, actual_k)] = coeff
            
            W_term = {}
            for (li, lk), coeff in w_dict.items():
                actual_k = k1 if lk == 0 else k2
                W_term[(li, actual_k)] = coeff
            
            U_terms.append(U_term)
            V_terms.append(V_term)
            W_terms.append(W_term)
    
    # For single column k=8: direct multiplication
    for i in range(2):
        for j in range(2):  # j=0,1 for A[i,j]*B[j,8]
            U_terms.append({(i, j): 1})
            V_terms.append({(j, 8): 1})
            W_terms.append({(i, 8): 1})
    
    # For j=2 (third row of A, third col of B): direct for all 9 columns
    for i in range(2):
        for k in range(9):
            U_terms.append({(i, 2): 1})
            V_terms.append({(2, k): 1})
            W_terms.append({(i, k): 1})
    
    rank = len(U_terms)
    # rank = 4*7 + 4 + 18 = 28 + 4 + 18 = 50
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        for (i, j), coeff in U_terms[col].items():
            U[i * m + j][col] = coeff
        for (j, k), coeff in V_terms[col].items():
            V[j * p + k][col] = coeff
        for (i, k), coeff in W_terms[col].items():
            W[k * n + i][col] = coeff
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    T_computed = np.zeros((n*m, m*p, p*n))
    for c in range(rank):
        u_vec = np.array([U[i][c] for i in range(n*m)])
        v_vec = np.array([V[i][c] for i in range(m*p)])
        w_vec = np.array([W[i][c] for i in range(p*n)])
        T_computed += np.einsum('i,j,k->ijk', u_vec, v_vec, w_vec)
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1
    return np.max(np.abs(T_computed - T_expected)) < 1e-10

def _rank1_decomposition(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    try:
        U, V, W = _construct_2x3x9()
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        else:
            U, V, W = _rank1_decomposition(n, m, p)
            return {"U": U, "V": V, "W": W}
    except Exception:
        U, V, W = _rank1_decomposition(n, m, p)
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
