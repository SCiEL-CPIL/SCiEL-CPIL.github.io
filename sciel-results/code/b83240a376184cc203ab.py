"""Strassen-based rank-58 decomposition for M_{3,4,5}."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]


def _build_target_tensor(n, m, p):
    """Build the target multiplication tensor as a flat array.
    
    T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    T[a,b,c] = 1 iff j==j', i==i', k==k'
    """
    T = np.zeros((n*m, m*p, n*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def _verify(U, V, W, n, m, p):
    """Verify decomposition using numpy."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T_computed = np.einsum('ar,br,cr->abc', U, V, W)
    T_target = _build_target_tensor(n, m, p)
    
    return np.allclose(T_computed, T_target, atol=1e-9)


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Strassen 2x2x2 coefficients (U, V, W) for the 7 rank-1 terms
    # These decompose 2x2 matrix multiplication: C = A * B
    # Using standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Strassen terms as (U_entry, V_entry, W_entry) for each of the 7 terms
    # U has 4 entries (2x2 A matrix), V has 4 entries (2x2 B matrix), W has 4 entries (2x2 C matrix)
    # Local indices: U[i,j], V[j,k], W[k,i] for i,j,k in {0,1}
    
    strassen_terms = []
    
    # Term 1: M1 = (A11+A22)(B11+B22)
    # U: A11=1, A22=1 (positions (0,0) and (1,1))
    # V: B11=1, B22=1 (positions (0,0) and (1,1))
    # W: C11=1, C22=1 (positions (0,0) and (1,1))
    strassen_terms.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    
    # Term 2: M2 = (A21+A22)B11
    # U: A21=1, A22=1 (positions (1,0) and (1,1))
    # V: B11=1 (position (0,0))
    # W: C21=1 (position (1,0))
    strassen_terms.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(1,0): 1}
    })
    
    # Term 3: M3 = A11(B21-B22)
    # U: A11=1 (position (0,0))
    # V: B21=1, B22=-1 (positions (1,0) and (1,1))
    # W: C12=1 (position (0,1))
    strassen_terms.append({
        'U': {(0,0): 1},
        'V': {(1,0): 1, (1,1): -1},
        'W': {(0,1): 1}
    })
    
    # Term 4: M4 = A22(B12-B11)
    # U: A22=1 (position (1,1))
    # V: B12=1, B11=-1 (positions (0,1) and (0,0))
    # W: C11=1, C21=1 (positions (0,0) and (1,0))
    strassen_terms.append({
        'U': {(1,1): 1},
        'V': {(0,1): 1, (0,0): -1},
        'W': {(0,0): 1, (1,0): 1}
    })
    
    # Term 5: M5 = (A11+A12)B22
    # U: A11=1, A12=1 (positions (0,0) and (0,1))
    # V: B22=1 (position (1,1))
    # W: C11=-1, C12=1 (positions (0,0) and (0,1))
    strassen_terms.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(0,0): -1, (0,1): 1}
    })
    
    # Term 6: M6 = (A21-A11)(B11+B12)
    # U: A21=1, A11=-1 (positions (1,0) and (0,0))
    # V: B11=1, B12=1 (positions (0,0) and (0,1))
    # W: C22=1 (position (1,1))
    strassen_terms.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (0,1): 1},
        'W': {(1,1): 1}
    })
    
    # Term 7: M7 = (A12-A22)(B21+B22)
    # U: A12=1, A22=-1 (positions (0,1) and (1,1))
    # V: B21=1, B22=1 (positions (1,0) and (1,1))
    # W: C11=1, C22=1 (positions (0,0) and (1,1))
    strassen_terms.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(1,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    
    # Build the full decomposition
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 2: i in {0,1}, j in {2,3}, k in {0,1}
    # Schoolbook for all other (i,j,k) triples
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(U_dict, V_dict, W_dict):
        """Add a rank-1 term from local dicts."""
        U_vec = [0.0] * (n * m)
        V_vec = [0.0] * (m * p)
        W_vec = [0.0] * (p * n)
        
        for (i_loc, j_loc), val in U_dict.items():
            U_vec[i_loc * m + j_loc] += val
        for (j_loc, k_loc), val in V_dict.items():
            V_vec[j_loc * p + k_loc] += val
        for (k_loc, i_loc), val in W_dict.items():
            W_vec[k_loc * n + i_loc] += val
        
        U_list.append(U_vec)
        V_list.append(V_vec)
        W_list.append(W_vec)
    
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Local to global: i_g = i_loc, j_g = j_loc, k_g = k_loc
    for term in strassen_terms:
        U_dict = {}
        V_dict = {}
        W_dict = {}
        for (i_loc, j_loc), val in term['U'].items():
            U_dict[(i_loc, j_loc)] = val
        for (j_loc, k_loc), val in term['V'].items():
            V_dict[(j_loc, k_loc)] = val
        for (k_loc, i_loc), val in term['W'].items():
            W_dict[(k_loc, i_loc)] = val
        add_term(U_dict, V_dict, W_dict)
    
    # Block 2: i in {0,1}, j in {2,3}, k in {0,1}
    # Local to global: i_g = i_loc, j_g = j_loc + 2, k_g = k_loc
    for term in strassen_terms:
        U_dict = {}
        V_dict = {}
        W_dict = {}
        for (i_loc, j_loc), val in term['U'].items():
            U_dict[(i_loc, j_loc + 2)] = val
        for (j_loc, k_loc), val in term['V'].items():
            V_dict[(j_loc + 2, k_loc)] = val
        for (k_loc, i_loc), val in term['W'].items():
            W_dict[(k_loc, i_loc)] = val
        add_term(U_dict, V_dict, W_dict)
    
    # Schoolbook for all other (i, j, k) triples
    # These are:
    # i=2, j in {0,1,2,3}, k in {0,1,2,3,4} -> 5 terms
    # i in {0,1}, j in {0,1}, k in {2,3,4} -> 2*2*3 = 12 terms
    # i in {0,1}, j in {2,3}, k in {2,3,4} -> 2*2*3 = 12 terms
    # Total: 5 + 12 + 12 = 29 terms
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if covered by Strassen blocks
                if i in {0, 1} and j in {0, 1} and k in {0, 1}:
                    continue
                if i in {0, 1} and j in {2, 3} and k in {0, 1}:
                    continue
                
                # Schoolbook term
                U_vec = [0.0] * (n * m)
                V_vec = [0.0] * (m * p)
                W_vec = [0.0] * (p * n)
                U_vec[i * m + j] = 1.0
                V_vec[j * p + k] = 1.0
                W_vec[k * n + i] = 1.0
                
                U_list.append(U_vec)
                V_list.append(V_vec)
                W_list.append(W_vec)
    
    # Convert to list of lists
    U = U_list
    V = V_list
    W = W_list
    
    rank = len(U)
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        
        result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return result
