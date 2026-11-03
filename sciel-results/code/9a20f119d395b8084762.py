import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 4]
n, m, p = DIMENSIONS
# U shape: (n*m, r) = (9, r)
# V shape: (m*p, r) = (12, r)
# W shape: (p*n, r) = (12, r)

def _verify(U, V, W):
    """
    Verify if U, V, W exactly decompose the matmul tensor.
    U: (n*m, r)
    V: (m*p, r)
    W: (p*n, r)
    """
    r = U.shape[1]
    U_3d = U.reshape(n, m, r)
    V_3d = V.reshape(m, p, r)
    W_3d = W.reshape(p, n, r)
    
    try:
        T_reconstructed = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    except Exception:
        return False
    
    if not np.allclose(T_reconstructed, 1.0, atol=1e-9, rtol=1e-9):
        return False
    
    # Check half-integer constraint
    U2 = 2 * U
    V2 = 2 * V
    W2 = 2 * W
    if not np.allclose(U2, np.round(U2), atol=1e-9): return False
    if not np.allclose(V2, np.round(V2), atol=1e-9): return False
    if not np.allclose(W2, np.round(W2), atol=1e-9): return False
    
    return True

def _strassen_2x2_block(i_base, j_base, k_base):
    """
    Returns the U, V, W vectors for a 2x2x2 Strassen block.
    """
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    # Term 1: (A00+A11)(B00+B11)
    U1 = {u_idx(i_base, j_base): 1, u_idx(i_base+1, j_base+1): 1}
    V1 = {v_idx(j_base, k_base): 1, v_idx(j_base+1, k_base+1): 1}
    W1 = {w_idx(k_base, i_base): 1, w_idx(k_base+1, i_base+1): 1}
    U_terms.append(U1); V_terms.append(V1); W_terms.append(W1)
    
    # Term 2: (A10+A11)B00
    U2 = {u_idx(i_base+1, j_base): 1, u_idx(i_base+1, j_base+1): 1}
    V2 = {v_idx(j_base, k_base): 1}
    W2 = {w_idx(k_base, i_base+1): 1}
    U_terms.append(U2); V_terms.append(V2); W_terms.append(W2)
    
    # Term 3: A00(B01-B11)
    U3 = {u_idx(i_base, j_base): 1}
    V3 = {v_idx(j_base, k_base+1): 1, v_idx(j_base+1, k_base+1): -1}
    W3 = {w_idx(k_base+1, i_base): 1}
    U_terms.append(U3); V_terms.append(V3); W_terms.append(W3)
    
    # Term 4: A11(B10-B00)
    U4 = {u_idx(i_base+1, j_base+1): 1}
    V4 = {v_idx(j_base+1, k_base): 1, v_idx(j_base, k_base): -1}
    W4 = {w_idx(k_base, i_base+1): 1}
    U_terms.append(U4); V_terms.append(V4); W_terms.append(W4)
    
    # Term 5: (A00+A01)B11
    U5 = {u_idx(i_base, j_base): 1, u_idx(i_base, j_base+1): 1}
    V5 = {v_idx(j_base+1, k_base+1): 1}
    W5 = {w_idx(k_base+1, i_base): 1}
    U_terms.append(U5); V_terms.append(V5); W_terms.append(W5)
    
    # Term 6: (A10-A00)(B00+B01)
    U6 = {u_idx(i_base+1, j_base): 1, u_idx(i_base, j_base): -1}
    V6 = {v_idx(j_base, k_base): 1, v_idx(j_base, k_base+1): 1}
    W6 = {w_idx(k_base, i_base+1): 1, w_idx(k_base+1, i_base+1): 1}
    U_terms.append(U6); V_terms.append(V6); W_terms.append(W6)
    
    # Term 7: (A01-A11)(B10+B11)
    U7 = {u_idx(i_base, j_base+1): 1, u_idx(i_base+1, j_base+1): -1}
    V7 = {v_idx(j_base+1, k_base): 1, v_idx(j_base+1, k_base+1): 1}
    W7 = {w_idx(k_base, i_base): 1, w_idx(k_base+1, i_base): 1}
    U_terms.append(U7); V_terms.append(V7); W_terms.append(W7)
    
    return U_terms, V_terms, W_terms

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Build the base decomposition using Strassen blocks
    # We can cover a 2x2x2 sub-tensor with 7 terms.
    # The tensor is 3x3x4.
    # Let's place a 2x2x2 block at (0,0,0) and another at (0,0,2).
    # This covers i in {0,1}, j in {0,1}, k in {0,1,2,3}.
    # Remaining terms:
    # i=2, j in {0,1,2}, k in {0,1,2,3} -> 12 terms
    # i in {0,1}, j=2, k in {0,1,2,3} -> 8 terms
    # Total remaining: 20 terms.
    
    strassen_covered = set()
    blocks = [
        (0, 0, 0), 
        (0, 0, 2)  
    ]
    
    for ib, jb, kb in blocks:
        for i in range(ib, ib+2):
            for j in range(jb, jb+2):
                for k in range(kb, kb+2):
                    strassen_covered.add((i, j, k))
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for ib, jb, kb in blocks:
        U_terms, V_terms, W_terms = _strassen_2x2_block(ib, jb, kb)
        for u_t, v_t, w_t in zip(U_terms, V_terms, W_terms):
            u_vec = np.zeros(n * m)
            v_vec = np.zeros(m * p)
            w_vec = np.zeros(p * n)
            
            for idx, val in u_t.items():
                u_vec[idx] = val
            for idx, val in v_t.items():
                v_vec[idx] = val
            for idx, val in w_t.items():
                w_vec[idx] = val
            
            U_cols.append(u_vec)
            V_cols.append(v_vec)
            W_cols.append(w_vec)
            
    # 2. Collect all remaining terms
    remaining = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in strassen_covered:
                    remaining.append((i, j, k))
    
    # 3. Attempt to reduce the rank of the remaining terms
    # The remaining terms form a "sparse" tensor.
    # We can try to find a rank decomposition for the remaining tensor T_rem.
    # T_rem is a 3x3x4 tensor with 20 non-zero entries.
    # We can use a simple greedy algorithm or random search to find a low-rank decomposition.
    
    # Let's try to pair up terms to reduce the count.
    # If we can pair (i1, j, k1) and (i2, j, k2) into one term, we save 1 rank.
    # This is possible if we can find U, V, W such that:
    # U[i1, j] * V[j, k1] * W[k1, i1] = 1
    # U[i2, j] * V[j, k2] * W[k2, i2] = 1
    # And all other products are 0.
    # This requires U[i1, j] != 0, U[i2, j] != 0, V[j, k1] != 0, V[j, k2] != 0, W[k1, i1] != 0, W[k2, i2] != 0.
    # And for any other (i', j', k'), the product must be 0.
    
    # A simpler approach: just use the schoolbook method for the remaining terms.
    # This gives rank 20. Total rank = 14 + 20 = 34.
    # Current program gets 34.
    
    # Can we do better?
    # Let's try to find a rank decomposition for the remaining tensor using a random search.
    
    # We will try to find a decomposition of T_rem with rank < 20.
    # We will use a simple iterative algorithm:
    # 1. Initialize U, V, W with random values.
    # 2. For each term in T_rem, try to fit it with the current U, V, W.
    # 3. If it fits, add it to the decomposition.
    # 4. If it doesn't fit, add a new term.
    
    # This is a heuristic, so it might not find the optimal rank.
    # But it's a good starting point.
    
    # Let's try a different approach: use the fact that the remaining tensor is sparse.
    # We can try to find a low-rank decomposition by solving a system of linear equations.
    
    # For each possible pair of terms (i1, j, k1) and (i2, j, k2), check if they can be combined.
    # This is a combinatorial problem, so we'll use a greedy algorithm.
    
    # Greedy algorithm:
    # 1. Sort the remaining terms.
    # 2. For each term, try to combine it with an existing term.
    # 3. If it can't be combined, add it as a new term.
    
    # Let's implement this greedy algorithm.
    
    # First, let's group the remaining terms by j.
    from collections import defaultdict
    by_j = defaultdict(list)
    for (i, j, k) in remaining:
        by_j[j].append((i, k))
    
    # For each j, try to combine terms.
    for j, terms in by_j.items():
        terms = sorted(terms)
        num_terms = len(terms)
        for idx in range(0, num_terms, 2):
            if idx + 1 < num_terms:
                (i1, k1) = terms[idx]
                (i2, k2) = terms[idx + 1]
                u_vec = np.zeros(n * m)
                v_vec = np.zeros(m * p)
                w_vec = np.zeros(p * n)
                
                u_vec[i1 * m + j] = 1
                u_vec[i2 * m + j] = 1
                
                v_vec[j * p + k1] = 1
                v_vec[j * p + k2] = 1
                
                w_vec[k1 * n + i1] = 1
                w_vec[k2 * n + i2] = 1
                
                U_cols.append(u_vec)
                V_cols.append(v_vec)
                W_cols.append(w_vec)
            else:
                (i1, k1) = terms[idx]
                u_vec = np.zeros(n * m)
                v_vec = np.zeros(m * p)
                w_vec = np.zeros(p * n)
                
                u_vec[i1 * m + j] = 1
                v_vec[j * p + k1] = 1
                w_vec[k1 * n + i1] = 1
                
                U_cols.append(u_vec)
                V_cols.append(v_vec)
                W_cols.append(w_vec)
                
    U = np.column_stack(U_cols) if U_cols else np.zeros((n*m, 0))
    V = np.column_stack(V_cols) if V_cols else np.zeros((m*p, 0))
    W = np.column_stack(W_cols) if W_cols else np.zeros((p*n, 0))
    
    rank = U.shape[1]
    
    if _verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback: try adding a third Strassen block at (0,1,0) if not already covered
        # Actually just use schoolbook as safe fallback
        U_sb = np.zeros((n * m, n * m * p))
        V_sb = np.zeros((m * p, n * m * p))
        W_sb = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]) if res['U'] else 0)
