import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # Use einsum
    # Build the full tensor check
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # Use einsum
    # Product: U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # = einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=tol)

def _strassen_2x2_terms():
    # Returns list of (u_dict, v_dict, w_dict) for 2x2x2 Strassen
    # u_dict: {(i,j): coeff}, v_dict: {(j,k): coeff}, w_dict: {(i,k): coeff}
    return [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]

def _expand_block(n, m, p, deadline):
    """
    Use two-level Strassen for 4x4x8.
    Top level: treat A as 2x2 blocks of 2x2, B as 2x2 blocks of 2x4 (columns), 
    but actually B is 4x8, so we treat it as 2x2 blocks where each block is 2x4.
    
    Actually, let's think more carefully.
    A is 4x4, B is 4x8, C is 4x8.
    
    We can use Strassen on the 4x4 x 4x4 part (left 4 columns of B), and similarly for the right 4 columns.
    But the current program already does this and gets rank 98.
    
    Let me try a different approach: use Strassen on the row dimension of A (n=4) and column dimension of B (p=8) simultaneously.
    Treat A as 2x2 blocks of 2x4 matrices, B as 2x2 blocks of 4x4 matrices... no, B is 4x8.
    
    Let me try: decompose into 7 Strassen products of (2x4) by (4x8), then use schoolbook for the m=4 dimension.
    Each (2x4)*(4x8) = (2x8) product requires 2*4*8=64 scalar multiplications via schoolbook.
    7 * 64 = 448. That's worse than 128.
    
    Alternative: use Strassen on the 2x2 block structure of A's rows and B's columns.
    A has 4 rows, B has 8 columns. Split A into top 2 rows and bottom 2 rows. Split B into left 4 cols and right 4 cols.
    This gives 4 sub-products, each 2x4 by 4x4, which is 2*4*4=32 each, total 128. Same as schoolbook.
    
    Let me try yet another approach. The current best is 98. Can we do better?
    
    For 4x4x4, the known best upper bound is 49 (two-level Strassen). For 4x4x8, it's 2*49=98.
    
    Actually, I recall that the tensor rank of matrix multiplication M_{n,m,p} has bounds.
    For M_{4,4,4}, the rank is known to be at most 49 and at least 48 (or maybe 49 is the best known).
    For M_{4,4,8}, by submultiplicativity, rank <= rank(M_{4,4,4}) * rank(M_{4,4,4}) / ... no, that's not right.
    
    Actually, M_{4,4,8} can be seen as M_{4,4,4} ⊗ M_{1,1,2} or something. 
    rank(M_{4,4,8}) <= rank(M_{4,4,4}) * rank(M_{1,1,2}) = 49 * 2 = 98.
    
    So 98 is a valid upper bound. Can we do better?
    
    Let me try to construct a decomposition with rank < 98 using a different blocking strategy.
    """
    n, m, p = 4, 4, 8
    deadline_local = deadline
    
    # Try the straightforward two-level Strassen: 7*7 = 49 for left 4x4, 49 for right 4x4, total 98
    # This is what the current program does.
    
    # Let me try to see if we can merge some terms.
    # In the left 49 and right 49, the U factors are identical (they only depend on A's block structure).
    # The V and W factors differ only in the k-index shift by 4.
    # So we have 98 terms where for each left term t, there's a right term with same U, shifted V and W.
    # We can't merge them directly because the V and W factors are different.
    
    # Let me try a completely different approach: use the fact that p=8=2*4, and use a 2x2 Strassen on the p-dimension.
    # Treat B as 4x2 blocks of 4x4 matrices. Then C is 4x2 blocks of 4x4 matrices.
    # For each pair of B-blocks (B0, B1), compute C0, C1 using Strassen-like combination.
    # But Strassen is for multiplying two matrices, not for this.
    
    # Actually, let's just stick with the 98-rank decomposition but implement it more cleanly.
    # The key question is: is 98 truly minimal? If not, can we find a better decomposition?
    
    # For now, let me just build the 98-rank decomposition efficiently.
    
    strassen_terms = _strassen_2x2_terms()
    
    all_terms = []
    
    for t_idx in range(7):
        if time.time() > deadline_local - 10:
            break
        a_blocks, b_blocks, c_blocks = strassen_terms[t_idx]
        
        for s_idx in range(7):
            if time.time() > deadline_local - 10:
                break
            s_u, s_v, s_w = strassen_terms[s_idx]
            
            u_global = {}
            for (r, c), coeff_s in s_u.items():
                for (u, v), coeff_a in a_blocks:
                    i = 2*u + r
                    j = 2*v + c
                    u_global[(i, j)] = u_global.get((i, j), 0) + coeff_s * coeff_a
            
            v_global = {}
            for (r, c), coeff_s in s_v.items():
                for (v, w), coeff_b in b_blocks:
                    j = 2*v + r
                    k = 2*w + c
                    v_global[(j, k)] = v_global.get((j, k), 0) + coeff_s * coeff_b
            
            w_global = {}
            for (r, c), coeff_s in s_w.items():
                for (u, w), coeff_c in c_blocks:
                    i = 2*u + r
                    k = 2*w + c
                    w_global[(i, k)] = w_global.get((i, k), 0) + coeff_s * coeff_c
            
            u_global = {k: v for k, v in u_global.items() if v != 0}
            v_global = {k: v for k, v in v_global.items() if v != 0}
            w_global = {k: v for k, v in w_global.items() if v != 0}
            
            if u_global and v_global and w_global:
                all_terms.append((u_global, v_global, w_global))
    
    terms_right = []
    for u_dict, v_dict, w_dict in all_terms:
        u_new = dict(u_dict)
        v_new = {}
        for (j, k), coeff in v_dict.items():
            v_new[(j, k + 4)] = coeff
        w_new = {}
        for (i, k), coeff in w_dict.items():
            w_new[(i, k + 4)] = coeff
        terms_right.append((u_new, v_new, w_new))
    
    total_terms = all_terms + terms_right
    rank = len(total_terms)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for col_idx, (u_dict, v_dict, w_dict) in enumerate(total_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j, col_idx] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k, col_idx] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i, col_idx] = coeff
    
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130
    
    # Build the 98-rank two-level Strassen decomposition
    U, V, W, rank = _expand_block(n, m, p, deadline)
    
    # Verify
    if not _verify_fast(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _build_schoolbook(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    print(f"Rank: {U.shape[1]}")
