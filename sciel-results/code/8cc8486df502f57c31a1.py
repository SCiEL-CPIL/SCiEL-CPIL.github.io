"""
Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK.
Uses a recursive block decomposition approach.
"""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Constructs the tensor decomposition for matrix multiplication of dimensions n x m and m x p.
    Uses a recursive strategy. If dimensions are powers of 2, it could use Strassen, 
    but for now it uses a recursive schoolbook-like split which results in the 
    standard rank n*m*p for non-power-of-2 or when optimized recursively.
    """
    n, m, p = DIMENSIONS
    
    # We will build the matrices U, V, W as lists of lists of floats.
    # U shape: (n*m) x rank
    # V shape: (m*p) x rank
    # W shape: (p*n) x rank
    
    # Helper to add a rank-1 term
    rank = 0
    U = []
    V = []
    W = []
    
    def add_term(u_row, v_row, w_row, u_val, v_val, w_val):
        nonlocal rank
        # Extend if needed (we know max rank is n*m*p)
        U.append([0.0] * (rank + 1))
        V.append([0.0] * (rank + 1))
        W.append([0.0] * (rank + 1))
        
        # Fill previous rows with 0 at new column
        for i in range(rank):
            U[i].append(0.0)
            V[i].append(0.0)
            W[i].append(0.0)
            
        U[u_row].append(u_val)
        V[v_row].append(v_val)
        W[w_row].append(w_val)
        rank += 1

    # Recursive function to decompose a sub-problem
    # i_start, i_end: range of rows in A
    # j_start, j_end: range of cols in A / rows in B
    # k_start, k_end: range of cols in B
    
    def decompose(i_s, i_e, j_s, j_e, k_s, k_e):
        n_sub = i_e - i_s
        m_sub = j_e - j_s
        p_sub = k_e - k_s
        
        # Base case: schoolbook for 1x1x1
        if n_sub == 1 and m_sub == 1 and p_sub == 1:
            # U index: (i, j) -> i * m + j
            u_idx = i_s * m + j_s
            # V index: (j, k) -> j * p + k
            v_idx = j_s * p + k_s
            # W index: (k, i) -> k * n + i
            w_idx = k_s * n + i_s
            
            add_term(u_idx, v_idx, w_idx, 1.0, 1.0, 1.0)
            return

        # Recursive split
        # Try to split the largest dimension
        if m_sub >= n_sub and m_sub >= p_sub and m_sub > 1:
            # Split M
            mid = j_s + m_sub // 2
            decompose(i_s, i_e, j_s, mid, k_s, k_e)
            decompose(i_s, i_e, mid, j_e, k_s, k_e)
        elif n_sub >= m_sub and n_sub >= p_sub and n_sub > 1:
            # Split N
            mid = i_s + n_sub // 2
            decompose(i_s, mid, j_s, j_e, k_s, k_e)
            decompose(mid, i_e, j_s, j_e, k_s, k_e)
        elif p_sub > 1:
            # Split P
            mid = k_s + p_sub // 2
            decompose(i_s, i_e, j_s, j_e, k_s, mid)
            decompose(i_s, i_e, j_s, j_e, mid, k_e)
        else:
            # Fallback schoolbook for small cases if split fails (shouldn't happen with >1 checks)
            for i in range(i_s, i_e):
                for j in range(j_s, j_e):
                    for k in range(k_s, k_e):
                        add_term(i * m + j, j * p + k, k * n + i, 1.0, 1.0, 1.0)

    decompose(0, n, 0, m, 0, p)
    
    # Convert to lists of lists (already done in add_term, but U/V/W are lists of lists)
    # The add_term function appends to U/V/W lists.
    # Ensure shapes are correct.
    # U should have n*m rows.
    # V should have m*p rows.
    # W should have p*n rows.
    
    # The recursion might not have populated all rows if we used a more complex algorithm,
    # but here we used exact splits covering all indices, so all rows in U,V,W corresponding 
    # to (i,j), (j,k), (k,i) will have been touched.
    # However, if a row was never touched (which shouldn't happen in schoolbook recursive), 
    # it would be missing. Let's ensure dimensions.
    
    expected_u = n * m
    expected_v = m * p
    expected_w = p * n
    
    # Pad with zeros if any row is missing (should not happen)
    while len(U) < expected_u:
        U.append([0.0] * rank)
    while len(V) < expected_v:
        V.append([0.0] * rank)
    while len(W) < expected_w:
        W.append([0.0] * rank)
        
    # The rank is the number of columns
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
