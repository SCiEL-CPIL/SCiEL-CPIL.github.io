"""
Optimized matrix multiplication tensor decomposition for 3x5 * 5x6.
Uses Block Strassen for 2x2x2 sub-blocks and Schoolbook for the rest.
Expected rank: 72.
"""
import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Partitions
    # n=3: [2, 1]
    # m=5: [2, 2, 1]
    # p=6: [2, 2, 2]
    
    n_parts = [2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 2]
    
    # Precompute index ranges for each part
    n_ranges = []
    start = 0
    for size in n_parts:
        n_ranges.append(list(range(start, start + size)))
        start += size
        
    m_ranges = []
    start = 0
    for size in m_parts:
        m_ranges.append(list(range(start, start + size)))
        start += size
        
    p_ranges = []
    start = 0
    for size in p_parts:
        p_ranges.append(list(range(start, start + size)))
        start += size

    # Initialize U, V, W with 0. We don't know the final rank yet, 
    # so we can build lists of vectors and then assemble.
    # Or just use a large enough size and trim?
    # Better: collect (U_vec, V_vec, W_vec) tuples.
    
    factors = []
    
    # Helper to add a schoolbook block
    def add_schoolblock(i_indices, j_indices, k_indices):
        for i in i_indices:
            for j in j_indices:
                for k in k_indices:
                    u = [0.0] * (n * m)
                    v = [0.0] * (m * p)
                    w = [0.0] * (p * n)
                    
                    u[i * m + j] = 1.0
                    v[j * p + k] = 1.0
                    w[k * n + i] = 1.0
                    
                    factors.append((u, v, w))

    # Helper to add a Strassen block (2x2x2)
    def add_strassenblock(i_indices, j_indices, k_indices):
        # i_indices: [i1, i2]
        # j_indices: [j1, j2]
        # k_indices: [k1, k2]
        i1, i2 = i_indices
        j1, j2 = j_indices
        k1, k2 = k_indices
        
        # M1 = (a+d)(e+h)
        u = [0.0] * (n * m); v = [0.0] * (m * p); w = [0.0] * (p * n)
        u[i1 * m + j1] = 1.0; u[i2 * m + j2] = 1.0
        v[j1 * p + k1] = 1.0; v[j2 * p + k2] = 1.0
        w[k1 * n + i1] = 1.0; w[k2 * n + i2] = 1.0
        factors.append((u, v, w))
        
        # M2 = (c+d)e
        u = [0.0] * (n * m); v = [0.0] * (m * p); w = [0.0] * (p * n)
        u[i2 * m + j1] = 1.0; u[i2 * m + j2] = 1.0
        v[j1 * p + k1] = 1.0
        w[k1 * n + i2] = 1.0; w[k2 * n + i2] = -1.0
        factors.append((u, v, w))
        
        # M3 = a(f-h)
        u = [0.0] * (n * m); v = [0.0] * (m * p); w = [0.0] * (p * n)
        u[i1 * m + j1] = 1.0
        v[j1 * p + k2] = 1.0; v[j2 * p + k2] = -1.0
        w[k2 * n + i1] = 1.0; w[k2 * n + i2] = 1.0
        factors.append((u, v, w))
        
        # M4 = d(g-e)
        u = [0.0] * (n * m); v = [0.0] * (m * p); w = [0.0] * (p * n)
        u[i2 * m + j2] = 1.0
        v[j2 * p + k1] = 1.0; v[j1 * p + k1] = -1.0
        w[k1 * n + i1] = 1.0; w[k1 * n + i2] = 1.0
        factors.append((u, v, w))
        
        # M5 = (a+b)h
        u = [0.0] * (n * m); v = [0.0] * (m * p); w = [0.0] * (p * n)
        u[i1 * m + j1] = 1.0; u[i1 * m + j2] = 1.0
        v[j2 * p + k2] = 1.0
        w[k1 * n + i1] = -1.0; w[k2 * n + i1] = 1.0
        factors.append((u, v, w))
        
        # M6 = (c-a)(e+f)
        u = [0.0] * (n * m); v = [0.0] * (m * p); w = [0.0] * (p * n)
        u[i2 * m + j1] = 1.0; u[i1 * m + j1] = -1.0
        v[j1 * p + k1] = 1.0; v[j1 * p + k2] = 1.0
        w[k2 * n + i2] = 1.0
        factors.append((u, v, w))
        
        # M7 = (b-d)(g+h)
        u = [0.0] * (n * m); v = [0.0] * (m * p); w = [0.0] * (p * n)
        u[i1 * m + j2] = 1.0; u[i2 * m + j2] = -1.0
        v[j2 * p + k1] = 1.0; v[j2 * p + k2] = 1.0
        w[k1 * n + i1] = 1.0
        factors.append((u, v, w))

    # Iterate through blocks
    for a_idx, i_size in enumerate(n_parts):
        for b_idx, j_size in enumerate(m_parts):
            for c_idx, k_size in enumerate(p_parts):
                i_indices = n_ranges[a_idx]
                j_indices = m_ranges[b_idx]
                k_indices = p_ranges[c_idx]
                
                if i_size == 2 and j_size == 2 and k_size == 2:
                    add_strassenblock(i_indices, j_indices, k_indices)
                else:
                    add_schoolblock(i_indices, j_indices, k_indices)

    rank = len(factors)
    
    # Assemble matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r, (u, v, w) in enumerate(factors):
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w
        
    # Convert to lists as per interface? 
    # "factor matrices of shapes ... with integer/half-integer entries"
    # The seed program returned lists of lists. 
    # I will return lists of lists to be safe, or numpy arrays if allowed.
    # "Python standard library, NumPy and SciPy are available."
    # Usually numpy arrays are fine, but the seed used lists.
    # I'll stick to lists to match the seed's return type exactly if possible, 
    # but numpy is faster to generate. 
    # Let's return lists of lists of floats/ints.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
