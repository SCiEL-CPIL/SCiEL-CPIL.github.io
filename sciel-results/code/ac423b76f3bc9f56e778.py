import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 7
    
    # Pre-allocate U, V, W as lists of lists, then convert to numpy or keep as lists?
    # The interface expects {'U', 'V', 'W'} with shapes.
    # The seed program returns lists of lists.
    # Let's use lists of lists to be safe, or numpy arrays if allowed.
    # "Python standard library, NumPy and SciPy are available."
    # Using NumPy is faster for construction and likely accepted.
    # However, the seed returns lists. I will return lists to match the seed's type exactly if possible,
    # or numpy arrays if the evaluator handles them.
    # The seed returns `{"U": U, "V": V, "W": W}` where U, V, W are lists of lists.
    # I will stick to lists of lists to be safe, but construct efficiently.
    
    # Actually, let's just build the terms and accumulate.
    
    # Define groups
    i_groups = [[0, 1], [2]]
    j_groups = [[0, 1], [2, 3]]
    k_groups = [[0, 1], [2, 3], [4, 5], [6]]
    
    # We need to know the total rank to initialize U, V, W.
    # Or we can append to lists.
    # Let's collect terms first.
    # Each term is a dict or tuple describing U, V, W entries.
    # U entry: (i*m + j, value)
    # V entry: (j*p + k, value)
    # W entry: (k*n + i, value)
    
    terms = []
    
    def add_strassen_block(i_list, j_list, k_list):
        # i_list, j_list, k_list are lists of length 2
        i1, i2 = i_list
        j1, j2 = j_list
        k1, k2 = k_list
        
        # M1 = (A11 + A22)(B11 + B22) -> C11 + C22
        # U: (i1,j1)=1, (i2,j2)=1
        # V: (j1,k1)=1, (j2,k2)=1
        # W: (k1,i1)=1, (k2,i2)=1
        terms.append({
            'U': [(i1*m + j1, 1), (i2*m + j2, 1)],
            'V': [(j1*p + k1, 1), (j2*p + k2, 1)],
            'W': [(k1*n + i1, 1), (k2*n + i2, 1)]
        })
        
        # M2 = (A21 + A22)(B11) -> C21 + C22(-1)
        # U: (i2,j1)=1, (i2,j2)=1
        # V: (j1,k1)=1
        # W: (k1,i2)=1, (k2,i2)=-1
        terms.append({
            'U': [(i2*m + j1, 1), (i2*m + j2, 1)],
            'V': [(j1*p + k1, 1)],
            'W': [(k1*n + i2, 1), (k2*n + i2, -1)]
        })
        
        # M3 = (A11)(B12 - B22) -> C12
        # U: (i1,j1)=1
        # V: (j1,k2)=1, (j2,k2)=-1
        # W: (k2,i1)=1
        terms.append({
            'U': [(i1*m + j1, 1)],
            'V': [(j1*p + k2, 1), (j2*p + k2, -1)],
            'W': [(k2*n + i1, 1)]
        })
        
        # M4 = (A22)(B21 - B11) -> C11(-1) + C21
        # U: (i2,j2)=1
        # V: (j2,k1)=1, (j1,k1)=-1
        # W: (k1,i1)=-1, (k1,i2)=1
        terms.append({
            'U': [(i2*m + j2, 1)],
            'V': [(j2*p + k1, 1), (j1*p + k1, -1)],
            'W': [(k1*n + i1, -1), (k1*n + i2, 1)]
        })
        
        # M5 = (A11 + A12)(B22) -> C11(-1) + C12
        # U: (i1,j1)=1, (i1,j2)=1
        # V: (j2,k2)=1
        # W: (k2,i1)=-1, (k2,i2)=1  -- Wait, C12 is (i1, k2)? No.
        # C indices are (i, k).
        # M5 contributes to C11 and C12.
        # C11 is (i1, k1). C12 is (i1, k2).
        # So W: (k1, i1)=-1, (k2, i1)=1.
        terms.append({
            'U': [(i1*m + j1, 1), (i1*m + j2, 1)],
            'V': [(j2*p + k2, 1)],
            'W': [(k1*n + i1, -1), (k2*n + i1, 1)]
        })
        
        # M6 = (A21 - A11)(B11 + B12) -> C22
        # U: (i2,j1)=1, (i1,j1)=-1
        # V: (j1,k1)=1, (j1,k2)=1
        # W: (k2,i2)=1 -- Wait, C22 is (i2, k2).
        # So W: (k2, i2)=1.
        terms.append({
            'U': [(i2*m + j1, 1), (i1*m + j1, -1)],
            'V': [(j1*p + k1, 1), (j1*p + k2, 1)],
            'W': [(k2*n + i2, 1)]
        })
        
        # M7 = (A12 - A22)(B21 + B22) -> C11 + C22(-1)
        # U: (i1,j2)=1, (i2,j2)=-1
        # V: (j2,k1)=1, (j2,k2)=1
        # W: (k1,i1)=1, (k2,i2)=-1
        terms.append({
            'U': [(i1*m + j2, 1), (i2*m + j2, -1)],
            'V': [(j2*p + k1, 1), (j2*p + k2, 1)],
            'W': [(k1*n + i1, 1), (k2*n + i2, -1)]
        })

    def add_schoolbook_block(i_list, j_list, k_list):
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    terms.append({
                        'U': [(i*m + j, 1)],
                        'V': [(j*p + k, 1)],
                        'W': [(k*n + i, 1)]
                    })

    for I in i_groups:
        for J in j_groups:
            for K in k_groups:
                if len(I) == 2 and len(J) == 2 and len(K) == 2:
                    add_strassen_block(I, J, K)
                else:
                    add_schoolbook_block(I, J, K)
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, term in enumerate(terms):
        for idx, val in term['U']:
            U[idx][c] = val
        for idx, val in term['V']:
            V[idx][c] = val
        for idx, val in term['W']:
            W[idx][c] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
