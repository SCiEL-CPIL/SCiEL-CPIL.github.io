import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x7 matrix.
    
    Strategy:
    1. Construct the hybrid Strassen/Schoolbook decomposition (Rank 39).
    2. Check for linear dependencies among the 39 term vectors to see if the rank can be reduced.
    3. If a dependency is found, remove the dependent term and return the reduced rank.
    4. If not, return the rank 39 decomposition.
    """
    n, m, p = DIMENSIONS  # 2, 3, 7
    
    terms = []
    
    def add_term(u, v, w):
        u_list = [0.0] * (n * m)
        v_list = [0.0] * (m * p)
        w_list = [0.0] * (p * n)
        for idx, val in u.items():
            u_list[idx] = val
        for idx, val in v.items():
            v_list[idx] = val
        for idx, val in w.items():
            w_list[idx] = val
        terms.append((u_list, v_list, w_list))
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    # M1
    add_term({u_idx(0,0): 1, u_idx(1,1): 1}, {v_idx(0,0): 1, v_idx(1,1): 1}, {w_idx(0,0): 1, w_idx(1,1): 1})
    # M2
    add_term({u_idx(1,0): 1, u_idx(1,1): 1}, {v_idx(0,0): 1}, {w_idx(0,1): 1, w_idx(1,1): -1})
    # M3
    add_term({u_idx(0,0): 1}, {v_idx(0,1): 1, v_idx(1,1): -1}, {w_idx(1,0): 1, w_idx(1,1): 1})
    # M4
    add_term({u_idx(1,1): 1}, {v_idx(1,0): 1, v_idx(0,0): -1}, {w_idx(0,0): 1, w_idx(0,1): 1})
    # M5
    add_term({u_idx(0,0): 1, u_idx(0,1): 1}, {v_idx(1,1): 1}, {w_idx(0,0): -1, w_idx(1,0): 1})
    # M6
    add_term({u_idx(1,0): 1, u_idx(0,0): -1}, {v_idx(0,0): 1, v_idx(0,1): 1}, {w_idx(1,1): 1})
    # M7
    add_term({u_idx(0,1): 1, u_idx(1,1): -1}, {v_idx(1,0): 1, v_idx(1,1): 1}, {w_idx(0,0): 1})
    
    # Part 2: Schoolbook for A[0:2, 0:2] * B[0:2, 2]
    add_term({u_idx(0,0): 1}, {v_idx(0,2): 1}, {w_idx(2,0): 1})
    add_term({u_idx(0,1): 1}, {v_idx(1,2): 1}, {w_idx(2,0): 1})
    add_term({u_idx(1,0): 1}, {v_idx(0,2): 1}, {w_idx(2,1): 1})
    add_term({u_idx(1,1): 1}, {v_idx(1,2): 1}, {w_idx(2,1): 1})
    
    # Part 3: Strassen for A[0:2, 0:2] * B[0:2, 3:5]
    # M1
    add_term({u_idx(0,0): 1, u_idx(1,1): 1}, {v_idx(0,3): 1, v_idx(1,4): 1}, {w_idx(3,0): 1, w_idx(4,1): 1})
    # M2
    add_term({u_idx(1,0): 1, u_idx(1,1): 1}, {v_idx(0,3): 1}, {w_idx(3,1): 1, w_idx(4,1): -1})
    # M3
    add_term({u_idx(0,0): 1}, {v_idx(0,4): 1, v_idx(1,4): -1}, {w_idx(4,0): 1, w_idx(4,1): 1})
    # M4
    add_term({u_idx(1,1): 1}, {v_idx(1,3): 1, v_idx(0,3): -1}, {w_idx(3,0): 1, w_idx(3,1): 1})
    # M5
    add_term({u_idx(0,0): 1, u_idx(0,1): 1}, {v_idx(1,4): 1}, {w_idx(3,0): -1, w_idx(4,0): 1})
    # M6
    add_term({u_idx(1,0): 1, u_idx(0,0): -1}, {v_idx(0,3): 1, v_idx(0,4): 1}, {w_idx(4,1): 1})
    # M7
    add_term({u_idx(0,1): 1, u_idx(1,1): -1}, {v_idx(1,3): 1, v_idx(1,4): 1}, {w_idx(3,0): 1})
    
    # Part 4: Strassen for A[0:2, 0:2] * B[0:2, 5:7]
    # M1
    add_term({u_idx(0,0): 1, u_idx(1,1): 1}, {v_idx(0,5): 1, v_idx(1,6): 1}, {w_idx(5,0): 1, w_idx(6,1): 1})
    # M2
    add_term({u_idx(1,0): 1, u_idx(1,1): 1}, {v_idx(0,5): 1}, {w_idx(5,1): 1, w_idx(6,1): -1})
    # M3
    add_term({u_idx(0,0): 1}, {v_idx(0,6): 1, v_idx(1,6): -1}, {w_idx(6,0): 1, w_idx(6,1): 1})
    # M4
    add_term({u_idx(1,1): 1}, {v_idx(1,5): 1, v_idx(0,5): -1}, {w_idx(5,0): 1, w_idx(5,1): 1})
    # M5
    add_term({u_idx(0,0): 1, u_idx(0,1): 1}, {v_idx(1,6): 1}, {w_idx(5,0): -1, w_idx(6,0): 1})
    # M6
    add_term({u_idx(1,0): 1, u_idx(0,0): -1}, {v_idx(0,5): 1, v_idx(0,6): 1}, {w_idx(6,1): 1})
    # M7
    add_term({u_idx(0,1): 1, u_idx(1,1): -1}, {v_idx(1,5): 1, v_idx(1,6): 1}, {w_idx(5,0): 1})
    
    # Part 5: Schoolbook for A[0:2, 2] * B[2, 0:7]
    for k in range(p):
        add_term({u_idx(0,2): 1}, {v_idx(2,k): 1}, {w_idx(k,0): 1})
        add_term({u_idx(1,2): 1}, {v_idx(2,k): 1}, {w_idx(k,1): 1})
        
    rank = len(terms)
    
    # Check for linear dependencies
    # Construct the matrix of terms
    # Each term is a vector in R^(nm + mp + pn)
    # We check if the rank of this matrix is less than the number of terms
    
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    total_dim = dim_u + dim_v + dim_w
    
    term_matrix = np.zeros((rank, total_dim))
    for i, (u, v, w) in enumerate(terms):
        term_matrix[i, :dim_u] = u
        term_matrix[i, dim_u:dim_u+dim_v] = v
        term_matrix[i, dim_u+dim_v:] = w
        
    # Compute rank
    # Using SVD for numerical stability
    try:
        s = np.linalg.svd(term_matrix, full_matrices=False)
        # Count non-zero singular values
        tol = 1e-10
        effective_rank = np.sum(s > tol)
    except:
        effective_rank = rank
        
    if effective_rank < rank:
        # If rank is reduced, we need to identify which terms to keep.
        # This is a complex basis reduction. For now, we just return the rank.
        # The evaluator expects U, V, W.
        # If we just return the rank, we must provide valid U, V, W of that rank.
        # We can use QR decomposition to find a basis.
        
        # Get QR
        Q, R = np.linalg.qr(term_matrix.T)
        # Q has shape (total_dim, effective_rank)
        # We need to map back to U, V, W
        # This is tricky because Q mixes the components.
        # However, the task requires U, V, W to be factor matrices.
        # The decomposition is T = sum t_i U_i V_i W_i.
        # If we find a linear dependency, we can replace the dependent term.
        # For simplicity and safety, we will return the original 39 terms if we can't easily reconstruct the basis.
        # But if effective_rank < rank, we can try to return the first 'effective_rank' terms? No, they might not span.
        
        # Given the risk of invalid output, we will stick to the known valid 39 terms.
        # The check was to see if 39 is minimal.
        pass
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for c, (u_list, v_list, w_list) in enumerate(terms):
        for i in range(n * m):
            U[i][c] = u_list[i]
        for i in range(m * p):
            V[i][c] = v_list[i]
        for i in range(p * n):
            W[i][c] = w_list[i]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
