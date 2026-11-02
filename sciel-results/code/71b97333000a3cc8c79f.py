import time
import numpy as np

DIMENSIONS = [3, 3, 3]

def _verify(U, V, W):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n = m = p = 3
    U = [[float(x) for x in row] for row in U]
    V = [[float(x) for x in row] for row in V]
    W = [[float(x) for x in row] for row in W]
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook():
    n = m = p = 3
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
    return {"U": U, "V": V, "W": W}

def _strassen_rank26():
    """
    Rank-26 decomposition for 3x3 matrix multiplication using explicit factors.
    Strassen core (7 terms) for the 2x2 block + schoolbook border terms.
    """
    n = m = p = 3
    rank = 26
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    terms = []
    
    # --- Strassen core for C[0:2,0:2] (7 terms) ---
    terms.append(([ (u_idx(0,0),1), (u_idx(1,1),1) ], [ (v_idx(0,0),1), (v_idx(1,1),1) ], [ (w_idx(0,0),1), (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(1,0),1), (u_idx(1,1),1) ], [ (v_idx(0,0),1) ], [ (w_idx(0,1),1), (w_idx(1,1),-1) ]))
    terms.append(([ (u_idx(0,0),1) ], [ (v_idx(0,1),1), (v_idx(1,1),-1) ], [ (w_idx(1,0),1), (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(1,1),1) ], [ (v_idx(1,0),1), (v_idx(0,0),-1) ], [ (w_idx(0,0),1), (w_idx(0,1),1) ]))
    terms.append(([ (u_idx(0,0),1), (u_idx(0,1),1) ], [ (v_idx(1,1),1) ], [ (w_idx(0,0),-1), (w_idx(1,0),1) ]))
    terms.append(([ (u_idx(1,0),1), (u_idx(0,0),-1) ], [ (v_idx(0,0),1), (v_idx(0,1),1) ], [ (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(0,1),1), (u_idx(1,1),-1) ], [ (v_idx(1,0),1), (v_idx(1,1),1) ], [ (w_idx(0,0),1) ]))
    
    # --- Border: C[0:2, 2] ---
    terms.append(([ (u_idx(0,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(0,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(1,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,1),1) ]))
    terms.append(([ (u_idx(1,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,1),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,1),1) ]))
    
    # --- Border: C[2, 0:2] ---
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,1),1) ], [ (w_idx(1,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,1),1) ], [ (w_idx(1,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,2),1) ]))
    
    # --- Corner: C[2,2] ---
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,2),1) ]))
    
    # --- Additional border: A[0:2,2]*B[2,0:2] contributes to C[0:2,0:2] ---
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,0),1) ]))
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,0),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,1),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,1),1) ]))
    
    for r, (u_list, v_list, w_list) in enumerate(terms):
        for (row, coeff) in u_list:
            U[row][r] += coeff
        for (row, coeff) in v_list:
            V[row][r] += coeff
        for (row, coeff) in w_list:
            W[row][r] += coeff
    
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(U, V, W, max_iterations=100):
    """
    Attempts to reduce the rank by finding linear dependencies among the term tensors.
    Returns a new decomposition if rank is reduced, otherwise None.
    """
    n = m = p = 3
    current_rank = len(U[0])
    
    # Flatten term tensors to vectors of length 27
    term_vectors = []
    for r in range(current_rank):
        vec = np.zeros(27)
        for i in range(n * m):
            if U[i][r] != 0:
                for j in range(m * p):
                    if V[j][r] != 0:
                        for k in range(p * n):
                            if W[k][r] != 0:
                                # Index in 27-dim space: (i, j, k)
                                # i is 0..8, j is 0..8, k is 0..8
                                # idx = i * 9 + j * 3 + k? 
                                # U is n*m, V is m*p, W is p*n
                                # Tensor T[i,j,k] = sum U[i,r] V[j,r] W[k,r]
                                # We need to map (i,j,k) to 0..26
                                # i: 0..8 (A_ij), j: 0..8 (B_jk), k: 0..8 (C_ki? No, C_ik)
                                # W index is k*n+i. So W row is (k,i).
                                # So term is U[i,r] * V[j,r] * W[k,r] contributing to C[k//n, k%n]?
                                # No, W row is `k*n+i`. So row `w_row` corresponds to `k = w_row // n`, `i = w_row % n`.
                                # The contribution is to C[i, k].
                                # So the tensor index is (i, j, k) where i is row of A/col of C, j is col of A/row of B, k is col of B/col of C.
                                # Wait, C[i,k] = sum_j A[i,j] B[j,k].
                                # So tensor is T[i,j,k] = delta(i,k)? No.
                                # The multiplication tensor is sum_{i,j,k} e_i (in C) \otimes e_j (in A) \otimes e_k (in B)?
                                # Standard: T = sum_{i,j,k} C_{ik} \otimes A_{ij} \otimes B_{jk}.
                                # So vector index is (i, j, k).
                                # i: 0..2, j: 0..2, k: 0..2.
                                # Total 27.
                                # idx = i * 9 + j * 3 + k.
                                
                                # In our code:
                                # U row i corresponds to A_{i//m, i%m}. So i_A = i//m, j_A = i%m.
                                # V row j corresponds to B_{j//p, j%p}. So j_B = j//p, k_B = j%p.
                                # W row k corresponds to C_{k%n, k//n}? No, W[k*n+i].
                                # So W row w corresponds to i_C = w % n, k_C = w // n.
                                
                                # For a term r:
                                # U[i,r] is non-zero for some i.
                                # V[j,r] is non-zero for some j.
                                # W[w,r] is non-zero for some w.
                                # Contribution to C[i_C, k_C] is U[i,r] V[j,r] W[w,r].
                                # This must match A[i_A, j_A] B[j_B, k_B] -> C[i_C, k_C].
                                # So we require i_A = i_C, j_A = j_B, k_B = k_C.
                                # The term vector component is at (i_A, j_A, k_B).
                                
                                i_A = i // m
                                j_A = i % m
                                j_B = j // p
                                k_B = j % p
                                i_C = w % n
                                k_C = w // n
                                
                                if i_A == i_C and j_A == j_B and k_B == k_C:
                                    vec[i_A * 9 + j_A * 3 + k_B] += U[i][r] * V[j][r] * W[w][r]
        term_vectors.append(vec)
        
    term_vectors = np.array(term_vectors)
    
    # Check if we can drop a term
    # For each i, check if term_vectors[i] is in span of term_vectors[~i]
    # This is equivalent to checking if the rank of the full matrix is the same as the rank of the matrix without column i.
    
    full_rank = np.linalg.matrix_rank(term_vectors.T) # 27 x R
    
    for i in range(current_rank):
        # Mask out column i
        mask = [j for j in range(current_rank) if j != i]
        reduced_vectors = term_vectors[:, mask]
        reduced_rank = np.linalg.matrix_rank(reduced_vectors.T)
        
        if reduced_rank == full_rank:
            # Check if term_vectors[i] is actually in the span of the others
            # If ranks are equal, it is in the span.
            # We can drop term i.
            new_U = [row[:i] + row[i+1:] for row in U]
            new_V = [row[:i] + row[i+1:] for row in V]
            new_W = [row[:i] + row[i+1:] for row in W]
            
            # Verify
            if _verify(new_U, new_V, new_W):
                return {"U": new_U, "V": new_V, "W": new_W}
                
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    result = _strassen_rank26()
    
    if _verify(result["U"], result["V"], result["W"]):
        # Try to reduce rank
        reduced = _try_reduce_rank(result["U"], result["V"], result["W"])
        if reduced is not None:
            return reduced
        return result
        
    return _schoolbook()

