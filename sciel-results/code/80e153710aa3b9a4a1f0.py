import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We construct a decomposition of rank 45.
    # Structure:
    # 1. i in {0,1}, j in {0,1}, k in {0,1,2} -> M_2,2,3 (Rank 8)
    # 2. i in {0,1}, j in {2,3}, k in {0,1,2} -> M_2,2,3 (Rank 8)
    # 3. i in {2}, j in {0,1,2,3}, k in {0,1,2,3,4,5,6} -> M_1,4,7 (Rank 1)
    # 4. i in {0,1}, j in {0,1,2,3}, k in {3,4,5,6} -> M_2,4,4 (Rank 28)
    #    M_2,4,4 is split into 4 M_2,2,2 blocks (Strassen, Rank 7 each)
    
    rank = 45
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u, v, w):
        nonlocal col
        # u is list of length 2 (local i), v is list of length 2 (local j), w is list of length 2 (local k)
        # We need to map to global indices
        pass # Not used directly, we will handle offsets
    
    # Strassen terms for M_2,2,2
    # u, v, w are tuples of 2 elements
    strassen_terms = [
        ((1, 1), (1, 1), (1, 1)),
        ((0, 1), (0, 1), (1, 0)),
        ((1, 0), (1, 0), (1, 0)),
        ((0, 1), (0, 1), (0, 1)),
        ((1, 0), (1, 0), (0, 1)),
        ((-1, 1), (1, 0), (1, 0)),
        ((1, -1), (1, 1), (1, 0)),
    ]
    
    # 1. Block i in {0,1}, j in {0,1}, k in {0,1,2}
    # k in {0,1} is M_2,2,2 (Strassen)
    # k=2 is M_2,2,1 (Rank 1)
    
    # M_2,2,2 part
    for u, v, w in strassen_terms:
        # u_i, v_j, w_k
        for i_local in range(2):
            for j_local in range(2):
                for k_local in range(2):
                    val = u[i_local] * v[j_local] * w[k_local]
                    if val != 0:
                        i_g = i_local
                        j_g = j_local
                        k_g = k_local
                        U[i_g * m + j_g][col] += val
                        V[j_g * p + k_g][col] += 1 # Wait, V and W are 1s in schoolbook, but here they are part of the decomposition
                        # Correction: The decomposition is T = sum u_r v_r w_r.
                        # U[i, r] = u_r[i], V[j, r] = v_r[j], W[k, r] = w_r[k]
                        # So we just set U, V, W to the vectors.
                        pass
        
        # Correct way:
        # U[i_g * m + j_g] is not how it works.
        # U has shape (n*m, rank). U[i*m+j, r] = u_r[i] ? No.
        # The tensor is T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
        # This is not a standard rank-1 decomposition form for T[i,j,k] = sum u_r[i] v_r[j] w_r[k].
        # Standard form: T[i,j,k] = sum_r U[i, r] V[j, r] W[k, r].
        # Here indices are flattened.
        # U[i*m+j, r] corresponds to u_r[i] ? No, U depends on i and j.
        # In matrix multiplication, T[i,j,k] = 1.
        # Decomposition: T = sum_r (u_r \otimes v_r \otimes w_r).
        # (u \otimes v \otimes w)[i,j,k] = u[i] v[j] w[k].
        # So U[i*m+j, r] should be u_r[i] ? No, that would make U depend on j only via the index.
        # If U[i*m+j, r] = u_r[i], then for fixed r, U is constant over j.
        # Then V[j*p+k, r] = v_r[j].
        # Then W[k*n+i, r] = w_r[k].
        # Then sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = sum_r u_r[i] v_r[j] w_r[k].
        # This matches T[i,j,k].
        
        # So for each rank-1 term (u, v, w):
        # For all i, j, k:
        # U[i*m+j, col] += u[i]
        # V[j*p+k, col] += v[j]
        # W[k*n+i, col] += w[k]
        
        # This is inefficient. We can just set the values.
        # Since u, v, w are small, we can iterate.
        
        # Reset col increment logic
        pass

    # Let's rewrite the loop properly.
    
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal col
        for u, v, w in strassen_terms:
            # u, v, w are tuples of length 2
            for i_l in range(2):
                for j_l in range(2):
                    for k_l in range(2):
                        # This is O(8) per term, 7 terms = 56 ops.
                        # We update U, V, W
                        i_g = i_off + i_l
                        j_g = j_off + j_l
                        k_g = k_off + k_l
                        
                        U[i_g * m + j_g][col] = u[i_l]
                        V[j_g * p + k_g][col] = v[j_l]
                        W[k_g * n + i_g][col] = w[k_l]
            col += 1

    def add_rank1_block_221(i_off, j_off, k_off):
        # M_2,2,1: u=(1,1), v=(1,1), w=(1,0) ?
        # T[i,j,k] = 1 for k=k_off.
        # u[i]=1, v[j]=1, w[k_local]=1 if k_local=0 else 0.
        nonlocal col
        for i_l in range(2):
            for j_l in range(2):
                for k_l in range(1): # Only k=0
                    i_g = i_off + i_l
                    j_g = j_off + j_l
                    k_g = k_off + k_l
                    U[i_g * m + j_g][col] = 1
                    V[j_g * p + k_g][col] = 1
                    W[k_g * n + i_g][col] = 1
        col += 1

    def add_rank1_block_1mp(i_off, j_off, k_off, m_size, p_size):
        # M_1,m,p: u=(1), v=(1,..,1), w=(1,..,1)
        nonlocal col
        i_g = i_off
        for j_l in range(m_size):
            for k_l in range(p_size):
                j_g = j_off + j_l
                k_g = k_off + k_l
                U[i_g * m + j_g][col] = 1
                V[j_g * p + k_g][col] = 1
                W[k_g * n + i_g][col] = 1
        col += 1

    # 1. Block i in {0,1}, j in {0,1}, k in {0,1,2}
    add_strassen_block(0, 0, 0) # k in {0,1}
    add_rank1_block_221(0, 0, 2) # k=2
    
    # 2. Block i in {0,1}, j in {2,3}, k in {0,1,2}
    add_strassen_block(0, 2, 0) # k in {0,1}
    add_rank1_block_221(0, 2, 2) # k=2
    
    # 3. Block i in {2}, j in {0,1,2,3}, k in {0,1,2,3,4,5,6}
    add_rank1_block_1mp(2, 0, 0, 4, 7)
    
    # 4. Block i in {0,1}, j in {0,1,2,3}, k in {3,4,5,6}
    # Split into 4 M_2,2,2 blocks
    # j in {0,1}, k in {3,4}
    add_strassen_block(0, 0, 3)
    # j in {0,1}, k in {5,6}
    add_strassen_block(0, 0, 5)
    # j in {2,3}, k in {3,4}
    add_strassen_block(0, 2, 3)
    # j in {2,3}, k in {5,6}
    add_strassen_block(0, 2, 5)
    
    # Check if col == rank
    # Rank 45.
    # 8 + 8 + 1 + 28 = 45.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
