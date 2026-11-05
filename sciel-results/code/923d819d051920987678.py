"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Define groups for partition 2+2+1
    # Group 0: indices 0, 1
    # Group 1: indices 2, 3
    # Group 2: indices 4
    groups = [[0, 1], [2, 3], [4]]
    group_sizes = [2, 2, 1]
    
    # Total rank is sum of ranks of block multiplications
    # We build U, V, W dynamically
    U = []
    V = []
    W = []
    
    # Helper to add a column to U, V, W
    def add_column(u_vec, v_vec, w_vec):
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
        
    # Iterate over block triples (u, v, w)
    for u in range(3):
        for v in range(3):
            for w in range(3):
                # Sizes
                nu = group_sizes[u]
                nv = group_sizes[v]
                nw = group_sizes[w]
                
                # Indices in global space
                i_indices = groups[u]
                j_indices = groups[v]
                k_indices = groups[w]
                
                # Initialize local vectors for this block product
                # We need to map local (r, c) to global index
                # U is indexed by i*m + j. Here i is from group u, j is from group v.
                # V is indexed by j*p + k. Here j is from group v, k is from group w.
                # W is indexed by k*n + i. Here k is from group w, i is from group u.
                
                # Create mapping functions
                def get_global_i(local_r):
                    return i_indices[local_r]
                def get_global_j(local_c):
                    return j_indices[local_c]
                def get_global_k(local_r):
                    return k_indices[local_r]
                
                # Generate the decomposition for this block
                if nu == 2 and nv == 2 and nw == 2:
                    # Strassen 2x2x2
                    # Local indices: A(r, c), B(c, k), C(r, k)
                    # r, c, k in {0, 1}
                    
                    # M1 = (A00 + A11)(B00 + B11)
                    u1 = [0]*(n*m); v1 = [0]*(m*p); w1 = [0]*(p*n)
                    # U
                    u1[get_global_i(0)*m + get_global_j(0)] = 1
                    u1[get_global_i(1)*m + get_global_j(1)] = 1
                    # V
                    v1[get_global_j(0)*p + get_global_k(0)] = 1
                    v1[get_global_j(1)*p + get_global_k(1)] = 1
                    # W
                    w1[get_global_k(0)*n + get_global_i(0)] = 1
                    w1[get_global_k(1)*n + get_global_i(1)] = 1
                    add_column(u1, v1, w1)
                    
                    # M2 = (A10 + A11)B00
                    u2 = [0]*(n*m); v2 = [0]*(m*p); w2 = [0]*(p*n)
                    u2[get_global_i(1)*m + get_global_j(0)] = 1
                    u2[get_global_i(1)*m + get_global_j(1)] = 1
                    v2[get_global_j(0)*p + get_global_k(0)] = 1
                    w2[get_global_k(0)*n + get_global_i(1)] = 1
                    add_column(u2, v2, w2)
                    
                    # M3 = A00(B01 - B11)
                    u3 = [0]*(n*m); v3 = [0]*(m*p); w3 = [0]*(p*n)
                    u3[get_global_i(0)*m + get_global_j(0)] = 1
                    v3[get_global_j(0)*p + get_global_k(1)] = 1
                    v3[get_global_j(1)*p + get_global_k(1)] = -1
                    w3[get_global_k(1)*n + get_global_i(0)] = 1
                    w3[get_global_k(1)*n + get_global_i(1)] = 1
                    add_column(u3, v3, w3)
                    
                    # M4 = A11(B10 - B00)
                    u4 = [0]*(n*m); v4 = [0]*(m*p); w4 = [0]*(p*n)
                    u4[get_global_i(1)*m + get_global_j(1)] = 1
                    v4[get_global_j(1)*p + get_global_k(0)] = 1
                    v4[get_global_j(0)*p + get_global_k(0)] = -1
                    w4[get_global_k(0)*n + get_global_i(0)] = 1
                    w4[get_global_k(0)*n + get_global_i(1)] = 1
                    add_column(u4, v4, w4)
                    
                    # M5 = (A00 + A01)B11
                    u5 = [0]*(n*m); v5 = [0]*(m*p); w5 = [0]*(p*n)
                    u5[get_global_i(0)*m + get_global_j(0)] = 1
                    u5[get_global_i(0)*m + get_global_j(1)] = 1
                    v5[get_global_j(1)*p + get_global_k(1)] = 1
                    w5[get_global_k(1)*n + get_global_i(0)] = -1
                    w5[get_global_k(1)*n + get_global_i(1)] = 1
                    add_column(u5, v5, w5)
                    
                    # M6 = (A10 - A00)(B00 + B01)
                    u6 = [0]*(n*m); v6 = [0]*(m*p); w6 = [0]*(p*n)
                    u6[get_global_i(1)*m + get_global_j(0)] = 1
                    u6[get_global_i(0)*m + get_global_j(0)] = -1
                    v6[get_global_j(0)*p + get_global_k(0)] = 1
                    v6[get_global_j(0)*p + get_global_k(1)] = 1
                    w6[get_global_k(1)*n + get_global_i(1)] = 1
                    add_column(u6, v6, w6)
                    
                    # M7 = (A01 - A11)(B10 + B11)
                    u7 = [0]*(n*m); v7 = [0]*(m*p); w7 = [0]*(p*n)
                    u7[get_global_i(0)*m + get_global_j(1)] = 1
                    u7[get_global_i(1)*m + get_global_j(1)] = -1
                    v7[get_global_j(1)*p + get_global_k(0)] = 1
                    v7[get_global_j(1)*p + get_global_k(1)] = 1
                    w7[get_global_k(0)*n + get_global_i(0)] = 1
                    add_column(u7, v7, w7)
                    
                else:
                    # Schoolbook
                    for r in range(nu):
                        for c in range(nv):
                            for k in range(nw):
                                uc = [0]*(n*m); vc = [0]*(m*p); wc = [0]*(p*n)
                                uc[get_global_i(r)*m + get_global_j(c)] = 1
                                vc[get_global_j(c)*p + get_global_k(k)] = 1
                                wc[get_global_k(k)*n + get_global_i(r)] = 1
                                add_column(uc, vc, wc)
                                
    rank = len(U)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
