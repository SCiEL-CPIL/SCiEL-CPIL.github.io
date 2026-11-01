import time
import numpy as np

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 25: 3 Strassen blocks (rank 7 each) + 1 Schoolbook block (rank 4)
    rank = 25
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a rank-1 term
    def add_term(u_vec, v_vec, w_vec, col):
        U[:, col] = u_vec
        V[:, col] = v_vec
        W[:, col] = w_vec

    # Strassen coefficients for 2x2x2
    # U indices: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    # V indices for block k0,k1: 
    #   j=0,k0=e, j=0,k1=f, j=1,k0=g, j=1,k1=h
    # W indices for block k0,k1:
    #   k0,i0=C00, k0,i1=C10, k1,i0=C01, k1,i1=C11
    
    def get_strassen_terms(k0, k1):
        # Returns list of (u, v, w) for the 7 Strassen terms
        # U is 4-dim, V is 4-dim (local), W is 4-dim (local)
        
        # M1 = (a+d)(e+h)
        u1 = [1, 0, 0, 1]
        v1 = [1, 0, 0, 1]
        w1 = [1, 0, 0, 1] # C00+1, C11+1
        
        # M2 = (c+d)e
        u2 = [0, 0, 1, 1]
        v2 = [1, 0, 0, 0]
        w2 = [0, 1, 0, -1] # C10+1, C11-1
        
        # M3 = a(f-h)
        u3 = [1, 0, 0, 0]
        v3 = [0, 1, 0, -1]
        w3 = [0, 0, 1, 1] # C01+1, C11+1
        
        # M4 = d(g-e)
        u4 = [0, 0, 0, 1]
        v4 = [-1, 0, 1, 0]
        w4 = [1, 1, 0, 0] # C00+1, C10+1
        
        # M5 = (a+b)h
        u5 = [1, 1, 0, 0]
        v5 = [0, 0, 0, 1]
        w5 = [-1, 0, 1, 0] # C00-1, C01+1
        
        # M6 = (c-a)(e+f)
        u6 = [-1, 0, 1, 0]
        v6 = [1, 1, 0, 0]
        w6 = [0, 0, 0, 1] # C11+1
        
        # M7 = (b-d)(g+h)
        u7 = [0, 1, 0, -1]
        v7 = [0, 0, 1, 1]
        w7 = [1, 0, 0, 0] # C00+1
        
        return [
            (u1, v1, w1),
            (u2, v2, w2),
            (u3, v3, w3),
            (u4, v4, w4),
            (u5, v5, w5),
            (u6, v6, w6),
            (u7, v7, w7)
        ]

    def map_local_to_global_v(local_v, k0, k1):
        # local_v is [e, f, g, h]
        # Global V indices:
        # j=0, k=k0 -> 0*p + k0 = k0
        # j=0, k=k1 -> 0*p + k1 = k1
        # j=1, k=k0 -> 1*p + k0 = p + k0
        # j=1, k=k1 -> 1*p + k1 = p + k1
        global_v = [0] * (m * p)
        global_v[k0] = local_v[0]
        global_v[k1] = local_v[1]
        global_v[p + k0] = local_v[2]
        global_v[p + k1] = local_v[3]
        return global_v

    def map_local_to_global_w(local_w, k0, k1):
        # local_w is [C00, C10, C01, C11]
        # Global W indices: W[k*n + i]
        # k=k0, i=0 -> k0*n + 0 = k0*n
        # k=k0, i=1 -> k0*n + 1
        # k=k1, i=0 -> k1*n + 0 = k1*n
        # k=k1, i=1 -> k1*n + 1
        global_w = [0] * (p * n)
        global_w[k0 * n + 0] = local_w[0]
        global_w[k0 * n + 1] = local_w[1]
        global_w[k1 * n + 0] = local_w[2]
        global_w[k1 * n + 1] = local_w[3]
        return global_w

    col = 0
    
    # Process 3 Strassen blocks for k=0,1; 2,3; 4,5
    for block_idx in range(3):
        k0 = block_idx * 2
        k1 = block_idx * 2 + 1
        
        terms = get_strassen_terms(k0, k1)
        for u_loc, v_loc, w_loc in terms:
            u_glob = [0] * (n * m)
            # Map local U [a, b, c, d] to global U [i*m+j]
            # i=0,j=0 (a) -> 0
            # i=0,j=1 (b) -> 1
            # i=1,j=0 (c) -> 2
            # i=1,j=1 (d) -> 3
            u_glob[0] = u_loc[0]
            u_glob[1] = u_loc[1]
            u_glob[2] = u_loc[2]
            u_glob[3] = u_loc[3]
            
            v_glob = map_local_to_global_v(v_loc, k0, k1)
            w_glob = map_local_to_global_w(w_loc, k0, k1)
            
            add_term(u_glob, v_glob, w_glob, col)
            col += 1

    # Process 1 Schoolbook block for k=6
    k0 = 6
    # C06 = a*B06 + b*B16
    # C16 = c*B06 + d*B16
    
    # Term 1: a * B06
    u1 = [1, 0, 0, 0]
    v1_loc = [1, 0] # B06, B16
    w1_loc = [1, 0] # C06, C16
    add_term(u1, map_local_to_global_v([v1_loc[0], 0, v1_loc[1], 0], k0, k0), map_local_to_global_w([w1_loc[0], 0, 0, w1_loc[1]], k0, k0), col)
    # Wait, map_local_to_global_v expects [e, f, g, h]. For single column, f=h=0.
    # So v_loc for single col k is [B0k, 0, B1k, 0] in the 4-element format?
    # No, my map_local_to_global_v takes 4 elements.
    # For k=6, we only have k0=6, k1 doesn't exist in the same way.
    # Let's just build V and W manually for the last block to be safe.
    
    # Reset col handling for the last block to be explicit
    # Col is now 21.
    
    # Term 1: a * B06
    u_t1 = [0]*4; u_t1[0] = 1
    v_t1 = [0]*(m*p); v_t1[0*p + 6] = 1 # j=0, k=6
    w_t1 = [0]*(p*n); w_t1[6*n + 0] = 1 # k=6, i=0
    add_term(u_t1, v_t1, w_t1, col)
    col += 1
    
    # Term 2: b * B16
    u_t2 = [0]*4; u_t2[1] = 1
    v_t2 = [0]*(m*p); v_t2[1*p + 6] = 1 # j=1, k=6
    w_t2 = [0]*(p*n); w_t2[6*n + 0] = 1 # k=6, i=0
    add_term(u_t2, v_t2, w_t2, col)
    col += 1
    
    # Term 3: c * B06
    u_t3 = [0]*4; u_t3[2] = 1
    v_t3 = [0]*(m*p); v_t3[0*p + 6] = 1 # j=0, k=6
    w_t3 = [0]*(p*n); w_t3[6*n + 1] = 1 # k=6, i=1
    add_term(u_t3, v_t3, w_t3, col)
    col += 1
    
    # Term 4: d * B16
    u_t4 = [0]*4; u_t4[3] = 1
    v_t4 = [0]*(m*p); v_t4[1*p + 6] = 1 # j=1, k=6
    w_t4 = [0]*(p*n); w_t4[6*n + 1] = 1 # k=6, i=1
    add_term(u_t4, v_t4, w_t4, col)
    col += 1
    
    # Convert to lists of lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
