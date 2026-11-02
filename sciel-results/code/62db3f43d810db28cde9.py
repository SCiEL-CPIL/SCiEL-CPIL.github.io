import time
import numpy as np

DIMENSIONS = [2, 3, 7]
DEADLINE = 150.0
SAFETY_MARGIN = 10.0

def construct():
    start = time.time()
    n, m, p = DIMENSIONS  # 2, 3, 7
    
    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def make_hybrid():
        """
        Hybrid Strassen/Schoolbook decomposition.
        Rank 39.
        """
        rank = 39
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        col = 0
        
        # Strassen 2x2x2 block
        def add_strassen_block(i_base, j_base, k_base):
            nonlocal col
            # Local indices 0,1 map to global i_base+i, j_base+j, k_base+k
            # u, v, w vectors are defined for local 2x2 matrices (4 elements)
            # We need to map these to the global U, V, W rows.
            
            # u is a 4-vector corresponding to A[i,j] for i,j in 0,1
            # Global U row for A[i_base+di, j_base+dj] is (i_base+di)*m + (j_base+dj)
            
            # v is a 4-vector corresponding to B[j,k] for j,k in 0,1
            # Global V row for B[j_base+dj, k_base+dk] is (j_base+dj)*p + (k_base+dk)
            
            # w is a 4-vector corresponding to C[i,k] for i,k in 0,1
            # Global W row for C[i_base+di, k_base+dk] is (k_base+dk)*n + (i_base+di)
            
            # Define the 7 terms (u, v, w) as lists of (local_idx, coeff)
            # Local idx for A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # Local idx for B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # Local idx for C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            
            terms = [
                # M1
                ([ (0, 1), (3, -1) ], [ (3, 1), (0, -1) ], [ (0, 1), (3, 1) ]),
                # M2
                ([ (0, 1), (3, 1) ], [ (3, 1) ], [ (2, 1), (3, -1) ]),
                # M3
                ([ (0, 1) ], [ (0, 1), (3, -1) ], [ (1, 1), (3, 1) ]),
                # M4
                ([ (3, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ]),
                # M5
                ([ (0, 1), (1, 1) ], [ (2, 1) ], [ (0, -1), (1, 1) ]),
                # M6
                ([ (2, 1), (3, -1) ], [ (0, 1), (1, 1) ], [ (2, 1) ]),
                # M7
                ([ (1, 1), (3, 1) ], [ (2, 1), (3, -1) ], [ (0, 1) ])
            ]
            
            for u_terms, v_terms, w_terms in terms:
                # Add to U
                for local_idx, coeff in u_terms:
                    di, dj = divmod(local_idx, 2) # 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> divmod gives (q,r)
                    # Wait, divmod(0,2)=(0,0), divmod(1,2)=(0,1), divmod(2,2)=(1,0), divmod(3,2)=(1,1). Correct.
                    global_i = i_base + di
                    global_j = j_base + dj
                    u_row = global_i * m + global_j
                    U[u_row, col] += coeff
                
                # Add to V
                for local_idx, coeff in v_terms:
                    dj, dk = divmod(local_idx, 2)
                    global_j = j_base + dj
                    global_k = k_base + dk
                    v_row = global_j * p + global_k
                    V[v_row, col] += coeff
                
                # Add to W
                for local_idx, coeff in w_terms:
                    di, dk = divmod(local_idx, 2)
                    global_i = i_base + di
                    global_k = k_base + dk
                    w_row = global_k * n + global_i
                    W[w_row, col] += coeff
                
                col += 1

        def add_schoolbook_block(i_set, j_set, k_set):
            nonlocal col
            for i in i_set:
                for j in j_set:
                    for k in k_set:
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1

        # 1. Strassen blocks (2x2x2)
        add_strassen_block(0, 0, 0) # i:{0,1}, j:{0,1}, k:{0,1}
        add_strassen_block(0, 0, 2) # i:{0,1}, j:{0,1}, k:{2,3}
        add_strassen_block(0, 0, 4) # i:{0,1}, j:{0,1}, k:{4,5}
        
        # 2. Schoolbook blocks
        # i:{0,1}, j:{0,1}, k:{6}
        add_schoolbook_block([0,1], [0,1], [6])
        
        # i:{0,1}, j:{2}, k:{0,1}
        add_schoolbook_block([0,1], [2], [0,1])
        
        # i:{0,1}, j:{2}, k:{2,3}
        add_schoolbook_block([0,1], [2], [2,3])
        
        # i:{0,1}, j:{2}, k:{4,5}
        add_schoolbook_block([0,1], [2], [4,5])
        
        # i:{0,1}, j:{2}, k:{6}
        add_schoolbook_block([0,1], [2], [6])
        
        if col != rank:
            return None
        return U, V, W

    # Try hybrid first
    U, V, W = make_hybrid()
    if U is None:
        U, V, W = make_schoolbook()
        
    return {"U": U, "V": V, "W": W}
