"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS

    def check_time():
        return time.time() > deadline

    def schoolbook():
        """Guaranteed valid schoolbook decomposition of rank n*m*p."""
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

    def verify_fast(U, V, W, n, m, p):
        """Fast verification using numpy einsum."""
        rank = len(U[0])
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check on-entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.einsum('r,r,r->', U_np[i*m+j], V_np[j*p+k], W_np[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        
        # Check off-entries (full tensor)
        T = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = 1.0
        if not np.allclose(T, target, atol=1e-9):
            return False
        return True

    def build_block_decomposition():
        """
        Build the rank-55 decomposition for M_{2,3,10}.
        It uses 5 blocks of M_{2,3,2}, each of rank 11.
        """
        # The 11 terms for M_{2,3,2}
        # Indices: U[i,j], V[j,k], W[k,i]
        # i in 0..1, j in 0..2, k in 0..1
        # Term format: (list of (i,j,val) for U, list of (j,k,val) for V, list of (k,i,val) for W)
        
        # Strassen-like terms for the 2x2 core (j=0,1) plus corrections for j=2
        # Reference: "The rank of the 2x3x2 matrix multiplication tensor is 11"
        
        terms = [
            # 1
            ([ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ]),
            # 2
            ([ (1,0,1), (1,1,1) ], [ (0,0,1) ], [ (0,1,1), (1,1,-1) ]),
            # 3
            ([ (0,0,1) ], [ (0,1,1), (1,1,-1) ], [ (1,0,1), (1,1,1) ]),
            # 4
            ([ (1,1,1) ], [ (1,0,1), (0,0,-1) ], [ (0,0,1), (0,1,1) ]),
            # 5
            ([ (0,0,1), (0,1,1) ], [ (1,1,1) ], [ (0,0,-1), (1,0,1) ]),
            # 6
            ([ (1,0,1), (0,0,-1) ], [ (0,0,1), (0,1,1) ], [ (1,1,1) ]),
            # 7
            ([ (0,1,1), (1,1,-1) ], [ (1,0,1), (1,1,1) ], [ (0,0,1) ]),
            # 8 (Outer product for j=2, k=0, i=0)
            ([ (0,2,1) ], [ (2,0,1) ], [ (0,0,1) ]),
            # 9 (Outer product for j=2, k=1, i=0)
            ([ (0,2,1) ], [ (2,1,1) ], [ (1,0,1) ]),
            # 10 (Outer product for j=2, k=0, i=1)
            ([ (1,2,1) ], [ (2,0,1) ], [ (0,1,1) ]),
            # 11 (Outer product for j=2, k=1, i=1)
            ([ (1,2,1) ], [ (2,1,1) ], [ (1,1,1) ]),
        ]
        
        num_blocks = p // 2 # 5 blocks
        block_size = 2
        rank_per_block = len(terms) # 11
        total_rank = num_blocks * rank_per_block
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col_offset = 0
        for b in range(num_blocks):
            for t_idx, (u_terms, v_terms, w_terms) in enumerate(terms):
                col = col_offset + t_idx
                for i, j, val in u_terms:
                    U[i * m + j][col] += val
                for j, k_local, val in v_terms:
                    k_global = b * block_size + k_local
                    V[j * p + k_global][col] += val
                for k_local, i, val in w_terms:
                    k_global = b * block_size + k_local
                    W[k_global * n + i][col] += val
            col_offset += rank_per_block
            
        return U, V, W

    # Build the rank-55 decomposition
    best_result = None
    try:
        U55, V55, W55 = build_block_decomposition()
        if verify_fast(U55, V55, W55, n, m, p):
            best_result = {"U": U55, "V": V55, "W": W55}
        else:
            best_result = schoolbook()
    except Exception:
        best_result = schoolbook()

    if check_time():
        return best_result

    # Optional: Attempt to find rank 54 (theoretical lower bound)
    # This is extremely unlikely to succeed with random search in 120s,
    # but we can try a very quick heuristic.
    # If we find a valid rank 54, we return it.
    # Given the time constraint, we will skip this to ensure we return the valid rank 55.
    
    return best_result
# EVOLVE-BLOCK-END
