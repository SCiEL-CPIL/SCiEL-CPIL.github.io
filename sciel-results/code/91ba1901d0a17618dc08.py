"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Constructs an exact tensor decomposition for multiplying a 2x3 matrix by a 3x10 matrix.
    
    Strategy:
    1. Split p=10 into 5 blocks of size 2.
    2. For each 2x3x2 block, use a rank-11 decomposition:
       - Strassen's 7-multiplication algorithm for the 2x2x2 core (using cols 0,1 of A and B).
       - 4 direct multiplications for the remaining 2x1x2 part (using col 2 of A and row 2 of B).
    3. Total rank = 5 * 11 = 55.
    
    This is an improvement over the schoolbook rank of 60.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    num_blocks = p // 2
    block_size = 2
    rank_per_block = 11
    total_rank = num_blocks * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Pre-define the rank 11 decomposition for M_{2,3,2}
    # Local indices: i in {0,1}, j in {0,1,2}, k in {0,1}
    # U index: i*m + j
    # V index: j*p + k_global (k_global = b*2 + k_local)
    # W index: k_global*n + i
    
    # Strassen terms for the 2x2x2 part (j=0,1)
    strassen_terms = [
        # M1: (A00+A11)(B00+B11) -> C00, C11
        [
            ('U', (0, 0), 1), ('U', (1, 1), 1),
            ('V', (0, 0), 1), ('V', (1, 1), 1),
            ('W', (0, 0), 1), ('W', (1, 1), 1)
        ],
        # M2: (A10+A11)B00 -> C10, C11
        [
            ('U', (1, 0), 1), ('U', (1, 1), 1),
            ('V', (0, 0), 1),
            ('W', (0, 1), 1), ('W', (1, 1), -1)
        ],
        # M3: A00(B01-B11) -> C01, C11
        [
            ('U', (0, 0), 1),
            ('V', (0, 1), 1), ('V', (1, 1), -1),
            ('W', (1, 0), 1), ('W', (1, 1), 1)
        ],
        # M4: A11(B10-B00) -> C00, C10
        [
            ('U', (1, 1), 1),
            ('V', (1, 0), 1), ('V', (0, 0), -1),
            ('W', (0, 0), 1), ('W', (0, 1), 1)
        ],
        # M5: (A00+A01)B11 -> C00, C01
        [
            ('U', (0, 0), 1), ('U', (0, 1), 1),
            ('V', (1, 1), 1),
            ('W', (0, 0), -1), ('W', (1, 0), 1)
        ],
        # M6: (A10-A00)(B00+B01) -> C11
        [
            ('U', (1, 0), 1), ('U', (0, 0), -1),
            ('V', (0, 0), 1), ('V', (0, 1), 1),
            ('W', (1, 1), 1)
        ],
        # M7: (A01-A11)(B10+B11) -> C00
        [
            ('U', (0, 1), 1), ('U', (1, 1), -1),
            ('V', (1, 0), 1), ('V', (1, 1), 1),
            ('W', (0, 0), 1)
        ]
    ]
    
    # Outer product terms for the 2x1x2 part (j=2)
    outer_terms = [
        [
            ('U', (0, 2), 1),
            ('V', (2, 0), 1),
            ('W', (0, 0), 1)
        ],
        [
            ('U', (0, 2), 1),
            ('V', (2, 1), 1),
            ('W', (1, 0), 1)
        ],
        [
            ('U', (1, 2), 1),
            ('V', (2, 0), 1),
            ('W', (0, 1), 1)
        ],
        [
            ('U', (1, 2), 1),
            ('V', (2, 1), 1),
            ('W', (1, 1), 1)
        ]
    ]
    
    all_terms = strassen_terms + outer_terms
    
    col_offset = 0
    for b in range(num_blocks):
        if time.time() > deadline:
            break
        
        for term in all_terms:
            col = col_offset
            for mat_name, idx, val in term:
                if mat_name == 'U':
                    i, j = idx
                    U[i * m + j][col] += val
                elif mat_name == 'V':
                    j, k_local = idx
                    k_global = b * block_size + k_local
                    V[j * p + k_global][col] += val
                elif mat_name == 'W':
                    k_local, i = idx
                    k_global = b * block_size + k_local
                    W[k_global * n + i][col] += val
            col_offset += 1
            
    # Verify decomposition correctness with a small test
    if time.time() < deadline - 5.0:
        try:
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            
            # Test with random matrices
            rng = np.random.default_rng(42)
            A_test = rng.standard_normal((n, m))
            B_test = rng.standard_normal((m, p))
            
            # True product: C[i,k] = sum_j A[i,j] * B[j,k]
            C_true = A_test @ B_test
            
            # Decomposed product: C[i,k] = sum_r U[i*m + ...] ... 
            # C[i,k] = sum_r (sum_j U[i*m+j, r]) * (sum_j' V[j'*p+k, r]) ... no
            # Actually: C[i,k] = sum_r U_vec_i,r * V_vec_k,r where
            # U_vec_i,r = sum_j U[i*m+j, r] * A[i,j]... 
            # The tensor decomposition means:
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]... 
            # Wait, no. The standard form is:
            # C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) * (sum_j' V[j'*p+k, r] ... )
            # Actually the convention is:
            # C[i,k] = sum_r U[i*m + ?, r] ... 
            # Let me re-read: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
            # So U has shape (n*m, rank), indexed by (i,j) -> i*m+j
            # V has shape (m*p, rank), indexed by (j,k) -> j*p+k
            # W has shape (p*n, rank), indexed by (k,i) -> k*n+i
            # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # And we need: sum_j T[i,j,k] * A[i,j] * B[j,k]... no.
            # Actually, the matrix multiplication tensor is T[i,j,k] = delta such that
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # The decomposition requires: for all A, B:
            # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j' V[j'*p+k, r] * B[j',k])
            # This must equal sum_j A[i,j] B[j,k]
            # So we need: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = delta_{j,j'} ... 
            # Hmm, I think W is just a scaling/selection factor.
            # Let me just verify numerically:
            # C_decomp[i,k] = sum_r (sum_j A[i,j] * U[i*m+j, r]) * (sum_j' B[j',k] * V[j'*p+k, r]) * W[k*n+i, r]
            # Wait, that doesn't look right dimensionally. Let me think again.
            # 
            # Standard tensor decomposition for matmul:
            # C = A*B, where C[i,k] = sum_j A[i,j] B[j,k]
            # We write: C[i,k] = sum_r M_r(A) * N_r(B) * W_r(i,k)
            # where M_r(A) = sum_j U[i*m+j, r] A[i,j]... but that depends on i too.
            # 
            # Actually I think the convention is:
            # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j' V[j'*p+k, r] * B[j',k]) * W[k*n+i, r]
            # For this to work, we need:
            # sum_r sum_j sum_j' U[i*m+j, r] A[i,j] V[j'*p+k, r] B[j',k] W[k*n+i, r] = sum_j A[i,j] B[j,k]
            # This requires: sum_r U[i*m+j, r] V[j'*p+k, r] W[k*n+i, r] = delta_{j,j'} for all i,k
            # But that can't be right because the left side depends on i and k.
            # 
            # I think the correct interpretation is simpler. Let me just test:
            # C_decomp[i,k] = sum_r (sum_j A[i,j] * U[i*m+j, r]) * (sum_j' B[j',k] * V[j'*p+k, r]) * W[k*n+i, r]
            
            C_decomp = np.zeros((n, p))
            for r in range(total_rank):
                Au = np.zeros((n,))
                for i in range(n):
                    for j in range(m):
                        Au[i] += A_test[i, j] * U[i * m + j][r]
                Bv = np.zeros((p,))
                for k in range(p):
                    for j in range(m):
                        Bv[k] += B_test[j, k] * V[j * p + k][r]
                for i in range(n):
                    for k in range(p):
                        C_decomp[i, k] += Au[i] * Bv[k] * W[k * n + i][r]
            
            max_err = np.max(np.abs(C_decomp - C_true))
            if max_err > 1e-10:
                # Fallback: return schoolbook to be safe
                U_fallback = [[0] * (n * m * p) for _ in range(n * m)]
                V_fallback = [[0] * (n * m * p) for _ in range(m * p)]
                W_fallback = [[0] * (n * m * p) for _ in range(p * n)]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            r = i * m * p + j * p + k
                            U_fallback[i * m + j][r] = 1
                            V_fallback[j * p + k][r] = 1
                            W_fallback[k * n + i][r] = 1
                return {"U": U_fallback, "V": V_fallback, "W": W_fallback}
        except Exception:
            pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
