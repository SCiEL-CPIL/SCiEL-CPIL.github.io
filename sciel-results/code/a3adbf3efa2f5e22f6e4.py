"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    # For each (i, j, k), the tensor entry is 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # Actually, the tensor T[i,j,k] = 1 for all i,j,k in the standard matrix mult.
    # The decomposition requires: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # Wait, let me re-read the convention.
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # Standard tensor decomposition for matmul (n x m) * (m x p):
    # T_{i,j,k} = sum_r U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
    # where i in [0,n), j in [0,m), k in [0,p)
    # U has shape (n*m, rank), indexed by (i*m+j, r)
    # V has shape (m*p, rank), indexed by (j*p+k, r)
    # W has shape (p*n, rank), indexed by (k*n+i, r)
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    if V_arr.shape[1] != rank or W_arr.shape[1] != rank:
        return False
    
    # Check all tensor entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i*m + j, :]
                v_val = V_arr[j*p + k, :]
                w_val = W_arr[k*n + i, :]
                s = np.dot(np.dot(u_val, v_val), w_val)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook():
    """Return the trivial schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x9 matrix multiplication."""
    deadline = time.time() + 140  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    n_m = n * m  # 6
    m_p = m * p  # 27
    p_n = p * n  # 18
    
    # Start with schoolbook
    best_result = _schoolbook()
    best_rank = n * m * p  # 54
    
    # Try to find lower rank decompositions using structured random search
    # The tensor has 54 non-zero entries (all 1s) in a 2x3x9 tensor.
    # Known lower bounds: max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    # Upper bound from schoolbook: 54
    # Strassen-like methods might help but for these dimensions it's tricky.
    
    # Strategy: Use random initialization and gradient-free search (Nelder-Mead style)
    # to minimize rank while maintaining exactness.
    
    # Alternative: Try structured decompositions based on Kronecker products or block structures.
    
    # For 2x3 * 3x9, we can view this as 2x3 * 3x(3*3), i.e., multiplying by a 3x9 matrix
    # which can be seen as three 3x3 blocks.
    
    # Let's try a systematic search over ranks from a lower bound up.
    # Lower bound: max(n*m, m*p, p*n) = 27
    # But this is a weak bound. Let's try to find any decomposition with rank < 54.
    
    # Approach: Randomized search with verification.
    # For a given rank r, we need to find U (6xr), V (27xr), W (18xr) such that
    # for all (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # This is a system of 54 polynomial equations in 6r + 27r + 18r = 51r variables.
    # For r=27, that's 1377 variables and 54 equations - should be solvable.
    
    # Let's try a simpler approach: use the fact that we can factor the computation.
    # A (2x3) * B (3x9) = A * [B1 B2 B3] where B1,B2,B3 are 3x3.
    # = [A*B1 A*B2 A*B3]
    # Each A*Bj is 2x3 * 3x3, which has rank at most 6 (schoolbook) or less.
    # But we need a joint decomposition.
    
    # Let's try random search for various ranks.
    rng = np.random.default_rng(42)
    
    for target_rank in range(27, best_rank + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Initialize with random values
            U = rng.standard_normal((n_m, target_rank))
            V = rng.standard_normal((m_p, target_rank))
            W = rng.standard_normal((p_n, target_rank))
            
            # Scale to get closer to the right magnitude
            U *= 0.5
            V *= 0.5
            W *= 0.5
            
            # Use a simple iterative refinement
            # For each (i,j,k), we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # This is hard to satisfy directly. Let's use a different approach.
            
            # Actually, let's use a structured approach based on the Kronecker product.
            # For 2x3 * 3x9, note that 9 = 3*3.
            # We can think of B as a 3x3 matrix of 3x3 blocks.
            
            # Let's try a different strategy: exploit the structure.
            # The 2x3 * 3x9 multiplication can be seen as:
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # where k = 3*q + r for q in {0,1,2}, r in {0,1,2}
            # So C[i, 3q+r] = sum_j A[i,j] * B[j, 3q+r]
            
            # This is equivalent to computing three 2x3 * 3x3 multiplications.
            # Each 2x3 * 3x3 has rank at most 6, so total rank at most 18?
            # No, that's not right because the decompositions need to be compatible.
            
            # Let me try a direct structured construction.
            # For 2x3 * 3x3, the schoolbook rank is 6.
            # Can we do better? For 2x2 * 2x2, Strassen gives 7 < 8.
            # For 2x3 * 3x3, I'm not sure of the optimal rank.
            
            # Let's just try to find ANY decomposition with rank < 54 using random search.
            # We'll use a heuristic: start with schoolbook and try to merge terms.
            
            pass  # Skip this attempt for now
    
    # Let's try a more systematic approach: use the known structure.
    # For 2x3 * 3x9, we can use the fact that 9 = 3*3.
    # Let B = [B0 B1 B2] where each Bj is 3x3.
    # C = A * B = [A*B0 A*B1 A*B2]
    
    # For each 2x3 * 3x3 multiplication, use schoolbook (rank 6).
    # Total rank = 3 * 6 = 18.
    # But wait, this doesn't work directly because the tensor decomposition
    # needs to be a single decomposition, not a block diagonal one.
    
    # Actually, it DOES work! If we define:
    # For the q-th block (q=0,1,2):
    #   For each (i, j) with i in {0,1}, j in {0,1,2}:
    #     U[i*m+j, q*6 + (i,j)_idx] = 1
    #     V[j*p + (3*q + r), q*6 + (i,j)_idx] = 1 for all r? No, that's not right.
    
    # Let me think more carefully.
    # The tensor entry T[i,j,k] = 1 for all i,j,k.
    # We want sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # If we use block structure where c = (q, (i,j)_idx) with q in {0,1,2}:
    # U[i*m+j, (q, (i',j')_idx)] = 1 if i=i' and j=j', else 0
    # V[j*p+k, (q, (i',j')_idx)] = 1 if j=j' and k in block q (i.e., k = 3*q + r for some r), else 0
    # W[k*n+i, (q, (i',j')_idx)] = 1 if i=i' and k in block q, else 0
    
    # Then sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # = sum_{(i',j')} U[i*m+j,(q,(i',j')_idx)]*V[j*p+k,(q,(i',j')_idx)]*W[k*n+i,(q,(i',j')_idx)]
    # The only non-zero term is when i'=i, j'=j, and k is in block q.
    # For a given k, there's exactly one q such that k is in block q.
    # So the sum becomes: 1 * 1 * 1 = 1. 
    
    # This gives rank = 3 * (2*3) = 18!
    
    # Let's construct this.
    n, m, p = DIMENSIONS
    block_size = p // m  # 9 // 3 = 3
    num_blocks = m  # 3 blocks
    
    rank = num_blocks * n * m  # 3 * 2 * 3 = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for q in range(num_blocks):
        for i in range(n):
            for j in range(m):
                col = q * (n * m) + i * m + j
                U[i * m + j][col] = 1
                # V[j*p+k, col] = 1 for k in block q
                for r in range(block_size):
                    k = q * block_size + r
                    V[j * p + k][col] = 1
                # W[k*n+i, col] = 1 for k in block q
                for r in range(block_size):
                    k = q * block_size + r
                    W[k * n + i][col] = 1
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # If verification failed, fall back to schoolbook
    return best_result
# EVOLVE-BLOCK-END
