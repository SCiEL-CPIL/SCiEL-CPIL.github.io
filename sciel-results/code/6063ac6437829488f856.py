"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns: {'U': list of lists, 'V': list of lists, 'W': list of lists}
    """
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    # Schoolbook decomposition as baseline
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    best_result = {'U': U_sb, 'V': V_sb, 'W': W_sb}
    best_rank = schoolbook_rank
    
    # Try to find better decompositions using random search
    # The tensor for matrix multiplication C = A * B where A is n x m, B is m x p
    # has components T[i,j,k] = sum_l A[i,l] * B[l,k]
    # In tensor form: T[i,k,l] = A[i,l] * B[l,k]
    # We want to decompose this as sum_r U[i,r] * V[k,r] * W[l,r]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    # Actually, let me reconsider the tensor structure
    # For C = A * B, C[i,k] = sum_j A[i,j] * B[j,k]
    # The multiplication tensor T has T[i,k,j] = A[i,j] * B[j,k]
    # We decompose as: T[i,k,j] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Wait, the convention says W[k*n+i, r], so W is indexed by (k,i)
    
    # Let me think about this more carefully:
    # U has shape (n*m, rank) - indexed by (i,j) where i in [0,n), j in [0,m)
    # V has shape (m*p, rank) - indexed by (j,k) where j in [0,m), k in [0,p)
    # W has shape (p*n, rank) - indexed by (k,i) where k in [0,p), i in [0,n)
    
    # The decomposition is: for all i,j,k:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta(j,k) if we're doing identity
    # Actually no. The tensor for matrix multiplication is:
    #   T[i,j,k] = 1 if this combination contributes to C[i,k]
    # More precisely, the multiplication tensor M has:
    #   M[i,j,k] = 1 for the "1" entries in the standard basis
    # The actual tensor is: for each (i,j,k), the coefficient is 1 if j is the shared index
    # Standard form: M[i,k,j] where the tensor decomposes as sum_r u_r[i] * v_r[j] * w_r[k]
    
    # Let me use a different approach: try small ranks and use random initialization
    # with gradient-free optimization or structured patterns
    
    # Try ranks from a lower bound up
    # Lower bound: max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # But we can do better with structured decompositions
    
    # Try Strassen-like ideas: for 3x3 * 3x7
    # We can try to use block structure
    
    # For now, let's try a random search with small ranks
    # and verify if the decomposition is exact
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly represents matrix multiplication."""
        # Convert to numpy for efficient computation
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # For each (i, j, k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # equals the correct value for matrix multiplication
        # The correct value is: 1 if this is a valid triple (i,j,k) contributing to C[i,k]
        # Actually, the tensor T[i,j,k] for matrix multiplication satisfies:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor is T[i,j,k] = A[i,j] * B[j,k] for the standard basis
        # In the decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # We need to check that for all i,j,k:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) is a "valid" triple
        # But what are the valid triples? For matrix multiplication, ALL triples (i,j,k)
        # contribute, with coefficient 1. So we need:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        
        # Wait, that's not right either. Let me think again.
        # The multiplication tensor for C = A*B where A is n x m and B is m x p:
        # The tensor T has T[i,j,k] such that C[i,k] = sum_j A[i,j] B[j,k]
        # In tensor form, this is: T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the coefficient
        # for e_i ⊗ e_j ⊗ e_k is 1 (since A[i,j] and B[j,k] are independent variables)
        
        # So we need: for all i in [0,n), j in [0,m), k in [0,p):
        #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # Let's verify this
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    
    # Try random searches for small ranks
    random.seed(42)
    
    for rank in range(21, min(64, schoolbook_rank + 1)):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization with small values
            U_rand = [[random.choice([-1, 0, 1]) + random.uniform(-0.5, 0.5) for _ in range(rank)] for _ in range(n*m)]
            V_rand = [[random.choice([-1, 0, 1]) + random.uniform(-0.5, 0.5) for _ in range(rank)] for _ in range(m*p)]
            W_rand = [[random.choice([-1, 0, 1]) + random.uniform(-0.5, 0.5) for _ in range(rank)] for _ in range(p*n)]
            
            if verify_decomposition(U_rand, V_rand, W_rand, rank):
                best_result = {'U': U_rand, 'V': V_rand, 'W': W_rand}
                best_rank = rank
                break  # Found a valid decomposition at this rank, move to next rank
        
        if best_rank < rank:
            break  # Found a better rank, no need to try larger ranks
    
    return best_result
# EVOLVE-BLOCK-END
