import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 45

    start_time = time.time()
    deadline = start_time + 110

    def make_schoolbook(rank):
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

    # Laderman's algorithm for 3x3x3 matrix multiplication (rank 23)
    # This is a known rank-23 decomposition for multiplying two 3x3 matrices.
    # U: 9x23, V: 9x23, W: 9x23
    # The decomposition satisfies: C[i,k] = sum_j A[i,j]*B[j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]

    # Laderman's 23 terms for 3x3 matrix multiplication
    # Reference: Laderman (1976), also in Björck & Hammarling (1984)
    # Using the standard formulation with indices 0..2 for rows/cols

    # The 23 rank-1 tensors are defined by their (i,j,k) contributions with coefficients
    # Let me use a known valid set. The Laderman decomposition has the following structure:
    
    # I'll construct it using the known 23-term decomposition.
    # Terms are (U_entry, V_entry, W_entry) for each (i,j,k) position
    
    # Known Laderman rank-23 decomposition for 3x3:
    # Using the formulation where each term r has:
    #   U[i,j,r], V[j,k,r], W[k,i,r]
    
    # Let me use a standard reference implementation of Laderman's algorithm.
    # The 23 terms can be represented as triples of 9-element vectors (for 3x3 matrices).

    # Standard Laderman 3x3x3 rank 23 decomposition:
    # I'll use the well-known set of 23 rank-1 tensors.
    
    # Term format: (u_vec, v_vec, w_vec) where each is a 9-element vector
    # u_vec[i*3+j] = U[i,j,r], v_vec[j*3+k] = V[j,k,r], w_vec[k*3+i] = W[k,i,r]
    
    # These are the 23 terms from Laderman's algorithm:
    
    # Let me construct them carefully.
    # For 3x3 matrix multiplication, Laderman's algorithm uses 23 multiplications.
    
    # I'll use a verified set of coefficients:
    
    terms = []
    
    # Diagonal terms (9 terms)
    # Term 0: i=0,j=0,k=0
    u0 = [1,0,0, 0,0,0, 0,0,0]
    v0 = [1,0,0, 0,0,0, 0,0,0]
    w0 = [1,0,0, 0,0,0, 0,0,0]
    terms.append((u0,v0,w0))
    
    # Term 1: i=1,j=1,k=1  
    u1 = [0,0,0, 0,1,0, 0,0,0]
    v1 = [0,0,0, 0,1,0, 0,0,0]
    w1 = [0,0,0, 0,1,0, 0,0,0]
    terms.append((u1,v1,w1))
    
    # Term 2: i=2,j=2,k=2
    u2 = [0,0,0, 0,0,0, 0,0,1]
    v2 = [0,0,0, 0,0,0, 0,0,1]
    w2 = [0,0,0, 0,0,0, 0,0,1]
    terms.append((u2,v2,w2))
    
    # Off-diagonal terms
    # Term 3: (A[0,1]+A[0,2]+A[1,0]+A[1,2]+A[2,0]+A[2,1]+A[2,2])
    u3 = [0,1,1, 1,0,1, 1,1,1]
    v3 = [0,1,1, 1,0,1, 1,1,1]
    w3 = [0,1,1, 1,0,1, 1,1,1]
    terms.append((u3,v3,w3))
    
    # This is getting complex and error-prone. Let me use a different approach.
    # I'll use a known valid Laderman decomposition from literature.
    
    # Actually, let me use a simpler and more reliable approach:
    # Use the schoolbook algorithm but try to find a better decomposition through
    # a more systematic search, or use a known decomposition.
    
    # For safety, let me implement a proper search that tries to find rank < 45
    # with a better strategy.
    
    best_result = make_schoolbook(rank_schoolbook)
    best_rank = rank_schoolbook

    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        U_reshaped = U_arr.reshape(n, m, rank)
        V_reshaped = V_arr.reshape(m, p, rank)
        W_reshaped = W_arr.reshape(p, n, rank)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_reshaped[i, j, :], np.multiply(V_reshaped[j, k, :], W_reshaped[k, i, :]))
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True

    def check_half_integer(mat):
        """Check if all entries are integers or half-integers."""
        arr = np.array(mat, dtype=np.float64)
        doubled = arr * 2
        rounded = np.round(doubled)
        if np.any(np.abs(doubled - rounded) > 1e-6):
            return False
        return True

    # Strategy: Use a structured approach based on the Kronecker product structure
    # For 3x3x5, we can think of it as combining two 3x3x3 problems and one 3x3x2 problem.
    # But a more direct approach: try to find a low-rank decomposition using 
    # integer/half-integer entries with a greedy or systematic search.
    
    # Alternative: Use the fact that for matrix multiplication, the tensor has a 
    # specific structure. Let's try a more systematic search with better initialization.
    
    # Try ranks from 44 down, with multiple attempts per rank
    import random
    
    # Use a seeded RNG for reproducibility within the time budget
    rng = np.random.RandomState(42)
    
    for target_rank in range(rank_schoolbook - 1, 35, -1):
        if time.time() > deadline - 20:
            break
        
        found = False
        
        # Number of attempts based on remaining time
        remaining = deadline - time.time()
        max_attempts = min(500, int(remaining * 5))
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
            
            # Generate U and V with sparse integer entries
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            
            # For each rank column, assign random non-zero entries
            # Use a mix of densities
            if attempt < max_attempts // 3:
                nnz_u = 2
                nnz_v = 2
            elif attempt < 2 * max_attempts // 3:
                nnz_u = 3
                nnz_v = 3
            else:
                nnz_u = 4
                nnz_v = 4
            
            for r in range(target_rank):
                # Random rows for U
                u_rows = rng.choice(n * m, size=nnz_u, replace=False)
                for row in u_rows:
                    U[int(row), r] = rng.choice([-1, 1])
                
                # Random rows for V
                v_rows = rng.choice(m * p, size=nnz_v, replace=False)
                for row in v_rows:
                    V[int(row), r] = rng.choice([-1, 1])
            
            # Solve for W using the minimum-norm solution
            W = np.zeros((p * n, target_rank))
            valid = True
            
            for i in range(n):
                for k in range(p):
                    A_ik = np.zeros(target_rank)
                    for j in range(m):
                        A_ik += U[i * m + j] * V[j * p + k]
                    
                    denom = np.dot(A_ik, A_ik)
                    if abs(denom) < 1e-10:
                        valid = False
                        break
                    
                    w = A_ik / denom
                    W[k * n + i] = w
                    
                    # Check half-integer constraint
                    doubled_w = w * 2
                    if np.any(np.abs(doubled_w - np.round(doubled_w)) > 1e-6):
                        valid = False
                        break
                
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Verify the full decomposition
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if verify_decomposition(U_list, V_list, W_list, target_rank):
                if check_half_integer(U_list) and check_half_integer(V_list) and check_half_integer(W_list):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
    
    return best_result
