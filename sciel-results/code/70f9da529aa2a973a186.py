"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns: {'U': list of lists, 'V': list of lists, 'W': list of lists}
    """
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Schoolbook decomposition as baseline
    schoolbook_rank = n * m * p  # 63
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
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        n_, m_, p_ = n, m, p
        if len(U) != n_ * m_:
            return False
        if len(V) != m_ * p_:
            return False
        if len(W) != p_ * n_:
            return False
        if len(U) == 0 or len(V) == 0 or len(W) == 0:
            return False
        rank_ = len(U[0])
        if len(V[0]) != rank_ or len(W[0]) != rank_:
            return False
        if rank_ <= 0:
            return False
        
        # Convert to numpy for faster verification
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    expected = 1.0
                    actual = np.dot(U_np[i * m_ + j], V_np[j * p_ + k] * W_np[k * n_ + i])
                    if abs(actual - expected) > tol:
                        return False
        return True
    
    def make_result(U, V, W):
        """Convert to list-of-lists format with clean values."""
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        # Clean up near-integer and near-half-integer values
        def clean_val(x):
            r = round(x * 2) / 2  # snap to half-integers
            if abs(x - r) < 1e-8:
                return r
            r = round(x)
            if abs(x - r) < 1e-8:
                return float(r)
            return x
        U_list = [[clean_val(x) for x in row] for row in U_list]
        V_list = [[clean_val(x) for x in row] for row in V_list]
        W_list = [[clean_val(x) for x in row] for row in W_list]
        return {'U': U_list, 'V': V_list, 'W': W_list}
    
    def try_decomposition(U, V, W):
        """Try a decomposition and update best if valid and lower rank."""
        nonlocal best_result, best_rank
        if verify_decomposition(U, V, W):
            rank = len(U[0])
            if rank < best_rank:
                best_result = make_result(U, V, W)
                best_rank = rank
                return True
        return False
    
    # Strategy: Try structured decompositions based on block patterns
    
    # Coefficient set: small integers and half-integers
    coeff_set = [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]
    
    random.seed(42)
    
    # Try to find a decomposition with rank < 63
    # Start with random search for various ranks
    
    # For efficiency, try ranks from 50 down to 30
    for target_rank in range(50, 29, -1):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations for this rank
        num_trials = 50 if target_rank > 40 else 20
        
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break
            
            # Random sparse initialization with better structure
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            for r in range(target_rank):
                # Choose random non-zero positions with some structure
                # Bias towards patterns that might work
                u_idx = random.randint(0, n*m-1)
                v_idx = random.randint(0, m*p-1)
                w_idx = random.randint(0, p*n-1)
                
                u_val = random.choice(coeff_set)
                v_val = random.choice(coeff_set)
                w_val = random.choice(coeff_set)
                
                if u_val == 0 or v_val == 0 or w_val == 0:
                    # Retry with non-zero
                    u_val = random.choice([x for x in coeff_set if x != 0])
                    v_val = random.choice([x for x in coeff_set if x != 0])
                    w_val = random.choice([x for x in coeff_set if x != 0])
                
                U[u_idx, r] = u_val
                V[v_idx, r] = v_val
                W[w_idx, r] = w_val
            
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if verify_decomposition(U_list, V_list, W_list):
                try_decomposition(U_list, V_list, W_list)
                break  # Found one for this rank, try lower
    
    # Return the best result
    return best_result

# EVOLVE-BLOCK-END
