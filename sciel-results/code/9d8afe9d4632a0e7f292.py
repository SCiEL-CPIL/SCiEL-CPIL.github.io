"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose matrix multiplication."""
    rank = len(U[0]) if len(U) > 0 else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # The matrix multiplication tensor T[i,j,k] = 1 for all (i,j,k)
                # C[i,k] = sum_j A[i,j] * B[j,k]
                # So T[i,j,k] = 1 for every valid triple
                actual = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(actual - 1.0) > tol:
                    return False
    return True

def _try_greedy_rank(target_rank, n, m, p, time_budget=30.0):
    """
    Try to find a rank-`target_rank` decomposition using greedy selection
    from a pool of simple (integer/half-integer) rank-1 candidates.
    
    The idea: the matrix multiplication tensor for n x m times m x p
    has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    A rank-1 term is defined by three vectors:
      u: length n*m, u[i*m+j] = a_{i,j}
      v: length m*p, v[j*p+k] = b_{j,k}  
      w: length p*n, w[k*n+i] = c_{k,i}
    
    The contribution of this rank-1 term to T[i,j,k] is a_{i,j} * b_{j,k} * c_{k,i}.
    
    We want sum over r of a^{(r)}_{i,j} * b^{(r)}_{j,k} * c^{(r)}_{k,i} = 1 for all (i,j,k).
    
    Strategy: generate candidate rank-1 terms with small integer/half-integer entries,
    then greedily select a subset that satisfies all 24 equations.
    """
    start_time = time.time()
    
    # Generate candidate vectors with entries in {-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2}
    # But that's too many. Let's use a smaller set: {-1, 0, 1} first, then {-1, -0.5, 0, 0.5, 1}
    
    values_small = [-1, 0, 1]
    values_medium = [-1, -0.5, 0, 0.5, 1]
    
    # For each candidate, we need vectors of length n*m=6, m*p=12, p*n=8
    # That's 3^6 * 3^12 * 3^8 = way too many to enumerate fully.
    
    # Instead, let's use a structured approach:
    # The trivial decomposition uses 24 terms where each term has exactly one 1 in each vector.
    # We can try to merge pairs of terms.
    
    # Alternative: use known algebraic constructions.
    # For 2x3 x 3x4, let's try to find a rank-20 or lower decomposition.
    
    # Let me try a different approach: set up the system of equations and solve.
    # We have 24 equations (one for each (i,j,k)) and rank * (6 + 12 + 8) = 26*rank unknowns.
    # For rank=20, that's 520 unknowns and 24 equations - very underdetermined.
    # But we also need the entries to be half-integers, which is a discrete constraint.
    
    # Let's try: start with the trivial 24-term decomposition and try to merge terms.
    # Two terms r1 and r2 can be merged if their combined contribution is still valid.
    
    # Actually, let's try a more direct approach: 
    # Use the fact that we can write the tensor as a sum of rank-1 terms.
    # Start with a random initialization and use iterative refinement.
    
    # For now, let's try a simple greedy approach:
    # 1. Start with all 24 trivial terms.
    # 2. Try to remove terms by adjusting the remaining ones.
    
    # Trivial decomposition:
    # Term (i,j,k): U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    
    # Let's try to find a lower rank by solving a linear system.
    # Fix U and V, solve for W, or vice versa.
    
    # Approach: 
    # 1. Choose U and V with small entries.
    # 2. For each (i,j,k), we need sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1.
    # 3. This is linear in W for fixed U, V.
    
    # Let's try: set U and V to have entries in {-1, 0, 1}, then solve for W.
    
    np.random.seed(42)
    
    for attempt in range(5000):
        if time.time() - start_time > time_budget:
            break
        
        rank = target_rank
        n_m = n * m  # 6
        m_p = m * p  # 12
        p_n = p * n  # 8
        
        # Generate random U and V with entries in {-1, 0, 1}
        U = np.random.choice([-1, 0, 1], size=(n_m, rank))
        V = np.random.choice([-1, 0, 1], size=(m_p, rank))
        
        # For each (i,j,k), we have the equation:
        # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        # Let A_{(i,j,k), r} = U[i*m+j,r] * V[j*p+k,r]
        # Then sum_r A_{(i,j,k), r} * W[k*n+i, r] = 1
        
        # This is a system of 24 equations in 8*rank unknowns (W entries).
        # But the unknowns are shared across equations in a structured way.
        
        # Let's set up the full system:
        # Variables: W[k*n+i, r] for k in [0,p), i in [0,n), r in [0,rank)
        # That's p*n*rank = 8*rank variables.
        
        # Equations: for each (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i, r] = 1
        
        num_vars = p_n * rank
        num_eqs = n * m * p  # 24
        
        # Build the system
        A_mat = np.zeros((num_eqs, num_vars))
        b_vec = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        var_idx = (k * n + i) * rank + r
                        A_mat[eq_idx, var_idx] = U[i*m+j, r] * V[j*p+k, r]
                    eq_idx += 1
        
        # Solve the underdetermined system: find a solution W that minimizes ||W||
        # Then check if W has half-integer entries (or can be rounded to half-integers)
        
        try:
            # Use least squares to find a minimum-norm solution
            W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        except:
            continue
        
        # Reshape W
        W = W_flat.reshape(p_n, rank)
        
        # Check if the solution is close to half-integers
        W_rounded = np.round(W * 2) / 2
        
        # Verify the decomposition with rounded values
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W_rounded.tolist()
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = sum(U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r] 
                                for r in range(rank))
                    if abs(actual - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _try_structured_rank(target_rank, n, m, p, time_budget=30.0):
    """
    Try structured constructions for lower rank.
    
    For 2x3 x 3x4 matrix multiplication, let's try specific algebraic constructions.
    
    One approach: use the fact that C = A*B where A is 2x3 and B is 3x4.
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    C[0,3] = A[0,0]*B[0,3] + A[0,1]*B[1,3] + A[0,2]*B[2,3]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    C[1,3] = A[1,0]*B[0,3] + A[1,1]*B[1,3] + A[1,2]*B[2,3]
    
    Let's try to share computations across columns of B.
    """
    start_time = time.time()
    
    # Try a specific construction:
    # For each row i of A, compute A[i,:] * B
    # A[0,:] * B = [A[0,0]*B[:,0] + A[0,1]*B[:,1] + A[0,2]*B[:,2]]
    
    # Let's try to use intermediate sums:
    # S0 = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] for each k
    # S1 = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] for each k
    
    # This is just the schoolbook approach, rank 24.
    
    # Let's try a Strassen-like approach:
    # Group the 3 columns of A and 4 rows of B.
    
    # Actually, let's try to find a rank-20 decomposition by solving a system.
    # We'll fix the structure of U and V to have a specific pattern, then solve for W.
    
    # Pattern: U has blocks corresponding to rows of A.
    # For row 0 of A (indices 0,1,2 in U), use a pattern.
    # For row 1 of A (indices 3,4,5 in U), use another pattern.
    
    # Let's try: U[i*m+j, r] depends on i and j in a structured way.
    
    # For now, let's try a simple approach: 
    # Use 20 terms where each term handles a subset of the 24 (i,j,k) triples.
    
    # This is getting complex. Let's fall back to the random search with refinement.
    
    return None

def _refinement_search(target_rank, n, m, p, time_budget=60.0):
    """
    More careful search: start from a near-solution and refine.
    
    1. Generate random U, V with half-integer entries.
    2. Solve for W using least squares.
    3. Round W to nearest half-integers.
    4. Check if the decomposition is valid.
    5. If not, try to fix the errors by adjusting U, V, W.
    """
    start_time = time.time()
    np.random.seed(123)
    
    n_m = n * m  # 6
    m_p = m * p  # 12
    p_n = p * n  # 8
    
    for attempt in range(20000):
        if time.time() - start_time > time_budget:
            break
        
        rank = target_rank
        
        # Generate U and V with entries in {-1, -0.5, 0, 0.5, 1}
        vals = np.array([-1, -0.5, 0, 0.5, 1])
        U = np.random.choice(vals, size=(n_m, rank))
        V = np.random.choice(vals, size=(m_p, rank))
        
        # Build the linear system for W
        num_vars = p_n * rank
        num_eqs = n * m * p
        
        A_mat = np.zeros((num_eqs, num_vars))
        b_vec = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        var_idx = (k * n + i) * rank + r
                        A_mat[eq_idx, var_idx] = U[i*m+j, r] * V[j*p+k, r]
                    eq_idx += 1
        
        # Check if the system is consistent
        # Use least squares
        try:
            W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        except:
            continue
        
        W = W_flat.reshape(p_n, rank)
        W_rounded = np.round(W * 2) / 2
        
        # Verify
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W_rounded.tolist()
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = sum(U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r] 
                                for r in range(rank))
                    if abs(actual - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin out of 150s
    
    n, m, p = DIMENSIONS
    
    # Try decreasing ranks
    for target_rank in [20, 19, 18, 17, 16]:
        if time.time() > deadline - 10:
            break
        
        # Try greedy approach
        result = _try_greedy_rank(target_rank, n, m, p, 
                                   time_budget=min(20, deadline - time.time() - 5))
        if result is not None:
            return result
        
        # Try refinement search
        result = _refinement_search(target_rank, n, m, p,
                                     time_budget=min(20, deadline - time.time() - 5))
        if result is not None:
            return result
    
    # Fallback: trivial rank-24 decomposition
    rank = n * m * p  # 24
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
# EVOLVE-BLOCK-END
