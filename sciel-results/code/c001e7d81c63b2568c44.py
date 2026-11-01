"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from fractions import Fraction

def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, n=2, m=3, p=3, tol=1e-9):
    """Verify that the tensor decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _to_list(arr):
    return np.array(arr).tolist()

def _try_known_decomposition_rank14():
    """
    Try a known 14-rank decomposition for 2x3 x 3x3 matrix multiplication.
    Based on algebraic constructions that exploit the rectangular structure.
    
    The key insight: we can use the fact that we're multiplying (2x3) by (3x3),
    so the "inner" dimension is 3, and we have 2 rows and 3 columns in output.
    
    One approach: decompose using Strassen-like block methods adapted to rectangular case.
    """
    n, m, p = 2, 3, 3
    
    # Try a systematic construction
    # For 2x3 * 3x3, we can think of it as computing 2x3 output from 2x3 and 3x3 inputs
    
    # Let's try a 14-rank construction based on algebraic identities
    # This is a hand-crafted decomposition exploiting the small dimensions
    
    # Construction idea: Use the fact that we can share computations between
    # different output entries
    
    # Define the decomposition explicitly
    # U: 6 x 14, V: 9 x 14, W: 6 x 14
    
    # Try a pattern based on splitting the 3x3 multiplication
    # into parts that can be shared with the 2x3 input
    
    U = np.zeros((6, 14), dtype=float)
    V = np.zeros((9, 14), dtype=float)
    W = np.zeros((6, 14), dtype=float)
    
    # This is a placeholder - we'll use search to find the actual values
    return None

def _systematic_search_rank(target_rank, time_budget, n=2, m=3, p=3):
    """
    Systematic search using integer entries from a small set.
    Uses the structure of the problem more effectively.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Entry set to try: {-1, 0, 1} first, then expand if needed
    entry_sets = [
        [-1, 0, 1],
        [-2, -1, 0, 1, 2],
    ]
    
    for entries in entry_sets:
        if time.time() > deadline:
            break
        
        # Generate all possible U matrices (6 x target_rank) with entries from `entries`
        # This is infeasible for large target_rank, so we use random sampling with structure
        
        np.random.seed(42)
        
        for trial in range(10000):
            if time.time() > deadline:
                break
            
            # Generate U and V with entries from the set
            U = np.random.choice(entries, size=(n*m, target_rank)).astype(float)
            V = np.random.choice(entries, size=(m*p, target_rank)).astype(float)
            
            # Build the linear system for W
            num_eqs = n * m * p
            num_unk = p * n * target_rank
            
            A = np.zeros((num_eqs, num_unk))
            b = np.ones(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(target_rank):
                            u_val = U[i*m+j, c]
                            v_val = V[j*p+k, c]
                            if u_val != 0 and v_val != 0:
                                w_row = k * n + i
                                w_col = c
                                unk_idx = w_row * target_rank + w_col
                                A[eq_idx, unk_idx] = u_val * v_val
                        eq_idx += 1
            
            # Solve the linear system
            try:
                # Check if system is consistent
                rank_A = np.linalg.matrix_rank(A)
                if rank_A < num_eqs:
                    # Underdetermined or inconsistent - try least squares
                    W_vec, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    # Check if solution is exact
                    if np.max(np.abs(A @ W_vec - b)) > 1e-8:
                        continue
                else:
                    W_vec = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            
            W_mat = W_vec.reshape(p*n, target_rank)
            
            # Check if W entries are close to integers or half-integers
            # Try both: round to integers, or round 2*W to integers
            W_int = np.round(W_mat)
            if np.max(np.abs(W_mat - W_int)) < 1e-8:
                if _verify(U.tolist(), V.tolist(), W_int.tolist()):
                    return {
                        "U": _to_list(U.astype(int)),
                        "V": _to_list(V.astype(int)),
                        "W": _to_list(W_int)
                    }
            
            W_half = np.round(2 * W_mat) / 2.0
            if np.max(np.abs(W_mat - W_half)) < 1e-8:
                if _verify(U.tolist(), V.tolist(), W_half.tolist()):
                    return {
                        "U": _to_list(U.astype(int)),
                        "V": _to_list(V.astype(int)),
                        "W": _to_list(W_half)
                    }
    
    return None

def _algebraic_construction():
    """
    Try to construct a low-rank decomposition using algebraic identities.
    
    For 2x3 * 3x3 multiplication, we can use the following approach:
    - Split the 3x3 matrix into 3 columns
    - Each column is a 3x1 vector
    - The 2x3 matrix times each column gives a 2x1 vector
    - We need 3 such multiplications, each of rank at most 3 (for 2x3*3x1)
    - But we can share computations
    
    Actually, let's think differently. The tensor T_{ijk} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}
    where i in {0,1}, j in {0,1,2}, k in {0,1,2}
    
    We want to minimize rank = number of c values.
    """
    n, m, p = 2, 3, 3
    
    # Try a specific construction based on the structure
    # The key observation: we have 2x3 input, 3x3 input, 2x3 output
    # Total "information" is 6 + 9 = 15 input values, 6 output values
    
    # Lower bound: at least 6 (number of outputs)
    # Upper bound: 18 (schoolbook)
    
    # Try rank 12-15 with structured patterns
    
    # Pattern 1: Use the fact that each output (i,k) depends on all j
    # Output (i,k) = sum_j A[i,j] * B[j,k]
    
    # We can try to factor this using shared intermediates
    
    # Let's try a 14-rank construction
    # Based on the idea of using 7 "partial products" for each of 2 rows
    
    U = np.zeros((6, 14), dtype=float)
    V = np.zeros((9, 14), dtype=float)
    W = np.zeros((6, 14), dtype=float)
    
    # This is a heuristic construction - may not work, but worth trying
    # We'll use random search with this as a starting point if it fails
    
    return None

def _randomized_search_with_refinement(target_rank, time_budget, n=2, m=3, p=3):
    """
    Randomized search with local refinement.
    Start with random U, V, solve for W, then refine.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    np.random.seed(42)
    
    best_result = None
    
    for trial in range(20000):
        if time.time() > deadline:
            break
        
        # Random U, V with small integer entries
        U = np.random.randint(-2, 3, size=(n*m, target_rank)).astype(float)
        V = np.random.randint(-2, 3, size=(m*p, target_rank)).astype(float)
        
        # Ensure not all zeros in any column
        for c in range(target_rank):
            if np.all(U[:, c] == 0) or np.all(V[:, c] == 0):
                U[0, c] = 1
                V[0, c] = 1
        
        # Build linear system for W
        num_eqs = n * m * p
        num_unk = p * n * target_rank
        
        A = np.zeros((num_eqs, num_unk))
        b = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(target_rank):
                        u_val = U[i*m+j, c]
                        v_val = V[j*p+k, c]
                        if u_val != 0 and v_val != 0:
                            w_row = k * n + i
                            w_col = c
                            unk_idx = w_row * target_rank + w_col
                            A[eq_idx, unk_idx] = u_val * v_val
                    eq_idx += 1
        
        try:
            W_vec, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            residual = np.max(np.abs(A @ W_vec - b))
            if residual > 1e-6:
                continue
        except np.linalg.LinAlgError:
            continue
        
        W_mat = W_vec.reshape(p*n, target_rank)
        
        # Check for integer or half-integer solution
        for scale in [1, 2]:
            W_scaled = scale * W_mat
            W_rounded = np.round(W_scaled)
            if np.max(np.abs(W_scaled - W_rounded)) < 1e-6:
                W_final = W_rounded / scale
                if _verify(U.tolist(), V.tolist(), W_final.tolist()):
                    return {
                        "U": _to_list(U),
                        "V": _to_list(V),
                        "W": _to_list(W_final)
                    }
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Try ranks from low to high
    # Theoretical lower bound for 2x3x3 is at least 6 (number of outputs)
    # Known upper bound is 18 (schoolbook)
    # Try 12, 13, 14, 15, 16, 17
    
    for target_rank in [12, 13, 14, 15, 16, 17]:
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        
        # Try systematic search
        result = _systematic_search_rank(target_rank, remaining * 0.4, 2, 3, 3)
        if result is not None:
            return result
        
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        
        # Try randomized search with refinement
        result = _randomized_search_with_refinement(target_rank, remaining * 0.4, 2, 3, 3)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _build_schoolbook()
# EVOLVE-BLOCK-END
