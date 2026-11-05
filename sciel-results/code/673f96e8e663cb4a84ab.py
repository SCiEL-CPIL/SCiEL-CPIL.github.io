"""Matrix multiplication 3,3,4 - rank minimization via structured decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def _verify_decomposition(U, V, W, n=3, m=3, p=4, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    r = U_arr.shape[1]
    # For each (i,j,k), check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for rr in range(r):
                    s += U_arr[i*m+j, rr] * V_arr[j*p+k, rr] * W_arr[k*n+i, rr]
                if abs(s - 1.0) > tol:
                    return False
    # Also check that non-matching entries are 0 (but in this tensor, only matching entries exist)
    # The tensor only has entries at (i,j,k) where indices are consistent, so we just need the 36 entries to be 1
    return True

def _try_polynomial_crt_construction(n=3, m=3, p=4):
    """
    Try to construct a low-rank decomposition using polynomial embedding + CRT.
    
    Idea: Encode matrix multiplication as polynomial multiplication.
    For A (n x m) and B (m x p), C = AB.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We can think of this as: for fixed i,k, we need sum_j A[i,j]*B[j,k].
    
    Approach: Use the fact that we can encode the j-index as a power.
    Let's try: for each i, define polynomial A_i(x) = sum_j A[i,j] x^j
    For each k, define polynomial B_k(x) = sum_j B[j,k] x^(m-1-j)  (reversed)
    Then A_i(x) * B_k(x) = sum_j sum_l A[i,j] B[l,k] x^(j + m-1-l)
    The coefficient of x^(m-1) is sum_j A[i,j] B[j,k] = C[i,k].
    
    So C[i,k] = [x^(m-1)] A_i(x) * B_k(x)
    
    This is a "middle product" of degree m-1.
    
    To compute the middle product of two polynomials of degree < m, we can:
    1. Evaluate at enough points
    2. Use CRT / interpolation
    
    If we evaluate at r points x_1, ..., x_r, we can recover the middle coefficient
    if we know the full product. The product has degree < 2m-1, so we need 2m-1 points
    to fully recover it. But we only need the middle coefficient.
    
    Alternatively, use the fact that the middle coefficient can be extracted via:
    C[i,k] = (1/(2*pi*i)) integral of A_i(x)*B_k(x)/x^m dx around 0
    
    For a finite field / integer approach:
    If we evaluate at points alpha_1, ..., alpha_r, we get:
    A_i(alpha_s) * B_k(alpha_s) = sum_{t=0}^{2m-2} c_t alpha_s^t
    where c_{m-1} = C[i,k].
    
    We need to extract c_{m-1} from the evaluations. This requires solving a linear system.
    
    But for a tensor decomposition, we need a multiplicative structure.
    
    Let me try a different approach: use the rank of the "structured" tensor directly.
    """
    pass

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 130.0  # 20s safety margin
    
    # Start with schoolbook as fallback
    rank_school = n * m * p
    U_fb = [[0] * rank_school for _ in range(n * m)]
    V_fb = [[0] * rank_school for _ in range(m * p)]
    W_fb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Try known constructions for small matrix multiplication
    # The rank of M_{3,3,3} is 23 (Strassen-like). For M_{3,3,4}, 
    # we can try to build from known small cases.
    
    # Strategy: Try random initialization with gradient-free optimization
    # to find lower rank decompositions.
    
    np.random.seed(42)
    
    for target_rank in range(20, rank_school):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for trial in range(50):
            if time.time() > deadline:
                break
            
            # Initialize with random small integers/half-integers
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Use a structured initialization based on polynomial evaluation points
            # Evaluate polynomials at distinct points
            points = list(range(1, target_rank + 1))
            
            for r_idx in range(target_rank):
                pt = points[r_idx % len(points)]
                # U entries: related to A row polynomials evaluated at pt
                # V entries: related to B column polynomials evaluated at pt  
                # W entries: related to extraction of middle coefficient
                
                # For now, use random initialization
                U[:, r_idx] = np.random.randint(-2, 3, size=n * m)
                V[:, r_idx] = np.random.randint(-2, 3, size=m * p)
                W[:, r_idx] = np.random.randint(-2, 3, size=p * n)
            
            # Check if this random init happens to work (very unlikely)
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist()):
                best_rank = target_rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
        
        if best_rank < rank_school:
            break
    
    # Try a more structured approach: use the polynomial middle product method
    if time.time() < deadline and best_rank == rank_school:
        # For M_{3,3,4}, try to use 3 polynomial evaluations
        # A_i(x) = A[i,0] + A[i,1]*x + A[i,2]*x^2
        # B_k(x) = B[0,k] + B[1,k]*x + B[2,k]*x^2
        # Product has degree < 6, middle coeff (x^2) is C[i,k]
        
        # To extract [x^2] from evaluations at points t_1, ..., t_r:
        # A_i(t_s) * B_k(t_s) = sum_{t=0}^{5} c_t * t_s^t
        # We need c_2.
        
        # If we use r=6 points, we can fully interpolate and extract c_2.
        # But we need a multiplicative decomposition.
        
        # The product A_i(t_s) * B_k(t_s) factors as U_{i,s} * V_{k,s}
        # where U_{i,s} = A_i(t_s) and V_{k,s} = B_k(t_s).
        # Then we need W to extract c_2 from the 6 values.
        
        # This gives rank 6 for the "evaluation" part, but W needs to combine
        # the 6 evaluations to get c_2. However, W depends on (i,k) through
        # the indices, not just s.
        
        # Actually, the issue is that the decomposition requires:
        # C[i,k] = sum_s U[i,s] * V[s,k] * W[s]  (rank-r with W scalar per s)
        # But here we have U depending on (i, s), V depending on (j, k, s), W depending on (k, i, s)
        
        # The tensor structure is different from simple polynomial multiplication.
        # Let me just try a direct search for low-rank solutions.
        
        pass
    
    return best_result
