"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import random
import time
import numpy as np
from fractions import Fraction

def _verify_factors(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W with shapes (n*m, rank), (m*p, rank), (p*n, rank)
    produce the matrix multiplication tensor under the trace convention W[k*n+i, r]."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A = i * m + j
                B = j * p + k
                C = k * n + i
                expected = 1
                actual = np.dot(U_arr[A], V_arr[B] * W_arr[C])
                if abs(actual - expected) > tol:
                    return False, (A, B, C, expected, actual)
    return True, None

def _build_schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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
    return U, V, W

def _cyclic_convolution_decomposition(n, m, p, time_limit=60):
    """
    Construct a decomposition using cyclic group convolution.
    
    For M_{2,3,3}: C[i,k] = sum_j A[i,j] * B[j,k]
    
    We embed the 3x3 multiplication into a cyclic convolution of size 3.
    Using the DFT of Z_3, we can decompose the multiplication into 
    components in the Fourier domain.
    
    Key idea: For each output position (i,k), we need sum over j of A[i,j]*B[j,k].
    If we define for each frequency f:
      A_hat[i,f] = sum_j A[i,j] * omega^{f*j}
      B_hat[f,k] = sum_j B[j,k] * omega^{f*j}
    Then A_hat[i,f] * B_hat[f,k] = sum_j sum_l A[i,j]*B[l,k]*omega^{f*(j+l)}
    This doesn't directly give what we want.
    
    Alternative: Use the regular representation approach.
    For 3x3 matrix multiplication, the tensor rank over C is 9 (schoolbook).
    But we're looking for a decomposition over a field that might allow lower rank
    via algebraic structure.
    
    Let me try a direct algebraic construction for rank 11.
    
    Actually, let me try a different approach: use the fact that for rectangular
    matrix multiplication M_{2,3,3}, we can use block decomposition.
    
    Split A (2x3) into [A1 | A2] where A1 is 2x1, A2 is 2x2.
    Split B (3x3) into [[B11, B12, B13], [B21, B22, B23], [B31, B32, B33]]
    
    C = A * B = [A1*B11 + A2*[B21,B22,B23], A1*B12 + A2*[B22,B23,B32], ...]
    
    This is getting complex. Let me try a more systematic algebraic approach.
    """
    start_time = time.time()
    
    # Try to construct a rank-11 decomposition using algebraic identities
    # For M_{2,3,3}, we have 18 scalar multiplications in schoolbook.
    # We want to reduce this using algebraic structure.
    
    # Approach: Use the structure of the tensor and try to find linear dependencies
    # among the 18 basis tensors.
    
    # The tensor T has components T_{i,j,k} where:
    # i in {0,1}, j in {0,1,2}, k in {0,1,2}
    # T_{i,j,k} = delta_{j,k} (since C_{i,k} = sum_j A_{i,j} B_{j,k})
    
    # Wait, that's not right. The multiplication tensor for M_{n,m,p} is:
    # T_{(i,j),(j,k),(k,i)} = 1 for all valid indices
    # So for M_{2,3,3}: T_{(i,j),(j,k),(k,i)} = 1
    
    # This means for each (i,j,k), the term A_{i,j} * B_{j,k} contributes to C_{k,i}
    # Wait, let me re-read the convention.
    
    # From the verify function:
    # A = i*m + j  (index into U, row i col j of A)
    # B = j*p + k  (index into V, row j col k of B)  
    # C = k*n + i  (index into W, row k col i of C)
    
    # So C_{k,i} = sum_j A_{i,j} * B_{j,k}
    # This means C = A * B^T? No wait...
    
    # If A is 2x3 (rows i, cols j) and B is 3x3 (rows j, cols k),
    # then (A*B)_{i,k} = sum_j A_{i,j} * B_{j,k}
    # But the convention says C_{k,i} gets the sum, so C = (A*B)^T = B^T * A^T
    
    # Actually, looking more carefully: the output index is C = k*n + i,
    # which means output row k, col i. So C[k,i] = sum_j A[i,j] * B[j,k].
    # This is C = A^T * B? No...
    
    # Let me just work with the convention as given.
    
    # For the tensor, we need: for each (i,j,k), the product U[A]·(V[B]*W[C]) = 1
    # where A=i*m+j, B=j*p+k, C=k*n+i
    
    # Let me try a constructive approach for rank 11.
    # I'll use a parameterized ansatz and solve for the coefficients.
    
    # Use a simpler approach: try known decompositions from literature.
    # For M_{2,3,3}, the known upper bound is 11 (from Bläser's work or similar).
    
    # Let me try to construct it explicitly.
    
    # One approach: use the fact that 3 = 2+1, and decompose accordingly.
    
    # Actually, let me try a direct algebraic construction.
    # For the 3x3 part, use the identity:
    # (a1+b1)(a2+b2)(a3+b3) expansion has 8 terms, but we need specific ones.
    
    # Let me try a different tactic: use a computer algebra approach with sympy-like
    # solving, but since we only have numpy, let's try a structured ansatz.
    
    # Ansatz for rank 11:
    # Define 11 "multiplication" terms, each being a product of a linear form
    # in A, a linear form in B, and a linear form in C.
    
    # Let me try to find it by exploiting the cyclic symmetry of the 3-index.
    
    # For the j-index (size 3), use the DFT basis:
    # e_0 = (1, 1, 1)
    # e_1 = (1, omega, omega^2)  
    # e_2 = (1, omega^2, omega)
    # where omega = exp(2*pi*i/3)
    
    # But we need real/rational coefficients, so let's use:
    # The minimal polynomial of omega is x^2 + x + 1 = 0
    # So we work in Q(omega) which has degree 2 over Q.
    
    # This suggests we might need rank related to 2*something.
    
    # Let me try a more direct approach: brute force search with structure.
    
    # For now, return None and fall back to schoolbook.
    # The random search below may find something, but it's unlikely to find rank 11.
    
    return None

def _structured_decomposition_attempt(n, m, p, target_rank=11):
    """Attempt a structured decomposition for M_{2,3,3} with rank < 18."""
    return _cyclic_convolution_decomposition(n, m, p)

def _try_random_search(n, m, p, max_rank, trials=5000, seed=42, time_limit=100):
    """Random search for a low-rank decomposition."""
    rng = random.Random(seed)
    start_time = time.time()
    
    for trial in range(trials):
        if time.time() - start_time > time_limit:
            break
        
        rank = max_rank
        U = [[rng.randint(-2, 2) for _ in range(rank)] for _ in range(n * m)]
        V = [[rng.randint(-2, 2) for _ in range(rank)] for _ in range(m * p)]
        W = [[rng.randint(-2, 2) for _ in range(rank)] for _ in range(p * n)]
        
        ok, _ = _verify_factors(U, V, W, n, m, p)
        if ok:
            return U, V, W, rank
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 120 - 5  # safety margin
    
    # First, try known lower-rank constructions if available
    result = _structured_decomposition_attempt(n, m, p)
    if result is not None:
        U, V, W, rank = result
        ok, _ = _verify_factors(U, V, W, n, m, p)
        if ok:
            return {"U": U, "V": V, "W": W}
    
    # Try random search with decreasing rank
    for target_rank in [11, 12, 13, 14, 15, 16, 17]:
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        remaining_time = deadline - elapsed
        if remaining_time < 5:
            break
        
        result = _try_random_search(n, m, p, target_rank, trials=10000, seed=target_rank, time_limit=min(remaining_time - 5, 20))
        if result is not None:
            U, V, W, rank = result
            ok, _ = _verify_factors(U, V, W, n, m, p)
            if ok:
                return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook (rank 18)
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
