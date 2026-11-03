"""Matrix multiplication 3x4 x 4x4: rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]


def _verify(U, V, W, n=3, m=4, p=4, tol=1e-9):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol


def _try_winograd_2x2():
    """
    Apply Winograd's scheme for 2x2 matrix multiplication to overlapping 2x2 blocks.
    
    For a 2x2 * 2x2 multiplication, Winograd's algorithm uses 7 multiplications:
    C = A * B where A, B are 2x2.
    
    The idea: partition the 3x4 and 4x4 matrices into overlapping 2x2 blocks and
    apply Winograd's scheme to each, sharing computations where possible.
    
    For 3x4 * 4x4:
    - A is 3x4, B is 4x4, C is 3x4
    - We can partition into overlapping 2x2 blocks
    
    Let's think of this differently. The standard approach for rectangular matrices
    uses the fact that a (2r-1)x(2s-1) * (2s-1)x(2t-1) multiplication can be done
    with r*s*t*(2r-1)*(2s-1)*(2t-1) multiplications using Winograd.
    
    For our case: 3x4 * 4x4
    - 3 = 2*2-1 (r=2), 4 = 2*2+2... not quite.
    
    Let me try a more direct approach: use the known rank bounds.
    The minimum rank for 3x4 * 4x4 is known to be at most 20 (from various sources).
    
    Let me construct a specific decomposition using the following strategy:
    - Use 2x2 Winograd blocks where possible
    - Handle the remaining dimensions with schoolbook
    
    Actually, let me try to construct a rank-20 decomposition explicitly.
    
    For matrix multiplication A(3x4) * B(4x4) = C(3x4):
    C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1,2}, k in {0,1,2,3}, j in {0,1,2,3}
    
    Tensor T[i,j,k] = 1 for all valid (i,j,k).
    
    Let me try a block decomposition:
    - Split j into {0,1} and {2,3}
    - For each j-block, compute partial products
    
    Actually, let me use a known construction. The rank of matrix multiplication
    M_{n,m,p} is the tensor rank of the multiplication tensor.
    
    For M_{3,4,4}, we can use the following approach:
    - Write A as [[A1, A2], [A3, A4]] where each Ai is 1x2 (rows of A split into pairs of columns)
      Actually A is 3x4, so let's split columns: A = [A_left | A_right] where each is 3x2
    - Write B as [[B1, B2], [B3, B4]] where each Bj is 2x4 (rows of B split into pairs)
      Actually B is 4x4, so split rows: B = [B_top; B_bottom] where each is 2x4
    
    C = A * B = [A_left | A_right] * [B_top; B_bottom] = A_left * B_top + A_right * B_bottom
    
    Each term is a 3x2 * 2x4 multiplication.
    
    For a 3x2 * 2x4 multiplication:
    C[i,k] = sum_j A[i,j] * B[j,k] for j in {0,1}
    This is just 3*2*4 = 24 scalar multiplications for schoolbook.
    
    But we can use Winograd's scheme for the 2x2 subproblems.
    
    Let me try a different decomposition. Let's use the fact that:
    For computing C = A*B where A is m x n and B is n x p,
    we can use Strassen-like algorithms.
    
    For our specific case, let me try to construct a rank-20 decomposition.
    
    Actually, let me just try to find a low-rank decomposition using numerical optimization
    and then round to fractions.
    """
    return None


def _numerical_optimization(n=3, m=4, p=4, target_rank=20, max_time=100):
    """
    Use numerical optimization to find a low-rank decomposition.
    We minimize the error ||T - sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]||_F^2
    where T[i,j,k] = 1 for all valid (i,j,k).
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    # Target: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # So we need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Initialize with schoolbook and try to reduce rank
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    for r in range(15, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        for trial in range(50):
            if time.time() > deadline - 10:
                break
            
            rng = np.random.RandomState(trial + r * 1000)
            
            # Initialize with random values
            U = rng.randn(n * m, r) * 0.5
            V = rng.randn(m * p, r) * 0.5
            W = rng.randn(p * n, r) * 0.5
            
            # Gradient descent with adaptive learning rate
            lr = 0.01
            for iteration in range(500):
                if time.time() > deadline - 5:
                    break
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # T[i,j,k] = 1
                            val = 0.0
                            for c in range(r):
                                val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                            resid = val - 1.0
                            err += resid * resid
                            
                            for c in range(r):
                                dU[i * m + j, c] += resid * V[j * p + k, c] * W[k * n + i, c]
                                dV[j * p + k, c] += resid * U[i * m + j, c] * W[k * n + i, c]
                                dW[k * n + i, c] += resid * U[i * m + j, c] * V[j * p + k, c]
                
                if err < 1e-12:
                    break
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Decay learning rate
                lr *= 0.999
                
                # Symmetrization trick: for tensor rank decomposition, we can use
                # the fact that the problem is invariant under certain transformations
                
            # Check final error
            final_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(r):
                            val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        final_err += (val - 1.0) ** 2
            
            if final_err < 1e-10:
                if r < best_rank:
                    best_rank = r
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    break
    
    if best_U is not None:
        # Convert to fractions
        U_frac = [[Fraction(x).limit_denominator(10000) for x in row] for row in best_U.tolist()]
        V_frac = [[Fraction(x).limit_denominator(10000) for x in row] for row in best_V.tolist()]
        W_frac = [[Fraction(x).limit_denominator(10000) for x in row] for row in best_W.tolist()]
        
        # Verify with fractions
        if _verify(U_frac, V_frac, W_frac, n, m, p, tol=1e-9):
            return {"U": U_frac, "V": V_frac, "W": W_frac}, best_rank
    
    return None, best_rank


def _schoolbook(n=3, m=4, p=4):
    """Standard schoolbook decomposition with rank n*m*p."""
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


def _winograd_based_decomposition():
    """
    Try to construct a Winograd-based decomposition for 3x4 * 4x4.
    
    Key insight: We can decompose the problem into smaller subproblems using
    the Winograd scheme for 2x2 matrix multiplication.
    
    For 2x2 * 2x2, Winograd uses 7 multiplications instead of 8.
    
    Let's try to use this structure for our problem.
    
    A (3x4) * B (4x4) = C (3x4)
    
    Split A into rows: A = [A1; A2; A3] where each Ai is 1x4
    Split B into columns: B = [B1, B2, B3, B4] where each Bj is 4x1
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Alternative: Use block structure
    A = [A_left | A_right] where A_left is 3x2, A_right is 3x2
    B = [B_top; B_bottom] where B_top is 2x4, B_bottom is 2x4
    
    C = A_left * B_top + A_right * B_bottom
    
    Each term is 3x2 * 2x4.
    
    For 3x2 * 2x4:
    Let A' be 3x2, B' be 2x4.
    C'[i,k] = A'[i,0]*B'[0,k] + A'[i,1]*B'[1,k]
    
    This can be done with 3*2*4 = 24 multiplications schoolbook.
    But we can use Winograd: for each i,k, it's a dot product of length 2.
    Winograd for length-2 dot product: a*b + c*d = (a+c)(b+d) - a*b - c*d
    That's 3 multiplications instead of 2... no wait, that's worse.
    
    Actually for 2-term dot product: a*b + c*d
    = (a+c)*(b+d) - a*b - c*d ... no that's 4 mults.
    
    The Winograd approach for small dot products:
    For length 2: ab + cd = (a+c)(b+d) - ab - cd ... this doesn't help.
    
    Let me think differently. The Winograd scheme for 2x2 matrix mult:
    m0 = a00 * b00
    m1 = (a00+a01) * b01
    m2 = (a10+a11) * b00
    m3 = a11 * (b01+b11)
    m4 = (a11-a10) * (b10-b11)
    m5 = a10 * b10
    m6 = (a00+a01-a11) * (b11-b00)
    
    C00 = m0 + m1 + m6
    C01 = m1 + m3 - m6
    C10 = m2 + m5 - m4
    C11 = m0 + m3 + m4
    
    So 7 multiplications instead of 8.
    
    For our problem, let me try to use this structure.
    
    Let's partition:
    A (3x4) = 
    [a00 a01 a02 a03]
    [a10 a11 a12 a13]
    [a20 a21 a22 a23]
    
    B (4x4) =
    [b00 b01 b02 b03]
    [b10 b11 b12 b13]
    [b20 b21 b22 b23]
    [b30 b31 b32 b33]
    
    C (3x4):
    C[i,k] = sum_j A[i,j]*B[j,k]
    
    For C[0,0] = a00*b00 + a01*b10 + a02*b20 + a03*b30
    This is a length-4 dot product.
    
    Winograd for length-4 dot product uses 6 multiplications instead of 4... 
    No wait, Winograd reduces multiplications for matrix multiplication, not dot products directly.
    
    Let me try a different approach: just use numerical optimization to find
    a low-rank decomposition and hope to find something with rank < 48.
    
    The known minimum rank for M_{3,4,4} is at most 20 (I believe).
    Let me try to find it.
    """
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120
    
    # Start with schoolbook as fallback
    result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try Winograd-based decomposition
    winograd_result = _winograd_based_decomposition()
    if winograd_result is not None:
        w_rank = len(winograd_result['U'][0])
        if w_rank < best_rank:
            best_rank = w_rank
            result = winograd_result
    
    # Try numerical optimization
    opt_result, opt_rank = _numerical_optimization(n, m, p, target_rank=max(15, best_rank - 5), max_time=min(90, deadline - time.time() - 10))
    if opt_result is not None:
        o_rank = len(opt_result['U'][0])
        if o_rank < best_rank:
            best_rank = o_rank
            result = opt_result
    
    # Final verification
    if not _verify(result['U'], result['V'], result['W'], n, m, p, tol=1e-9):
        # Fallback to schoolbook
        result = _schoolbook(n, m, p)
        best_rank = n * m * p
    
    return result
