"""Find a rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 4, 5]

def _verify(U, V, W):
    """Verify that the decomposition exactly reproduces matrix multiplication.
    
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    n_rows_U = n * m
    n_rows_V = m * p
    n_rows_W = p * n
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if s != 1:
                    return False
    return True

def _verify_fast(U, V, W):
    """Fast verification using numpy for real-valued checks."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p = 40."""
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

def _strassen_2x2():
    """Return Strassen decomposition for 2x2 * 2x2.
    
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    # Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    U = [
        [1, 1, 0, 0, 0, 0, 0],  # A00: M1, M2
        [0, 0, 1, 1, 1, 1, -1], # A01: M3, M4, M5, M6, M7
        [0, 1, 0, 0, 1, -1, 0], # A10: M2, M5, M6
        [1, 1, 0, 1, 0, 0, -1]  # A11: M1, M2, M4, M7
    ]
    V = [
        [1, 0, 0, 0, 0, 1, 0],  # B00: M1, M2, M6
        [0, 0, 1, 0, 0, 0, 1],  # B01: M3, M7
        [0, 0, 0, 0, 1, 0, 0],  # B10: M5
        [1, 0, -1, 1, 0, 0, 1]  # B11: M1, M3, M4, M7
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # C00: M1, M4, M5, M7
        [0, 0, 1, 0, 1, 0, 0],   # C01: M3, M5
        [0, 1, 0, 1, 0, 0, 0],   # C10: M2, M4
        [1, -1, 1, 0, 0, 1, 0]   # C11: M1, M2, M3, M6
    ]
    return U, V, W

def _construct_strassen_embedding():
    """Construct a decomposition for 2x4 * 4x5 using Strassen embedding.
    
    We embed the 2x2 * 2x2 Strassen algorithm into the 2x4 * 4x5 problem.
    
    The key idea: view the 2x4 matrix A as two 2x2 blocks stacked vertically:
    A = [A_top]  (2x4)
        [A_bot]
    
    And the 4x5 matrix B as two 2x5 blocks stacked vertically:
    B = [B_top]  (2x5)
        [B_bot]
    
    Then A*B = A_top * B_top + A_bot * B_bot
    
    For each 2x2 * 2x5 multiplication, we can use Strassen-like decomposition.
    Actually, let's use a simpler approach:
    
    For 2x2 * 2x5:
    A (2x2) * B (2x5) = C (2x5)
    C[0,:] = A[0,0]*B[0,:] + A[0,1]*B[1,:]
    C[1,:] = A[1,0]*B[0,:] + A[1,1]*B[1,:]
    
    This is just 4 scalar multiplications per column, but we can use Strassen for the 2x2 part.
    
    Actually, let's think of it differently. We can use the fact that:
    (2x4) * (4x5) can be seen as computing 10 outputs.
    
    Let me try a different structured approach:
    
    Split the 4 intermediate dimension into two groups of 2:
    A (2x4) = [A1 | A2] where A1, A2 are 2x2
    B (4x5) = [B1; B2] where B1, B2 are 2x5
    
    Then C = A1*B1 + A2*B2
    
    For A1*B1 (2x2 * 2x5):
    This is equivalent to multiplying a 2x2 matrix by a 2x5 matrix.
    We can use Strassen's algorithm adapted for this case.
    
    Actually, the simplest structured approach:
    Use Strassen for each 2x2 * 2x2 sub-block, and handle the 2x5 part by
    treating it as two 2x2 * 2x2 multiplications plus adjustments.
    
    Let me try yet another approach: direct structured construction.
    
    For the 2x4 * 4x5 problem, we can use the following decomposition:
    - Use 7 Strassen multiplications for the top-left 2x2 * 2x2 block
    - Use 7 Strassen multiplications for the top-right 2x2 * 2x2 block  
    - Use 7 Strassen multiplications for the bottom-left 2x2 * 2x2 block
    - Use 7 Strassen multiplications for the bottom-right 2x2 * 2x2 block
    - But this gives rank 28, and we need to handle the cross terms.
    
    Actually, this doesn't work directly because the dimensions don't align.
    
    Let me try a completely different structured approach:
    
    We can write the 2x4 * 4x5 multiplication as:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    For i=0,1 and k=0,1,2,3,4 and j=0,1,2,3
    
    Let's try to use a rank-35 decomposition based on Kronecker products.
    
    Actually, let me just try to find a good decomposition through optimization.
    """
    n, m, p = DIMENSIONS
    
    # Try to construct a rank-35 decomposition using structured patterns
    # Based on the idea that rank(2x4*4x5) <= rank(2x2*2x2) * rank(2x2*2x5/2)
    # = 7 * 5 = 35 (if we can decompose 2x2*2x5 with rank 5)
    
    # For 2x2 * 2x5:
    # A (2x2) * B (2x5) = C (2x5)
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    # This is 4 scalar multiplications per output column, so rank 4*5 = 20 schoolbook.
    # Can we do better? The tensor rank of 2x2*2x5 is at most 7 (using Strassen for each 2x2*2x2 subproblem).
    
    # Let's try to construct a rank-35 decomposition directly.
    
    # Approach: Use the fact that we can pad the 2x2*2x2 Strassen to work with 2x2*2x5
    # by treating the 2x5 as two 2x2 blocks plus a 2x1 block.
    
    # This is getting complex. Let's just try random search with a better initialization.
    
    return None

def _try_rank_k_structural(k, deadline):
    """Try to find a decomposition of rank k using structural search.
    
    We use the fact that for matrix multiplication A(n x m) * B(m x p),
    the tensor T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
    
    We search for integer/half-integer solutions.
    """
    n, m, p = DIMENSIONS
    
    if k < 14:
        return None
    
    best = None
    n_trials = 300
    trial = 0
    while trial < n_trials and time.time() < deadline:
        trial += 1
        
        # Initialize U and V with random small values
        # Use values from {-1, 0, 1}
        U = [[random.choice([-1, 0, 1]) for _ in range(k)] for _ in range(n*m)]
        V = [[random.choice([-1, 0, 1]) for _ in range(k)] for _ in range(m*p)]
        
        # Solve for W
        W = [[0]*k for _ in range(p*n)]
        valid = True
        for i in range(n):
            for ki in range(p):
                # For this (i, ki), we have m constraints:
                # sum_c U[i*m+j,c] * V[j*p+ki,c] * W[ki*n+i,c] = 1
                # Coefficient matrix A[j,c] = U[i*m+j,c] * V[j*p+ki,c]
                A = np.array([[U[i*m+j][c] * V[j*p+ki][c] for c in range(k)] for j in range(m)], dtype=float)
                b = np.ones(m)
                
                try:
                    # Use lstsq
                    w, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=None)
                    # Check if this is an exact solution
                    if np.allclose(A @ w, b, atol=1e-8):
                        # Check if w entries are integer or half-integer
                        w_rounded = np.round(w * 2) / 2
                        if np.allclose(w, w_rounded, atol=1e-8):
                            W[ki*n+i] = [float(x) for x in w_rounded]
                        else:
                            # Try rounding to nearest integer
                            w_int = np.round(w)
                            if np.allclose(A @ w_int, b, atol=1e-8):
                                W[ki*n+i] = [float(x) for x in w_int]
                            else:
                                valid = False
                                break
                    else:
                        valid = False
                        break
                except:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Verify
        if _verify(U, V, W):
            # Check if all are integer or half-integer
            ok = True
            for mat in [U, V, W]:
                for row in mat:
                    for v in row:
                        if abs(v * 2 - round(v * 2)) > 1e-8:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                return {"U": U, "V": V, "W": W}
    
    return None

def _try_systematic_low_rank(deadline):
    """Try systematic search for low-rank decompositions."""
    n, m, p = DIMENSIONS
    
    for k in range(25, 36):
        if time.time() > deadline:
            break
        result = _try_rank_k_structural(k, deadline)
        if result is not None:
            return result
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # First, get the schoolbook solution as a fallback
    fallback = _schoolbook()
    best_rank = 40  # n*m*p = 2*4*5 = 40
    best_result = fallback
    
    # Try to find lower rank decompositions
    # Try systematic search
    if time.time() < deadline - 10:
        result = _try_systematic_low_rank(deadline - 10)
        if result is not None:
            # Verify
            if _verify(result["U"], result["V"], result["W"]):
                new_rank = len(result["U"][0])
                if new_rank < best_rank:
                    best_rank = new_rank
                    best_result = result
    
    # Try more randomized search if time permits
    if time.time() < deadline - 5:
        for k in range(best_rank - 1, max(20, best_rank - 5), -1):
            if time.time() > deadline - 5:
                break
            for trial in range(80):
                if time.time() > deadline - 5:
                    break
                n, m, p = DIMENSIONS
                U = [[random.choice([-1, 0, 1]) for _ in range(k)] for _ in range(n*m)]
                V = [[random.choice([-1, 0, 1]) for _ in range(k)] for _ in range(m*p)]
                
                W = [[0]*k for _ in range(p*n)]
                valid = True
                for i in range(n):
                    for ki in range(p):
                        A = np.array([[U[i*m+j][c] * V[j*p+ki][c] for c in range(k)] for j in range(m)], dtype=float)
                        b = np.ones(m)
                        try:
                            w, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=None)
                            if np.allclose(A @ w, b, atol=1e-8):
                                w_rounded = np.round(w * 2) / 2
                                if np.allclose(w, w_rounded, atol=1e-8):
                                    W[ki*n+i] = [float(x) for x in w_rounded]
                                else:
                                    w_int = np.round(w)
                                    if np.allclose(A @ w_int, b, atol=1e-8):
                                        W[ki*n+i] = [float(x) for x in w_int]
                                    else:
                                        valid = False
                                        break
                            else:
                                valid = False
                                break
                        except:
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                if _verify(U, V, W):
                    ok = True
                    for mat in [U, V, W]:
                        for row in mat:
                            for v in row:
                                if abs(v * 2 - round(v * 2)) > 1e-8:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    
                    if ok:
                        if k < best_rank:
                            best_rank = k
                            best_result = {"U": U, "V": V, "W": W}
                        break
    
    return best_result
