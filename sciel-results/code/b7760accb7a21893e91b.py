"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import product

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # The tensor equation: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal 1 if i,j,k correspond to a valid multiplication term
    # Actually, the convention is: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T[i,j,k] = 1 for all i,j,k, and the decomposition is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
                if max_err > tol:
                    return False
    return True

def _get_strassen_2x2x2():
    """Return Strassen's 7-term decomposition for 2x2x2 matrix multiplication."""
    # Strassen's formulas for C = A*B where A,B are 2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    # m1 = (a+d)(e+h), m2 = (c+d)*e, m3 = a*(f-h), m4 = d*(g-e)
    # m5 = (a+b)*h, m6 = (c-d)*(f+h), m7 = (b-d)*(e+g)
    # i = m1+m4-m5+m7, j = m3+m5, k = m2+m4, l = m1-m2+m3+m6
    
    # We need U, V, W such that:
    # For each of the 7 terms, we have U[i,j], V[j,k], W[k,i]
    # A[i,j] is at position i*2+j in U, B[j,k] is at j*2+k in V, C[k,i] is at k*2+i in W
    
    U = [[0.0]*7 for _ in range(4)]  # 2*2 = 4 entries for A
    V = [[0.0]*7 for _ in range(4)]  # 2*2 = 4 entries for B
    W = [[0.0]*7 for _ in range(4)]  # 2*2 = 4 entries for C (with trace convention W[k*n+i])
    
    # m1 = (a+d)(e+h): A has a at (0,0)=idx0, d at (1,1)=idx3; B has e at (0,0)=idx0, h at (1,1)=idx3
    U[0][0] = 1.0; U[3][0] = 1.0
    V[0][0] = 1.0; V[3][0] = 1.0
    # C[0,0] = i gets m1, so W[0*2+0]=W[0] += 1; C[1,1]=l gets -m1, so W[1*2+1]=W[3] += -1
    W[0][0] = 1.0; W[3][0] = -1.0
    
    # m2 = (c+d)*e: A has c at (1,0)=idx2, d at (1,1)=idx3; B has e at (0,0)=idx0
    U[2][1] = 1.0; U[3][1] = 1.0
    V[0][1] = 1.0
    # C[1,0] = k gets m2, so W[0*2+1]=W[1] += 1
    W[1][1] = 1.0
    
    # m3 = a*(f-h): A has a at (0,0)=idx0; B has f at (0,1)=idx1, h at (1,1)=idx3
    U[0][2] = 1.0
    V[1][2] = 1.0; V[3][2] = -1.0
    # C[0,1] = j gets m3, so W[1*2+0]=W[2] += 1; C[1,1]=l gets m3, so W[3] += 1
    W[2][2] = 1.0; W[3][2] = 1.0
    
    # m4 = d*(g-e): A has d at (1,1)=idx3; B has g at (1,0)=idx2, e at (0,0)=idx0
    U[3][3] = 1.0
    V[2][3] = 1.0; V[0][3] = -1.0
    # C[0,0]=i gets m4, so W[0] += 1; C[1,0]=k gets m4, so W[1] += 1
    W[0][3] = 1.0; W[1][3] = 1.0
    
    # m5 = (a+b)*h: A has a at (0,0)=idx0, b at (0,1)=idx1; B has h at (1,1)=idx3
    U[0][4] = 1.0; U[1][4] = 1.0
    V[3][4] = 1.0
    # C[0,0]=i gets -m5, so W[0] += -1
    W[0][4] = -1.0
    
    # m6 = (c-d)*(f+h): A has c at (1,0)=idx2, d at (1,1)=idx3; B has f at (0,1)=idx1, h at (1,1)=idx3
    U[2][5] = 1.0; U[3][5] = -1.0
    V[1][5] = 1.0; V[3][5] = 1.0
    # C[1,1]=l gets m6, so W[3] += 1
    W[3][5] = 1.0
    
    # m7 = (b-d)*(e+g): A has b at (0,1)=idx1, d at (1,1)=idx3; B has e at (0,0)=idx0, g at (1,0)=idx2
    U[1][6] = 1.0; U[3][6] = -1.0
    V[0][6] = 1.0; V[2][6] = 1.0
    # C[0,0]=i gets m7, so W[0] += 1
    W[0][6] = 1.0
    
    return U, V, W

def _embed_strassen_into_3x4x5():
    """
    Try to embed Strassen's 2x2x2 decomposition into the 3x4x5 problem.
    
    Idea: The 3x4 matrix can be seen as having 3 rows of 4 elements.
    The 4x5 matrix has 4 rows of 5 elements.
    
    We can split the middle dimension (size 4) into two halves of size 2.
    Then we have:
    - Left part: 3x2 (rows 0-1 of the 4-element rows)
    - Right part: 2x5 (first 2 rows of the 4x5 matrix)
    - Left part 2: 3x2 (rows 2-3)
    - Right part 2: 2x5 (last 2 rows)
    
    But this doesn't directly give us a 2x2x2 structure.
    
    Alternative: Think of the 3x4 as a 3x(2+2) and 4x5 as (2+2)x5.
    Then C[3x5] = A_left[3x2]*B_top[2x5] + A_right[3x2]*B_bottom[2x5]
    
    Each term is a 3x2 times 2x5 = 3x5 multiplication.
    The rank of 3x2x5 multiplication is at most 3*2*5 = 30 by schoolbook.
    So two such terms give rank 60, same as schoolbook.
    
    This doesn't help directly. Let me try a different embedding.
    
    What if we use the fact that 3 = 2+1 and 5 = 2+3?
    Then we have blocks:
    - 2x4 * 4x2 = 2x2 (use Strassen, rank 7)
    - 2x4 * 4x3 = 2x3 (rank 6 by schoolbook, or maybe less)
    - 1x4 * 4x2 = 1x2 (rank 2)
    - 1x4 * 4x3 = 1x3 (rank 3)
    Total: 7 + 6 + 2 + 3 = 18, but we need to account for the fact that these share the middle dimension.
    
    This is getting complicated. Let me just return the schoolbook for now and add a greedy search.
    """
    return None

def _greedy_search(n, m, p, target_rank, time_budget, seed=42):
    """
    Greedy search for a tensor decomposition with rank <= target_rank.
    Returns (U, V, W) if found, else None.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Try multiple random initializations
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        # Initialize with small random values
        U = np.random.randn(n*m, target_rank) * 0.1
        V = np.random.randn(m*p, target_rank) * 0.1
        W = np.random.randn(p*n, target_rank) * 0.1
        
        # Use a simple iterative method to try to satisfy the equations
        # For each (i,j,k), we want sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        
        # This is a non-convex problem, so we'll use a heuristic approach
        # Try to update one factor at a time
        
        converged = False
        for iteration in range(1000):
            if time.time() > deadline:
                break
            
            # Compute current error for each (i,j,k)
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        err = abs(val - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-8:
                converged = True
                break
            
            # Update U: for each (i,j), try to adjust U[i*m+j, :] to reduce error
            # This is complex, so let's just do a random perturbation for now
            U += np.random.randn(n*m, target_rank) * 0.01
            V += np.random.randn(m*p, target_rank) * 0.01
            W += np.random.randn(p*n, target_rank) * 0.01
        
        if converged:
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # For 3x4x5, the schoolbook rank is 60.
    # Finding a lower rank is computationally expensive and unlikely within the time limit.
    # Return the known valid schoolbook decomposition immediately.
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
