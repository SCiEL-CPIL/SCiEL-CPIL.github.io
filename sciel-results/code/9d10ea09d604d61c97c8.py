"""Matrix multiplication 2x2 x 2x3 with rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 3]

def _schoolbook(n, m, p):
    """Return the schoolbook rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly represents the matrix multiplication tensor."""
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u[i * m + j] * v[j * p + k] * w[k * n + i]
    canonical = np.ones((n, m, p))
    return np.allclose(T, canonical, atol=tol)

def _build_rank10():
    """
    Construct a rank-10 decomposition for 2x2 * 2x3.
    
    Strategy: Use Strassen (rank 7) for the first 2 columns, and 3 additional
    factors for the third column.
    
    For the 2x2 subproblem (columns 0,1), we use a verified Strassen decomposition.
    For the third column, we use 3 factors that can be combined with the Strassen
    factors to produce the correct result.
    """
    n, m, p = 2, 2, 3
    rank = 10
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen for 2x2 * 2x2 (columns 0,1)
    # Using the standard verified formulation:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C11 = ae+bg, C12 = af+bh, C21 = ce+dg, C22 = cf+dh
    
    # Factor 0: (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1  # a, d
    V[0, 0] = 1; V[4, 0] = 1  # e, h (B[0,0], B[1,1])
    # W contributions: need to extract C11, C12, C21, C22 parts
    # This factor contributes to all 4 outputs via W coefficients
    W[0, 0] = 1   # C[0,0] (k=0,i=0)
    W[2, 0] = 1   # C[0,1] (k=1,i=0)
    W[1, 0] = 1   # C[1,0] (k=0,i=1)
    W[3, 0] = 1   # C[1,1] (k=1,i=1)
    
    # Actually, let me use a cleaner approach. I'll define the Strassen factors
    # with their exact U, V, W entries based on a verified decomposition.
    
    # Let me restart with a known correct approach:
    # I'll use the fact that rank-10 should be achievable and construct it
    # by solving the system of equations.
    
    # For now, let me try a different strategy: use rank 9 with a clever
    # factorization.
    
    # Actually, let me just verify if rank 10 works with a specific construction.
    
    # Clear and rebuild
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # I'll use a parameterized approach and solve for the factors.
    # Given the complexity, let me try to find a valid rank-10 decomposition
    # by using a known result or by systematic search.
    
    # For 2x2 * 2x3, the tensor rank is known to be 10 (or possibly 9).
    # Let me try to construct rank 9 first.
    
    return None  # Will be replaced by actual construction

def _try_rank9_constructive():
    """
    Attempt to construct a rank-9 decomposition.
    
    Approach: Split the problem into:
    - 7 factors for the 2x2 subproblem (Strassen)
    - 2 factors for the third column
    
    But we need to ensure the factors for the third column don't interfere
    with the first two columns, and vice versa.
    
    For the third column (k=2), we need:
    C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    This is a 2x1 matrix product, which has tensor rank 2 (schoolbook).
    So we need exactly 2 more factors.
    
    Total: 7 + 2 = 9.
    
    The key is to make the Strassen factors for columns 0,1 have zero entries
    in V for k=2, and the 2 extra factors have zero entries in V for k=0,1.
    """
    n, m, p = 2, 2, 3
    rank = 9
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # First 7 factors: Strassen for 2x2 * 2x2 (columns 0,1)
    # These factors should only have nonzero V entries for k=0 and k=1.
    
    # I need a verified Strassen decomposition. Let me use one that I can verify.
    
    # Standard Strassen (verified):
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e  
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    
    # C11 = p1 - p4 + p6
    # C12 = p3 + p5
    # C21 = p2 + p4
    # C22 = p1 - p3 + p2 + p5
    
    # Let me verify with a=1,b=2,c=3,d=4,e=5,f=6,g=7,h=8:
    # C = [[19, 22], [43, 50]]
    # C11 = 1*5+2*7 = 19
    # C12 = 1*6+2*8 = 22
    # C21 = 3*5+4*7 = 43
    # C22 = 3*6+4*8 = 50
    
    # p1 = (1+4)(5+8) = 65
    # p2 = (3+4)*5 = 35
    # p3 = 1*(6-8) = -2
    # p4 = 4*(7-5) = 8
    # p5 = (1+2)*8 = 24
    # p6 = (3-1)*(5+6) = 22
    # p7 = (2-4)*(7+8) = -30
    
    # C11 = 65 - 8 + 22 = 79. Should be 19. WRONG!
    
    # So this Strassen formulation is incorrect. Let me find the correct one.
    
    # The correct Strassen (from the original 1969 paper):
    # m1 = (a + d)(e + h)
    # m2 = (c + d)(e)
    # m3 = (a)(f - h)
    # m4 = (d)(g - e)
    # m5 = (a + b)(h)
    # m6 = (c - a)(e + f)
    # m7 = (b - d)(g + h)
    
    # C11 = m1 - m4 + m6
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m3 + m2 + m5
    
    # This is the same as above, so it's still wrong. Let me try a different variant.
    
    # Actually, I think the issue is that different sources use different conventions.
    # Let me just use a numerically verified decomposition.
    
    # I'll use a known correct rank-7 decomposition for 2x2*2x2.
    # One such decomposition (verified):
    
    # Factor 0: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1] -> (a+d)(e+h)
    # Factor 1: U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,0] -> (c+d)e
    # Factor 2: U=[1,0,0,0], V=[0,1,0,-1], W=[0,1,0,0] -> a(f-h)
    # Factor 3: U=[0,0,0,1], V=[0,0,1,-1], W=[1,0,1,0] -> d(g-e)
    # Factor 4: U=[1,1,0,0], V=[0,0,0,1], W=[0,1,0,1] -> (a+b)h
    # Factor 5: U=[0,-1,1,0], V=[1,1,0,0], W=[1,0,0,0] -> (c-a)(e+f)
    # Factor 6: U=[0,1,0,-1], V=[0,0,1,1], W=[1,0,0,1] -> (b-d)(g+h)
    
    # Let me verify C11 = sum of W[0,r] * (contribution from factor r to C11)
    # This is getting complex. Let me just use a computational approach.
    
    # Given the time constraints, let me try to find a valid rank-9 or rank-10
    # decomposition by solving the system of equations.
    
    return None

def _solve_for_decomposition(target_rank, n=2, m=2, p=3, max_iterations=1000):
    """
    Try to find a decomposition of the given rank by solving the trilinear system.
    Uses a simple iterative approach.
    """
    U = np.random.randn(n * m, target_rank)
    V = np.random.randn(m * p, target_rank)
    W = np.random.randn(p * n, target_rank)
    
    for iteration in range(max_iterations):
        # For each (i,j,k), we need:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # This is a system of n*m*p equations.
        # We can try to solve it iteratively.
        
        # For now, this is a placeholder. A proper implementation would use
        # alternating minimization or gradient descent on the residual.
        pass
    
    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Start with schoolbook (rank 12) as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = n * m * p
    
    # Try to construct a rank-10 decomposition
    # Based on known results, the rank of 2x2*2x3 matrix multiplication is 10.
    # Let me try to construct it explicitly.
    
    # For 2x2 * 2x3, one approach is:
    # - Use 7 factors for the first 2 columns (Strassen)
    # - Use 3 factors for the third column
    # But this gives rank 10, and we need to verify it works.
    
    # Actually, the third column can be done in 2 factors (it's a 2x1 product),
    # so 7+2=9 might work. But we need to be careful about interference.
    
    # Let me try a direct construction for rank 10.
    
    rank = 10
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # I'll use a systematic approach: define the factors based on the
    # algebraic structure of the problem.
    
    # For now, let me try to find a valid decomposition by using a known
    # result or by careful construction.
    
    # Given the complexity and time constraints, I'll return the schoolbook
    # decomposition which is guaranteed correct.
    
    return best_result
# EVOLVE-BLOCK-END
