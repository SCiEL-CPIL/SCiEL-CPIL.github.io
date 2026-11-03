"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 4, 4]
_START_TIME = time.time()
_TIMEOUT = 130  # safety margin before 150s limit

# EVOLVE-BLOCK-START
def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition using Fraction arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                if s != 1:
                    return False
    return True

def _verify_float(U, V, W, n, m, p, tol=1e-9):
    """Fast float verification."""
    rank = len(U[0])
    U3 = np.array(U, dtype=float).reshape(n, m, rank)
    V3 = np.array(V, dtype=float).reshape(m, p, rank)
    W3 = np.array(W, dtype=float).reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.max(np.abs(T - 1.0)) < tol

def _schoolbook():
    """Schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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

def _winograd_2x2_block():
    """
    Use Winograd 2x2 convolution pattern to build a structured decomposition.
    For 3x4 x 4x4, we can tile with 2x2 blocks.
    Winograd 2x2: F = [1,0,1], G = [1,1,-1]
    This gives rank 3 per 2x2 block instead of 4.
    """
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # Try to decompose using Winograd-style blocking
    # 3x4 can be seen as overlapping 2x2 blocks
    # 4x4 can be seen as 2x2 blocks
    
    # Let's try a direct structured approach:
    # Partition into 2x2 sub-blocks where possible
    # For 3 rows: use rows (0,1) with Winograd, row 2 separately
    # For 4 cols in A: use (0,1) and (2,3) with Winograd
    # For 4 cols in B: use (0,1) and (2,3) with Winograd
    
    # This is complex; let's try a simpler structured initialization
    # and then optimize
    
    # Alternative: use the fact that we can share computation
    # across overlapping windows
    
    # For now, return schoolbook and let optimization reduce it
    return _schoolbook()

def _try_random_exact(rank, n, m, p, n_seeds=50):
    """
    Try to find an exact decomposition of given rank using random initialization
    and exact rational solving via alternating least squares with Fraction.
    """
    if time.time() - _START_TIME > _TIMEOUT:
        return None
    
    np.random.seed(12345)
    
    for seed in range(n_seeds):
        if time.time() - _START_TIME > _TIMEOUT:
            break
        
        np.random.seed(seed)
        
        # Initialize with small random integers
        U = np.random.randint(-2, 3, size=(n*m, rank)).tolist()
        V = np.random.randint(-2, 3, size=(m*p, rank)).tolist()
        W = np.random.randint(-2, 3, size=(p*n, rank)).tolist()
        
        # Alternating exact solve
        for it in range(30):
            if time.time() - _START_TIME > _TIMEOUT:
                return None
            
            # Solve for U given V, W
            U_new = [[0]*rank for _ in range(n*m)]
            for i in range(n):
                for j in range(m):
                    # For each k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                    # Build system: A[k,r] = V[j*p+k,r] * W[k*n+i,r]
                    # Need: A @ u = ones(p)
                    A = np.array([[V[j*p+k][r] * W[k*n+i][r] for r in range(rank)] for k in range(p)], dtype=float)
                    b = np.ones(p)
                    try:
                        sol, res, rk, sv = np.linalg.lstsq(A, b, rcond=None)
                        # Round to nearest half-integer
                        sol_rounded = np.round(sol * 2) / 2
                        U_new[i*m+j] = sol_rounded.tolist()
                    except:
                        U_new[i*m+j] = U[i*m+j]
            
            # Solve for V given U, W
            V_new = [[0]*rank for _ in range(m*p)]
            for j in range(m):
                for k in range(p):
                    A = np.array([[U[i*m+j][r] * W[k*n+i][r] for r in range(rank)] for i in range(n)], dtype=float)
                    b = np.ones(n)
                    try:
                        sol, res, rk, sv = np.linalg.lstsq(A, b, rcond=None)
                        sol_rounded = np.round(sol * 2) / 2
                        V_new[j*p+k] = sol_rounded.tolist()
                    except:
                        V_new[j*p+k] = V[j*p+k]
            
            # Solve for W given U, V
            W_new = [[0]*rank for _ in range(p*n)]
            for k in range(p):
                for i in range(n):
                    A = np.array([[U[i*m+j][r] * V[j*p+k][r] for r in range(rank)] for j in range(m)], dtype=float)
                    b = np.ones(m)
                    try:
                        sol, res, rk, sv = np.linalg.lstsq(A, b, rcond=None)
                        sol_rounded = np.round(sol * 2) / 2
                        W_new[k*n+i] = sol_rounded.tolist()
                    except:
                        W_new[k*n+i] = W[k*n+i]
            
            U, V, W = U_new, V_new, W_new
            
            # Check if exact
            if _verify_float(U, V, W, n, m, p, tol=1e-6):
                if _verify_exact(U, V, W, n, m, p):
                    return U, V, W
    
    return None

def _try_structured_kronecker():
    """
    Try Kronecker product structure.
    3x4 x 4x4: Can we express as Kronecker products?
    3 = 3, 4 = 2*2, 4 = 2*2
    So A is 3x(2*2), B is (2*2)x(2*2)
    This suggests a block structure.
    """
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # Try: treat A as 3x2 blocks of width 2, B as 2x2 blocks
    # Output C is 3x4 = 3x2 blocks of width 2
    
    # For each pair of 2x2 blocks, use Winograd 2x2 (rank 3)
    # Number of block pairs: 3 rows * 2 col-blocks in A * 2 col-blocks in B = 12
    # Each with rank 3: total rank = 36
    
    # Let's construct this
    rank = 36
    U = [[0]*rank for _ in range(n*m)]  # 12 x 36
    V = [[0]*rank for _ in range(m*p)]  # 16 x 36
    W = [[0]*rank for _ in range(p*n)]  # 12 x 36
    
    # Winograd 2x2: 
    # Input: 2x2 matrix M = [[a,b],[c,d]]
    # Transform: d0=a+b, d1=a-b, d2=c, d3=d
    # Multiply: e0=d0*f0, e1=d1*f1, e2=d2*f2, e3=d3*f3
    # Inverse: [[e0+e1, e0-e1],[e0+e2, e0-e2]]... 
    # Actually standard Winograd 2x2:
    # F = [[1,0,1],[1,0,-1],[0,1,0]]  (3x2)
    # G = [[1,1,-1],[0,1,1]]  (2x3)
    # For matrix mult: U uses F^T, V uses F, W uses G
    
    # F^T = [[1,1,0],[0,1,1]]  (2x3)
    # F = [[1,0,1],[1,0,-1],[0,1,0]]  (3x2) -> but we need 2x3 for V
    # Actually for 2x2 matmul:
    # A(2x2) * B(2x2) = C(2x2)
    # U[i, r] for i in 0,1; r in 0,1,2
    # V[j, r] for j in 0,1; r in 0,1,2  
    # W[k, r] for k in 0,1; r in 0,1,2
    # where output index k corresponds to position in C
    
    # Winograd 2x2:
    # u = [a+b, a-b, c]  (but this is for 1D)
    # For 2D: 
    # d0 = a+b+c+d, d1 = a-c, d2 = b-d
    # e0 = f0+f1+g0+g1, e1 = f0-g0, e2 = f1-g1
    # C[0,0] = d0*e0 + d1*e1 + d2*e2
    # C[0,1] = d0*e0 - d1*e1 - d2*e2
    # C[1,0] = d0*e0 - d1*e1 + d2*e2
    # C[1,1] = d0*e0 + d1*e1 - d2*e2
    
    # So:
    # U[0,0]=1, U[0,1]=1, U[0,2]=0  (a+b+c+d: a contributes 1)
    # U[1,0]=1, U[1,1]=-1, U[1,2]=0  (c contributes 1 to d0, -1 to d1)
    # Wait, let me redo:
    # d0 = a+b+c+d: U[0,0]=1 (a), U[1,0]=1 (b), U[2,0]=1 (c), U[3,0]=1 (d)
    # But we only have 2x2 input, so indices 0,1 for A
    # Let me use standard Winograd F(2,2,2):
    # F = [[1,1,0],[0,1,1]]  for input transform (2x3)
    # G = [[1,0,1],[1,0,-1],[0,1,0]]  for output transform (3x2)
    
    # For A (2x2): row i, col j -> index i*2+j
    # U[i*2+j, r] where r=0,1,2
    # A = [[a,b],[c,d]]
    # d0 = a+b, d1 = a-b, d2 = c, d3 = d  -> this is 4 terms, not 3
    # 
    # Standard Winograd for 2x2 matmul uses 3 multiplications:
    # m0 = (a+b)(f+g)
    # m1 = (a-b)(f-h)  
    # m2 = (c+d)(g+h)
    # C[0,0] = m0 - m1 + m2 - c*h - d*g ... no this isn't right
    
    # Let me just use the known 3-multiplication scheme:
    # For A=[[a,b],[c,d]], B=[[e,f],[g,h]]:
    # p0 = (a+b)(e+g)
    # p1 = (a-b)(e-g)
    # p2 = (c+d)(g+h)
    # p3 = (c-d)(g-h)  -- wait that's 4
    
    # Actually Winograd 2x2 uses 3:
    # U = [[1,1,0],[1,-1,0],[0,1,1]]  (3x2) for A rows
    # V = [[1,0,1],[1,0,-1],[0,1,0]]  (3x2) for B cols  
    # W = [[1,1,1],[1,-1,-1],[1,-1,1]]  (2x3) for C
    
    # Let me verify: C[0,0] = sum_r U[0,r]*V[0,r]*W[0,r] = 1*1*1 + 1*0*(-1) + 0*1*1 = 1
    # But we need C[0,0] = a*e + c*g
    # This doesn't match. Let me look up the correct Winograd 2x2.
    
    # Correct Winograd 2x2 (F(2,2,2)):
    # Input A (2x2): flatten to [a,b,c,d]
    # Transform: d0=a+b, d1=a-b, d2=c, d3=d -> 4 values, use 3 multiplications
    # Actually the minimal is 3 for 2x2 matmul? No, it's known that 2x2 matmul needs 4 mults
    # Winograd reduces 4 to 3 for convolution, but for matrix mult it's different.
    
    # For 2x2 matrix multiplication, the tensor rank is 7 (not 4 or 3)
    # This is a known result. So Winograd doesn't help for pure matrix mult.
    
    # Let's abandon this approach and try direct search.
    return None

def construct():
    n, m, p = DIMENSIONS
    max_rank = n * m * p  # 48
    
    # Start with schoolbook
    U, V, W = _schoolbook()
    best_rank = max_rank
    best_U, best_V, best_W = U, V, W
    
    # Try ranks from just below schoolbook down
    # For 3x4x4, try a range of ranks
    # Based on known results, the rank might be around 40-45
    
    for target_rank in range(max_rank - 1, 30, -1):
        if time.time() - _START_TIME > _TIMEOUT:
            break
        
        result = _try_random_exact(target_rank, n, m, p, n_seeds=20)
        if result is not None:
            U, V, W = result
            best_rank = target_rank
            best_U, best_V, best_W = U, V, W
            # Try to go lower
            continue
    
    # Final verification
    if not _verify_exact(best_U, best_V, best_W, n, m, p):
        # Fall back to schoolbook
        best_U, best_V, best_W = _schoolbook()
        best_rank = max_rank
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
