"""Matrix multiplication 3,4,6 - rank-minimal exact tensor decomposition."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 12
    mp = m * p  # 24
    pn = p * n  # 18
    schoolbook_rank = n * m * p  # 72
    
    deadline = time.time() + 130
    
    def make_schoolbook():
        rank = schoolbook_rank
        U = [[Fraction(0)] * rank for _ in range(nm)]
        V = [[Fraction(0)] * rank for _ in range(mp)]
        W = [[Fraction(0)] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = Fraction(1)
                    V[j*p+k][col] = Fraction(1)
                    W[k*n+i][col] = Fraction(1)
                    col += 1
        return {"U": U, "V": V, "W": W}, rank
    
    best_result, best_rank = make_schoolbook()
    
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    def error(U, V, W):
        rank = len(U[0])
        max_e = Fraction(0)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    e = abs(s - 1)
                    if e > max_e:
                        max_e = e
        return max_e
    
    # For a given rank, try to find a solution using structured initialization
    # Key insight: We can fix W to be a "simple" matrix and solve for U, V.
    
    # Strategy: For rank r, try to find U, V, W such that the decomposition works.
    # Use the fact that if we fix W, then for each (i,k), the vector w_{i,k} = W[k*n+i, :] 
    # is fixed, and we need U[i*m+j, :] * V[j*p+k, :] (element-wise product, summed) = 1.
    
    # This is still hard. Let me try a different approach:
    # Use the "identity" structure. For rank r, if r is a multiple of something...
    
    # Actually, let me try to use known results. For 3x4x6 matrix multiplication,
    # the rank is known to be at most some value. Let me try to construct a good one.
    
    # One approach: Use block structure. 3x4x6 can be seen as blocks.
    
    # Let me try a simpler approach: random search with rational entries, 
    # then local refinement.
    
    def random_init(rank, seed):
        np.random.seed(seed)
        U = [[Fraction(np.random.randint(-2, 3)) for _ in range(rank)] for _ in range(nm)]
        V = [[Fraction(np.random.randint(-2, 3)) for _ in range(rank)] for _ in range(mp)]
        W = [[Fraction(np.random.randint(-2, 3)) for _ in range(rank)] for _ in range(pn)]
        return U, V, W
    
    def update_U(V, W, rank):
        """Update U given V, W to minimize error."""
        U = [[Fraction(0)] * rank for _ in range(nm)]
        for i in range(n):
            for j in range(m):
                # Solve: for all k, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # This is a linear system: A x = b where A[k,c] = V[j*p+k,c]*W[k*n+i,c]
                A = []
                for k in range(p):
                    row = []
                    for c in range(rank):
                        row.append(V[j*p+k][c] * W[k*n+i][c])
                    A.append(row)
                b = [Fraction(1) for _ in range(p)]
                # Solve using Gaussian elimination with Fractions
                # System is p x rank, want least squares or exact if possible
                # For simplicity, use numpy with float conversion
                A_f = np.array([[float(x) for x in row] for row in A], dtype=float)
                b_f = np.array([float(x) for x in b], dtype=float)
                try:
                    x, _, _, _ = np.linalg.lstsq(A_f, b_f, rcond=None)
                    U_row = []
                    for val in x:
                        # Round to nearest half-integer or integer
                        # Try to find a simple fraction
                        f = Fraction(val).limit_denominator(100)
                        U_row.append(f)
                    U[i*m+j] = U_row
                except:
                    pass
        return U
    
    def update_V(U, W, rank):
        """Update V given U, W to minimize error."""
        V = [[Fraction(0)] * rank for _ in range(mp)]
        for j in range(m):
            for k in range(p):
                # Solve: for all i, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                A = []
                for i in range(n):
                    row = []
                    for c in range(rank):
                        row.append(U[i*m+j][c] * W[k*n+i][c])
                    A.append(row)
                b = [Fraction(1) for _ in range(n)]
                A_f = np.array([[float(x) for x in row] for row in A], dtype=float)
                b_f = np.array([float(x) for x in b], dtype=float)
                try:
                    x, _, _, _ = np.linalg.lstsq(A_f, b_f, rcond=None)
                    V_row = []
                    for val in x:
                        f = Fraction(val).limit_denominator(100)
                        V_row.append(f)
                    V[j*p+k] = V_row
                except:
                    pass
        return V
    
    def update_W(U, V, rank):
        """Update W given U, V to minimize error."""
        W = [[Fraction(0)] * rank for _ in range(pn)]
        for k in range(p):
            for i in range(n):
                # Solve: for all j, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                A = []
                for j in range(m):
                    row = []
                    for c in range(rank):
                        row.append(U[i*m+j][c] * V[j*p+k][c])
                    A.append(row)
                b = [Fraction(1) for _ in range(m)]
                A_f = np.array([[float(x) for x in row] for row in A], dtype=float)
                b_f = np.array([float(x) for x in b], dtype=float)
                try:
                    x, _, _, _ = np.linalg.lstsq(A_f, b_f, rcond=None)
                    W_row = []
                    for val in x:
                        f = Fraction(val).limit_denominator(100)
                        W_row.append(f)
                    W[k*n+i] = W_row
                except:
                    pass
        return W
    
    # Try to find a better rank
    for rank in range(24, best_rank):
        if time.time() > deadline:
            break
        
        for trial in range(20):
            if time.time() > deadline:
                break
            
            U, V, W = random_init(rank, 1000 + rank * 100 + trial)
            
            # Local search: iterate updates
            for it in range(20):
                if time.time() > deadline:
                    break
                U = update_U(V, W, rank)
                V = update_V(U, W, rank)
                W = update_W(U, V, rank)
                
                if error(U, V, W) == 0:
                    if verify(U, V, W):
                        best_rank = rank
                        best_result = {"U": U, "V": V, "W": W}
                        break
            if time.time() > deadline:
                break
            if best_rank == rank:
                break
    
    # Convert to float lists
    result = {
        "U": [[float(x) for x in row] for row in best_result["U"]],
        "V": [[float(x) for x in row] for row in best_result["V"]],
        "W": [[float(x) for x in row] for row in best_result["W"]]
    }
    
    return result
# EVOLVE-BLOCK-END
