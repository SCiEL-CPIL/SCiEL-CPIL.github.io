<<<<<<< SEARCH
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists of lists as per the return interface example, 
    # but numpy arrays are faster for construction if we convert later?
    # The interface expects {'U': U, 'V': V, 'W': W} with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    # The seed used lists. I will use lists to be safe and match the "integer/half-integer" entry type directly.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: row i*m + j
                # V index: row j*p + k
                # W index: row k*n + i (trace convention W[k*n+i, r])
                
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                
                column += 1
                
                if column % 100 == 0 and time.time() > deadline:
                    # If we exceed time, return what we have? 
                    # But this loop is very fast (48 iterations). 
                    # This check is just a safeguard.
                    pass
                    
    return {"U": U, "V": V, "W": W}
=======
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Build the matrix multiplication tensor T[i,j,k] = 1 if (i,j,k) corresponds to a term
    # The tensor has dimensions (n*m) x (m*p) x (p*n)
    # T[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    
    # We want to find U, V, W such that for all a,b,c:
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Start with schoolbook and try to reduce rank
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook solution as fallback
    U_school = np.zeros((n*m, schoolbook_rank))
    V_school = np.zeros((m*p, schoolbook_rank))
    W_school = np.zeros((p*n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_U = U_school.tolist()
    best_V = V_school.tolist()
    best_W = W_school.tolist()
    
    # Try to find lower rank decompositions
    # Strategy: random initialization + ALS, then exact rational reconstruction
    
    def validate_exact(U_list, V_list, W_list, rank):
        """Check if the decomposition is exact using rational arithmetic"""
        if time.time() > deadline:
            return False
        
        U_arr = np.array(U_list, dtype=object)
        V_arr = np.array(V_list, dtype=object)
        W_arr = np.array(W_arr, dtype=object)
        
        # Convert to Fractions for exact arithmetic
        U_frac = [[Fraction(U_arr[i][j]) for j in range(rank)] for i in range(n*m)]
        V_frac = [[Fraction(V_arr[i][j]) for j in range(rank)] for i in range(m*p)]
        W_frac = [[Fraction(W_arr[i][j]) for j in range(rank)] for i in range(p*n)]
        
        # Check all tensor entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    
                    # Should be 1
                    val = sum(U_frac[a][r] * V_frac[b][r] * W_frac[c][r] for r in range(rank))
                    if val != 1:
                        return False
        
        # Check that non-existent terms are 0
        # For each (a,b,c) not corresponding to a valid (i,j,k), sum should be 0
        valid_abc = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    valid_abc.add((i*m+j, j*p+k, k*n+i))
        
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    if (a,b,c) not in valid_abc:
                        val = sum(U_frac[a][r] * V_frac[b][r] * W_frac[c][r] for r in range(rank))
                        if val != 0:
                            return False
        
        return True
    
    def try_rank(rank, max_iters=100, n_restarts=5):
        """Try to find a rank-r decomposition using ALS"""
        if time.time() > deadline:
            return None
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            np.random.seed(42 + restart)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Update U: for each a, solve U[a,:] to match T[a,:,:] = U[a,:] * (V ⊙ W)^T
                # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # For fixed a: T[a,b,c] = sum_r U[a,r] * (V[b,r] * W[c,r])
                # So T[a,:,:] = U[a,:] @ (V ⊙ W).T where (V⊙W)[r,:] = V[:,r] * W[:,r]
                
                # Compute M[a] = V ⊙ W (outer product along rank dimension)
                # M[r, b, c] = V[b,r] * W[c,r]
                # T[a,b,c] = sum_r U[a,r] * M[r,b,c]
                
                for a in range(n*m):
                    # Build system: T[a,b,c] = sum_r U[a,r] * M[r,b,c]
                    # Vectorize: T[a, b*p*n + c] = sum_r U[a,r] * M[r, b*p*n + c]
                    # M[r, b, c] = V[b,r] * W[c,r]
                    
                    # Flatten M to (rank, m*p*p*n)
                    M_flat = np.zeros((rank, m*p*p*n))
                    for r in range(rank):
                        for b in range(m*p):
                            for c in range(p*n):
                                M_flat[r, b*p*n + c] = V[b,r] * W[c,r]
                    
                    T_a = np.zeros(m*p*p*n)
                    for b in range(m*p):
                        for c in range(p*n):
                            # T[a,b,c] = 1 if there exists i,j,k with a=i*m+j, b=j*p+k, c=k*n+i
                            i = a // m
                            j = a % m
                            # b = j'*p + k, so j' = b // p, k = b % p
                            j_prime = b // p
                            k = b % p
                            # c = k'*n + i', so k' = c // n, i' = c % n
                            k_prime = c // n
                            i_prime = c % n
                            
                            if j == j_prime and k == k_prime and i == i_prime:
                                T_a[b*p*n + c] = 1.0
                    
                    # Solve: T_a = U[a,:] @ M_flat
                    # This is an overdetermined system, use least squares
                    try:
                        U[a, :] = np.linalg.lstsq(M_flat.T, T_a, rcond=None)[0]
                    except:
                        pass
                
                # Update V
                for b in range(m*p):
                    M_flat = np.zeros((rank, n*m*p*n))
                    for r in range(rank):
                        for a in range(n*m):
                            for c in range(p*n):
                                M_flat[r, a*p*n + c] = U[a,r] * W[c,r]
                    
                    T_b = np.zeros(n*m*p*n)
                    for a in range(n*m):
                        for c in range(p*n):
                            i = a // m
                            j = a % m
                            j_prime = b // p
                            k = b % p
                            k_prime = c // n
                            i_prime = c % n
                            
                            if j == j_prime and k == k_prime and i == i_prime:
                                T_b[a*p*n + c] = 1.0
                    
                    try:
                        V[b, :] = np.linalg.lstsq(M_flat.T, T_b, rcond=None)[0]
                    except:
                        pass
                
                # Update W
                for c in range(p*n):
                    M_flat = np.zeros((rank, n*m*m*p))
                    for r in range(rank):
                        for a in range(n*m):
                            for b in range(m*p):
                                M_flat[r, a*m*p + b] = U[a,r] * V[b,r]
                    
                    T_c = np.zeros(n*m*m*p)
                    for a in range(n*m):
                        for b in range(m*p):
                            i = a // m
                            j = a % m
                            j_prime = b // p
                            k = b % p
                            k_prime = c // n
                            i_prime = c % n
                            
                            if j == j_prime and k == k_prime and i == i_prime:
                                T_c[a*m*p + b] = 1.0
                    
                    try:
                        W[c, :] = np.linalg.lstsq(M_flat.T, T_c, rcond=None)[0]
                    except:
                        pass
            
            # After ALS, try to find exact rational solution
            # Use the numerical solution as a guide
            # Try to snap to small rationals
            
            U_int = np.zeros((n*m, rank))
            V_int = np.zeros((m*p, rank))
            W_int = np.zeros((p*n, rank))
            
            # Try to find rational approximations
            for i in range(n*m):
                for r in range(rank):
                    val = U[i,r]
                    # Try small denominators
                    found = False
                    for den in [1, 2, 4, 8]:
                        num = round(val * den)
                        if abs(val - num/den) < 1e-6:
                            U_int[i,r] = num/den
                            found = True
                            break
                    if not found:
                        U_int[i,r] = round(val)
            
            for i in range(m*p):
                for r in range(rank):
                    val = V[i,r]
                    found = False
                    for den in [1, 2, 4, 8]:
                        num = round(val * den)
                        if abs(val - num/den) < 1e-6:
                            V_int[i,r] = num/den
                            found = True
                            break
                    if not found:
                        V_int[i,r] = round(val)
            
            for i in range(p*n):
                for r in range(rank):
                    val = W[i,r]
                    found = False
                    for den in [1, 2, 4, 8]:
                        num = round(val * den)
                        if abs(val - num/den) < 1e-6:
                            W_int[i,r] = num/den
                            found = True
                            break
                    if not found:
                        W_int[i,r] = round(val)
            
            # Validate
            if validate_exact(U_int.tolist(), V_int.tolist(), W_int.tolist(), rank):
                return (U_int.tolist(), V_int.tolist(), W_int.tolist())
        
        return None
    
    # Try ranks from schoolbook down
    for rank in range(schoolbook_rank - 1, 20, -1):
        if time.time() > deadline:
            break
        
        result = try_rank(rank, max_iters=50, n_restarts=3)
        if result is not None:
            best_rank = rank
            best_U, best_V, best_W = result
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
>>>>>>> REPLACE
