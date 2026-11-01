import time
import numpy as np
from fractions import Fraction
import sys

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    # U: n*m x R, V: m*p x R, W: p*n x R
    # Indices: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    # Target Tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # Else 0
    
    def verify_exact(U, V, W):
        """Verify that U, V, W correctly decompose the tensor using exact arithmetic."""
        R = len(U[0]) if U else 0
        if len(V[0]) != R or len(W[0]) != R:
            return False
        
        # Check all n*m*p required entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    actual = Fraction(0)
                    for r in range(R):
                        actual += U[a][r] * V[b][r] * W[c][r]
                    if actual != Fraction(1):
                        return False
        
        # Check that all other entries are zero
        # We can iterate over all a, b, c and skip the non-zero ones
        # Or just iterate all and check
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i1, j1 = divmod(a, m)
                    j2, k1 = divmod(b, p)
                    k2, i2 = divmod(c, n)
                    
                    # If this is a non-zero entry, we already checked it
                    if i1 == i2 and j1 == j2 and k1 == k2:
                        continue
                        
                    actual = Fraction(0)
                    for r in range(R):
                        actual += U[a][r] * V[b][r] * W[c][r]
                    if actual != Fraction(0):
                        return False
        return True

    def verify_fast_float(U, V, W):
        """Fast verification using float, for initial screening."""
        R = len(U[0]) if U else 0
        if len(V[0]) != R or len(W[0]) != R:
            return False
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Compute tensor T_hat[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # Using einsum: 'ar,br,cr->abc'
        T_hat = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        
        # Create target tensor T
        T_target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_target[a, b, c] = 1.0
        
        # Check difference
        if not np.allclose(T_hat, T_target, atol=1e-6):
            return False
        return True

    # Try to find rank reduction
    # Start from 41 down to 1
    # We use random search with small integers
    
    # Define the set of values to use
    # Small integers are more likely to yield exact solutions if they exist
    values = [-2, -1, 0, 1, 2]
    
    np.random.seed(42)
    
    for target_rank in range(41, 0, -1):
        if time.time() > deadline - 5.0:
            break
            
        # Number of attempts depends on rank
        # Lower rank is harder, so maybe more attempts? Or fewer because it's unlikely?
        # Let's try a fixed number of attempts per rank, adjusted by time.
        max_attempts = 2000
        attempts = 0
        
        while attempts < max_attempts:
            if time.time() > deadline:
                break
            
            attempts += 1
            
            # Generate random sparse matrices
            # Density: start low, increase if failing
            # For rank R, we need enough non-zeros to satisfy 42 eqs.
            # Let's say each row has ~2-4 non-zeros.
            
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Fill with random values
            # We can use a structured sparsity:
            # For each rank index r, pick a few positions to be non-zero.
            for r in range(target_rank):
                # U has 6 rows
                num_u = np.random.randint(1, 4)
                u_idx = np.random.choice(n * m, size=num_u, replace=False)
                U[u_idx, r] = np.random.choice(values, size=num_u)
                
                # V has 21 rows
                num_v = np.random.randint(1, 4)
                v_idx = np.random.choice(m * p, size=num_v, replace=False)
                V[v_idx, r] = np.random.choice(values, size=num_v)
                
                # W has 14 rows
                num_w = np.random.randint(1, 4)
                w_idx = np.random.choice(p * n, size=num_w, replace=False)
                W[w_idx, r] = np.random.choice(values, size=num_w)
            
            # Fast check
            if verify_fast_float(U, V, W):
                # Convert to Fractions for exact check
                U_frac = [[Fraction(int(x)) for x in row] for row in U]
                V_frac = [[Fraction(int(x)) for x in row] for row in V]
                W_frac = [[Fraction(int(x)) for x in row] for row in W]
                
                if verify_exact(U_frac, V_frac, W_frac):
                    # Found a solution
                    U_out = [[float(x) for x in row] for row in U_frac]
                    V_out = [[float(x) for x in row] for row in V_frac]
                    W_out = [[float(x) for x in row] for row in W_frac]
                    return {"U": U_out, "V": V_out, "W": W_out}
            
            # If we are struggling, maybe increase density or change strategy
            # For now, just continue
    
    # If no lower rank found, return schoolbook
    rank_schoolbook = n * m * p
    U_sb = [[Fraction(0)] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[Fraction(0)] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[Fraction(0)] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = Fraction(1)
                V_sb[j * p + k][col] = Fraction(1)
                W_sb[k * n + i][col] = Fraction(1)
                col += 1
    
    U_out = [[float(x) for x in row] for row in U_sb]
    V_out = [[float(x) for x in row] for row in V_sb]
    W_out = [[float(x) for x in row] for row in W_sb]
    return {"U": U_out, "V": V_out, "W": W_out}
