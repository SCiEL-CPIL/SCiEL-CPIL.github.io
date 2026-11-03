import time
import numpy as np

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[a,b,c]."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    return T

def _check_removable(U, V, W, T, max_time=10.0):
    """Check if removing term r still gives the correct tensor.
    Returns the index of the removable term, or -1."""
    start = time.time()
    R = len(U[0])
    T_flat = T.flatten()
    
    # Compute each term's contribution
    term_sums = []
    full_sum = np.zeros_like(T_flat)
    for r in range(R):
        t = np.outer(np.outer(U[:,r], V[:,r]), W[:,r]).flatten()
        term_sums.append(t)
        full_sum += t
    
    # Verify full_sum equals T_flat
    if np.linalg.norm(full_sum - T_flat) > 1e-9:
        return -1
    
    # Try removing each term
    for r_remove in range(R):
        if time.time() - start > max_time:
            break
        sum_without = full_sum - term_sums[r_remove]
        if np.linalg.norm(sum_without - T_flat) < 1e-9:
            return r_remove
    
    return -1

def construct():
    n, m, p = 2, 3, 8
    
    # Dimensions of factors
    dim_U = n * m  # 6
    dim_V = m * p  # 24
    dim_W = p * n  # 16
    
    # Start with known rank-44 solution
    rank = 44
    
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]
    
    col = 0
    
    # Part 1: 2x1x8 part (j=0)
    for k in range(p):
        U[0 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        U[1 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
    
    # Part 2: 2x2x8 part (j=1,2)
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        U[0 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        U[1 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1
        
        U[0 * m + 1][col] = 1
        V[1 * p + k1][col] = 1
        V[2 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        U[1 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        V[1 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        U[0 * m + 1][col] = 1
        U[0 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1
        
        U[1 * m + 1][col] = 1
        U[0 * m + 1][col] = -1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        U[0 * m + 2][col] = 1
        U[1 * m + 2][col] = -1
        V[2 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
    
    # Convert to numpy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    T = _build_tensor(n, m, p)
    
    # Try to remove one term
    r_remove = _check_removable(U_np, V_np, W_np, T, max_time=10.0)
    
    if r_remove >= 0:
        keep = [r for r in range(rank) if r != r_remove]
        U_new = [[U_np[a, r] for r in keep] for a in range(dim_U)]
        V_new = [[V_np[b, r] for r in keep] for b in range(dim_V)]
        W_new = [[W_np[c, r] for r in keep] for c in range(dim_W)]
        
        if _verify(U_new, V_new, W_new):
            return {"U": U_new, "V": V_new, "W": W_new}
    
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Check that the decomposition is exact."""
    n, m, p = 2, 3, 8
    R = len(U[0]) if U else 0
    
    for a in range(n * m):
        i = a // m
        j = a % m
        for b in range(m * p):
            j2 = b // p
            k = b % p
            for c in range(p * n):
                i2 = c % n
                k2 = c // n
                expected = 1 if (i == i2 and j == j2 and k == k2) else 0
                
                val = 0
                for r in range(R):
                    val += U[a][r] * V[b][r] * W[c][r]
                
                if val != expected:
                    return False
    return True

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
    if _verify(res["U"], res["V"], res["W"]):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
