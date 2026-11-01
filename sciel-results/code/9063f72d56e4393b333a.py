import time
import numpy as np

DIMENSIONS = [2, 4, 7]

def _build_trivial():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx, r] = 1
                V[b_idx, r] = 1
                W[c_idx, r] = 1
                r += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    R = U.shape[1]
    # Compute tensor product sum
    # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Build expected tensor (sparse)
    # Expected[i*m+j, j*p+k, k*n+i] = 1
    Expected = np.zeros((n * m, m * p, p * n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Expected[i * m + j, j * p + k, k * n + i] = 1
                
    return np.array_equal(T, Expected)

def _search_decomposition(n, m, p, time_limit):
    start_time = time.time()
    dim_a = n * m
    dim_b = m * p
    dim_c = p * n
    
    # We will search for ranks lower than n*m*p
    # Start from a lower bound estimate
    # Lower bound is max(dim_a, dim_b, dim_c) = 28
    # Let's try ranks from 30 to 55
    
    # Use a set of small integers for coefficients
    coeffs = [-1, 0, 1]
    
    # Random search loop
    # Prioritize higher ranks as they are more likely to have a solution
    # and if we find one, we return it.
    ranks_to_try = [55, 54, 52, 50, 49, 45, 40, 35]
    
    for target_rank in ranks_to_try:
        if time.time() - start_time > time_limit:
            break
            
        while time.time() - start_time < time_limit:
            # Generate random U, V, W
            U = np.random.choice(coeffs, size=(dim_a, target_rank))
            V = np.random.choice(coeffs, size=(dim_b, target_rank))
            W = np.random.choice(coeffs, size=(dim_c, target_rank))
            
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
                
            # Check time to avoid long loops
            if time.time() - start_time > time_limit:
                break

    return None

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin
    
    n, m, p = DIMENSIONS
    trivial = _build_trivial()
    trivial_rank = trivial["U"].shape[1]
    
    # Try to find a better decomposition
    # Reserve 10 seconds for overhead/safety
    search_time = min(100, deadline - (time.time() - start_time))
    
    result = _search_decomposition(n, m, p, search_time)
    
    if result is not None:
        # Verify again to be safe
        if _verify(result["U"], result["V"], result["W"], n, m, p):
            if result["U"].shape[1] < trivial_rank:
                return {"U": result["U"].tolist(), "V": result["V"].tolist(), "W": result["W"].tolist()}
    
    # Fallback to trivial
    return {"U": trivial["U"].tolist(), "V": trivial["V"].tolist(), "W": trivial["W"].tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    T = np.einsum('ar,br,cr->abc', U, V, W)
    Expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Expected[i*m + j, j*p + k, k*n + i] = 1
    
    if np.array_equal(T, Expected):
        print(f"Valid decomposition with rank {U.shape[1]}")
    else:
        print("Invalid decomposition")
        diff = np.abs(T - Expected)
        print(f"Max diff: {diff.max()}, Sum diff: {diff.sum()}")
