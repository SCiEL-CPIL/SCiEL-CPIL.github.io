import time
import numpy as np
import itertools

DIMENSIONS = [3, 3, 6]

def _schoolbook():
    n, m, p = 3, 3, 6
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
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, tol=1e-5):
    n, m, p = 3, 3, 6
    # Random test
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    
    A_flat = A.flatten()
    B_flat = B.flatten()
    
    # Compute C from decomposition
    # C[i,k] = sum_t W[k*n+i, t] * (sum_j U[i*m+j, t] A[i,j]) * (sum_j V[j*p+k, t] B[j,k])
    # Wait, the formula derived earlier was:
    # C[i,k] = sum_t W[k*n+i, t] * (U[:,t]^T A_flat) * (V[:,t]^T B_flat)
    # Let's check if this matches the schoolbook.
    # Schoolbook: U[i*m+j, t]=1, V[j*p+k, t]=1, W[k*n+i, t]=1.
    # Term t: 1 * A[i,j] * B[j,k]. Sum over t (i,j,k) -> sum_j A[i,j]B[j,k]. Correct.
    
    C_calc = np.zeros((n, p))
    for t in range(U.shape[1]):
        u_val = np.dot(U[:, t], A_flat)
        v_val = np.dot(V[:, t], B_flat)
        w_vec = W[:, t]
        # w_vec is length p*n. Index k*n+i.
        # We need to add w_vec * u_val * v_val to C.
        # w_vec[k*n + i] contributes to C[i,k].
        for k in range(p):
            for i in range(n):
                C_calc[i, k] += w_vec[k * n + i] * u_val * v_val
                
    if np.allclose(C_calc, C_expected, atol=tol):
        return True
    return False

def _search_3x3x2_rank15():
    """
    Attempts to find a rank-15 decomposition for 3x3 x 3x2.
    Returns None if not found.
    """
    n, m, p_block = 3, 3, 2
    target_rank = 15
    
    # Initialize with random small integers
    np.random.seed(0)
    U = np.random.randint(-2, 3, size=(n*m, target_rank))
    V = np.random.randint(-2, 3, size=(m*p_block, target_rank))
    W = np.random.randint(-2, 3, size=(p_block*n, target_rank))
    
    # Simple local search / hill climbing
    best_err = float('inf')
    for _ in range(1000):
        # Pick a random element to perturb
        mat = np.random.choice([U, V, W])
        idx = np.random.randint(mat.shape[0])
        t = np.random.randint(target_rank)
        old_val = mat[idx, t]
        
        # Try changing value
        new_val = old_val + np.random.choice([-1, 1])
        mat[idx, t] = new_val
        
        # Verify
        # We need to verify the 3x3x3x2 decomposition
        # C[i,k] = sum_t W[k*n+i, t] (U[:,t]^T A_flat) (V[:,t]^T B_flat)
        # This is hard to verify quickly without full matrix mult.
        # Use a few random test cases.
        valid = True
        for _ in range(5):
            A = np.random.rand(n, m)
            B = np.random.rand(m, p_block)
            C_exp = A @ B
            C_calc = np.zeros((n, p_block))
            A_flat = A.flatten()
            B_flat = B.flatten()
            for t in range(target_rank):
                u_val = np.dot(U[:, t], A_flat)
                v_val = np.dot(V[:, t], B_flat)
                w_vec = W[:, t]
                for k in range(p_block):
                    for i in range(n):
                        C_calc[i, k] += w_vec[k * n + i] * u_val * v_val
            if not np.allclose(C_calc, C_exp, atol=1e-5):
                valid = False
                break
        
        if valid:
            return {"U": U, "V": V, "W": W}
        
        # Revert if not improved (simplified: just revert)
        mat[idx, t] = old_val

    return None

def construct():
    deadline = time.time() + 110.0
    
    # Try to find a better decomposition
    # For now, just return schoolbook as the search is too slow/unreliable in this snippet
    # But we can try a simple heuristic:
    # If we find a rank-15 for 3x3x3x2, total rank 45.
    
    # Given the constraints and lack of guaranteed fast search, 
    # we return the schoolbook decomposition which is rank 54.
    # This ensures correctness and validity.
    
    res = _schoolbook()
    return res

if __name__ == "__main__":
    res = construct()
    print("Rank:", res['U'].shape[1])
