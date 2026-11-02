"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _schoolbook_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    must equal 1 for all (i,j,k) and 0 otherwise (but the structure ensures only
    (i,j,k) with i in [0,n), j in [0,m), k in [0,p) are non-zero).
    
    Actually, the matrix multiplication tensor has entries T[i,j,k] = 1 if we're
    computing (A*B)[i,k] = sum_j A[i,j]*B[j,k], so the tensor is indexed by (i,j,k)
    and T[i,j,k] = 1 for all valid i,j,k.
    
    We verify: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    """
    n, m, p = n, m, p
    # Build the full tensor via the decomposition
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We can compute this efficiently using matrix operations.
    
    # For each c, the outer product structure:
    # U_c is a vector of length n*m, reshape to (n, m)
    # V_c is a vector of length m*p, reshape to (m, p)
    # W_c is a vector of length p*n, reshape to (p, n)
    
    # T[i,j,k] = sum_c U_c[i,j] * V_c[j,k] * W_c[k,i]
    
    # Let's verify by checking a few entries or all entries.
    # For exactness, we need all entries to be 1.
    
    # Efficient check: compute T as a 3D array
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(U.shape[1]):
        Uc = U[:, c].reshape(n, m)
        Vc = V[:, c].reshape(m, p)
        Wc = W[:, c].reshape(p, n)
        # T[i,j,k] += Uc[i,j] * Vc[j,k] * Wc[k,i]
        # This is a rank-1 update in 3D
        # T += Uc[:, :, None] * Vc[None, :, :] * Wc.T[None, :, :]
        # Wait, Wc[k,i] so Wc.T[i,k]... no.
        # Wc is (p, n), so Wc[k, i]. We need Wc[k, i] for fixed k,i.
        # T[i,j,k] += Uc[i,j] * Vc[j,k] * Wc[k,i]
        
        # Let's do it element-wise for correctness, but it might be slow.
        # For small n,m,p it's fine.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += Uc[i, j] * Vc[j, k] * Wc[k, i]
    
    # Check if T is all ones
    return np.allclose(T, 1.0, atol=1e-9)


def _is_half_integer(x, tol=1e-6):
    """Check if x is close to a half-integer (multiple of 0.5)."""
    y = 2.0 * x
    return np.allclose(y, np.round(y), atol=tol)


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_rank = U.shape[1]
    best_result = (U, V, W)
    
    # Try to find a lower rank decomposition
    # For 3x3 * 3x7, the known minimal rank is not trivially lower than 63
    # Let's try a greedy/randomized approach to see if we can reduce rank
    
    # Strategy: try random rank r from min_possible to current best
    # The theoretical lower bound for matrix multiplication rank is at least max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # But actually for matrix multiplication, the rank is at least the border rank, which for 3x3*3x7...
    # Let's just try to find any valid decomposition with rank < 63
    
    # Try a simple approach: use the fact that matrix multiplication can be seen as
    # contracting along the middle dimension. 
    
    # For now, return the schoolbook decomposition as it's guaranteed correct.
    # A more sophisticated search would try random matrices and solve for W, etc.
    
    # Let's try a quick random search for lower rank
    np.random.seed(42)
    
    for rank in range(21, best_rank):
        if time.time() > deadline:
            break
        
        # Try random U and V of shape (n*m, rank) and (m*p, rank)
        # Then solve for W such that the decomposition holds
        # This is a linear system in W
        
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            U_rand = np.random.randn(n * m, rank)
            V_rand = np.random.randn(m * p, rank)
            
            # We need: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # This is linear in W.
            # For each (i,j,k), we have: sum_c W[k*n+i, c] * (U[i*m+j, c] * V[j*p+k, c]) = 1
            # Let A_{(i,j,k), (k',i',c)} = U[i*m+j, c] * V[j*p+k, c] if k'=k and i'=i, else 0
            # This is a big linear system.
            
            # Actually, let's set up the linear system more carefully.
            # Unknowns: W[k*n+i, c] for k in [0,p), i in [0,n), c in [0,rank)
            # Total unknowns: p*n*rank
            
            # Equations: for each (i,j,k): sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1
            # Total equations: n*m*p = 63
            
            # Build the linear system
            num_unknowns = p * n * rank
            num_equations = n * m * p
            
            A = np.zeros((num_equations, num_unknowns), dtype=np.float64)
            b = np.ones(num_equations, dtype=np.float64)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(rank):
                            # W[k*n+i, c] has index k*n + i for the row part, c for column
                            # In the flattened unknowns: index = (k*n + i) * rank + c
                            unknown_idx = (k * n + i) * rank + c
                            A[eq_idx, unknown_idx] = U_rand[i * m + j, c] * V_rand[j * p + k, c]
                        eq_idx += 1
            
            # Solve the linear system
            try:
                W_flat, residuals, rank_solver, sv = np.linalg.lstsq(A, b, rcond=None)
                W_rand = W_flat.reshape(p * n, rank)
                
                # Verify the solution
                if _verify_decomposition(U_rand, V_rand, W_rand, n, m, p):
                    # Check if entries are half-integers
                    if (_is_half_integer(U_rand) and _is_half_integer(V_rand) and _is_half_integer(W_rand)):
                        best_rank = rank
                        best_result = (U_rand, V_rand, W_rand)
                        break
                    else:
                        # Try to round to half-integers and verify
                        U_rounded = np.round(2 * U_rand) / 2
                        V_rounded = np.round(2 * V_rand) / 2
                        W_rounded = np.round(2 * W_rand) / 2
                        if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
                            best_rank = rank
                            best_result = (U_rounded, V_rounded, W_rounded)
                            break
            except np.linalg.LinAlgError:
                continue
    
    # Convert to lists of lists for the return format
    U_list = best_result[0].tolist()
    V_list = best_result[1].tolist()
    W_list = best_result[2].tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
