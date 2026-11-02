"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]
import time as _time
import numpy as _np

# EVOLVE-BLOCK-START

def _to_halfint(x):
    """Round to nearest half-integer if close, else None."""
    r = 2.0 * x
    c = int(round(r))
    if abs(2.0 * x - c) < 1e-9:
        return c / 2.0
    return None

def _verify_decomp(n, m, p, U, V, W, tol=1e-9):
    """
    Verify that for all (i,k): 
        sum_j A[i,j]*B[j,k] = sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r]
    for all (i,j,k).
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    Convention: W[k*n+i, r]
    """
    rank = U.shape[1]
    Ua = _np.array(U, dtype=float)
    Va = _np.array(V, dtype=float)
    Wa = _np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = Ua[i*m + j]
                v = Va[j*p + k]
                w = Wa[k*n + i]
                # sum_r u[r] * v[r] * w[r]
                s = _np.dot(_np.multiply(u, v), w)
                if abs(s - 1.0) > tol:
                    return False
    # Also verify orthogonality-like conditions to ensure exactness
    # Actually, the decomposition must satisfy: for each r, the tensor is
    # T_r[i,j,k] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # And sum_r T_r = delta(i,j,k in valid range)... 
    # Wait, the matrix multiplication tensor M[i,j,k] = 1 if valid, but actually:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is: sum_r A_r[i,j] * B_r[j,k] * C_r[k,i]
    # where C_r[k,i] corresponds to W[k*n+i, r]
    # For this to work: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # Wait no! That's not right. Let me re-derive.
    # We need: for all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{ij} * delta_{jk}... 
    # No. The matrix multiplication tensor T satisfies: C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    # where T[i,j,k] = 1 for all valid i,j,k. 
    # The decomposition says: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[i,k,r] (with W indexed as W[k*n+i,r])
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for ALL i in [0,n), j in [0,m), k in [0,p)
    # That's the condition. Let me verify the schoolbook: each (i,j,k) has its own column r, with 1s in the right places.
    # So for a given (i,j,k), the term is 1, all others are 0. Sum = 1. Correct.
    
    # So the verification is: for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = Ua[i*m + j]
                v = Va[j*p + k]
                w = Wa[k*n + i]
                s = _np.dot(_np.multiply(u, v), w)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _check_halfint(U, V, W, tol=1e-9):
    """Check if all entries are half-integers."""
    for mat in [U, V, W]:
        for row in mat:
            for x in row:
                r = _to_halfint(x)
                if r is None:
                    return False
    return True

def _construct_rank_attempt(n, m, p, rank, method="greedy", max_iter=5000, tol=1e-8, timeout=100):
    """
    Try to construct a rank decomposition.
    Returns (U, V, W) as lists of lists if found, else None.
    """
    start = _time.time()
    nm = n * m
    mp = m * p
    pn = p * n
    
    if method == "greedy":
        # Initialize with schoolbook and try to merge columns
        U = [[0.0] * rank for _ in range(nm)]
        V = [[0.0] * rank for _ in range(mp)]
        W = [[0.0] * rank for _ in range(pn)]
        
        # Start with rank = n*m*p schoolbook, but we want exactly `rank` columns
        # For now, just return None for greedy - too complex to implement correctly in limited time
        # Let's use a random approach instead
        pass
    
    if method == "random":
        U = _np.random.randn(nm, rank) * 0.1
        V = _np.random.randn(mp, rank) * 0.1
        W = _np.random.randn(pn, rank) * 0.1
        
        for it in range(max_iter):
            if _time.time() - start > timeout:
                break
            # Compute error: E[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1
            # Gradient of error w.r.t. U[i*m+j,r] = V[j*p+k,r] * W[k*n+i,r]
            # Use SGD-like update
            lr = 0.001 * (1.0 / (1 + it * 0.001))
            
            # Pick random (i,j,k) and update
            i = _np.random.randint(n)
            j = _np.random.randint(m)
            k = _np.random.randint(p)
            
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            
            # Current value
            val = _np.dot(_np.multiply(U[u_idx], V[v_idx]), W[w_idx])
            err = 1.0 - val
            
            # Update all r
            dU = V[v_idx] * W[w_idx] * err
            dV = U[u_idx] * W[w_idx] * err
            dW = U[u_idx] * V[v_idx] * err
            
            U[u_idx] += lr * dU
            V[v_idx] += lr * dV
            W[w_idx] += lr * dW
        
        # Snap to half-integers
        Us = [[_to_halfint(x) for x in row] for row in U]
        Vs = [[_to_halfint(x) for x in row] for row in V]
        Ws = [[_to_halfint(x) for x in row] for row in W]
        
        # Check if any entry failed to snap
        valid = True
        for mat in [Us, Vs, Ws]:
            for row in mat:
                if any(x is None for x in row):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            if _verify_decomp(n, m, p, Us, Vs, Ws, tol=1e-6):
                return [Us, Vs, Ws]
        
        return None
    
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
    
    start = _time.time()
    deadline = start + 120  # must finish within 120s
    
    # First, establish the schoolbook baseline
    U0, V0, W0 = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U0, "V": V0, "W": W0}
    
    # Try to find a better (lower rank) decomposition
    # Known lower bounds: for m x p multiplication, rank >= max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # Try ranks from 30 up to 59 (below schoolbook 60)
    
    for rank in range(30, best_rank):
        if _time.time() - start > deadline - 30:
            break
        # Try random initialization with this rank
        found = _construct_rank_attempt(n, m, p, rank, method="random", 
                                        max_iter=2000, timeout=10)
        if found is not None:
            best_rank = rank
            Uf, Vf, Wf = found
            best_result = {"U": Uf, "V": Vf, "W": Wf}
            # Continue searching for even lower rank
            continue
    
    return best_result

# EVOLVE-BLOCK-END
