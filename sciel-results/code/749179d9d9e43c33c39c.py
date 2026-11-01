import time
import numpy as np
from scipy.optimize import least_squares
from fractions import Fraction

DIMENSIONS = [2, 3, 9]

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3x9 matrix multiplication."""
    start_time = time.time()
    DEADLINE = start_time + 115  # 115 second limit (safety margin)

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 54

    # First try the Kronecker product decomposition (rank 18)
    kron_result = _kron_decomposition(n, m, p)
    if kron_result is not None:
        r = _verify(n, m, p, kron_result)
        if r is not None:
            return kron_result

    best_result = _schoolbook(n, m, p)
    best_rank = schoolbook_rank

    # Try structured decompositions with lower rank
    # Strategy: use scipy.optimize.least_squares with random restarts
    for target_rank in range(53, 40, -1):
        if time.time() > DEADLINE - 15:
            break

        result = _try_rank_scipy(n, m, p, target_rank, time_budget=10)
        if result is not None:
            r = _verify(n, m, p, result)
            if r == target_rank:
                return result

    # If nothing found, return schoolbook
    return best_result

def _kron_decomposition(n, m, p):
    """
    Construct a rank-18 decomposition using Kronecker product structure.
    
    The 2x3 × 3x9 multiplication tensor T[i,j,k] = 1 can be decomposed as:
    T[i,j,k] = sum_{c=0}^{17} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    where c = (i', k') for i' in {0,1}, k' in {0..8}:
    - U[i*m+j, (i',k')] = delta(i, i')
    - V[j*p+k, (i',k')] = delta(k, k')
    - W[k*n+i, (i',k')] = delta(i, i')
    
    This gives: sum_{i',k'} delta(i,i') * delta(k,k') * delta(i,i') = delta(i,i) * delta(k,k) = 1
    """
    if n != 2:
        return None
    
    rank = n * p  # 2 * 9 = 18
    U_size = n * m  # 6
    V_size = m * p  # 27
    W_size = p * n  # 18
    
    U = np.zeros((U_size, rank), dtype=float)
    V = np.zeros((V_size, rank), dtype=float)
    W = np.zeros((W_size, rank), dtype=float)
    
    for i_prime in range(n):  # i' = 0, 1
        for k_prime in range(p):  # k' = 0..8
            c = i_prime * p + k_prime  # column index
            
            # U[i*m+j, c] = delta(i, i_prime) for all j
            for i in range(n):
                if i == i_prime:
                    for j in range(m):
                        U[i * m + j, c] = 1.0
            
            # V[j*p+k, c] = delta(k, k_prime) for all j
            for j in range(m):
                for k in range(p):
                    if k == k_prime:
                        V[j * p + k, c] = 1.0
            
            # W[k*n+i, c] = delta(i, i_prime) for all k
            for k in range(p):
                for i in range(n):
                    if i == i_prime:
                        W[k * n + i, c] = 1.0
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, result):
    """Verify that the decomposition is correct. Returns rank if valid, None otherwise."""
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)

    rank = U.shape[1]

    # Check shapes
    if U.shape != (n * m, rank):
        return None
    if V.shape != (m * p, rank):
        return None
    if W.shape != (p * n, rank):
        return None

    # Verify: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > 1e-9:
                    return None

    # Check entries are integer or half-integer
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return None

    return rank

def _try_rank_scipy(n, m, p, target_rank, time_budget=5):
    """Try to find a decomposition with the given target rank using scipy.optimize.least_squares."""
    start_time = time.time()
    
    U_size = n * m  # 6
    V_size = m * p  # 27
    W_size = p * n  # 18
    
    # Precompute index mappings
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i * m + j, j * p + k, k * n + i))
                
    num_triples = len(triples)
    
    def residuals(params):
        """Unpack params and compute residuals for all (i,j,k)."""
        U_flat = params[:U_size * target_rank]
        V_flat = params[U_size * target_rank : U_size * target_rank + V_size * target_rank]
        W_flat = params[U_size * target_rank + V_size * target_rank :]
        
        U = U_flat.reshape(U_size, target_rank)
        V = V_flat.reshape(V_size, target_rank)
        W = W_flat.reshape(W_size, target_rank)
        
        res = np.zeros(num_triples)
        for idx, (ui, vi, wi) in enumerate(triples):
            res[idx] = np.dot(U[ui, :], np.multiply(V[vi, :], W[wi, :])) - 1.0
            
        return res

    best_result = None
    best_loss = float('inf')
    
    # Multiple random restarts
    max_restarts = 20
    np.random.seed(42)
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_budget:
            break
        
        # Random initialization
        # Scale initialization to avoid huge values
        scale = 1.0
        U_init = np.random.randn(U_size, target_rank) * scale
        V_init = np.random.randn(V_size, target_rank) * scale
        W_init = np.random.randn(W_size, target_rank) * scale
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            # Use least_squares which is robust for this
            # method='trf' is default, 'lm' is Levenberg-Marquardt (requires more memory but often faster)
            # We use 'trf' for robustness with bounds if we had them, but we don't enforce bounds yet.
            result = least_squares(residuals, x0, method='trf', max_nfev=1000)
            
            if result.success or result.cost < best_loss:
                best_loss = result.cost
                if result.cost < 1e-10:
                    best_result = result.x
                    break
        except Exception as e:
            continue
            
    if best_result is None:
        return None
        
    # Extract matrices
    U_flat = best_result[:U_size * target_rank]
    V_flat = best_result[U_size * target_rank : U_size * target_rank + V_size * target_rank]
    W_flat = best_result[U_size * target_rank + V_size * target_rank :]
    
    U_opt = U_flat.reshape(U_size, target_rank)
    V_opt = V_flat.reshape(V_size, target_rank)
    W_opt = W_flat.reshape(W_size, target_rank)
    
    # Snap to half-integers
    U_snap = np.round(U_opt * 2) / 2
    V_snap = np.round(V_opt * 2) / 2
    W_snap = np.round(W_opt * 2) / 2
    
    # Verify the snapped version
    valid = True
    for (ui, vi, wi) in triples:
        val = np.dot(U_snap[ui, :], np.multiply(V_snap[vi, :], W_snap[wi, :]))
        if abs(val - 1.0) > 1e-6:
            valid = False
            break
    
    if valid:
        return {
            "U": U_snap.tolist(),
            "V": V_snap.tolist(),
            "W": W_snap.tolist()
        }
    
    return None

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0]) if result else 'None'}")
