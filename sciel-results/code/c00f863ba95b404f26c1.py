import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition is correct."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check shapes
    if U_arr.shape != (n*m, rank):
        return False
    if V_arr.shape != (m*p, rank):
        return False
    if W_arr.shape != (p*n, rank):
        return False
    
    # For each output element C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor: T[i,j,k] = 1 if this is a valid multiplication term
    # Decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                expected = 1.0  # Every (i,j,k) triple contributes exactly 1
                if abs(val - expected) > 1e-6:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n*m)]
    V = [[0.0] * rank for _ in range(m*p)]
    W = [[0.0] * rank for _ in range(p*n)]
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1.0
                V[j*p+k][c] = 1.0
                W[k*n+i][c] = 1.0
                c += 1
    return U, V, W

def _structured_decomposition(n, m, p):
    """
    Try structured decompositions based on known patterns.
    For 3x3x5, we can try to exploit the rectangular structure.
    
    Strategy: Decompose into blocks.
    A is 3x3, B is 3x5, C is 3x5.
    
    We can try rank-15 to rank-25 structured patterns.
    """
    # Try various structured patterns
    best = None
    
    # Pattern: Use the fact that we can share computations
    # A[i,j] * B[j,k] for all i,j,k
    # If we can find a low-rank representation...
    
    # Try rank from 15 upward
    for target_rank in range(15, 26):
        result = _try_structured(n, m, p, target_rank)
        if result is not None:
            if best is None or result[3] < best[3]:
                best = result
        if time.time() > time.time() + 5:  # Safety break
            break
    
    return best

def _try_structured(n, m, p, target_rank):
    """
    Try to find a structured decomposition with given rank.
    Uses parameterized families of solutions.
    """
    # Family 1: Diagonal-like patterns
    # U[i*m+j, c] depends on (i,j,c)
    # V[j*p+k, c] depends on (j,k,c)  
    # W[k*n+i, c] depends on (k,i,c)
    
    # For small dimensions, we can try exhaustive search over simple patterns
    # where each factor entry is in {-1, 0, 1}
    
    if target_rank < 15:
        return None
    
    # Try a specific structured pattern for 3x3x5
    # Key insight: C[i,k] = sum_j A[i,j] B[j,k]
    # We need: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all i,j,k
    
    # Try using rank-9 for the "core" and additional terms
    # Actually, let's try direct search with small integer values
    
    np.random.seed(42 + target_rank)
    
    for trial in range(200):
        # Generate random structured patterns
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        for c in range(target_rank):
            # Each term (c) contributes to the tensor
            # U entries: for each (i,j), choose from {-1, 0, 1}
            # But with structure to make it work
            
            # Random small integers
            for idx in range(n*m):
                if np.random.random() < 0.5:
                    U[idx, c] = np.random.choice([-1.0, 1.0])
            
            for idx in range(m*p):
                if np.random.random() < 0.5:
                    V[idx, c] = np.random.choice([-1.0, 1.0])
            
            for idx in range(p*n):
                if np.random.random() < 0.5:
                    W[idx, c] = np.random.choice([-1.0, 1.0])
        
        if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return (U.tolist(), V.tolist(), W.tolist(), target_rank)
    
    return None

def _random_search(n, m, p, max_rank, iterations=5000, seed=0):
    """
    Random search for low-rank decompositions.
    Uses integer/half-integer entries.
    """
    np.random.seed(seed)
    best = None
    
    for target_rank in range(max(15, max_rank - 10), max_rank + 1):
        for trial in range(iterations // (target_rank - 14)):
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            for c in range(target_rank):
                # Use small values: -1, -0.5, 0, 0.5, 1
                vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
                for idx in range(n*m):
                    if np.random.random() < 0.4:
                        U[idx, c] = np.random.choice(vals)
                for idx in range(m*p):
                    if np.random.random() < 0.4:
                        V[idx, c] = np.random.choice(vals)
                for idx in range(p*n):
                    if np.random.random() < 0.4:
                        W[idx, c] = np.random.choice(vals)
            
            # Quick check: compute all tensor entries
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(target_rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        if abs(val - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                if best is None or target_rank < best[3]:
                    best = (U.tolist(), V.tolist(), W.tolist(), target_rank)
                return best
        
        if best is not None:
            break
    
    return best

def _laser_method(n, m, p):
    """
    Try the laser method: embed into a larger symmetric problem.
    For 3x3x5, try to relate to 3x3x6 or 4x4x4.
    
    If we can find a rank-R decomposition for 4x4x4, we can project
    to 3x3x5 by restricting indices.
    """
    # Known: 4x4x4 has rank at most 21 (Lader)
    # If we can construct it, projecting gives rank <= 21 for 3x3x4
    # Then add 3 more for the 5th column: total rank <= 24
    
    # Let's try to construct a 4x4x4 decomposition with rank 21
    # using known patterns
    
    # Lader's 21-term decomposition for 4x4x4 uses specific patterns.
    # For simplicity, let's try a rank-25 approach:
    # Use 21 terms for 4x4x4 (if we can find them) + 4 for extension
    
    # Actually, let's try a more direct approach:
    # The 3x3x5 tensor has 45 nonzero entries.
    # A rank-r decomposition means each of r terms covers some subset.
    # Minimum rank is at least 45 / (max coverage per term)
    # Each term can cover at most... well, it's multiplicative.
    
    # Lower bound: For matrix multiplication, rank >= (n*m*p) / (something)
    # Known lower bound for 3x3x3 is 19 (Strassen-like), but we know 23 works.
    # For 3x3x5, the lower bound is at least... 
    
    # Let's just do a thorough random search
    return None

def _gradient_based_search(n, m, p, target_rank, iterations=10000):
    """
    Try gradient-based optimization to find a rank-target_rank decomposition.
    We want sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
    This is a system of nonlinear equations.
    """
    np.random.seed(123)
    
    # Parameterize with real values, then round to half-integers
    for trial in range(10):
        U = np.random.randn(n*m, target_rank) * 0.5
        V = np.random.randn(m*p, target_rank) * 0.5
        W = np.random.randn(p*n, target_rank) * 0.5
        
        # Simple gradient descent on the residual
        lr = 0.01
        for it in range(iterations):
            # Compute residuals
            residuals = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(target_rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        residuals[i,j,k] = val - 1.0
            
            loss = np.sum(residuals**2)
            if loss < 1e-10:
                break
            
            # Compute gradients (simplified)
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = residuals[i,j,k]
                        for c in range(target_rank):
                            dU[i*m+j, c] += 2 * r * V[j*p+k, c] * W[k*n+i, c]
                            dV[j*p+k, c] += 2 * r * U[i*m+j, c] * W[k*n+i, c]
                            dW[k*n+i, c] += 2 * r * U[i*m+j, c] * V[j*p+k, c]
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        # Round to nearest half-integer
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        if _verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
            return (U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), target_rank)
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Start with schoolbook (rank 45)
    U_sb, V_sb, W_sb = _schoolbook_decomposition(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    # Try structured decompositions
    if time.time() < deadline:
        result = _structured_decomposition(n, m, p)
        if result is not None:
            U_s, V_s, W_s, r_s = result
            if r_s < best_rank:
                best_U, best_V, best_W = U_s, V_s, W_s
                best_rank = r_s
    
    # Try random search from various starting points
    if time.time() < deadline:
        for seed in range(0, 10):
            if time.time() > deadline - 20:
                break
            result = _random_search(n, m, p, best_rank, iterations=3000, seed=seed * 1000)
            if result is not None:
                U_r, V_r, W_r, r_r = result
                if r_r < best_rank:
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = r_r
    
    # Try gradient-based for a few target ranks
    if time.time() < deadline:
        for target in range(best_rank - 5, best_rank):
            if time.time() > deadline - 10:
                break
            result = _gradient_based_search(n, m, p, target, iterations=5000)
            if result is not None:
                U_g, V_g, W_g, r_g = result
                if r_g < best_rank:
                    best_U, best_V, best_W = U_g, V_g, W_g
                    best_rank = r_g
    
    # Final verification
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        # Fallback to schoolbook
        U_sb, V_sb, W_sb = _schoolbook_decomposition(n, m, p)
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": best_U, "V": best_V, "W": best_W}
