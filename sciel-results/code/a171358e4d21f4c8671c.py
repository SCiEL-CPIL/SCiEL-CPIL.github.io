import time
import numpy as np
from itertools import product
from fractions import Fraction

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _build_target_tensor(n, m, p):
    """Build the matmul tensor as a list of (a, b, c) indices where T=1."""
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))
    return triples

def _verify(U, V, W, n, m, p, tol=1e-7):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    triples = _build_target_tensor(n, m, p)
    for (a, b, c) in triples:
        val = np.dot(U_arr[a, :], np.multiply(V_arr[b, :], W_arr[c, :]))
        if abs(val - 1.0) > tol:
            return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using Fraction arithmetic."""
    U_frac = [[Fraction(int(x), 1) if float(x) == int(x) else Fraction(x).limit_denominator(1000) for x in row] for row in U]
    V_frac = [[Fraction(int(x), 1) if float(x) == int(x) else Fraction(x).limit_denominator(1000) for x in row] for row in V]
    W_frac = [[Fraction(int(x), 1) if float(x) == int(x) else Fraction(x).limit_denominator(1000) for x in row] for row in W]
    
    triples = _build_target_tensor(n, m, p)
    for (a, b, c) in triples:
        val = sum(U_frac[a][r] * V_frac[b][r] * W_frac[c][r] for r in range(len(U_frac[0])))
        if val != 1:
            return False
    return True

def _snap_to_half_integers(arr, tol=0.01):
    """Snap values to nearest half-integer if close enough."""
    result = np.copy(arr)
    for idx in np.ndindex(result.shape):
        val = result[idx]
        nearest_half = round(val * 2) / 2.0
        if abs(val - nearest_half) < tol:
            result[idx] = nearest_half
    return result

def _try_rank_r(n, m, p, r, max_iters=200, lr=0.05):
    """Try to find rank-r decomposition using gradient descent."""
    if _time_left() < 5:
        return None
    
    U = np.random.randn(n * m, r) * 0.1
    V = np.random.randn(m * p, r) * 0.1
    W = np.random.randn(p * n, r) * 0.1
    
    triples = _build_target_tensor(n, m, p)
    
    for it in range(max_iters):
        if _time_left() < 5:
            break
        # Compute loss and gradients
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for (a, b, c) in triples:
            current = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
            error = current - 1.0
            loss += error * error
            
            for rr in range(r):
                dU = error * V[b, rr] * W[c, rr]
                dV = error * U[a, rr] * W[c, rr]
                dW = error * U[a, rr] * V[b, rr]
                grad_U[a, rr] += dU
                grad_V[b, rr] += dV
                grad_W[c, rr] += dW
        
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        if loss < 1e-10:
            break
    
    U_s = _snap_to_half_integers(U)
    V_s = _snap_to_half_integers(V)
    W_s = _snap_to_half_integers(W)
    
    if _verify(U_s.tolist(), V_s.tolist(), W_s.tolist(), n, m, p, tol=1e-6):
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-6):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _kronecker_decomposition(n, m, p):
    """
    View 2x3x6 multiplication as Kronecker product of 2x3x1 and 1x3x6.
    
    For 2x3x1 (n1=2, m1=3, p1=1):
      Rank-3 decomposition:
        U1[i,j] = 1 if (i,j) matches, V1[j,0] = 1, W1[0,i] = 1
      Actually, for p1=1, it's just outer products.
      
    For 1x3x6 (n2=1, m2=3, p2=6):
      Rank-6 decomposition:
        U2[0,j] = 1, V2[j,k] = 1, W2[k,0] = 1
      
    Kronecker product gives rank 3*6 = 18, but we can do better.
    
    Actually, let me think more carefully. The matmul tensor T[i,j,k] for 2x3*3x6
    can be factored. Let's try a structured approach.
    """
    n, m, p = 2, 3, 6
    
    # Try rank 12: exploit that n=2 is small
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # For each k, we have 2 outputs (i=0,1) and 3 inputs (j=0,1,2)
    
    # Strategy: for each k (0..5), use a rank-2 decomposition of the 2x3 matrix
    # (rows i, cols j) that computes both C[0,k] and C[1,k]
    
    # Actually, let's try the Kronecker approach more carefully.
    # The tensor T is indexed by (i,j,k) with i in {0,1}, j in {0,1,2}, k in {0..5}
    # T[i,j,k] = 1 for all valid (i,j,k)
    
    # We can write this as: T[i,j,k] = T1[i,j,0] * T2[0,j,k] where
    # T1 is 2x3x1 and T2 is 1x3x6
    # But this isn't quite right because of the shared j index.
    
    # Let me try a direct structured construction for rank 12.
    # For each k, we need to compute:
    #   C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    #   C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # Using rank-2 per k (total rank 12):
    # For each k, use two terms:
    #   Term 1: captures (A[0,0]*B[0,k] + A[1,0]*B[0,k]) 
    #   Term 2: captures (A[0,1]*B[1,k] + A[1,1]*B[1,k] + A[0,2]*B[2,k] + A[1,2]*B[2,k])
    # This doesn't work cleanly.
    
    # Let me just try gradient descent with the Kronecker structure as initialization.
    if _time_left() < 10:
        return None
    
    r = 12
    # Initialize with Kronecker-like structure
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # For each k (0..5), use 2 terms
    for k in range(p):
        for term in range(2):
            r_idx = k * 2 + term
            # Term 0: handles j=0 for both i values
            # Term 1: handles j=1,2 for both i values
            if term == 0:
                # U[i*m + 0] = 1 for i=0,1
                U[0 * m + 0, r_idx] = 1
                U[1 * m + 0, r_idx] = 1
                # V[0*p + k] = 1
                V[0 * p + k, r_idx] = 1
                # W[k*n + i] = 1 for i=0,1
                W[k * n + 0, r_idx] = 1
                W[k * n + 1, r_idx] = 1
            else:
                # Handle j=1 and j=2
                # This is trickier - we need to combine two products
                # Use: A[0,1]*B[1,k] + A[1,1]*B[1,k] + A[0,2]*B[2,k] + A[1,2]*B[2,k]
                # = (A[0,1] + A[1,1]) * B[1,k] + (A[0,2] + A[1,2]) * B[2,k]
                # But we can't factor this with a single term in rank-1...
                # Actually we can use: (A[0,1]*B[1,k] + A[0,2]*B[2,k]) for i=0
                # and (A[1,1]*B[1,k] + A[1,2]*B[2,k]) for i=1
                # But that's 2 more terms, making it rank 4 per k.
                
                # Let me try a different split. Use 3 terms per k for rank 18, or
                # try to be clever.
                
                # Actually, for rank 12, let's use a different approach.
                # Use the fact that we can share across k values.
                
                # Let me just add small random perturbation and use gradient descent
                U[0 * m + 1, r_idx] = 0.5
                U[1 * m + 1, r_idx] = 0.5
                U[0 * m + 2, r_idx] = 0.5
                U[1 * m + 2, r_idx] = 0.5
                V[1 * p + k, r_idx] = 0.5
                V[2 * p + k, r_idx] = 0.5
                W[k * n + 0, r_idx] = 0.5
                W[k * n + 1, r_idx] = 0.5
    
    # Add small random perturbation to break symmetry
    U += np.random.randn(n * m, r) * 0.01
    V += np.random.randn(m * p, r) * 0.01
    W += np.random.randn(p * n, r) * 0.01
    
    # Gradient descent polish
    triples = _build_target_tensor(n, m, p)
    lr = 0.01
    for it in range(300):
        if _time_left() < 5:
            break
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for (a, b, c) in triples:
            current = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
            error = current - 1.0
            loss += error * error
            
            for rr in range(r):
                dU = error * V[b, rr] * W[c, rr]
                dV = error * U[a, rr] * W[c, rr]
                dW = error * U[a, rr] * V[b, rr]
                grad_U[a, rr] += dU
                grad_V[b, rr] += dV
                grad_W[c, rr] += dW
        
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        if loss < 1e-12:
            break
    
    U_s = _snap_to_half_integers(U)
    V_s = _snap_to_half_integers(V)
    W_s = _snap_to_half_integers(W)
    
    if _verify(U_s.tolist(), V_s.tolist(), W_s.tolist(), n, m, p, tol=1e-6):
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-6):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _try_structured_rank_r(n, m, p, r):
    """Try structured decomposition exploiting the small n=2 dimension."""
    if _time_left() < 5:
        return None
    
    if r == 12:
        result = _kronecker_decomposition(n, m, p)
        if result is not None:
            return result
    
    best = None
    n_restarts = 5
    for restart in range(n_restarts):
        if _time_left() < 10:
            break
        result = _try_rank_r(n, m, p, r, max_iters=100, lr=0.03)
        if result is not None:
            return result
    return best

def construct():
    n, m, p = 2, 3, 6
    
    # Try rank 12 first (Kronecker structure)
    if _time_left() > 30:
        result = _kronecker_decomposition(n, m, p)
        if result is not None:
            return result
    
    # Try other ranks
    for r in [13, 14, 15, 16, 17, 18]:
        if _time_left() < 30:
            break
        result = _try_structured_rank_r(n, m, p, r)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
