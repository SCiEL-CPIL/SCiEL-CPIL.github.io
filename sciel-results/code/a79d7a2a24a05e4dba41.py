import numpy as np
from scipy.optimize import minimize
import time
import itertools

DIMENSIONS = [3, 4, 5]

def _build_target_tensor():
    """Build the matrix multiplication tensor Y[a, b, c] where a=i*m+j, b=j*p+k, c=k*n+i."""
    n, m, p = DIMENSIONS
    Y = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                Y[a, b, c] = 1.0
    return Y

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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

def _verify_decomposition(U, V, W, Y, tol=1e-9):
    """Verify that the decomposition exactly matches the target tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = np.einsum('ar,br,cr->abc', U, V, W)
    return np.max(np.abs(T - Y)) < tol

def _try_fixed_support(rank, Y, n, m, p, deadline):
    """
    Try to find a rank-rank decomposition by fixing the support pattern
    and solving for the coefficients using linear algebra.
    
    For each rank-1 tensor j, we choose a box support:
    A_j ⊆ {0,...,n*m-1}, B_j ⊆ {0,...,m*p-1}, C_j ⊆ {0,...,p*n-1}
    such that the tensor only has non-zeros in A_j × B_j × C_j.
    
    The equation is: sum_j (u_j ⊗ v_j ⊗ w_j) = Y
    where u_j is supported on A_j, etc.
    """
    A_dim = n * m
    B_dim = m * p
    C_dim = p * n
    
    # The non-zero positions of Y
    nz_positions = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nz_positions.append((a, b, c))
    
    # For a given rank r, we need to assign each of the 60 positions to one of r "boxes"
    # Each box j has a set of positions S_j
    # For position (a,b,c) in S_j, we need u_j[a]*v_j[b]*w_j[c] to contribute to Y[a,b,c]
    # But if multiple boxes cover the same position, their contributions sum
    
    # Simplification: assume each position is covered by exactly one box
    # Then for each box j, we have a system of equations
    # This is still nonlinear, but we can try a specific structure
    
    # Alternative: use the fact that for matrix multiplication, there's a natural
    # block structure. Let's try to exploit that.
    
    # For 3x4x5, think of it as multiplying a 3x4 by 4x5 matrix.
    # The (i,k) entry of the result is sum_j A[i,j]*B[j,k]
    # In tensor form: Y[i*m+j, j*p+k, k*n+i] = 1
    
    # A natural decomposition: for each j (the inner dimension), we have
    # a rank-4 contribution (since j ranges over 4 values)
    # But we can do better by combining j values
    
    # Let's try a direct search over support patterns
    # For small ranks, we can enumerate possible box structures
    
    np.random.seed(42)
    
    # Try random assignments of positions to boxes
    for attempt in range(30):
        if time.time() > deadline:
            break
        
        # Randomly assign each of the 60 positions to one of rank boxes
        assignment = np.random.randint(0, rank, size=len(nz_positions))
        
        # Build the support sets
        boxes = [set() for _ in range(rank)]
        for idx, (a, b, c) in enumerate(nz_positions):
            boxes[assignment[idx]].add((a, b, c))
        
        # For each box, determine the min/max indices to define the box
        # Actually, let's just use the positions directly
        
        # For each box j, we need to find u_j, v_j, w_j such that
        # for all (a,b,c) in box_j: u_j[a]*v_j[b]*w_j[c] = 1 (if Y[a,b,c]=1)
        # and for all other positions: contribution is 0
        
        # This is only possible if the box has a special structure
        # Let's try a simpler approach: for each box, set u_j[a]=1, v_j[b]=1, w_j[c]=1
        # for all (a,b,c) in the box. But this may not work if the box is not a "complete" box
        
        # Better: for each box, check if it forms a valid rank-1 pattern
        # A set S of positions is a valid rank-1 support if there exist sets A, B, C
        # such that S ⊆ A×B×C and for each a in A, there's b in B, c in C with (a,b,c) in S
        # etc. This is complex.
        
        # Let's try a different tactic: use the known structure of matrix multiplication
        # The tensor Y has the property that Y[a,b,c]=1 iff a=i*m+j, b=j*p+k, c=k*n+i
        # for some i,j,k. This means the three indices are linked through i,j,k.
        
        # For a rank-1 tensor to be a valid factor, its support must respect this linking.
        # Specifically, if we fix a set of (i,j,k) triples, the support is:
        # {(i*m+j, j*p+k, k*n+i) : (i,j,k) in T}
        # This is always a valid rank-1 support (with u[a]=1 if a appears, etc.)
        
        # So let's partition the 60 triples into rank groups, and for each group,
        # set u_j[a]=1 if a appears in any triple in the group, etc.
        
        U = np.zeros((A_dim, rank))
        V = np.zeros((B_dim, rank))
        W = np.zeros((C_dim, rank))
        
        for idx, (a, b, c) in enumerate(nz_positions):
            j = assignment[idx]
            U[a, j] = 1
            V[b, j] = 1
            W[c, j] = 1
        
        # Now check if this gives a valid decomposition
        # The issue: if position (a,b,c) is covered by multiple boxes, the sum may exceed 1
        # Let's verify
        if _verify_decomposition(U, V, W, Y):
            # Remove zero columns
            U_nz = np.any(U != 0, axis=0)
            V_nz = np.any(V != 0, axis=0)
            W_nz = np.any(W != 0, axis=0)
            used = U_nz | V_nz | W_nz
            if np.sum(used) < rank:
                U = U[:, used]
                V = V[:, used]
                W = W[:, used]
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _try_optimization(rank, Y, n, m, p, deadline, max_restarts=3, verbose=False):
    """Try to find a decomposition of given rank using L-BFGS-B optimization."""
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        np.random.seed(42 + rank * 100 + restart)
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        def objective(params):
            U_flat = params[:n * m * rank].reshape(n * m, rank)
            V_flat = params[n * m * rank : n * m * rank + m * p * rank].reshape(m * p, rank)
            W_flat = params[n * m * rank + m * p * rank :].reshape(p * n, rank)
            T = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat)
            diff = T - Y
            return np.sum(diff * diff)
        
        def gradient(params):
            U_flat = params[:n * m * rank].reshape(n * m, rank)
            V_flat = params[n * m * rank : n * m * rank + m * p * rank].reshape(m * p, rank)
            W_flat = params[n * m * rank + m * p * rank :].reshape(p * n, rank)
            T = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat)
            diff = T - Y
            gU = np.einsum('abc,br,cr->ar', diff, V_flat, W_flat)
            gV = np.einsum('abc,ar,cr->br', diff, U_flat, W_flat)
            gW = np.einsum('abc,ar,br->cr', diff, U_flat, V_flat)
            return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', jac=gradient, 
                           options={'maxiter': 150, 'maxfun': 150, 'gtol': 1e-7})
            
            if res.fun < 1e-4:
                U_opt = res.x[:n * m * rank].reshape(n * m, rank)
                V_opt = res.x[n * m * rank : n * m * rank + m * p * rank].reshape(m * p, rank)
                W_opt = res.x[n * m * rank + m * p * rank :].reshape(p * n, rank)
                
                # Try rounding to nearest half-integer
                for denom in [2, 4, 1]:
                    U_int = np.round(U_opt * denom) / denom
                    V_int = np.round(V_opt * denom) / denom
                    W_int = np.round(W_opt * denom) / denom
                    
                    if _verify_decomposition(U_int, V_int, W_int, Y):
                        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                    
        except Exception:
            pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    Y = _build_target_tensor()
    
    start_time = time.time()
    deadline = start_time + 135  # Safety margin for 150s limit
    
    # Try fixed support search for various ranks
    for rank in range(25, 61, 2):
        if time.time() > deadline:
            break
        
        result = _try_fixed_support(rank, Y, n, m, p, deadline)
        if result is not None:
            return result
    
    # Try optimization for a few promising ranks
    for rank in [30, 35, 40, 45, 50, 55]:
        if time.time() > deadline:
            break
        
        result = _try_optimization(rank, Y, n, m, p, deadline, max_restarts=5)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _schoolbook()
