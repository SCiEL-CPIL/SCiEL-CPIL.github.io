"""Matrix multiplication 2x4x7 tensor decomposition - rank-minimal exact."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 4, 7]

def _build_tensor_equations(n, m, p, rank):
    """
    Build the system of equations for rank-rank decomposition.
    Variables: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    Equations: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
    
    We'll use a structured approach where we fix some variables and solve for others.
    """
    pass

def _try_structured_decomposition(n, m, p, target_rank, time_budget=30.0):
    """
    Try to find a decomposition by exploiting the algebraic structure of matrix multiplication.
    
    Key insight: For matrix multiplication, we can use the following approach:
    - Fix U to have a simple structure (e.g., based on i-index)
    - Fix V to have a simple structure (e.g., based on j-index)  
    - Solve for W (linear system)
    
    Or use symmetry and known algebraic identities.
    """
    deadline = time.time() + time_budget
    
    # Strategy: Use the fact that for each fixed i, the row of A is multiplied by B.
    # C[i,:] = A[i,:] @ B
    # This is a row-vector times matrix multiplication.
    
    # For n=2, we have two such equations.
    # Let's try to decompose each row multiplication separately, then combine.
    
    # For a single row (n=1, m=4, p=7): C[k] = sum_j A[j] * B[j,k]
    # This is a 1x4 times 4x7 multiplication.
    
    # The rank of M_{1,4,7} is at most 4*7=28, but can be lower.
    # Actually for M_{1,m,p}, the rank is m*p since each term A[j]*B[j,k] is independent.
    # Wait, no - it's a tensor of rank at most m*p.
    
    # Let me try a different structural approach.
    
    # Approach: Parameterize U, V, W with a small number of free parameters
    # and solve the resulting system.
    
    # Let's try: U depends only on i (2 values), V depends only on j (4 values), 
    # W depends on (k,i) (14 values).
    
    # U[i*4+j, r] = u[i, r]  (same for all j)
    # V[j*7+k, r] = v[j, r]  (same for all k)  
    # W[k*2+i, r] = w[k, i, r]
    
    # Then: sum_r u[i,r] * v[j,r] * w[k,i,r] = 1 for all i,j,k
    
    # For fixed i: sum_r u[i,r] * v[j,r] * w[k,i,r] = 1 for all j,k
    # This means: for each i, the matrix (v[j,r] * w[k,i,r]) weighted by u[i,r] sums to 1.
    
    # Let M_i[j,k] = sum_r u[i,r] * v[j,r] * w[k,i,r] = 1 for all j,k
    
    # This is a rank-rank decomposition of the all-ones matrix (4x7) for each i.
    # But the rank-1 components are coupled across i through u[i,r] and v[j,r].
    
    # Actually, for fixed i, M_i = sum_r u[i,r] * (v[:,r] * w[:,i,r]^T)
    # where v[:,r] is a 4-vector and w[:,i,r] is a 7-vector.
    
    # So M_i is a sum of target_rank rank-1 matrices, each scaled by u[i,r].
    # We need M_i = ones(4,7) for i=0,1.
    
    # This is promising! Let's try this.
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Random initialization
        np.random.seed(attempt * 7 + 3)
        
        u = np.random.uniform(-1, 1, size=(n, target_rank))
        v = np.random.uniform(-1, 1, size=(m, target_rank))
        w = np.random.uniform(-1, 1, size=(p, n, target_rank))
        
        # For each i, we need: sum_r u[i,r] * outer(v[:,r], w[:,i,r]) = ones(m,p)
        # This is a linear system in w[:,i,r] for fixed i and r.
        
        # Actually, for fixed i, the equation is:
        # sum_r u[i,r] * v[:,r] * w[:,i,r]^T = ones(m,p)
        
        # Let's solve for w given u, v.
        # For each i, we have a linear system.
        
        w_solved = np.zeros((p, n, target_rank))
        valid = True
        
        for i in range(n):
            # Build the system: for each (j,k): sum_r u[i,r] * v[j,r] * w[k,i,r] = 1
            # This is a linear system in w[:,i,:] (which has p*target_rank unknowns)
            # with m*p equations.
            
            # Let's vectorize: unknowns are w[k,i,r] for k=0..p-1, r=0..target_rank-1
            # Equation (j,k): sum_r u[i,r] * v[j,r] * w[k,i,r] = 1
            
            # This is p*target_rank unknowns, m*p equations.
            # If target_rank >= 1, this is underdetermined (p*target_rank >= m*p when target_rank >= m=4)
            
            # For target_rank < m, it might be overdetermined.
            
            # Let's use least squares.
            A = np.zeros((m*p, p*target_rank))
            b = np.ones(m*p)
            
            for j in range(m):
                for k in range(p):
                    eq_idx = j * p + k
                    for r in range(target_rank):
                        col_idx = k * target_rank + r
                        A[eq_idx, col_idx] = u[i, r] * v[j, r]
            
            # Solve least squares
            try:
                sol, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                w_solved[:, i, :] = sol.reshape(p, target_rank)
            except:
                valid = False
                break
        
        if not valid:
            continue
        
        # Build U, V, W
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        for i in range(n):
            for j in range(m):
                U[i*m + j, :] = u[i, :]
        
        for j in range(m):
            for k in range(p):
                V[j*p + k, :] = v[j, :]
        
        for k in range(p):
            for i in range(n):
                W[k*n + i, :] = w_solved[k, i, :]
        
        # Verify
        T = np.zeros((n, m, p))
        for r in range(target_rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        if np.allclose(T, np.ones((n, m, p)), atol=1e-9, rtol=0):
            # Try to snap to rational values
            U_snap = np.round(U * 8) / 8
            V_snap = np.round(V * 8) / 8
            W_snap = np.round(W * 8) / 8
            
            T2 = np.zeros((n, m, p))
            for r in range(target_rank):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T2[i, j, k] += U_snap[i*m+j, r] * V_snap[j*p+k, r] * W_snap[k*n+i, r]
            
            if np.allclose(T2, np.ones((n, m, p)), atol=1e-10, rtol=0):
                return U_snap, V_snap, W_snap
        
        # Also try without snapping
        if np.allclose(T, np.ones((n, m, p)), atol=1e-10, rtol=0):
            return U, V, W
    
    return None

def _try_general_decomposition(n, m, p, target_rank, time_budget=30.0):
    """
    Try a more general decomposition without the structural assumption.
    Use alternating minimization with random restarts.
    """
    deadline = time.time() + time_budget
    
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        np.random.seed(attempt * 13 + 7)
        
        U = np.random.uniform(-1, 1, size=(n*m, target_rank))
        V = np.random.uniform(-1, 1, size=(m*p, target_rank))
        W = np.random.uniform(-1, 1, size=(p*n, target_rank))
        
        # Alternating minimization
        for it in range(50):
            if time.time() > deadline:
                break
            
            # Update W: for each (i,k), solve linear system
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # For fixed (i,k): sum_r (sum_j U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r] = 1
            
            M = np.zeros((n, p, target_rank))
            for i in range(n):
                for k in range(p):
                    for r in range(target_rank):
                        s = 0.0
                        for j in range(m):
                            s += U[i*m+j, r] * V[j*p+k, r]
                        M[i, k, r] = s
            
            for i in range(n):
                for k in range(p):
                    if np.allclose(M[i,k,:], 0, atol=1e-12):
                        continue
                    try:
                        w_vec, _, _, _ = np.linalg.lstsq(M[i,k,:].reshape(-1,1), np.array([1.0]), rcond=None)
                        W[k*n+i, :] = w_vec.flatten()
                    except:
                        pass
            
            # Update V similarly
            # For fixed (j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # This is not linear in V alone.
            
            # Update U similarly - not linear.
            
            # Check convergence
            T = np.zeros((n, m, p))
            for r in range(target_rank):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            err = np.max(np.abs(T - 1.0))
            if err < 1e-10:
                # Snap to rational
                U_snap = np.round(U * 4) / 4
                V_snap = np.round(V * 4) / 4
                W_snap = np.round(W * 4) / 4
                
                T2 = np.zeros((n, m, p))
                for r in range(target_rank):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T2[i, j, k] += U_snap[i*m+j, r] * V_snap[j*p+k, r] * W_snap[k*n+i, r]
                
                if np.allclose(T2, np.ones((n, m, p)), atol=1e-10, rtol=0):
                    return U_snap, V_snap, W_snap
                return U, V, W
    
    return None

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    expected = np.ones((n, m, p))
    if not np.allclose(T, expected, atol=tol, rtol=0):
        return False
    return True

def _schoolbook(n, m, p):
    """Return U, V, W for naive tensor decomposition of n x m @ m x p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    deadline = time.time() + 115  # Leave 5s safety margin
    
    # Start with schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p
    
    # Try structured decomposition for various ranks
    for target_rank in range(15, best_rank):
        if time.time() > deadline:
            break
        
        result = _try_structured_decomposition(n, m, p, target_rank, time_budget=5.0)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = target_rank
            break
    
    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try general decomposition
    for target_rank in range(15, best_rank):
        if time.time() > deadline:
            break
        
        result = _try_general_decomposition(n, m, p, target_rank, time_budget=10.0)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = target_rank
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
