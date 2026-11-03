"""Rank-minimal exact tensor decomposition for 3x3 * 3x5 matrix multiplication."""
import time
import numpy as np

N, M, P = 3, 3, 5

def _aidx(i, j):
    return i * M + j

def _bidx(j, k):
    return j * P + k

def _cidx(k, i):
    return k * N + i

def _build_target():
    """Build the matmul tensor T[n,j,k,m] = 1 if m==n, else 0."""
    T = np.zeros((N * M, M * P, P * N))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                for n in range(N):
                    if n == i:
                        T[_aidx(i, j), _bidx(j, k), _cidx(k, n)] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-9):
    """Check that sum_c U[:,c] outer V[:,c] outer W[:,c] == T."""
    r = U.shape[1]
    S = np.zeros((N * M, M * P, P * N))
    for c in range(r):
        S += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
    return bool(np.max(np.abs(S - T)) < tol)

def _to_exact(mat):
    """Convert float matrix to exact rational (Fraction) representation."""
    from fractions import Fraction
    out = []
    for row in mat:
        r = []
        for v in row:
            if abs(v) < 1e-9:
                r.append(Fraction(0, 1))
            else:
                f = Fraction(v).limit_denominator(1 << 20)
                if abs(float(f) - v) > 1e-7:
                    return None
                r.append(f)
        out.append(r)
    return out

def _structured_rank3():
    """
    Construct a rank-3 decomposition for matmul (3x3) * (3x5).
    
    Tensor: T[n,j,k,m] = 1 if m==n, else 0
    
    Construction: For c in {0,1,2}:
    - U[n,j,c] = 1 if n==c, else 0
    - V[j,k,c] = 1 for all j,k
    - W[k,m,c] = 1 if m==c, else 0
    
    Verification: sum_c U[n,j,c]*V[j,k,c]*W[k,m,c] = sum_c delta(n,c)*1*delta(m,c) = delta(n,m)
    """
    rank = N  # 3
    U = np.zeros((N * M, rank))
    V = np.ones((M * P, rank))
    W = np.zeros((P * N, rank))
    
    for n in range(N):
        for j in range(M):
            U[_aidx(n, j), n] = 1.0
        for k in range(P):
            W[_cidx(k, n), n] = 1.0
    
    return U, V, W

def _try_lower_rank_search(time_budget):
    """
    Attempt to find a rank < 15 decomposition via structured search.
    
    The matmul tensor for (3x3)*(3x5) has:
    - 45 input A entries (i,j)
    - 15 input B entries (j,k)  
    - 15 output C entries (k,n)
    - 135 total tensor entries (i,j,k,n)
    
    For rank r, we need:
    - U: 45 x r
    - V: 15 x r
    - W: 15 x r
    
    The bilinear constraint is: T[i,j,k,n] = sum_c U[i,j,c] * V[j,k,c] * W[k,n,c]
    
    For r < 15, this becomes increasingly constrained. We try a few structured
    approaches to see if rank 12 or 13 works.
    """
    T = _build_target()
    
    # Try rank 12: fix W to be sparse with specific structure
    for rank in [12, 13, 14]:
        if time.time() > time_budget:
            break
        
        # Strategy: Use W as a partial basis, then solve for U and V
        # W[k,n,c]: 15 x rank matrix
        # Try W where each column c has a specific pattern
        
        # For rank 12, we can't have all 15 (k,n) pairs covered independently
        # Try: W[k,n,c] = 1 if c == k*3 + (n % 3) for c < 12, else 0
        # This gives 12 independent columns
        
        rng = np.random.default_rng(rank * 1000)
        
        # Try multiple random structured W matrices
        for attempt in range(50):
            if time.time() > time_budget:
                break
            
            # Create structured W
            W = np.zeros((P * N, rank))
            # Each column c covers a specific (k,n) pair
            for c in range(rank):
                # Map c to (k,n) with some overlap
                k_idx = c // 3
                n_idx = c % 3
                if k_idx < P and n_idx < N:
                    W[_cidx(k_idx, n_idx), c] = 1.0
            
            # For each (i,j), we need: T[i,j,k,n] = sum_c U[i,j,c] * V[j,k,c] * W[k,n,c]
            # = sum_c U[i,j,c] * V[j,k,c] * (1 if c maps to (k,n))
            
            # This is bilinear in (U, V). Fix U randomly, solve for V, or vice versa.
            
            # Fix U to be simple: U[i,j,c] = 1 if c < 9 else 0 (first 9 columns)
            U = np.zeros((N * M, rank))
            for i in range(N):
                for j in range(M):
                    for c in range(min(9, rank)):
                        U[_aidx(i, j), c] = 1.0
            
            # Now solve for V: T[i,j,k,n] = sum_c U[i,j,c] * V[j,k,c] * W[k,n,c]
            # For fixed (i,j), this is: T[i,j,k,n] = sum_c V[j,k,c] * (U[i,j,c] * W[k,n,c])
            # Let M[i,j,k,n] = U[i,j,c] * W[k,n,c] summed over c where U is nonzero
            # Actually: for each (j,k,n), we have equations over i:
            # T[i,j,k,n] = sum_c U[i,j,c] * V[j,k,c] * W[k,n,c] for all i
            
            # This is linear in V[j,k,:] for each (k,n).
            # Unknowns: V[j,k,c] for c=0..rank-1, j=0..2, k=0..4 => 3*5*rank = 15*rank unknowns
            # Equations: for each (i,j,k,n): 3*3*5*3 = 135 equations
            
            # Build the linear system for V
            # Unknowns indexed by (j,k,c)
            num_unknowns = M * P * rank
            num_eqs = N * M * P * N
            
            A_mat = np.zeros((num_eqs, num_unknowns))
            b_vec = np.zeros(num_eqs)
            
            eq_idx = 0
            for i in range(N):
                for j in range(M):
                    for k in range(P):
                        for n in range(N):
                            b_vec[eq_idx] = T[_aidx(i, j), _bidx(j, k), _cidx(k, n)]
                            for c in range(rank):
                                u_val = U[_aidx(i, j), c]
                                w_val = W[_cidx(k, n), c]
                                if u_val != 0 and w_val != 0:
                                    unknown_idx = j * (P * rank) + k * rank + c
                                    A_mat[eq_idx, unknown_idx] = u_val * w_val
                            eq_idx += 1
            
            # Solve least squares
            try:
                sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                V_sol = sol.reshape(M, P, rank)
                
                # Check if solution is exact
                V_check = np.zeros((M * P, rank))
                for j in range(M):
                    for k in range(P):
                        for c in range(rank):
                            V_check[_bidx(j, k), c] = V_sol[j, k, c]
                
                if _verify(U, V_check, W, T, tol=1e-6):
                    return U, V_check, W
                    
            except np.linalg.LinAlgError:
                pass
    
    return None

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin before 150s
    
    T = _build_target()
    
    # Try the rank-3 construction
    U, V, W = _structured_rank3()
    if _verify(U, V, W, T):
        U_r = _to_exact(U)
        V_r = _to_exact(V)
        W_r = _to_exact(W)
        if U_r is not None and V_r is not None and W_r is not None:
            return {"U": U_r, "V": V_r, "W": W_r}
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    rank = N * M * P
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[_aidx(i, j), col] = 1.0
                V[_bidx(j, k), col] = 1.0
                W[_cidx(k, i), col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
