import numpy as np
import time
from fractions import Fraction
import sys

DIMENSIONS = [2, 5, 5]

def _verify_exact(U, V, W, n, m, p, tol=1e-9):
    """
    Verifies if U, V, W form a valid decomposition.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    
    Condition: For all i, k, a, b, c, d:
    sum_r U[a*m+b, r] * V[c*p+d, r] * W[k*n+i, r] = 1 if a==i, b==c, d==k else 0
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    if U_arr.shape[1] != V_arr.shape[1] or U_arr.shape[1] != W_arr.shape[1]:
        return False
        
    R = U_arr.shape[1]
    
    # Pre-transpose for dot products
    U_T = U_arr.T # (R, n*m)
    V_T = V_arr.T # (R, m*p)
    W_T = W_arr.T # (R, p*n)
    
    # Iterate over all output indices (i, k) and input indices (a, b, c, d)
    # This is O(n*p * n*m*m*p * R) which is fine for small n,m,p
    for i in range(n):
        for k in range(p):
            w_col = W_T[:, k * n + i]
            for a in range(n):
                for b in range(m):
                    u_col = U_T[:, a * m + b]
                    for c in range(m):
                        for d in range(p):
                            v_col = V_T[:, c * p + d]
                            actual = np.dot(w_col, u_col * v_col)
                            
                            expected = 1.0 if (a == i and b == c and d == k) else 0.0
                            
                            if abs(actual - expected) > tol:
                                return False
    return True

def _solve_for_U_W_given_V(V, n, m, p, R):
    """
    Given a fixed V matrix (m*p x R), solve for U (n*m x R) and W (p*n x R)
    such that the tensor decomposition holds.
    
    The equations are linear in U and W.
    Let's fix U and solve for W? Or fix W and solve for U?
    Actually, the equations are bilinear in U and W.
    sum_r U_{ab,r} V_{cd,r} W_{ki,r} = delta_{ai} delta_{bc} delta_{dk}
    
    This is equivalent to: For each r, the term is a rank-1 contribution.
    
    We can rephrase this as a system of linear equations if we treat the product U*V as one variable? No.
    
    Let's try a different approach:
    For a fixed V, the condition is:
    For all (i, k, a, b, c, d):
    sum_r W_{ki,r} * (U_{ab,r} * V_{cd,r}) = Target_{i,k,a,b,c,d}
    
    This is linear in W for fixed U, and linear in U for fixed W.
    It is NOT linear in (U, W) simultaneously.
    
    However, we can use a "Guess and Check" strategy with structured U and W.
    Or, we can use the fact that if we pick U and V, W is determined by:
    W_{ki,r} = ? No, W is a sum.
    
    Let's try a simpler structural approach:
    We know that the tensor T is the sum of rank-1 tensors.
    T = sum_r u_r \otimes v_r \otimes w_r
    
    If we fix V to be the identity matrix (padded with zeros), does it help?
    If V is identity-like, V_{cd,r} is non-zero only if c == d and r corresponds to that (c,d).
    This forces a specific structure.
    """
    pass # This function is a placeholder for a more complex solver if needed.
    # For this specific task, we will use a different strategy.

def _make_schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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

def _try_strassen_like(n, m, p, R):
    """
    Attempt to find a decomposition of rank R by exploiting symmetries.
    For 2x5x5, the schoolbook rank is 50.
    Known lower bound for M(2,5,5) is not trivially less than 50 in standard literature,
    but we can try to construct one.
    
    Strategy: 
    The tensor is T[i,k,a,b,c,d] = delta_{ia} delta_{bc} delta_{dk}.
    
    We can group the terms.
    For each b (0..4), we have a 2x5x5 sub-problem? No.
    
    Let's try to share terms between i=0 and i=1.
    For a fixed b and k, we need:
    Term 1: i=0, a=0, c=b, d=k -> U[0,b] V[b,k] W[k,0] = 1
    Term 2: i=1, a=1, c=b, d=k -> U[1,b] V[b,k] W[k,1] = 1
    
    If we set U[0,b] = U[1,b] = 1, V[b,k] = 1, then:
    W[k,0] = 1 and W[k,1] = 1.
    This uses 1 rank term for (b,k) covering both i=0 and i=1?
    Let's check the cross terms.
    If we use a rank term r with U[0,b]=1, U[1,b]=1, V[b,k]=1, W[k,0]=1, W[k,1]=1.
    
    Contribution to T[i,k',a,b',c',d']:
    Sum over r.
    If r corresponds to (b,k):
    U[a,b'] is non-zero only if b'=b and a in {0,1}.
    V[c',d'] is non-zero only if c'=b and d'=k.
    W[k',i'] is non-zero only if k'=k and i' in {0,1}.
    
    So the term contributes:
    U[a,b] * V[b,d'] * W[k',i']
    Non-zero if b'=b, c'=b, d'=k, k'=k.
    Value is U[a,b] * 1 * W[k,i'].
    
    We want T[i,k,a,b,c,d] = 1 if a=i, b=c, d=k.
    
    If we sum over all b in 0..4 and k in 0..4 using this "paired" structure:
    For a specific (b, k), let r_{b,k} be the index.
    U[0*m+b, r] = 1
    U[1*m+b, r] = 1
    V[b*p+k, r] = 1
    W[k*n+0, r] = 1
    W[k*n+1, r] = 1
    
    Let's calculate the sum for a specific target (i, k, a, b, c, d).
    Sum_{r} U[a*m+b_r, r] V[c*p+d_r, r] W[k*n+i, r]
    Here r indexes the pair (b', k').
    U is non-zero if b_r = b and a in {0,1}.
    V is non-zero if c = b_r and d = k_r.
    W is non-zero if k = k_r and i in {0,1}.
    
    So the term is non-zero only if:
    1. b_r = b (from U)
    2. c = b_r = b (from V) => c=b
    3. d = k_r (from V)
    4. k = k_r (from W) => d=k
    5. a in {0,1} (always true for n=2)
    6. i in {0,1} (always true for n=2)
    
    If these conditions are met, the value is:
    U[a,b] * V[b,k] * W[k,i] = 1 * 1 * 1 = 1.
    
    So, for any (i, k, a, b, c, d) such that c=b and d=k, the sum is 1.
    Wait, the condition for the tensor is a=i AND b=c AND d=k.
    
    My constructed sum gives 1 if b=c AND d=k. It does NOT check a=i.
    It gives 1 for BOTH a=0 and a=1 (since U[0,b]=1 and U[1,b]=1).
    So if i=0, a=1, b=c, d=k, the sum is 1, but the target is 0.
    So this simple pairing FAILS.
    
    We need to distinguish a=i.
    We can use coefficients.
    Let U[0,b] = 1, U[1,b] = 0? No, then we don't share.
    
    Let's try:
    U[0,b] = 1, U[1,b] = -1?
    V[b,k] = 1
    W[k,0] = 1, W[k,1] = -1?
    
    Term for (b,k):
    U[a,b] V[b,d'] W[k',i']
    If a=0: 1 * 1 * W[k,i']
    If a=1: -1 * 1 * W[k,i']
    
    We want:
    If i=0, a=0: 1. If i=0, a=1: 0.
    If i=1, a=0: 0. If i=1, a=1: 1.
    
    Let W[k,0] = 1, W[k,1] = 0? No, W is shared.
    
    Let's use two rank terms per (b,k) pair to separate i?
    That goes back to schoolbook.
    
    Is there a way to use 1 term for 2 (i) values?
    We need a linear combination of U and W that yields delta_{ai}.
    This is possible if we allow fractional coefficients and use more than 2 rows?
    But U has fixed rows.
    
    Actually, the tensor T is separable in (a,i) and (b,c) and (d,k).
    T = (sum_a,i delta_{ai} e_a e_i) \otimes (sum_b,c delta_{bc} e_b e_c) \otimes (sum_d,k delta_{dk} e_d e_k)
    
    The rank of the tensor product of matrices is the product of their ranks.
    Rank(T) = Rank(Delta_n) * Rank(Delta_m) * Rank(Delta_p)
    Rank of Identity matrix I_m is m.
    So Rank(T) = n * m * p.
    
    This suggests the rank is EXACTLY n*m*p for these dimensions if we restrict to integer/half-integer?
    No, tensor rank is not multiplicative in general.
    However, for the "standard" matrix multiplication tensor, the rank is known to be close to n*m*p.
    For 2x5x5, is it possible to go below 50?
    
    According to literature, the rank of M(2,5,5) is not definitively known to be less than 50 in simple constructions, but Strassen-like methods usually apply to square matrices.
    
    Given the time limit and the requirement for exact integer/half-integer entries, finding a rank < 50 decomposition is extremely difficult and likely non-existent for small n,m,p without very specific algebraic identities.
    
    However, the prompt asks to "Find a rank-minimal exact tensor decomposition".
    If the minimum is 50, the score is 1.0.
    
    Let's try one more structural idea:
    Can we merge terms for different b?
    If we set V to be a random matrix with 1/2 entries, can we solve for U and W?
    
    Let's implement a solver that:
    1. Picks a random sparse V with entries in {0, 0.5, 1}.
    2. Sets up the linear system for U and W?
       The equations are:
       sum_r U_{ab,r} V_{cd,r} W_{ki,r} = Target
       
       This is quadratic.
       
    Alternative:
    Fix U and V. Solve for W.
    For each (k, i), the vector W_{k,i} (of length R) must satisfy:
    sum_r W_{k,i,r} * (U_{ab,r} * V_{cd,r}) = Target_{k,i,a,b,c,d} for all a,b,c,d.
    
    This is a linear system for W_{k,i} given U and V.
    Let M_{k,i} be a matrix where rows are indexed by (a,b,c,d) and columns by r.
    M_{k,i}[(a,b,c,d), r] = U_{ab,r} * V_{cd,r}.
    Then M_{k,i} * W_{k,i} = Target_{k,i}.
    
    If M_{k,i} is invertible (or has a solution), we can find W_{k,i}.
    We need W_{k,i} to have integer/half-integer entries.
    
    Algorithm:
    1. Initialize best = schoolbook (rank 50).
    2. For R in range(25, 50):
       For trial in range(100):
          Generate random U, V with entries in {0, 0.5, 1} (sparse).
          For each (k, i):
             Construct M_{k,i}.
             Solve M_{k,i} w = t using least squares or exact rational solve.
             Check if solution w exists and has integer/half-integer entries.
          If all (k,i) yield valid W, check full decomposition.
    
    This is a solid structural change: "Random U,V + Linear Solve for W".
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    def elapsed():
        return time.time() - start_time
        
    def still_going():
        return elapsed() < deadline

    best = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Pre-compute target vectors for all (k, i)
    # Target[k, i] is a vector of length (n*m*m*p) corresponding to (a,b,c,d)
    # Index mapping: idx = ((a*m + b)*m + c)*p + d
    targets = {}
    for k in range(p):
        for i in range(n):
            t = np.zeros(n*m*m*p)
            for a in range(n):
                for b in range(m):
                    for c in range(m):
                        for d in range(p):
                            if a == i and b == c and d == k:
                                idx = ((a*m + b)*m + c)*p + d
                                t[idx] = 1.0
            targets[(k, i)] = t
    
    # Try to find lower rank
    for R in range(25, best_rank):
        if not still_going():
            break
            
        for trial in range(50):
            if not still_going():
                break
                
            # Generate random sparse U and V
            # Using Fraction for exactness in solve?
            # Let's stick to floats for speed, then verify.
            
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            
            # Fill U and V with random sparse patterns
            # Each column r of U and V should be non-zero in a few places.
            for r in range(R):
                # U column
                num_nnz_u = 2
                indices_u = np.random.choice(n*m, num_nnz_u, replace=False)
                for idx in indices_u:
                    U[idx, r] = 1.0 # Keep integers for simplicity first
                
                # V column
                num_nnz_v = 2
                indices_v = np.random.choice(m*p, num_nnz_v, replace=False)
                for idx in indices_v:
                    V[idx, r] = 1.0
            
            # Try to solve for W
            W_valid = True
            W = np.zeros((p*n, R))
            
            for k in range(p):
                for i in range(n):
                    # Build M_{k,i}
                    # M is (n*m*m*p, R)
                    M = np.zeros((n*m*m*p, R))
                    for a in range(n):
                        for b in range(m):
                            u_ab = U[a*m+b] # (R,)
                            for c in range(m):
                                for d in range(p):
                                    v_cd = V[c*p+d] # (R,)
                                    idx = ((a*m + b)*m + c)*p + d
                                    M[idx] = u_ab * v_cd
                                    
                    # Solve M w = t
                    # Use np.linalg.lstsq
                    try:
                        w_sol, residuals, rank_M, s = np.linalg.lstsq(M, targets[(k, i)], rcond=None)
                    except:
                        W_valid = False
                        break
                        
                    # Check if solution is exact (residuals ~ 0)
                    if residuals.size > 0 and residuals[0] > 1e-9:
                        W_valid = False
                        break
                        
                    # Check if w_sol has integer/half-integer values
                    # Round to nearest 0.5 and check
                    w_rounded = np.round(w_sol * 2) / 2.0
                    if np.allclose(w_sol, w_rounded, atol=1e-9):
                        W[k*n+i] = w_rounded
                    else:
                        # Allow small deviation? No, must be exact.
                        W_valid = False
                        break
                        
            if W_valid:
                # Final verification
                if _verify_exact(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = R
                    break # Found best for this R, go lower
            
    return best

def construct():
    return _try_strassen_like(2, 5, 5, 50)

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
