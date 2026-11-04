import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    # T[a,b,c] = 1 if (a%m)==(b//p) and (b%p)==(c//n) and (a//m)==(c%n), else 0
    # a in [0, n*m), b in [0, m*p), c in [0, p*n)
    
    nm = n * m  # 16
    mp = m * p  # 20
    pn = p * n  # 20
    
    # Build the target tensor as a list of nonzero entries
    # T[a,b,c] = 1 iff a%m == b//p and b%p == c//n and a//m == c%n
    nonzero = []
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if (a % m) == (b // p) and (b % p) == (c // n) and (a // m) == (c % n):
                    nonzero.append((a, b, c))
    
    R_target = len(nonzero)  # 80 nonzero entries
    
    def verify(U, V, W, R):
        """Check if U,V,W with rank R exactly decompose T."""
        # U: (nm, R), V: (mp, R), W: (pn, R)
        for r in range(R):
            for a in range(nm):
                ua = U[a, r]
                if ua == 0:
                    continue
                for b in range(mp):
                    uv = ua * V[b, r]
                    if uv == 0:
                        continue
                    for c in range(pn):
                        val = uv * W[c, r]
                        if val == 0:
                            continue
                        # Check if this (a,b,c) should be nonzero
                        should_be_1 = (a % m == b // p and b % p == c // n and a // m == c % n)
                        if should_be_1:
                            if val != 1:
                                return False
                        else:
                            if val != 0:
                                return False
        # Also need to verify that all nonzero entries are actually 1 (sum over r)
        # This is expensive; do a full check
        # Build sum tensor
        S = np.zeros((nm, mp, pn), dtype=np.float64)
        for r in range(R):
            S += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
        # Check
        for (a, b, c) in nonzero:
            if abs(S[a, b, c] - 1) > 1e-9:
                return False
        # Check zeros
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if not (a % m == b // p and b % p == c // n and a // m == c % n):
                        if abs(S[a, b, c]) > 1e-9:
                            return False
        return True
    
    def verify_fast(U, V, W, R):
        """Faster verification using numpy outer products."""
        S = np.zeros((nm, mp, pn), dtype=np.float64)
        for r in range(R):
            S += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        for (a, b, c) in nonzero:
            if abs(S[a, b, c] - 1) > 1e-9:
                return False
        # Spot-check some zeros
        zero_checks = 0
        for a in range(0, nm, 3):
            for b in range(0, mp, 3):
                for c in range(0, pn, 3):
                    if not (a % m == b // p and b % p == c // n and a // m == c % n):
                        if abs(S[a, b, c]) > 1e-9:
                            return False
                        zero_checks += 1
        if zero_checks < 10:
            # Do more checks
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        if not (a % m == b // p and b % p == c // n and a // m == c % n):
                            if abs(S[a, b, c]) > 1e-9:
                                return False
        return True
    
    # Try ranks from low to high
    candidate_ranks = [20, 25, 30, 35, 40, 50, 60, 80]
    
    def try_rank(R, max_trials=50):
        if time.time() > deadline:
            return None
        
        for trial in range(max_trials):
            if time.time() > deadline:
                return None
            
            # Random initialization with structure: 
            # Each term should contribute to some entries
            U = np.random.randn(nm, R) * 0.5
            V = np.random.randn(mp, R) * 0.5
            W = np.random.randn(pn, R) * 0.5
            
            # Gradient descent to minimize sum of squared errors
            # Error = sum over all (a,b,c) of (S[a,b,c] - T[a,b,c])^2
            lr = 0.01
            for step in range(200):
                if time.time() > deadline:
                    return None
                
                S = np.zeros((nm, mp, pn), dtype=np.float64)
                for r in range(R):
                    S += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                
                # Target
                T = np.zeros((nm, mp, pn), dtype=np.float64)
                for (a, b, c) in nonzero:
                    T[a, b, c] = 1.0
                
                err = S - T
                loss = np.sum(err**2)
                
                if loss < 1e-10:
                    break
                
                # Gradients
                # dS/dr_u[i,r] = sum_{b,c} err[i,b,c] * V[b,r] * W[c,r]
                # dS/dr_v[j,r] = sum_{a,c} err[a,j,c] * U[a,r] * W[c,r]
                # dS/dr_w[k,r] = sum_{a,b} err[a,b,k] * U[a,r] * V[b,r]
                
                for r in range(R):
                    # dU
                    grad_U = np.zeros(nm)
                    for a in range(nm):
                        grad_U[a] = np.sum(err[a, :, :] * V[:, r][None, :] * W[:, r][None, :])
                    U[:, r] -= lr * grad_U
                    
                    grad_V = np.zeros(mp)
                    for b in range(mp):
                        grad_V[b] = np.sum(err[:, b, :] * U[:, r][:, None] * W[:, r][None, :])
                    V[:, r] -= lr * grad_V
                    
                    grad_W = np.zeros(pn)
                    for c in range(pn):
                        grad_W[c] = np.sum(err[:, :, c] * U[:, r][:, None] * V[:, r][:, None])
                    W[:, r] -= lr * grad_W
            
            # Round to nearest half-integer
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            if verify_fast(U_r, V_r, W_r, R):
                # Convert to list of lists
                U_list = U_r.tolist()
                V_list = V_r.tolist()
                W_list = W_r.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Schoolbook as fallback
    def schoolbook():
        R = n * m * p
        U = [[0] * R for _ in range(nm)]
        V = [[0] * R for _ in range(mp)]
        W = [[0] * R for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try each rank
    for R in candidate_ranks:
        if time.time() > deadline:
            break
        result = try_rank(R, max_trials=30)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return schoolbook()
