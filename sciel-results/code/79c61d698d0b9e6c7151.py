"""Program for small nonzero signed harmonic subsums using LLL lattice reduction."""
# EVOLVE-BLOCK-START
import numpy as np
from math import gcd

def construct():
    """
    Find 30 coefficients in {-1, 0, 1} minimizing |sum c_i / i| (nonzero).
    
    Approach: Formulate as a shortest vector problem (SVP) in a lattice.
    Construct a lattice where short vectors correspond to good coefficient assignments.
    Use LLL reduction to find short vectors, then extract and verify candidates.
    """
    n = 30
    
    # Compute L = lcm(1, 2, ..., 30)
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
    
    # w_i = L // i for i=1..30
    w = np.array([L // i for i in range(1, n + 1)], dtype=np.float64)
    
    # Construct lattice basis for SVP formulation
    # We want to find c in {-1,0,1}^n minimizing |sum c_i * w_i|
    # 
    # Lattice construction:
    # Basis vectors: v_i = (M * e_i, -w_i) for i=1..n, plus additional vectors
    # to handle the constraint that coefficients should be small.
    #
    # Actually, let's use a cleaner formulation:
    # Create a lattice in R^{n+1} with basis:
    # b_i = (M * e_i, -w_i) for i = 0, ..., n-1
    # where e_i is the i-th standard basis vector in R^n
    #
    # A vector sum c_i * b_i = (M*c_0, M*c_1, ..., M*c_{n-1}, -sum c_i * w_i)
    #
    # If M is large, the norm is dominated by M^2 * sum c_i^2, so LLL will
    # prefer vectors with small coefficients. Among those with the same number
    # of nonzeros, it will minimize |sum c_i * w_i|.
    
    M = float(L)  # Large scaling factor
    
    dim = n + 1
    basis = np.zeros((n, dim), dtype=np.float64)
    for i in range(n):
        basis[i, i] = M
        basis[i, n] = -w[i]
    
    # Implement proper LLL algorithm
    def lll_reduced(B, delta=0.99):
        """
        LLL lattice basis reduction.
        B: m x n matrix of basis vectors (rows)
        Returns reduced basis.
        """
        m, dim = B.shape
        B = B.astype(np.float64).copy()
        
        # Initialize mu (Gram-Schmidt coefficients)
        mu = np.zeros((m, m), dtype=np.float64)
        
        # Initialize lambda (squared norms of GS vectors)
        lam = np.zeros(m, dtype=np.float64)
        
        # Compute initial Gram-Schmidt
        for i in range(m):
            B_i = B[i].copy()
            for j in range(i):
                if lam[j] > 1e-15:
                    mu[i, j] = np.dot(B_i, B[j]) / lam[j]
                    B_i -= mu[i, j] * B[j]
            lam[i] = np.dot(B_i, B_i)
        
        # LLL main loop
        for k in range(1, m):
            # Size reduce B[k] against B[0..k-1]
            for j in range(k - 1, -1, -1):
                if lam[j] < 1e-15:
                    continue
                while abs(mu[k, j]) >= 0.5:
                    q = int(np.round(mu[k, j]))
                    B[k] -= q * B[j]
                    # Update mu for rows above k
                    for i in range(k - 1, -1, -1):
                        mu[i, j] -= q * mu[i, k] if i < k else 0
                    # Recompute mu[k, j]
                    if lam[j] > 1e-15:
                        mu[k, j] = np.dot(B[k], B[j]) / lam[j]
            
            # Check Lovasz condition
            for j in range(k - 1, -1, -1):
                if lam[j] < 1e-15:
                    continue
                # Lovasz condition: (1-delta)*lam[j] <= lam[k-1] + mu[k-1,j]^2*lam[j]
                # Simplified: check if lam[k] < delta * lam[j]
                if lam[k] < delta * lam[j]:
                    # Swap B[k] and B[j]
                    B[k], B[j] = B[j].copy(), B[k].copy()
                    mu[k, :], mu[j, :] = mu[j, :].copy(), mu[k, :].copy()
                    lam[k], lam[j] = lam[j], lam[k]
                    # Update mu for all pairs involving k and j
                    for i in range(m):
                        if i != k and i != j:
                            mu[i, k], mu[i, j] = mu[i, j], mu[i, k]
                    # Re-size-reduce B[k]
                    for jj in range(k - 1, -1, -1):
                        if lam[jj] < 1e-15:
                            continue
                        while abs(mu[k, jj]) >= 0.5:
                            q = int(np.round(mu[k, jj]))
                            B[k] -= q * B[jj]
                            if lam[jj] > 1e-15:
                                mu[k, jj] = np.dot(B[k], B[jj]) / lam[jj]
                    break
            # Recompute GS for row k
            B_k = B[k].copy()
            for j in range(k):
                if lam[j] > 1e-15:
                    mu[k, j] = np.dot(B_k, B[j]) / lam[j]
                    B_k -= mu[k, j] * B[j]
            lam[k] = np.dot(B_k, B_k)
        
        return B
    
    # Run LLL
    reduced_basis = lll_reduced(basis)
    
    # Extract candidates from the reduced basis
    # For each vector in the reduced basis, extract coefficients from first n coordinates
    best_val = float('inf')
    best_coeffs = None
    best_sum = 0.0
    
    for vec in reduced_basis:
        # First n coordinates should be M * c_i
        c_float = vec[:n] / M
        # Round to nearest integer
        c = np.round(c_float).astype(int)
        
        # Check if coefficients are in {-1, 0, 1}
        if np.all(np.abs(c) <= 1):
            # Compute the actual sum
            s = float(np.dot(c, w))
            if abs(s) > 1e-10:  # Nonzero
                abs_s = abs(s)
                if abs_s < best_val:
                    best_val = abs_s
                    best_coeffs = c.tolist()
                    best_sum = s
    
    # Also try all-pairs combinations of basis vectors
    for i in range(len(reduced_basis)):
        for j in range(i + 1, len(reduced_basis)):
            vec = reduced_basis[i] + reduced_basis[j]
            c_float = vec[:n] / M
            c = np.round(c_float).astype(int)
            if np.all(np.abs(c) <= 1):
                s = float(np.dot(c, w))
                if abs(s) > 1e-10:
                    abs_s = abs(s)
                    if abs_s < best_val:
                        best_val = abs_s
                        best_coeffs = c.tolist()
                        best_sum = s
    
    # If LLL didn't find a good candidate, use local search as fallback
    if best_coeffs is None:
        # Start with a simple candidate: c_29 = 1, c_30 = -1
        best_coeffs = [0] * 28 + [1, -1]
        best_sum = 1/29 - 1/30
        best_val = abs(best_sum)
        
        # Local search improvement
        import random
        for restart in range(500):
            if restart == 0:
                coeffs = best_coeffs[:]
            else:
                coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
            
            current_sum = float(sum(c * w[i] for i, c in enumerate(coeffs)) / L)
            abs_current = abs(current_sum)
            
            improved = True
            while improved:
                improved = False
                for i in range(n):
                    for new_val in [-1, 0, 1]:
                        if new_val == coeffs[i]:
                            continue
                        old_val = coeffs[i]
                        coeffs[i] = new_val
                        new_sum = current_sum - old_val * w[i] / L + new_val * w[i] / L
                        abs_new = abs(new_sum)
                        
                        if new_sum != 0 and abs_new < abs_current:
                            current_sum = new_sum
                            abs_current = abs_new
                            improved = True
                            break
                        else:
                            coeffs[i] = old_val
                    if improved:
                        break
            
            if current_sum != 0:
                abs_final = abs(current_sum)
                if abs_final < best_val:
                    best_val = abs_final
                    best_coeffs = coeffs[:]
                    best_sum = current_sum
    
    if best_coeffs is None:
        return [0] * 28 + [1, -1]
    
    return best_coeffs

# EVOLVE-BLOCK-END
