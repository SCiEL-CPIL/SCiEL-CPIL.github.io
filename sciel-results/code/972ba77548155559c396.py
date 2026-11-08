"""Matrix multiplication 2x3 x 3x3 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 3
    # Tensor T has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 if this is the "multiplication" tensor entry
    # Actually the multiplication tensor is: sum over a,b of A[i,a]*B[a,b]*E_{i,b}
    # Tensor form: T_{i,a; a,b; b} -> but we need the standard form
    # 
    # The matrix multiplication tensor for (n,m) x (m,p) -> (n,p):
    # T[i, a, b] where we multiply A[i,a] * B[a,b] to get C[i,b]
    # In the factorization: C[i,b] = sum_c U[i,a,c] * V[a,b,c] * W[b,i,c] ... 
    # 
    # Wait, let me re-read the convention. The task says:
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # W[k*n+i, r] - so W is indexed by (k, i) where k is column of C, i is row
    # 
    # The decomposition: C[i,k] = sum_{a,c} A[i,a] * V[a* p + k, c] * W[k*n + i, c]
    # But U[i*m+a, c] multiplies A[i,a]
    # So: C[i,k] = sum_{a, c} U[i*m+a, c] * A[i,a] * V[a*p+k, c] * W[k*n+i, c]
    # 
    # Actually the standard tensor decomposition for matmul:
    # The tensor T has entries T[i, a, b] = delta (it's the identity in the middle)
    # We want: T[i,a,b] = sum_c U[i,a,c] * V[a,b,c] * W[b,i,c] ... 
    # 
    # Let me think differently. The multiplication map is:
    # (A, B) -> A*B where A is n×m, B is m×p
    # This is a trilinear map. The tensor is T in (R^{n×m})^* ⊗ (R^{m×p})^* ⊗ (R^{n×p})
    # T[A,B] = A*B
    # 
    # In coordinates: (A*B)[i,k] = sum_a A[i,a] * B[a,k]
    # So the tensor T has components: T[i,k | i',a' | a'',k'] = delta_{i,i'} delta_{k,k'} delta_{a,a''} ... 
    # 
    # For the factorization format given:
    # U: (n*m) x rank, indexed by (i,a) -> U[i*m+a, c]
    # V: (m*p) x rank, indexed by (a,k) -> V[a*p+k, c]  
    # W: (p*n) x rank, indexed by (k,i) -> W[k*n+i, c]
    # 
    # The reconstruction: C[i,k] = sum_{a,c} A[i,a] * V[a*p+k, c] * W[k*n+i, c] * ... 
    # Hmm, where does U fit? 
    # 
    # I think the convention is:
    # C[i,k] = sum_{a, c} U[i*m+a, c] * A[i,a] * (something with V and B) * W[k*n+i, c]
    # 
    # Actually, I think it's:
    # C[i,k] = sum_c W[k*n+i, c] * (sum_a U[i*m+a, c] * A[i,a]) * (sum_b V[a*p+b, c] * B[a,b])
    # No, that doesn't work dimensionally.
    # 
    # Let me reconsider. The standard format for matmul tensor decomposition:
    # We have three tensors U, V, W such that:
    # (A*B)[i,k] = sum_{a, c} A[i,a] * U[i,a,c] * V[a,k,c] * W[i,k,c]
    # But U is (n*m, rank) so U[i*m+a, c], V is (m*p, rank) so V[a*p+k, c], W is (p*n, rank) so W[k*n+i, c]
    # 
    # So: (A*B)[i,k] = sum_{a, c} A[i,a] * B[a,k] * ... no.
    # 
    # I think the correct interpretation is:
    # C[i,k] = sum_c W[k*n+i, c] * (sum_a U[i*m+a, c] * A[i,a]) * (sum_a V[a*p+k, c] * B[a,k])
    # But that has two sums over a which is wrong.
    # 
    # Alternative: The tensor for matmul is:
    # T = sum_{i,a,k} E_{i,a} ⊗ E_{a,k} ⊗ E_{i,k} (as a vector in R^{nm} ⊗ R^{mp} ⊗ R^{pn})
    # Wait, that's not right either.
    # 
    # Let me use the standard definition. The matrix multiplication tensor T_{n,m,p} is in
    # (R^{n×m})^* ⊗ (R^{m×p})^* ⊗ R^{n×p}
    # T(A,B) = A*B
    # 
    # In the basis, T = sum_{i,a,k} e^*_{i,a} ⊗ e^*_{a,k} ⊗ e_{i,k}
    # where e^*_{i,a} is the dual basis element picking out A[i,a], etc.
    # 
    # A rank-1 decomposition: T = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c ∈ (R^{n×m})^*, v_c ∈ (R^{m×p})^*, w_c ∈ R^{n×p}
    # 
    # In coordinates: T[i,a; a',k; i',k'] = sum_c u_c[i,a] * v_c[a',k] * w_c[i',k']
    # And T[i,a; a',k; i',k'] = delta_{i,i'} delta_{a,a'} delta_{k,k'} (when a=a')
    # Wait no. T(A,B)[i,k] = sum_a A[i,a] B[a,k]
    # So T = sum_{i,a,k} (e^*_{i,a}) ⊗ (e^*_{a,k}) ⊗ (e_{i,k})
    # where e^*_{i,a} picks A[i,a], e^*_{a,k} picks B[a,k], e_{i,k} is the output basis.
    # 
    # So the tensor components are: T_{(i,a),(a',k),(i',k')} = delta_{a,a'} delta_{i,i'} delta_{k,k'}
    # 
    # For a rank-r decomposition: T = sum_{c=1}^r u_c ⊗ v_c ⊗ w_c
    # u_c is in (R^{nm})^* so u_c[i,a] for i in [0,n), a in [0,m)
    # v_c is in (R^{mp})^* so v_c[a,k] for a in [0,m), k in [0,p)
    # w_c is in R^{pn} so w_c[k,i] for k in [0,p), i in [0,n)
    # 
    # The equation: sum_c u_c[i,a] * v_c[a',k] * w_c[k',i'] = delta_{a,a'} delta_{i,i'} delta_{k,k'}
    # 
    # In the factor matrix format:
    # U[i*m+a, c] = u_c[i,a]
    # V[a*p+k, c] = v_c[a,k]
    # W[k*n+i, c] = w_c[k,i]
    # 
    # So the constraint is: for all i,a,a',k,k',i':
    # sum_c U[i*m+a, c] * V[a'*p+k, c] * W[k'*n+i', c] = delta_{a,a'} * delta_{i,i'} * delta_{k,k'}
    
    # Let's verify with schoolbook:
    # For each (i,a,k), set U[i*m+a, (i,a,k)] = 1, V[a*p+k, (i,a,k)] = 1, W[k*n+i, (i,a,k)] = 1
    # Then sum_c ... = sum over c where c=(i0,a0,k0): U[i*m+a, c]*V[a'*p+k, c]*W[k'*n+i', c]
    # = 1 if (i,a)=(i0,a0) AND (a',k)=(a0,k0) AND (k',i')=(k0,i0)
    # = 1 if i=i0, a=a0, a'=a0, k=k0, k'=k0, i'=i0
    # = delta_{a,a'} delta_{i,i'} delta_{k,k'} * delta_{i,i'} ... 
    # Hmm, we get delta_{a,a'} delta_{i,i'} delta_{k,k'} which is correct!
    
    # So the constraint is:
    # For all (i, a, a', k, k', i'):
    # sum_c U[i*m+a, c] * V[a'*p+k, c] * W[k'*n+i', c] = delta_{a,a'} * delta_{i,i'} * delta_{k,k'}
    
    # This is a system of equations. Let's count: n*m*m*p*p*n = 2*3*3*3*3*2 = 324 equations
    # Unknowns: 3 * r (for U, V, W each having r columns, but they're independent)
    # Actually U has n*m*r = 6r entries, V has m*p*r = 9r, W has p*n*r = 6r, total 21r
    
    # For r=18: 378 unknowns, 324 equations - overdetermined but schoolbook works
    # We want minimal r.
    
    # Strategy: Try ranks from low to high. For each rank r, try to find a solution.
    # This is a nonlinear system (bilinear in U,V,W). 
    # 
    # Approach: Fix U and V, solve for W (linear). Or use iterative methods.
    # 
    # Better approach: Use the fact that we can set up the problem as:
    # For each (a, a', k, k', i, i'), we have a constraint.
    # 
    # Alternative: Use random initialization + gradient descent, then rational reconstruction.
    # 
    # Let me try a different approach: Use known results. For 2x3 x 3x3, the rank is known.
    # The schoolbook is 18. Can we do better?
    # 
    # For small matrix multiplication, known ranks:
    # 2x2 x 2x2: rank 7 (Strassen)
    # 2x3 x 3x2: rank ? 
    # 2x3 x 3x3: rank ?
    # 
    # Let me try to find a decomposition by solving the system numerically and then exactifying.
    
    def try_rank(r, max_iter=5000, tol=1e-10):
        """Try to find a rank-r decomposition using iterative refinement."""
        if time.time() > deadline:
            return None
        
        np.random.seed(42 + r)
        
        # Initialize U, V, W randomly
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # The constraint: for all (i, a, a', k, k', i'):
        # sum_c U[i*m+a, c] * V[a'*p+k, c] * W[k'*n+i', c] = delta_{a,a'} * delta_{i,i'} * delta_{k,k'}
        # 
        # Let's rewrite. Fix (a, a', i, i'). Then for each (k, k'):
        # sum_c U[i*m+a, c] * V[a'*p+k, c] * W[k'*n+i', c] = delta_{a,a'} * delta_{i,i'} * delta_{k,k'}
        # 
        # This is complicated. Let me use a different formulation.
        # 
        # Think of it as: The tensor T flattened. 
        # T has shape (n*m, m*p, p*n) in the factorization basis? No.
        # 
        # Actually, let me think of it as a matrix equation.
        # 
        # For fixed c, the rank-1 contribution to T is:
        # T_c[i, a, a', k, k', i'] = U[i*m+a, c] * V[a'*p+k, c] * W[k'*n+i', c]
        # 
        # We need sum_c T_c = T_target
        # 
        # T_target[i, a, a', k, k', i'] = delta_{a,a'} * delta_{i,i'} * delta_{k,k'}
        
        # Let me try a simpler approach: Use the structure to reduce the problem.
        # 
        # For the equation to hold, consider specific cases:
        # Case 1: a = a', i = i', k = k'. Then LHS must be 1.
        # Case 2: Any other combination, LHS must be 0.
        
        # This is a very constrained system. Let me try to solve it using 
        # alternating least squares or similar.
        
        # Actually, let me try a different approach: enumerate small structures.
        # 
        # For 2x3 x 3x3, let me try to find a rank 15, 16, 17 decomposition.
        # 
        # I'll use a numerical approach: 
        # 1. Start with schoolbook rank 18
        # 2. Try to merge terms or find a lower rank via SVD-like techniques
        
        # Alternative: Use the fact that the tensor can be viewed as a matrix and use SVD.
        # 
        # Let me reshape the tensor. T is in (R^{nm})^* ⊗ (R^{mp})^* ⊗ R^{pn}
        # Reshape to a matrix: M of size (nm*mp) x (pn) where M[(i,a),(a',k); (k',i')] 
        # But that's not quite right.
        # 
        # Let me define: For each pair ((i,a), (a',k)), the "output" is a vector in R^{pn}
        # T[(i,a),(a',k)] = sum_{k',i'} T[i,a,a',k,k',i'] * e_{k',i'}
        # = delta_{a,a'} * delta_{i,i'} * ... 
        # 
        # This is getting complex. Let me just try random search with gradient descent.
        
        def compute_error(U, V, W):
            err = 0.0
            for i in range(n):
                for a in range(m):
                    for ap in range(m):
                        for k in range(p):
                            for kp in range(p):
                                for ip in range(n):
                                    lhs = 0.0
                                    for c in range(r):
                                        lhs += U[i*m+a, c] * V[ap*p+k, c] * W[kp*n+ip, c]
                                    rhs = 1.0 if (a == ap and i == ip and k == kp) else 0.0
                                    err += (lhs - rhs) ** 2
            return err
        
        # This is O(n*m*m*p*p*n*r) = O(324*r) per evaluation, which is fine.
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_err = compute_error(U, V, W)
        
        # Use Nelder-Mead or simple gradient-free optimization
        # Actually, let's use a smarter approach: alternating minimization
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Update W: For each (k', i'), solve linear system
            # sum_c U[i*m+a, c] * V[a'*p+k, c] * W[k'*n+i', c] = delta_{a,a'} delta_{i,i'} delta_{k,k'}
            # For fixed (i, a, a', k), this is linear in W[k'*n+i', c] for all c
            # But it's coupled across (i,a,a',k).
            
            # Let's do: Fix U and V, solve for W.
            # The system is: For all (i, a, a', k, k', i'):
            # sum_c (U[i*m+a, c] * V[a'*p+k, c]) * W[k'*n+i', c] = delta_{a,a'} delta_{i,i'} delta_{k,k'}
            # 
            # Let M_{(i,a,a',k), (k',i'), c} = U[i*m+a, c] * V[a'*p+k, c]
            # Then sum_c M[...] * W[k'*n+i', c] = rhs
            # 
            # This is a linear system in the W variables. There are p*n*r = 6r unknowns.
            # And n*m*m*p = 2*3*3*3 = 54 equations per (k',i') ... no.
            # 
            # Actually, for each (k', i'), we have equations from all (i, a, a', k):
            # sum_c U[i*m+a, c] * V[a'*p+k, c] * W[k'*n+i', c] = delta_{a,a'} delta_{i,i'} delta_{k,k'}
            # 
            # For fixed (k', i'), this gives n*m*m*p = 54 equations in r unknowns W[k'*n+i', 0..r-1]
            # But the RHS depends on (i, a, a', k, k', i') and we need i=i' and k=k' for nonzero.
            # 
            # So for fixed (k', i'), the equations are:
            # For (i, a, a', k) with i=i' and k=k': sum_c U[i'*m+a, c] * V[a'*p+k', c] * W[k'*n+i', c] = delta_{a,a'}
            # For (i, a, a', k) with i!=i' or k!=k': sum_c U[i*m+a, c] * V[a'*p+k, c] * W[k'*n+i', c] = 0
            # 
            # This is a system of 54 equations in r unknowns. If r < 54, it may be overdetermined.
            # We can solve in least squares sense.
            
            # Let me set up the linear system for W.
            # Unknowns: W_flat of size p*n*r = 6r
            # Equations: n*m*m*p*p*n = 324
            
            # Build A matrix: 324 x 6r
            # A[eq_idx, col_idx] where eq_idx corresponds to (i,a,a',k,k',i') and col_idx to (kp,ip,c)
            
            # This is getting heavy. Let me try a simpler heuristic.
            
            # Use random restarts and local search
            pass
        
        # Let me try a completely different approach: use known small-rank decompositions
        # or construct one systematically.
        
        # For now, let's just return the schoolbook if we can't find better quickly.
        return None
    
    # Let me try a more direct approach: search for low-rank decompositions
    # using the structure of the problem.
    
    # The key insight: The tensor T[i,a,a',k,k',i'] = delta_{a,a'} delta_{i,i'} delta_{k,k'}
    # 
    # This can be written as: T = sum_{a} (sum_i e_{i,a} ⊗ e_{i,a}^*) ⊗ ... 
    # 
    # Let me think of it as: T = sum_{i,a,k} E_{i,a} ⊗ E_{a,k} ⊗ E_{i,k}
    # where E_{i,a} is the matrix unit in R^{n×m}, etc.
    # 
    # For a rank-1 decomposition term: u ⊗ v ⊗ w
    # u ∈ (R^{n×m})^*, v ∈ (R^{m×p})^*, w ∈ R^{n×p}
    # 
    # The sum of such terms must equal T.
    
    # Let me try to find a decomposition by solving a linear system for each rank.
    # 
    # Approach: 
    # 1. For a given rank r, assume a structure for U, V, W.
    # 2. Solve the resulting equations.
    
    # Actually, let me just try to find any decomposition with rank < 18 using 
    # numerical optimization and rational reconstruction.
    
    def find_decomposition(r, seed=0):
        """Find a rank-r decomposition using numerical methods."""
        np.random.seed(seed)
        
        # Initialize
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        
        # Use alternating minimization
        for it in range(2000):
            if time.time() > deadline:
                return None
            
            # Update W given U, V
            # For each (kp, ip), solve: 
            # sum_c U[i*m+a, c] * V[ap*p+k, c] * W[kp*n+ip, c] = delta_{a,ap} delta_{i,ip} delta_{k,kp}
            # for all (i, a, ap, k)
            
            # This is a linear system. Let's build it.
            # Unknowns: W[kp*n+ip, :] for all kp, ip -> total p*n*r unknowns
            # Equations: n*m*m*p equations (for each i, a, ap, k, and for each kp, ip... no)
            
            # Actually, the full system has n*m*m*p*p*n equations.
            # Let me just use least squares on the full system.
            
            # Build the linear system: A * W_flat = b
            # A has shape (n*m*m*p*p*n, p*n*r)
            # b has shape (n*m*m*p*p*n,)
            
            num_eq = n * m * m * p * p * n
            A = np.zeros((num_eq, p * n * r))
            b = np.zeros(num_eq)
            
            eq_idx = 0
            for i in range(n):
                for a in range(m):
                    for ap in range(m):
                        for k in range(p):
                            for kp in range(p):
                                for ip in range(n):
                                    rhs = 1.0 if (a == ap and i == ip and k == kp) else 0.0
                                    b[eq_idx] = rhs
                                    for c in range(r):
                                        A[eq_idx, (kp * n + ip) * r + c] = U[i * m + a, c] * V[ap * p + k, c]
                                    eq_idx += 1
            
            # Solve least squares
            W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            W = W_flat.reshape(p * n, r)
            
            # Update U given V, W
            num_eq = n * m * m * p * p * n
            A = np.zeros((num_eq, n * m * r))
            b = np.zeros(num_eq)
            eq_idx = 0
            for i in range(n):
                for a in range(m):
                    for ap in range(m):
                        for k in range(p):
                            for kp in range(p):
                                for ip in range(n):
                                    rhs = 1.0 if (a == ap and i == ip and k == kp) else 0.0
                                    b[eq_idx] = rhs
                                    for c in range(r):
                                        A[eq_idx, (i * m + a) * r + c] = V[ap * p + k, c] * W[kp * n + ip, c]
                                    eq_idx += 1
            
            U_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            U = U_flat.reshape(n * m, r)
            
            # Update V given U, W
            num_eq = n * m * m * p * p * n
            A = np.zeros((num_eq, m * p * r))
            b = np.zeros(num_eq)
            eq_idx = 0
            for i in range(n):
                for a in range(m):
                    for ap in range(m):
                        for k in range(p):
                            for kp in range(p):
                                for ip in range(n):
                                    rhs = 1.0 if (a == ap and i == ip and k == kp) else 0.0
                                    b[eq_idx] = rhs
                                    for c in range(r):
                                        A[eq_idx, (ap * p + k) * r + c] = U[i * m + a, c] * W[kp * n + ip, c]
                                    eq_idx += 1
            
            V_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            V = V_flat.reshape(m * p, r)
        
        # Check if we found a valid decomposition
        max_err = 0.0
        for i in range(n):
            for a in range(m):
                for ap in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                lhs = 0.0
                                for c in range(r):
                                    lhs += U[i*m+a, c] * V[ap*p+k, c] * W[kp*n+ip, c]
                                rhs = 1.0 if (a == ap and i == ip and k == kp) else 0.0
                                max_err = max(max_err, abs(lhs - rhs))
        
        if max_err < 1e-8:
            # Rational reconstruction
            U_frac = np.array([[Fraction(U[i, c]).limit_denominator(1000) for c in range(r)] for i in range(n*m)])
            V_frac = np.array([[Fraction(V[i, c]).limit_denominator(1000) for c in range(r)] for i in range(m*p)])
            W_frac = np.array([[Fraction(W[i, c]).limit_denominator(1000) for c in range(r)] for i in range(p*n)])
            
            # Verify with exact arithmetic
            max_err_exact = 0.0
            for i in range(n):
                for a in range(m):
                    for ap in range(m):
                        for k in range(p):
                            for kp in range(p):
                                for ip in range(n):
                                    lhs = Fraction(0)
                                    for c in range(r):
                                        lhs += U_frac[i*m+a][c] * V_frac[ap*p+k][c] * W_frac[kp*n+ip][c]
                                    rhs = Fraction(1) if (a == ap and i == ip and k == kp) else Fraction(0)
                                    max_err_exact = max(max_err_exact, abs(lhs - rhs))
            
            if max_err_exact == 0:
                return {
                    "U": [[float(x) for x in row] for row in U_frac],
                    "V": [[float(x) for x in row] for row in V_frac],
                    "W": [[float(x) for x in row] for row in W_frac]
                }
        
        return None
    
    # Try ranks from low to high
    for r in range(10, 18):
        if time.time() > deadline:
            break
        
        result = find_decomposition(r, seed=42)
        if result is not None:
            return result
        
        # Try a few more seeds
        for seed in [0, 1, 2, 3, 4]:
            if time.time() > deadline:
                break
            result = find_decomposition(r, seed=seed)
            if result is not None:
                return result
    
    # Fall back to schoolbook rank 18
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
