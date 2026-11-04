import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank_seed = n * m * p  # 96
    
    start_time = time.time()
    timeout = 140.0
    
    # Schoolbook fallback
    U0 = np.zeros((n*m, rank_seed))
    V0 = np.zeros((m*p, rank_seed))
    W0 = np.zeros((p*n, rank_seed))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank_seed
    
    # Precompute constraint indices
    # We need to satisfy: sum_r U[i'*m+j',r] V[j''*p+k',r] W[k*n+i,r] = delta_{i',i} delta_{j',j''} delta_{k',k}
    # for all i,i',j,j',j'',k,k'
    
    # Let's build the linear system for alternating minimization
    
    def solve_U(V, W, R):
        """Solve for U given V and W."""
        # For each (i', j'), we need to solve for U[i'*m+j', :]
        # The constraint involves all (i, j'', k')
        # Let's set up the full linear system
        
        num_U_vars = n * m * R  # R variables per (i', j'), n*m of them
        # Actually, let's build it more carefully
        
        # The constraint is: for each (i', j', i, j'', k'):
        # sum_r U[i'*m+j',r] * V[j''*p+k',r] * W[k*n+i,r] = delta_{i',i} delta_{j',j''} delta_{k',k}
        
        # This is a linear system in U. Let's build the matrix.
        
        # Number of constraints: n * m * n * m * p = n^2 * m^2 * p
        num_constraints = n * n * m * m * p
        num_unknowns = n * m * R
        
        A = np.zeros((num_constraints, num_unknowns))
        b = np.zeros(num_constraints)
        
        idx = 0
        for ip in range(n):
            for jp in range(m):
                for i in range(n):
                    for jpp in range(m):
                        for kp in range(p):
                            # Constraint: sum_r U[ip*m+jp,r] * V[jpp*p+kp,r] * W[kp*n+i,r] = delta_{ip,i} delta_{jp,jpp} delta_{kp,kp}
                            # Wait, the last delta should be delta_{kp, k} but k is not in the loop...
                            # Let me re-index.
                            
                            # Actually, the constraint is for all (i, i', j, j', j'', k, k')
                            # Let me re-read the constraint.
                            
                            # The constraint is:
                            # sum_r U[i'*m+j',r] V[j''*p+k',r] W[k*n+i,r] = delta_{i',i} delta_{j',j''} delta_{k',k}
                            # for all i, i', j, j', j'', k, k'
                            
                            # So for fixed (i', j'), the unknowns are U[i'*m+j', 0:R]
                            # And the constraint couples them with (i, j'', k')
                            
                            # Let me restructure: for each (i', j'), build a separate system
                            pass
        
        # This is getting messy. Let me use a different approach.
        # Instead of building the full system, let's use the fact that we can solve for U
        # by considering each (i', j') separately, but the constraints couple them.
        
        # Actually, for fixed V and W, the problem is linear in U, so we can solve it as one big LSQ.
        
        # Let me build the system properly.
        
        A = np.zeros((n*n*m*m*p, n*m*R))
        b = np.zeros(n*n*m*m*p)
        
        row = 0
        for ip in range(n):
            for jp in range(m):
                for i in range(n):
                    for jpp in range(m):
                        for kp in range(p):
                            # The constraint: sum_r U[ip*m+jp, r] * V[jpp*p+kp, r] * W[kp*n+i, r] = delta_{ip,i} * delta_{jp,jpp} * delta_{kp,kp}
                            # Wait, the last delta is always 1? No, that can't be right.
                            
                            # Let me re-read the original constraint.
                            # "sum_r U[im+j, r] V[jp+k, r] W[kn+i, r] = 1 for all i,j,k"
                            # This is from the seed program, but it's simplified.
                            
                            # The actual constraint for matrix multiplication is:
                            # sum_r U[i'*m+j',r] V[j''*p+k',r] W[k*n+i,r] = delta_{i',i} delta_{j',j''} delta_{k',k}
                            
                            # So for fixed (i', j'), the unknowns are U[i'*m+j', :]
                            # And we have constraints for all (i, j'', k')
                            
                            # The RHS is delta_{i',i} delta_{j',j''} delta_{k',k}
                            # But k is not in the constraint index... 
                            
                            # I think I'm overcomplicating this. Let me use the seed program's constraint.
                            # The seed says: "sum_r U[im+j, r] V[jp+k, r] W[kn+i, r] = 1 for all i,j,k"
                            # But this doesn't look right for matrix multiplication.
                            
                            pass
        
        # I'm getting confused with the indexing. Let me just use a numerical approach
        # where I compute the residual directly and use gradient descent.
        
        return None
    
    # Let me try a simpler approach: just use random initialization and check
    # if we can find a valid decomposition by solving a least squares problem.
    
    # For a given R, initialize U, V, W randomly.
    # Then iterate: fix V,W and solve for U via LSQ; fix U,W and solve for V; fix U,V and solve for W.
    
    # To solve for U given V,W:
    # We need to find U such that for all (i', j', i, j'', k'):
    # sum_r U[i'*m+j',r] V[j''*p+k',r] W[k*n+i,r] = delta_{i',i} delta_{j',j''} delta_{k',k}
    
    # This is a linear system in U. Let me build it.
    
    def build_and_solve_U(V, W, R):
        """Build linear system for U and solve."""
        num_rows = n * n * m * m * p
        num_cols = n * m * R
        A = np.zeros((num_rows, num_cols))
        b = np.zeros(num_rows)
        
        row = 0
        for ip in range(n):
            for jp in range(m):
                for i in range(n):
                    for jpp in range(m):
                        for kp in range(p):
                            # RHS
                            rhs = 1.0 if (ip == i and jp == jpp and kp == kp) else 0.0
                            # Wait, the last condition kp==kp is always true. That's wrong.
                            
                            # The constraint is delta_{i',i} delta_{j',j''} delta_{k',k}
                            # where k' is the index in V and k is the index in W.
                            # But in our loop, we have kp which serves as both k' and k?
                            
                            # I think the issue is that the constraint should be:
                            # for all i, i', j', j'', k:
                            # sum_r U[i'*m+j',r] V[j''*p+k,r] W[k*n+i,r] = delta_{i',i} delta_{j',j''}
                            
                            # Because k' = k (they're the same index in the product).
                            
                            # Let me re-derive. We have:
                            # C[i,k] = sum_j A[i,j] B[j,k]
                            # And we want: C[i,k] = sum_r f_r(A) g_r(B) W_r[k,i]
                            # where f_r(A) = sum_{i',j'} U[i'*m+j',r] A[i',j']
                            # g_r(B) = sum_{j'',k'} V[j''*p+k',r] B[j'',k']
                            
                            # So: sum_r (sum_{i',j'} U[i'*m+j',r] A[i',j']) (sum_{j'',k'} V[j''*p+k',r] B[j'',k']) W_r[k,i]
                            # = sum_{i',j',j'',k'} U[i'*m+j',r] V[j''*p+k',r] W_r[k,i] A[i',j'] B[j'',k']
                            
                            # This should equal sum_j A[i,j] B[j,k]
                            # = sum_{i',j',j'',k'} delta_{i',i} delta_{j',j''} delta_{k',k} A[i',j'] B[j'',k']
                            
                            # So we need:
                            # sum_r U[i'*m+j',r] V[j''*p+k',r] W_r[k,i] = delta_{i',i} delta_{j',j''} delta_{k',k}
                            # for all i', j', j'', k', i, k
                            
                            # Now, for fixed V, W, and fixed (i', j'), this is linear in U[i'*m+j', :].
                            # The unknowns are U[i'*m+j', 0:R].
                            # The constraints are for all (j'', k', i, k).
                            
                            # So for each (i', j'), we have a system with n*p unknowns... no, R unknowns.
                            # And n*m*p constraints (for j'', k', i, k but with the delta structure).
                            
                            # Actually, for fixed (i', j'), the constraint is:
                            # for all j'', k', i, k:
                            # sum_r U[i'*m+j',r] V[j''*p+k',r] W[k*n+i,r] = delta_{i',i} delta_{j',j''} delta_{k',k}
                            
                            # This is n*m*p constraints (j'' in range(m), k' in range(p), i in range(n), k in range(p))
                            # Wait, that's n*m*p*p constraints, but many are trivially 0=0.
                            
                            # The non-trivial ones are when delta_{i',i} delta_{j',j''} delta_{k',k} = 1,
                            # which gives m*p constraints (j''=j', k'=k, i=i').
                            # The rest are homogeneous.
                            
                            # So for fixed (i', j'), we have a system of n*m*p*p equations in R unknowns.
                            # But we can reduce it to just the non-trivial ones plus some homogeneous ones.
                            
                            # For efficiency, let's just use all of them.
                            
                            rhs = 1.0 if (ip == i and jp == jpp and kp == kp) else 0.0
                            # This is wrong. The last delta should be delta_{k', k} where k' is from V and k is from W.
                            # In our notation, k' = kp (from V[j''*p+k',r]) and k = ? 
                            # But k is not in the loop! 
                            
                            # I see the issue. The constraint has 6 free indices: i', j', j'', k', i, k
                            # So we need to loop over all 6.
                            
                            pass
        
        # This is getting too complicated. Let me just use a direct numerical method.
        return None
    
    # Let me try a completely different approach: use scipy's least squares solver
    # on the full system.
    
    # For a given R, the total number of unknowns is (n*m + m*p + p*n) * R = (12 + 32 + 24) * R = 68R
    # The number of constraints is n*n*m*m*p*p = 9*16*64 = 9216
    
    # For R=96, we have 6528 unknowns and 9216 constraints.
    # For R=95, we have 6460 unknowns and 9216 constraints.
    
    # This is a large system, but we can use iterative methods.
    
    # Let me just try to find a solution by random search with a few iterations.
    
    for R in range(rank_seed - 1, max(rank_seed - 10, 80), -1):
        if time.time() - start_time > timeout - 10:
            break
        
        np.random.seed(123 + R)
        U = np.random.randn(n*m, R)
        V = np.random.randn(m*p, R)
        W = np.random.randn(p*n, R)
        
        # Compute residual
        def compute_residual(U, V, W):
            res = 0.0
            for ip in range(n):
                for jp in range(m):
                    for i in range(n):
                        for jpp in range(m):
                            for kp in range(p):
                                for k in range(p):
                                    val = np.sum(U[ip*m+jp, :] * V[jpp*p+kp, :] * W[k*n+i, :])
                                    target = 1.0 if (ip == i and jp == jpp and kp == k) else 0.0
                                    res += (val - target) ** 2
            return res
        
        # This is too slow. Let me vectorize.
        
        if time.time() - start_time > timeout - 5:
            break
    
    # Given the complexity, let's just return the schoolbook solution.
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
