import numpy as np
import time
import sys

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k]
    # We want to minimize || T - sum_r U[:, :, r] * V[:, :, r] * W[:, :, r] ||
    # Actually, the tensor is defined by the bilinear map.
    # The condition is:
    # sum_r W[i, k, r] * U[i, j, r] * V[j, k, r] == 1 if j matches, 0 otherwise.
    
    # Let's represent the target as a 3D array T_target[i, j, k]
    # But the constraint is on the "slice" for each i,k.
    # For fixed i,k, the vector of coeffs for j is [0...1...0] with 1 at j.
    # Actually, for fixed i,k, we have m=4 terms.
    # The term a[i,j] b[j,k] contributes to C[i,k].
    # So for each i,k, we want the sum over r of W[i,k,r] * U[i,:,r] * V[:,k,r] to be e_j (basis vector)?
    # No.
    # C[i,k] = sum_j a[i,j] b[j,k].
    # We compute C[i,k] = sum_r W[i,k,r] * (sum_j U[i,j,r] a[i,j]) * (sum_k' V[j,k',r] b[j,k'])?
    # No, the multiplication is z_r = u_r(a) v_r(b).
    # u_r(a) = sum_j U[i,j,r] a[i,j] ? No, u_r depends on all a.
    # u_r(a) = sum_{i', j'} U[i', j', r] a[i', j'].
    # v_r(b) = sum_{j', k'} V[j', k', r] b[j', k'].
    # z_r = u_r(a) v_r(b).
    # C[i,k] = sum_r W[i,k,r] z_r.
    # Coeff of a[i,j] b[j,k] in C[i,k] is sum_r W[i,k,r] U[i,j,r] V[j,k,r].
    # This must be 1.
    # Coeff of a[i,j] b[j',k] (j != j') is sum_r W[i,k,r] U[i,j,r] V[j',k,r].
    # This must be 0.
    
    # So for each i, k, and each j, j':
    # sum_r W[i,k,r] U[i,j,r] V[j',k,r] = delta_{j,j'}
    
    # This is a linear system in W for fixed U,V.
    # Let M[i,k,j,j'] = sum_r U[i,j,r] V[j',k,r] W[i,k,r].
    # We want M[i,k,j,j'] = delta_{j,j'}.
    
    # We can solve for W given U,V using least squares.
    # The variables are W[i,k,r].
    # The equations are:
    # For each i, k, j, j':
    # sum_r W[i,k,r] * (U[i,j,r] * V[j',k,r]) = delta_{j,j'}
    
    # This is a linear system A * w = b.
    # A has shape (n * p * m * m, R) ?
    # No, for each i,k, we have m*m equations and R unknowns (W[i,k, :]).
    # So we can solve for each i,k independently.
    
    def solve_W(U, V):
        W = np.zeros((n, p, R))
        for i in range(n):
            for k in range(p):
                # Build A for this i,k
                # Rows correspond to (j, j') pairs.
                # A[row, r] = U[i, j, r] * V[j', k, r]
                A = np.zeros((m*m, R))
                b = np.zeros(m*m)
                idx = 0
                for j in range(m):
                    for jp in range(m):
                        for r in range(R):
                            A[idx, r] = U[i, j, r] * V[jp, k, r]
                        if j == jp:
                            b[idx] = 1
                        else:
                            b[idx] = 0
                        idx += 1
                # Solve A * W[i,k,:] = b
                # Use least squares
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W[i, k, :] = sol
        return W

    def solve_U(V, W):
        U = np.zeros((n, m, R))
        for i in range(n):
            for j in range(m):
                # Equations for U[i,j,r]:
                # For each k, jp:
                # sum_r U[i,j,r] * V[jp,k,r] * W[i,k,r] = delta_{j,jp}
                # Wait, the equation is:
                # sum_r W[i,k,r] U[i,j,r] V[jp,k,r] = delta_{j,jp}
                # For fixed i,j, the unknowns are U[i,j,r].
                # The equations are for all k, jp.
                # A[row, r] = W[i,k,r] * V[jp,k,r]
                # b[row] = delta_{j,jp}
                A = np.zeros((p*m, R))
                b = np.zeros(p*m)
                idx = 0
                for k in range(p):
                    for jp in range(m):
                        for r in range(R):
                            A[idx, r] = W[i, k, r] * V[jp, k, r]
                        if j == jp:
                            b[idx] = 1
                        else:
                            b[idx] = 0
                        idx += 1
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U[i, j, :] = sol
        return U

    def solve_V(U, W):
        V = np.zeros((m, p, R))
        for k in range(p):
            for jp in range(m): # This is the j index in V[j,k,r]
                # Wait, V index is j,k.
                # Unknowns are V[j,k,r] for all j.
                # Equations:
                # sum_r W[i,k,r] U[i,j,r] V[jp,k,r] = delta_{j,jp}
                # For fixed k, jp, unknowns are V[j,k,r] for all j?
                # No, V[jp,k,r] is one variable vector.
                # But it appears in equations for all i, j.
                # Let's fix k, jp.
                # Unknowns: V[jp,k,r] for r=0..R-1.
                # Equations: For each i, j:
                # sum_r W[i,k,r] U[i,j,r] V[jp,k,r] = delta_{j,jp}
                # A[row, r] = W[i,k,r] * U[i,j,r]
                # b[row] = delta_{j,jp}
                A = np.zeros((n*m, R))
                b = np.zeros(n*m)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for r in range(R):
                            A[idx, r] = W[i, k, r] * U[i, j, r]
                        if j == jp:
                            b[idx] = 1
                        else:
                            b[idx] = 0
                        idx += 1
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V[jp, k, :] = sol
        return V

    def error(U, V, W):
        err = 0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        val = np.sum(W[i, k, :] * U[i, j, :] * V[jp, k, :])
                        target = 1 if j == jp else 0
                        err += (val - target) ** 2
        return np.sqrt(err)

    # Try ranks
    for R in range(28, 57):
        # Initialize
        np.random.seed(42 + R)
        U = np.random.randn(n, m, R)
        V = np.random.randn(m, p, R)
        W = np.random.randn(n, p, R)
        
        # ALS
        converged = False
        for it in range(100):
            W = solve_W(U, V)
            U = solve_U(V, W)
            V = solve_V(U, W)
            
            e = error(U, V, W)
            if e < 1e-6:
                converged = True
                break
        
        if converged:
            # Round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Check if rounded solution still works
            e_rounded = error(U_rounded, V_rounded, W_rounded)
            if e_rounded < 1e-6:
                # Format output
                U_out = [[0]*R for _ in range(n*m)]
                V_out = [[0]*R for _ in range(m*p)]
                W_out = [[0]*R for _ in range(p*n)]
                
                for i in range(n):
                    for j in range(m):
                        U_out[i*m+j] = [float(x) for x in U_rounded[i,j,:]]
                for j in range(m):
                    for k in range(p):
                        V_out[j*p+k] = [float(x) for x in V_rounded[j,k,:]]
                for k in range(p):
                    for i in range(n):
                        W_out[k*n+i] = [float(x) for x in W_rounded[i,k,:]]
                        
                return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback to schoolbook
    R = n * m * p
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
