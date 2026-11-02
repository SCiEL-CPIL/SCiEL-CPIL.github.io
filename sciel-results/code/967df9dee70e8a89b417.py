import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 140.0  # 10s margin
    
    # Target Tensor
    # T[i, j, k] = 1 if (i, j, k) is a valid index for the product
    # Indices: i in [0, n), j in [0, m), k in [0, p)
    # The tensor is in the space V1 x V2 x V3
    # V1 dimension n*m, V2 dimension m*p, V3 dimension p*n
    # Wait, the indices for the tensor T in the decomposition sum are:
    # Sum_c U_c[i*m+j] V_c[j*p+k] W_c[k*n+i]
    # This sum must equal 1 for all i,j,k.
    # And for any other combination of indices in the full tensor space?
    # The tensor T is an element of (V1*) x (V2*) x (V3*)?
    # No, the decomposition is T = sum U_c x V_c x W_c.
    # The "entries" of T are T_{a,b,c} where a in V1, b in V2, c in V3.
    # V1 is the space of A (n x m). V2 is the space of B (m x p). V3 is the space of C (p x n).
    # The multiplication map is linear in A and B, and the result is C.
    # So T is a 3-tensor.
    # The non-zero components are T_{i,j; j',k; k',i'} = 1 if j=j', k=k', i=i'.
    # So there are n*m*p non-zero components.
    
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                T_target[i*m + j, j*p + k, k*n + i] = 1.0

    def verify(U, V, W):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Compute reconstruction
        # T_rec[a, b, c] = sum_k U[a, k] V[b, k] W[c, k]
        # This is a tensor contraction.
        # We can compute this efficiently.
        # T_rec = np.einsum('ak,bk,ck->abc', U, V, W)
        # But this creates a large array (42, 21, 14) = 12348 elements.
        # This is small enough.
        T_rec = np.einsum('ak,bk,ck->abc', U, V, W)
        
        # Check against target
        # Only check non-zero parts of target?
        # No, we must check that zero parts are also zero.
        # But T_target is sparse.
        # We can just check the difference.
        diff = np.abs(T_rec - T_target)
        max_diff = np.max(diff)
        
        # Check half-integer constraint
        # U, V, W entries should be close to k/2
        def check_half_int(M):
            # M is a matrix
            # Check if M * 2 is close to integer
            M2 = M * 2
            diff_int = np.abs(M2 - np.round(M2))
            return np.max(diff_int) < 1e-5
        
        if not (check_half_int(U) and check_half_int(V) and check_half_int(W)):
            return False, max_diff
            
        if max_diff < 1e-5:
            return True, max_diff
        return False, max_diff

    def snap_to_half_int(M):
        return np.round(M * 2) / 2.0

    def solve_rank(r):
        # Initialize with first r columns of schoolbook
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= r:
                        break
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
                if col >= r:
                    break
            if col >= r:
                break
        
        # Newton's Method
        # Variables: U, V, W flattened
        # We can treat them as separate blocks.
        
        for iteration in range(100):
            if time.time() - start_time > time_limit:
                break
                
            # Compute residual
            T_rec = np.einsum('ak,bk,ck->abc', U, V, W)
            R = T_rec - T_target
            
            # Flatten residual
            R_flat = R.flatten()
            
            # Check convergence
            if np.max(np.abs(R_flat)) < 1e-8:
                break
                
            # Compute Jacobian J
            # dR/du_a, dR/dv_b, dR/dw_c
            # R[a,b,c] = sum_k U[a,k] V[b,k] W[c,k]
            # dR[a,b,c]/dU[a',k] = delta_{a,a'} V[b,k] W[c,k]
            # This is sparse.
            
            # Instead of building the full Jacobian, we can use Gauss-Newton or a simple step.
            # Given the small size, let's build the Jacobian.
            # Size: (n*m*m*p*p*n) x (r*(n*m + m*p + p*n))
            # 588 x (r * 41). For r=41, 588 x 1681.
            # This is feasible.
            
            J = np.zeros((n*m*m*p*p*n, r*(n*m + m*p + p*n)))
            
            # Part for U
            # J[R_idx, U_idx]
            # R_idx = (a, b, c) -> a*(m*p*p*n) + b*(p*n) + c
            # U_idx = (a, k) -> k*(n*m) + a
            for a in range(n*m):
                for b in range(m*p):
                    for c in range(p*n):
                        r_idx = a*(m*p*p*n) + b*(p*n) + c
                        for k in range(r):
                            u_idx = k*(n*m) + a
                            J[r_idx, u_idx] = V[b, k] * W[c, k]
                            
            # Part for V
            # dR[a,b,c]/dV[b',k] = U[a,k] delta_{b,b'} W[c,k]
            for a in range(n*m):
                for b in range(m*p):
                    for c in range(p*n):
                        r_idx = a*(m*p*p*n) + b*(p*n) + c
                        for k in range(r):
                            v_idx = (n*m) + k*(m*p) + b
                            J[r_idx, v_idx] = U[a, k] * W[c, k]
                            
            # Part for W
            # dR[a,b,c]/dW[c',k] = U[a,k] V[b,k] delta_{c,c'}
            for a in range(n*m):
                for b in range(m*p):
                    for c in range(p*n):
                        r_idx = a*(m*p*p*n) + b*(p*n) + c
                        for k in range(r):
                            w_idx = (n*m + m*p) + k*(p*n) + c
                            J[r_idx, w_idx] = U[a, k] * V[b, k]
                            
            # Solve J * dx = -R_flat
            # Use least squares
            try:
                dx, residuals, rank, sv = np.linalg.lstsq(J, -R_flat, rcond=None)
            except np.linalg.LinAlgError:
                break
                
            # Update U, V, W
            U_flat = U.flatten()
            V_flat = V.flatten()
            W_flat = W.flatten()
            
            U_flat += dx[:n*m*r]
            V_flat += dx[n*m*r : n*m*r + m*p*r]
            W_flat += dx[n*m*r + m*p*r : ]
            
            U = U_flat.reshape(n*m, r)
            V = V_flat.reshape(m*p, r)
            W = W_flat.reshape(p*n, r)
            
            # Optional: Snap to half-integers periodically?
            # No, snap at the end.
            
        # Snap and verify
        U_snapped = snap_to_half_int(U)
        V_snapped = snap_to_half_int(V)
        W_snapped = snap_to_half_int(W)
        
        is_valid, err = verify(U_snapped, V_snapped, W_snapped)
        if is_valid:
            return U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()
        return None

    best_result = None
    # Try ranks from 41 down to 30
    for r in range(41, 29, -1):
        if time.time() - start_time > time_limit - 10:
            break
        res = solve_rank(r)
        if res is not None:
            best_result = {'U': res[0], 'V': res[1], 'W': res[2]}
            # If we find a valid one, we can stop or try lower?
            # "rank-minimal" -> we want the lowest rank.
            # So we should continue to lower ranks if possible.
            # But lower ranks are harder.
            # Let's continue.
            
    if best_result is None:
        # Fall back to schoolbook
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
        best_result = {'U': U, 'V': V, 'W': W}
        
    return best_result
