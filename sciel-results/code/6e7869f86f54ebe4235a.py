"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = 2, 3, 7
    # Build the multiplication tensor T[i,j,k] = 1
    # Tensor shape: (n, m, p)
    T = np.ones((n, m, p))
    
    # Schoolbook decomposition for fallback
    def schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Time limit for search
    t_start = time.time()
    timeout = 145.0
    
    # Try to find a decomposition of rank r = 41
    # The tensor is 2x3x7.
    # Dimensions:
    # U: 6 x r
    # V: 21 x r
    # W: 14 x r
    
    # We use a random search approach.
    # Initialize U, V, W with random small integers.
    # Since entries must be integers or half-integers, we can scale by 2.
    # Let U, V, W be integer matrices.
    # We want U @ V^T (reshaped) to match T.
    
    # Reshaping logic:
    # U has shape (nm, r). Let U_idx[i, j, c] = U[i*m + j, c]
    # V has shape (mp, r). Let V_idx[j, k, c] = V[j*p + k, c]
    # W has shape (pn, r). Let W_idx[k, i, c] = W[k*n + i, c]
    
    # Sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
    
    # We can view this as a tensor completion or optimization problem.
    # Given U and V, W is determined by the constraints.
    # For each (k, i), W[k*n+i, :] is a vector w_ki.
    # The constraint is sum_j,c U[i,j,c] V[j,k,c] w_ki,c = 1
    # This is a linear system for w_ki?
    # No, it's coupled across c for fixed (k,i).
    # Actually, for fixed i, k:
    # sum_c ( sum_j U[i,j,c] V[j,k,c] ) w_ki,c = 1
    # Let A_ik,c = sum_j U[i,j,c] V[j,k,c].
    # Then A_ik . w_ki = 1.
    # This is one equation with r unknowns. Under-determined.
    # However, U and V are also variables.
    
    # Strategy: Randomly initialize U and V with small integers.
    # Solve for W using least squares (integer constraint relaxation).
    # Check if W can be made integer/half-integer and satisfies exactly.
    # If not, perturb U or V.
    
    # A more robust approach:
    # Iterate:
    # 1. Random U, V (integers)
    # 2. For each (k,i), solve min ||A_ik w - 1||_2 subject to w integer?
    #    This is hard.
    # 3. Relax to real numbers, get W.
    #    If W is close to half-integers, round and check exactness.
    
    # Let's try a simpler heuristic:
    # Since rank 42 is the schoolbook, rank 41 is likely possible if there's symmetry.
    # But 2x3x7 is small.
    
    # Let's try a specific structured approach.
    # Maybe we can merge two schoolbook terms?
    # Term (i=0, j=0, k=0) and (i=1, j=1, k=1)?
    # U: (0,0) and (1,1)
    # V: (0,0) and (1,1)
    # W: (0,0) and (1,1)
    # Product:
    # For (0,0,0): 1*1*1 + 0 = 1 (if we merge, we add terms, not subtract)
    # We need to cancel out.
    # If we add a term with -1s, we can cancel.
    
    # Let's just do the random search for rank 41.
    best_solution = None
    
    for trial in range(2000):
        if time.time() - t_start > timeout:
            break
            
        r = 41
        U = np.random.randint(-1, 2, size=(n*m, r))
        V = np.random.randint(-1, 2, size=(m*p, r))
        
        # Construct W
        W = np.zeros((p*n, r))
        # For each k, i:
        success = True
        for k in range(p):
            for i in range(n):
                # A_ik vector of length r
                A = np.zeros(r)
                for j in range(m):
                    A += U[i*m + j, :] * V[j*p + k, :]
                
                # We need w such that A . w = 1
                # To keep integers small, pick a random solution.
                # General solution: w = w_p + null_space * z
                # Simple particular solution: if A is not zero, w = A / (A.A)
                # But we need integers.
                # If A has an element 1 or -1, we can set that element to +/-1 and others 0.
                # If A is all zeros, impossible (unless 1=0, so fail).
                
                # Find index with max abs value to minimize norm?
                # Or just pick the first non-zero.
                nz = np.nonzero(A)[0]
                if len(nz) == 0:
                    success = False
                    break
                idx = nz[0]
                # We need A[idx] * w[idx] = 1
                # If A[idx] is 1, w[idx]=1. If -1, w[idx]=-1.
                # If A[idx] is 2, w[idx]=0.5 (allowed).
                # If A[idx] is 3, w[idx]=1/3 (NOT allowed).
                # So we need A[idx] to be +/-1 or +/-2.
                
                # Let's check if any non-zero element is +/-1 or +/-2.
                valid_idx = -1
                for idx in nz:
                    val = A[idx]
                    if val in (1, -1, 2, -2):
                        valid_idx = idx
                        break
                
                if valid_idx == -1:
                    success = False
                    break
                
                W[k*n + i, valid_idx] = 1.0 / A[valid_idx]
        
        if not success:
            continue
        
        # Verify the solution
        # Reconstruct tensor
        T_rec = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(r):
                        val += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                    T_rec[i, j, k] = val
                    
        if np.allclose(T_rec, T, atol=1e-9):
            # Check if W entries are integer or half-integer
            W_int = np.round(W * 2)
            if np.allclose(W * 2, W_int, atol=1e-9):
                W = W_int / 2.0
                # Also check U, V (they are already integers)
                best_solution = {"U": U, "V": V, "W": W}
                break
                
    if best_solution is not None:
        return best_solution
    else:
        return schoolbook()
# EVOLVE-BLOCK-END
