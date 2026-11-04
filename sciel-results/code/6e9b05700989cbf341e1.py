"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s margin before 150s limit
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best = {"U": U_sb, "V": V_sb, "W": W_sb, "rank": rank_schoolbook}
    
    def verify_decomposition(U, V, W, rank, tol=1e-6):
        """Verify that U,V,W give an exact decomposition of the MM tensor."""
        if time.time() > deadline:
            return False
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def snap_to_halfint(arr):
        """Snap values to nearest half-integer."""
        return np.round(np.array(arr) * 2) / 2
    
    # Try to construct a rank-20 decomposition using structured approach
    # For 4x4x5, we can use a block-Winograd approach
    # Strategy: use the 4x4x4 rank-16 decomposition and extend to 5 columns
    
    # First, try known low-rank constructions
    # The minimal rank for 4x4x4 is 16 (Strassen-like)
    # For 4x4x5, we expect rank around 20-24
    
    # Let's try a direct construction using the following idea:
    # Split the 5 output columns into groups and use tensor product structure
    
    # Approach: Use a rank-r decomposition where r is minimized
    # Try ranks from 16 up to schoolbook
    for r in range(16, rank_schoolbook + 1):
        if time.time() > deadline:
            break
        
        # Try to find a rank-r decomposition using structured initialization
        # Initialize with a pattern that has good chance of working
        
        rng = np.random.RandomState(r)
        found = False
        
        # Try multiple random initializations with different seeds
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            seed = r * 100 + attempt
            rng = np.random.RandomState(seed)
            
            # Initialize with structured pattern
            # Use a Kronecker-like structure as starting point
            U_init = rng.randn(n * m, r) * 0.5
            V_init = rng.randn(m * p, r) * 0.5
            W_init = rng.randn(p * n, r) * 0.5
            
            # Use a smarter initialization: try to match the tensor structure
            # The tensor is T[i,j,k] = 1 for all i,j,k
            # So we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            
            # Try a different approach: use the fact that the MM tensor has
            # a specific algebraic structure and try to exploit it
            
            # For now, try gradient-free approach: solve a system of equations
            # We have n*m*p = 80 equations in r*r*... variables
            # This is underdetermined for small r, so we can try to find solutions
            
            # Actually, let's try a more direct approach:
            # Fix U and V, then solve for W linearly
            # Or use alternating minimization
            
            # Let's try: fix U and W, solve for V
            # For each (j,k), V[j*p+k,:] appears in equations for all i
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
            
            # This is a linear system in V[j*p+k,:]
            # A_{i,c} = U[i*m+j,c] * W[k*n+i,c]
            # sum_c A_{i,c} * V[j*p+k,c] = 1 for all i
            
            # So for each (j,k), we solve a linear system
            
            # Let's try this alternating approach
            U_curr = U_init.copy()
            V_curr = V_init.copy()
            W_curr = W_init.copy()
            
            # Alternate: fix U,W -> solve V; fix U,V -> solve W; fix V,W -> solve U
            for it in range(200):
                if time.time() > deadline:
                    break
                
                # Solve for V: for each (j,k), solve linear system
                for j in range(m):
                    for k in range(p):
                        # Build system: for each i, sum_c U[i*m+j,c]*W[k*n+i,c]*V[j*p+k,c] = 1
                        A = np.array([[U_curr[i*m+j, c] * W_curr[k*n+i, c] for c in range(r)] for i in range(n)])
                        b = np.ones(n)
                        # Solve A @ V[j*p+k,:] = b
                        try:
                            V_curr[j*p+k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Solve for W: for each (i,k), solve linear system
                for i in range(n):
                    for k in range(p):
                        # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all j
                        A = np.array([[U_curr[i*m+j, c] * V_curr[j*p+k, c] for c in range(r)] for j in range(m)])
                        b = np.ones(m)
                        try:
                            W_curr[k*n+i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Solve for U: for each (i,j), solve linear system
                for i in range(n):
                    for j in range(m):
                        # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all k
                        A = np.array([[V_curr[j*p+k, c] * W_curr[k*n+i, c] for c in range(r)] for k in range(p)])
                        b = np.ones(p)
                        try:
                            U_curr[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Check error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_curr[i*m+j], V_curr[j*p+k] * W_curr[k*n+i])
                            err += (val - 1.0) ** 2
                
                if err < 1e-10:
                    # Snap to half-integers and verify
                    U_snap = snap_to_halfint(U_curr)
                    V_snap = snap_to_halfint(V_curr)
                    W_snap = snap_to_halfint(W_curr)
                    
                    if verify_decomposition(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), r):
                        best = {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist(),
                            "rank": r
                        }
                        found = True
                        break
            
            if found:
                break
        
        if found:
            break
    
    # Return the best result
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END

