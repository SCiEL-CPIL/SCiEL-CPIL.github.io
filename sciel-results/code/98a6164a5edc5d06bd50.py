"""Matrix multiplication 2x3x9: find rank-minimal exact tensor decomposition."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Tensor entry T[i,j,k] = 1 if we're multiplying A[i,j]*B[j,k] -> C[i,k]
    # W index: k*n + i (trace convention)
    
    # The tensor has entries: for each (i,j,k) in [0,n)×[0,m)×[0,p),
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # We want to find rank r such that there exist U (nm×r), V (mp×r), W (pn×r)
    # with sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = T[i,j,k] for all valid (i,j,k)
    
    # Strategy: Try small ranks. For each rank, set up equations.
    # Use finite field to find candidates, then verify over integers.
    
    # First, try a structured approach based on Kronecker substitution / polynomial multiplication
    # 2x3 * 3x9: Think of A as 2 polynomials of degree <3, B as 3 polynomials of degree <9?
    # Actually, let's think differently.
    
    # A is 2x3: A[i,j] for i in {0,1}, j in {0,1,2}
    # B is 3x9: B[j,k] for j in {0,1,2}, k in {0,...,8}
    # C = A*B: C[i,k] = sum_j A[i,j]*B[j,k]
    
    # The tensor T has indices (i,j,k) -> contribution to C[i,k]
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # Let's try to find a decomposition using the structure.
    # One approach: use the fact that 3 and 9 are related (9 = 3*3).
    # Group B's columns into 3 groups of 3: B[j, 3*q + r] for q in {0,1,2}, r in {0,1,2}
    # Then C[i, 3*q+r] = sum_j A[i,j] * B[j, 3*q+r]
    
    # This is like: for each (i,q), C[i,3q+r] = sum_j A[i,j]*B[j,3q+r]
    # which is a 2x3 * 3x1 multiplication for each r, but B[j,3q+r] varies with q.
    
    # Let me try a direct search for low rank using linear algebra over a small prime field.
    
    start_time = time.time()
    deadline = start_time + 120  # 120 second budget
    
    # The tensor T is a 3-tensor of size (nm) x (mp) x (pn) = 6 x 27 x 18
    # Non-zero entries: 54 (one for each (i,j,k))
    
    # For a rank-1 term c: U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # The equation is: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = T[i,j,k]
    
    # This is a system of 54 equations (one per non-zero T entry) plus we need
    # the sum to be 0 for all other (a,b,c) entries.
    # Total entries in the 3-tensor: 6*27*18 = 2916
    
    # Let's set up: for rank r, we have 3*r*r variables (U,V,W each have r entries per row, but we can normalize)
    # Actually, the decomposition is not linear in (U,V,W) jointly, but it IS linear in W if we fix U and V.
    
    # Better approach: iterate. Fix U and V, solve for W linearly.
    # Or: use the fact that the tensor is sparse.
    
    # Let me try a different structured approach:
    # Use the polynomial ring Z[x]/(x^3 - 1) or similar.
    
    # Actually, let me try to directly construct a low-rank decomposition.
    # For 2x3 * 3x9, the rank is at most 54 (schoolbook).
    # Known results: rank of 2x3 * 3x9 is between... let's try to find it.
    
    # Let me try rank 27, 36, 45, etc. and see if I can find a solution.
    
    # Use a prime field for computation
    prime = 1000003
    
    # Build the tensor T as a sparse structure
    # T[a, b, c] = 1 where a = i*m+j, b = j*p+k, c = k*n+i
    tensor_entries = {}  # (a,b,c) -> value
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_entries[(a, b, c)] = 1
    
    # For a given rank r, we want to find U (nm x r), V (mp x r), W (pn x r)
    # such that for all (a,b,c): sum_s U[a,s]*V[b,s]*W[c,s] = T[a,b,c]
    
    # This is a nonlinear system. Let's use an iterative approach:
    # 1. Initialize U, V randomly
    # 2. Solve for W linearly: for each c, W[c,s] such that sum_s V[b,s]*W[c,s] = ... 
    #    Actually, for fixed U and V, the equation for W is:
    #    For each (a,b,c) with T[a,b,c] != 0: sum_s U[a,s]*V[b,s]*W[c,s] = 1
    #    This is linear in W[c,s] for fixed c, but couples different c's through the same s.
    
    # Let me think differently. Fix U and V. Then for each c, we have:
    # sum_s (sum_a' T[a',b',c] * U[a',s] * V[b',s]) ... no this doesn't simplify easily.
    
    # Alternative: use the structure. Let me try a specific construction.
    
    # For 2x3 * 3x9, note that 9 = 3*3. Let's write k = 3*q + r where q,r in {0,1,2}.
    # C[i, 3q+r] = sum_j A[i,j] * B[j, 3q+r]
    
    # Define B_q[j, r] = B[j, 3q+r]. Then C[i, 3q+r] = sum_j A[i,j] * B_q[j,r].
    # This is like doing 3 separate 2x3 * 3x3 multiplications (one for each q), 
    # but B_q is different for each q.
    
    # A 2x3 * 3x3 multiplication has rank at most 18 (schoolbook), but can be done in rank 16 or so.
    # If we could share structure between the 3 multiplications, we might save.
    
    # Let me try: use the same U and V structure for all q, and only W varies.
    
    # Actually, let me just try to find a solution by solving a linear system over a finite field
    # with a fixed U and V, then iterating.
    
    # Simpler: let me try to construct a solution of rank 36 or so using a block structure.
    
    # U: 6 x r, V: 27 x r, W: 18 x r
    # Let's try r = 36.
    
    # I'll use a greedy/iterative approach over a finite field.
    
    def solve_rank(r, max_iters=200):
        """Try to find a rank-r decomposition over the finite field."""
        # Initialize U, V with random values
        np.random.seed(42)
        U = np.random.randint(1, prime, size=(n*m, r)) % prime
        V = np.random.randint(1, prime, size=(m*p, r)) % prime
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # For fixed U and V, solve for W.
            # The equation is: for all (a,b,c), sum_s U[a,s]*V[b,s]*W[c,s] = T[a,b,c]
            # Group by c: for each c, we need sum_s (sum over (a,b) with this c) ... 
            # Actually, for each c, the equation involves W[c,:] coupled across s.
            
            # Let me set it up as: for each c, we have equations:
            # sum_s M_c[s] * W[c,s] = t_c where M_c[s] = sum_{(a,b): (a,b,c) in tensor} U[a,s]*V[b,s]
            # Wait, that's not right either. The tensor entry T[a,b,c] is 1 for specific (a,b,c).
            
            # For a fixed c, the relevant (a,b) pairs are those where (a,b,c) is a tensor entry.
            # For each such (a,b), we have: sum_s U[a,s]*V[b,s]*W[c,s] = 1
            # This is a system where W[c,s] appears in multiple equations (one for each (a,b) with this c).
            
            # Let me build the system properly.
            # Variables: W[0..pn-1, 0..r-1], total pn*r variables.
            # Equations: for each (a,b,c) in tensor_entries: sum_s U[a,s]*V[b,s]*W[c,s] = 1
            # Plus: for all other (a,b,c) not in tensor_entries: sum_s U[a,s]*V[b,s]*W[c,s] = 0
            
            # Total equations: nm*mp*pn = 6*27*18 = 2916
            # Total variables: pn*r = 18*r
            
            # This is a linear system in W. Let me build the matrix.
            # But 2916 equations and 18*r variables. For r=36, that's 648 variables.
            
            # Build A_matrix (2916 x 18*r) and b_vector (2916,)
            # This might be too large. Let me be smarter.
            
            # Actually, for each c, the equations involving W[c,:] are:
            # For each (a,b) such that (a,b,c) is a tensor entry: sum_s U[a,s]*V[b,s]*W[c,s] = 1
            # For each (a,b) such that (a,b,c) is NOT a tensor entry: sum_s U[a,s]*V[b,s]*W[c,s] = 0
            
            # So for each c, we have nm*mp = 6*27 = 162 equations in r unknowns (W[c,0..r-1]).
            # Total: pn * nm * mp = 18*6*27 = 2916 equations, 18*r unknowns.
            
            # For r=36: 2916 equations, 648 unknowns. Overdetermined.
            
            # Let me build the system for W.
            # For each c in range(pn):
            #   For each a in range(nm), b in range(mp):
            #     coeff[s] = U[a,s]*V[b,s] for s in range(r)
            #     rhs = 1 if (a,b,c) in tensor_entries else 0
            #     equation: sum_s coeff[s]*W[c,s] = rhs
            
            # Build the full matrix.
            A_mat = np.zeros((n*m*m*p*p*n, r), dtype=np.float64)  # This is too big: 2916 x r
            # Wait, it's 2916 rows and r columns? No, each row has r entries.
            # Total: 2916 rows, r columns. For r=36, that's 2916x36 = 104976 entries. OK.
            
            b_vec = np.zeros(n*m*m*p*p*n)
            row = 0
            for c in range(p*n):
                for a in range(n*m):
                    for b in range(m*p):
                        for s in range(r):
                            A_mat[row, s] = (U[a, s] * V[b, s]) % prime
                        if (a, b, c) in tensor_entries:
                            b_vec[row] = 1
                        row += 1
            
            # Solve A_mat @ W_col = b_vec for each column of W (they're independent!)
            # Wait, W is (pn x r), and each row W[c,:] is independent.
            # So I have pn independent systems, each of size (nm*mp) x r.
            
            W = np.zeros((p*n, r), dtype=np.float64)
            for c in range(p*n):
                # Extract the sub-matrix for this c
                start = c * n * m * m * p
                end = start + n * m * m * p
                A_c = A_mat[start:end, :]  # (162 x r)
                b_c = b_vec[start:end]  # (162,)
                
                # Solve least squares
                try:
                    W[c, :], residuals, rank_c, sv = np.linalg.lstsq(A_c, b_c, rcond=None)
                except:
                    return None
            
            # Check residual
            residual = 0
            for c in range(p*n):
                for a in range(n*m):
                    for b in range(m*p):
                        val = sum(U[a, s] * V[b, s] * W[c, s] for s in range(r)) % prime
                        expected = 1 if (a, b, c) in tensor_entries else 0
                        residual += (val - expected) ** 2
            
            if residual < 1e-6:
                return (U, V, W, residual)
            
            # Update U and V randomly (or via some heuristic)
            # Let's just re-randomize
            U = np.random.randint(1, prime, size=(n*m, r)) % prime
            V = np.random.randint(1, prime, size=(m*p, r)) % prime
        
        return None
    
    # Try various ranks
    for r in [18, 27, 36, 45, 54]:
        if time.time() > deadline:
            break
        result = solve_rank(r, max_iters=50)
        if result is not None:
            U, V, W, res = result
            # Verify over integers by checking all entries
            # Convert to integers (they should be integers if the solution is exact)
            U_int = U.astype(int)
            V_int = V.astype(int)
            W_int = W.astype(int)
            
            # Check if the decomposition is exact over integers
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        val = sum(U_int[a, s] * V_int[b, s] * W_int[c, s] for s in range(r))
                        if val != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Also check that all other entries are 0
            for a in range(n*m):
                for b in range(m*p):
                    for c in range(p*n):
                        if (a, b, c) not in tensor_entries:
                            val = sum(U_int[a, s] * V_int[b, s] * W_int[c, s] for s in range(r))
                            if val != 0:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Found a valid integer decomposition!
                return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
    
    # If we couldn't find a lower rank, fall back to schoolbook
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
# EVOLVE-BLOCK-END
