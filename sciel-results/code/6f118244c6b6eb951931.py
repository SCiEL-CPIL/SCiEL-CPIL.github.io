"""Rank-minimal tensor decomposition for 2x2 * 2x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115

    def verify(U, V, W, tol=1e-9):
        rank = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    a = i * m + k
                    b = k * p + j
                    c = j * n + i
                    val = np.sum(Ua[a] * Va[b] * Wa[c])
                    if abs(val - 1.0) > tol:
                        return False
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i = a // m
                    k = a % m
                    k2 = b // p
                    j = b % p
                    i2 = c % n
                    j2 = c // n
                    if i != i2 or k != k2 or j != j2:
                        val = np.sum(Ua[a] * Va[b] * Wa[c])
                        if abs(val) > tol:
                            return False
        return True

    def make_rank14():
        rank = 14
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        U[0][0] = 1; U[3][0] = 1
        U[2][1] = 1; U[3][1] = 1
        U[0][2] = 1
        U[3][3] = 1
        U[0][4] = 1; U[1][4] = 1
        U[2][5] = 1; U[0][5] = -1
        U[1][6] = 1; U[3][6] = -1
        U[0][7] = 1; U[3][7] = 1
        U[2][8] = 1; U[3][8] = 1
        U[0][9] = 1
        U[3][10] = 1
        U[0][11] = 1; U[1][11] = 1
        U[2][12] = 1; U[0][12] = -1
        U[1][13] = 1; U[3][13] = -1
        V[0][0] = 1; V[5][0] = 1
        V[0][1] = 1
        V[1][2] = 1; V[5][2] = -1
        V[4][3] = 1; V[0][3] = -1
        V[5][4] = 1
        V[0][5] = 1; V[1][5] = 1
        V[4][6] = 1; V[5][6] = 1
        V[2][7] = 1; V[7][7] = 1
        V[2][8] = 1
        V[3][9] = 1; V[7][9] = -1
        V[6][10] = 1; V[2][10] = -1
        V[7][11] = 1
        V[2][12] = 1; V[3][12] = 1
        V[6][13] = 1; V[7][13] = 1
        W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
        W[1][1] = 1; W[1][3] = 1
        W[2][2] = 1; W[2][4] = 1
        W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
        W[4][7] = 1; W[4][10] = 1; W[4][11] = -1; W[4][13] = 1
        W[5][8] = 1; W[5][10] = 1
        W[6][9] = 1; W[6][11] = 1
        W[7][7] = 1; W[7][8] = -1; W[7][9] = 1; W[7][12] = 1
        return {"U": U, "V": V, "W": W}

    def build_rank12_structure():
        """
        Rank-12 decomposition structure for 2x2 * 2x4.
        
        Terms 0-6: Strassen for block 1 (B columns 0,1)
        Terms 0,1,3,4: SHARED - A-forms reused, B-forms extended to columns 2,3
        Terms 7,8,9: New terms for block 2 (B columns 2,3)
        
        Each term r has a sparse support: set of (a,b,c) indices it contributes to.
        """
        rank = 12
        # U: 4 rows (a00, a01, a10, a11), V: 8 rows (b00..b03, b10..b13), W: 8 rows (c00..c03, c10..c13)
        # a-index: i*m+k -> 0=a00, 1=a01, 2=a10, 3=a11
        # b-index: k*p+j -> 0=b00, 1=b01, 2=b02, 3=b03, 4=b10, 5=b11, 6=b12, 7=b13
        # c-index: j*n+i -> 0=c00, 1=c10, 2=c01, 3=c11, 4=c02, 5=c12, 6=c03, 7=c13
        
        # Strassen A-forms for 2x2:
        # M1: a00+a11, M2: a10+a11, M3: a00, M4: a11, M5: a00+a01, M6: a10-a00, M7: a01-a11
        # Strassen B-forms: b00+b11, b00, b01-b11, b10-b00, b11, b00+b01, b10+b11
        # Strassen W: c00=M1-M4+M5, c10=M2+M3, c01=M4+M6, c11=M1+M7
        
        # Term definitions: (U_vector, V_vector, W_vector) as sparse dicts
        # U has 4 entries, V has 8 entries, W has 8 entries
        
        terms = []
        
        # Term 0: M1 shared - U=[1,0,0,1], V=[1,0,0,0,0,1,1,0], W=[1,0,0,0,1,0,0,0]
        # Block 1: (a00+a11)*(b00+b11) -> c00, c11
        # Block 2: (a00+a11)*(b02+b12) -> c02, c12
        terms.append({
            'U': {0: 1, 3: 1},
            'V': {0: 1, 5: 1, 6: 1},
            'W': {0: 1, 4: 1}
        })
        
        # Term 1: M2 shared - U=[0,0,1,1], V=[1,0,0,0,0,0,0,0], W=[0,1,0,0,0,1,0,0]
        # Block 1: (a10+a11)*b00 -> c10
        # Block 2: (a10+a11)*b02 -> c12
        terms.append({
            'U': {2: 1, 3: 1},
            'V': {0: 1},
            'W': {1: 1, 5: 1}
        })
        
        # Term 2: M3 block1 only - U=[1,0,0,0], V=[0,0,0,0,1,0,0,0], W=[0,0,0,0,0,0,0,1]
        # (a00)*b10 -> c01
        terms.append({
            'U': {0: 1},
            'V': {4: 1},
            'W': {7: 1}
        })
        
        # Term 3: M4 shared - U=[0,0,0,1], V=[0,0,0,0,0,1,1,0], W=[0,0,0,1,0,0,0,1]
        # Block 1: (a11)*b11 -> c11
        # Block 2: (a11)*b12 -> c12
        terms.append({
            'U': {3: 1},
            'V': {5: 1, 6: 1},
            'W': {3: 1, 7: 1}
        })
        
        # Term 4: M5 shared - U=[1,1,0,0], V=[1,0,0,0,0,1,1,0], W=[-1,0,0,0,1,0,0,0]
        # Block 1: (a00+a01)*(b00+b11) -> c00, c11
        # Block 2: (a00+a01)*(b02+b12) -> c02, c12
        terms.append({
            'U': {0: 1, 1: 1},
            'V': {0: 1, 5: 1, 6: 1},
            'W': {0: -1, 4: 1}
        })
        
        # Term 5: M6 block1 only - U=[-1,0,1,0], V=[1,0,0,0,0,1,0,0], W=[0,0,1,0,0,0,0,0]
        # (a10-a00)*(b00+b11) -> c01
        terms.append({
            'U': {0: -1, 2: 1},
            'V': {0: 1, 5: 1},
            'W': {2: 1}
        })
        
        # Term 6: M7 block1 only - U=[0,1,0,-1], V=[0,0,0,0,1,1,0,0], W=[0,0,0,1,0,0,0,0]
        # (a01-a11)*(b10+b11) -> c11
        terms.append({
            'U': {1: 1, 3: -1},
            'V': {4: 1, 5: 1},
            'W': {3: 1}
        })
        
        # Term 7: New - U=[0,0,0,1], V=[0,0,1,0,0,0,1,0], W=[0,0,0,0,1,0,0,0]
        # (a11)*(b02+b12) -> c02
        terms.append({
            'U': {3: 1},
            'V': {2: 1, 6: 1},
            'W': {4: 1}
        })
        
        # Term 8: New - U=[1,0,0,0], V=[0,0,0,0,1,0,0,1], W=[0,0,0,0,0,0,1,0]
        # (a00)*(b10+b13) -> c03
        terms.append({
            'U': {0: 1},
            'V': {4: 1, 7: 1},
            'W': {6: 1}
        })
        
        # Term 9: New - U=[0,1,0,0], V=[0,0,0,0,0,1,0,0], W=[0,0,0,0,0,0,0,1]
        # (a01)*b11 -> c13
        terms.append({
            'U': {1: 1},
            'V': {5: 1},
            'W': {7: 1}
        })
        
        # Term 10: New - U=[0,0,1,0], V=[0,0,0,0,0,1,0,1], W=[0,0,0,0,0,0,1,0]
        # (a10)*(b11+b13) -> c13
        terms.append({
            'U': {2: 1},
            'V': {5: 1, 7: 1},
            'W': {7: 1}
        })
        
        # Term 11: New - U=[0,1,0,-1], V=[0,0,1,0,0,0,1,0], W=[0,0,0,0,1,0,0,0]
        # (a01-a11)*(b02+b12) -> c02
        terms.append({
            'U': {1: 1, 3: -1},
            'V': {2: 1, 6: 1},
            'W': {4: 1}
        })
        
        return terms
    
    def solve_rank12():
        """
        Solve for the rank-12 decomposition using the structured ansatz.
        
        Each term r has a fixed sparsity pattern (which (a,b,c) it contributes to).
        The unknowns are the values at those positions.
        
        For each output entry c_ji, the constraint is:
        sum over terms r that contribute to c_ji: U_r[a_r] * V_r[b_r] * W_r[c_r] = 1 (or 0 for invalid)
        
        We set W_r[c_r] = 1 for all terms (absorbed into U and V), making the system
        quadratic in U and V. Use Newton's method.
        """
        terms = build_rank12_structure()
        rank = len(terms)
        
        # Build the system of equations
        # For each output entry (i,j), the valid constraint is:
        # C[i,j] = sum_k A[i,k] * B[k,j]
        # In tensor form: for each (a,b,c) where a=i*m+k, b=k*p+j, c=j*n+i:
        #   sum_r U[a,r]*V[b,r]*W[c,r] = 1
        # For invalid (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] = 0
        
        # With W=1 at support positions, each term r contributes U_r[a_r]*V_r[b_r] to output c_r[c_r_pos]
        # But we need to be more careful: a term can contribute to multiple outputs via its W vector
        
        # Let's enumerate all equations
        # Unknowns: for each term r, for each nonzero position in U_r, V_r (W is fixed to 1)
        
        # Actually, let me use a cleaner formulation.
        # For each term r, define:
        #   u_r = vector of U values at nonzero positions (with W=1)
        #   v_r = vector of V values at nonzero positions
        # The contribution of term r to output (i,j) is:
        #   sum over (a_r, b_r) in support of term r where c_r = j*n+i: u_r[a_r] * v_r[b_r]
        
        # This is getting complex. Let me use a direct numerical approach instead.
        
        # Direct approach: parameterize all free variables, build residual function, use Newton.
        
        # Free variables:
        # For each term r:
        #   U_r has entries at positions in terms[r]['U']
        #   V_r has entries at positions in terms[r]['V']
        #   W_r has entries at positions in terms[r]['W']
        
        # Total free variables:
        nvars = 0
        var_info = []  # (term_idx, 'U'/'V'/'W', position)
        for r, term in enumerate(terms):
            for pos in term['U']:
                var_info.append((r, 'U', pos))
                nvars += 1
            for pos in term['V']:
                var_info.append((r, 'V', pos))
                nvars += 1
            for pos in term['W']:
                var_info.append((r, 'W', pos))
                nvars += 1
        
        # Equations: for each valid (a,b,c) triple, the sum must equal 1
        # For each invalid (a,b,c), the sum must equal 0
        
        # Build the list of (a,b,c) triples that each term contributes to
        # Term r contributes to (a,b,c) if a in U_r, b in V_r, c in W_r
        # The contribution is U_r[a] * V_r[b] * W_r[c]
        
        # For each equation (a,b,c), collect which terms contribute and which variables are involved
        # This is a multilinear system - each equation is a product of 3 variables (one from each matrix)
        
        # Let me just do a direct numerical solve with all variables free.
        
        # Initialize with the hardcoded values
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for r, term in enumerate(terms):
            for pos, val in term['U'].items():
                U[pos, r] = val
            for pos, val in term['V'].items():
                V[pos, r] = val
            for pos, val in term['W'].items():
                W[pos, r] = val
        
        # Build the full variable vector
        # Variables: U.flatten(), V.flatten(), W.flatten() but only nonzero positions are free
        # For simplicity, let's use all positions as variables (some will stay 0)
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Build equations
        equations = []
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    a = i * m + k
                    b = k * p + j
                    c = j * n + i
                    equations.append((a, b, c, 1.0))
        
        # Add some invalid equations to help constrain
        # Pick a subset of invalid equations
        invalid_count = 0
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i = a // m
                    k = a % m
                    k2 = b // p
                    j = b % p
                    i2 = c % n
                    j2 = c // n
                    if i != i2 or k != k2 or j != j2:
                        if invalid_count < 40:
                            equations.append((a, b, c, 0.0))
                            invalid_count += 1
        
        def residuals(x):
            Ux = x[:n*m*rank].reshape(n*m, rank)
            Vx = x[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
            Wx = x[n*m*rank+m*p*rank:].reshape(p*n, rank)
            res = []
            for (a, b, c, target) in equations:
                val = np.sum(Ux[a] * Vx[b] * Wx[c])
                res.append(val - target)
            return np.array(res)
        
        def jacobian(x):
            Ux = x[:n*m*rank].reshape(n*m, rank)
            Vx = x[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
            Wx = x[n*m*rank+m*p*rank:].reshape(p*n, rank)
            neq = len(equations)
            nvar = x.size
            J = np.zeros((neq, nvar))
            for eq_idx, (a, b, c, target) in enumerate(equations):
                for r in range(rank):
                    # d/dr of U[a,r]*V[b,r]*W[c,r]
                    # w.r.t. U[a,r]: V[b,r]*W[c,r]
                    ju = Vx[b, r] * Wx[c, r]
                    jv = Ux[a, r] * Wx[c, r]
                    jw = Ux[a, r] * Vx[b, r]
                    J[eq_idx, a*rank+r] = ju
                    J[eq_idx, n*m*rank + b*rank+r] = jv
                    J[eq_idx, n*m*rank+m*p*rank + c*rank+r] = jw
            return J
        
        # Newton's method
        x = x0.copy()
        for iteration in range(50):
            if time.time() > deadline - 5:
                break
            res = residuals(x)
            max_res = np.max(np.abs(res))
            if max_res < 1e-14:
                break
            try:
                J = jacobian(x)
                # Solve J * dx = -res
                dx, residuals_lst, rank_solve, sv = np.linalg.lstsq(J, -res, rcond=None)
                # Damping
                step_size = 1.0
                for _ in range(20):
                    x_new = x + step_size * dx
                    res_new = residuals(x_new)
                    if np.max(np.abs(res_new)) < max_res:
                        x = x_new
                        break
                    step_size *= 0.5
                else:
                    x = x + step_size * dx
            except np.linalg.LinAlgError:
                break
        
        # Extract solution
        U_sol = x[:n*m*rank].reshape(n*m, rank)
        V_sol = x[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
        W_sol = x[n*m*rank+m*p*rank:].reshape(p*n, rank)
        
        # Clean up: zero out entries that should be zero (outside support)
        for r, term in enumerate(terms):
            for a in range(n*m):
                if a not in term['U']:
                    U_sol[a, r] = 0.0
            for b in range(m*p):
                if b not in term['V']:
                    V_sol[b, r] = 0.0
            for c in range(p*n):
                if c not in term['W']:
                    W_sol[c, r] = 0.0
        
        # Round small values
        U_sol[np.abs(U_sol) < 1e-10] = 0.0
        V_sol[np.abs(V_sol) < 1e-10] = 0.0
        W_sol[np.abs(W_sol) < 1e-10] = 0.0
        
        U_list = U_sol.tolist()
        V_list = V_sol.tolist()
        W_list = W_sol.tolist()
        
        if verify(U_list, V_list, W_list, tol=1e-8):
            return {"U": U_list, "V": V_list, "W": W_list}
        return None
    
    def try_rank11():
        """Try to find a rank-11 decomposition (unlikely but worth checking)."""
        if time.time() > deadline - 30:
            return None
        # Rank 11 would require sharing 3 products between blocks.
        # This is theoretically possible but I don't have a known construction.
        # Skip for now.
        return None
    
    # Try rank-12 first
    result12 = solve_rank12()
    if result12 is not None:
        return result12
    
    # Try rank-11
    result11 = try_rank11()
    if result11 is not None:
        return result11
    
    # Fallback to rank-14
    return make_rank14()
# EVOLVE-BLOCK-END
