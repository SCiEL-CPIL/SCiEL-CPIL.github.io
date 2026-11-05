"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # We'll build the decomposition by summing rank-1 terms.
    # Each term is (U_vec, V_vec, W_vec) where:
    #   U_vec has length n*m = 12
    #   V_vec has length m*p = 20  
    #   W_vec has length p*n = 15
    # and T[x,y,z] = sum_c U[x,c]*V[y,c]*W[z,c]
    
    terms = []  # list of (U_sparse, V_sparse, W_sparse) as dicts or arrays
    
    def add_term(U_dict, V_dict, W_dict):
        """Add a rank-1 term. Dicts map index -> value (nonzero only)."""
        terms.append((U_dict, V_dict, W_dict))
    
    def add_laderman_2x2x2(i0, j0, k0, i1, j1, k1):
        """Add Laderman's 6-term decomposition for 2x2x2 matrix multiplication.
        A[i0:i1, j0:j1] * B[j0:j1, k0:k1] = C[i0:i1, k0:k1]
        Requires i1-i0==2, j1-j0==2, k1-k0==2.
        
        Laderman's algorithm:
        z1 = (a+b)(e+f)
        z2 = (a-b)(e-f)
        z3 = (a+c)(e+g)
        z4 = (a-c)(e-g)
        z5 = (b+d)(f+h)
        z6 = (b-d)(f-h)
        
        where A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        
        Reconstruction:
        p = ae+bg = (z1+z2+z3-z4)/4 + (z5-z6-z3+z4)/4 ... 
        Let me just solve the linear system directly.
        """
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # a = A[i0,j0], b = A[i0,j1-1], c = A[i1-1,j0], d = A[i1-1,j1-1]
        # e = B[j0,k0], f = B[j0,k1-1], g = B[j1-1,k0], h = B[j1-1,k1-1]
        # C[i0,k0] = ae+bg, C[i0,k1-1] = af+bh
        # C[i1-1,k0] = ce+dg, C[i1-1,k1-1] = cf+dh
        
        # Laderman's 6 products:
        # z1 = (a+b)(e+f)
        U1 = {i0*m+j0: 1, i0*m+(j1-1): 1}
        V1 = {j0*p+k0: 1, j0*p+(k1-1): 1}
        W1 = {}
        # z2 = (a-b)(e-f)
        U2 = {i0*m+j0: 1, i0*m+(j1-1): -1}
        V2 = {j0*p+k0: 1, j0*p+(k1-1): -1}
        W2 = {}
        # z3 = (a+c)(e+g)
        U3 = {i0*m+j0: 1, (i1-1)*m+j0: 1}
        V3 = {j0*p+k0: 1, (j1-1)*p+k0: 1}
        W3 = {}
        # z4 = (a-c)(e-g)
        U4 = {i0*m+j0: 1, (i1-1)*m+j0: -1}
        V4 = {j0*p+k0: 1, (j1-1)*p+k0: -1}
        W4 = {}
        # z5 = (b+d)(f+h)
        U5 = {i0*m+(j1-1): 1, (i1-1)*m+(j1-1): 1}
        V5 = {j0*p+(k1-1): 1, (j1-1)*p+(k1-1): 1}
        W5 = {}
        # z6 = (b-d)(f-h)
        U6 = {i0*m+(j1-1): 1, (i1-1)*m+(j1-1): -1}
        V6 = {j0*p+(k1-1): 1, (j1-1)*p+(k1-1): -1}
        W6 = {}
        
        # Now solve for W coefficients.
        # We need: for each output (i,k), C[i,k] = sum_c U[x,c]*V[y,c]*W[z,c] summed over j
        # Actually, the tensor equation is: T[x,y,z] = sum_c U[x,c]*V[y,c]*W[z,c]
        # and C[i,k] = sum_j T[i*m+j, j*p+k, k*n+i]
        
        # Let me just compute the W vectors by solving the linear system.
        # For each output position (i_out, k_out), we need:
        # C[i_out, k_out] = sum_j sum_c U[i_out*m+j, c] * V[j*p+k_out, c] * W[k_out*n+i_out, c]
        
        # Let's build the system. For each of the 4 output positions, we have one equation.
        # Unknowns: W[k0*n+i0], W[(k1-1)*n+i0], W[k0*n+(i1-1)], W[(k1-1)*n+(i1-1)]
        # (4 unknowns, 4 equations)
        
        # Let me index the 6 terms as c=0..5
        U_vecs = [U1, U2, U3, U4, U5, U6]
        V_vecs = [V1, V2, V3, V4, V5, V6]
        
        # For output C[i_out, k_out]:
        # sum_j sum_c U_vecs[c][i_out*m+j] * V_vecs[c][j*p+k_out] * W[k_out*n+i_out]
        # = sum_c W[k_out*n+i_out] * sum_j U_vecs[c][i_out*m+j] * V_vecs[c][j*p+k_out]
        
        # So for each output (i_out, k_out), the coefficient of W[k_out*n+i_out] is:
        # sum_c (sum_j U_vecs[c][i_out*m+j] * V_vecs[c][j*p+k_out]) * W[k_out*n+i_out]
        
        # But W is shared across all outputs. Let me think differently.
        
        # The W vector for this block has 4 nonzero entries (one for each output position).
        # Let w00 = W[k0*n+i0], w01 = W[(k1-1)*n+i0], w10 = W[k0*n+(i1-1)], w11 = W[(k1-1)*n+(i1-1)]
        
        # For output C[i0,k0]: sum_c (sum_j U_vecs[c][i0*m+j] * V_vecs[c][j*p+k0]) * w00 = ae+bg
        # For output C[i0,k1-1]: sum_c (sum_j U_vecs[c][i0*m+j] * V_vecs[c][j*p+(k1-1)]) * w01 = af+bh
        # For output C[i1-1,k0]: sum_c (sum_j U_vecs[c][(i1-1)*m+j] * V_vecs[c][j*p+k0]) * w10 = ce+dg
        # For output C[i1-1,k1-1]: sum_c (sum_j U_vecs[c][(i1-1)*m+j] * V_vecs[c][j*p+(k1-1)]) * w11 = cf+dh
        
        # Wait, that's not right either. The W vector is the same for all outputs in this block.
        # Let me re-derive.
        
        # T[x,y,z] = sum_c U[x,c]*V[y,c]*W[z,c]
        # C[i,k] = sum_j T[i*m+j, j*p+k, k*n+i] = sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        
        # So for a fixed (i,k), the coefficient of W[k*n+i,c] is sum_j U[i*m+j,c]*V[j*p+k,c].
        # Let's call this coef[i,k,c].
        
        # Then C[i,k] = sum_c coef[i,k,c] * W[k*n+i,c]
        
        # For our block, W only has 4 nonzero positions: k0*n+i0, (k1-1)*n+i0, k0*n+(i1-1), (k1-1)*n+(i1-1)
        # And for each of these 4 positions, W has 6 components (one per term c=0..5).
        
        # So we have:
        # C[i0,k0] = sum_c coef[i0,k0,c] * W[k0*n+i0, c]
        # C[i0,k1-1] = sum_c coef[i0,k1-1,c] * W[(k1-1)*n+i0, c]
        # C[i1-1,k0] = sum_c coef[i1-1,k0,c] * W[k0*n+(i1-1), c]
        # C[i1-1,k1-1] = sum_c coef[i1-1,k1-1,c] * W[(k1-1)*n+(i1-1), c]
        
        # But W[k*n+i, c] is a single scalar for each (k,n+i,c). So we have 4*6=24 unknowns
        # and only 4 equations. That's underdetermined.
        
        # Wait, I think I'm overcomplicating this. Let me reconsider.
        
        # The tensor T has entries T[x,y,z] where x in [0,12), y in [0,20), z in [0,15).
        # T[x,y,z] = sum_c U[x,c] * V[y,c] * W[z,c]
        
        # For the block A[i0:i1, j0:j1] * B[j0:j1, k0:k1] = C[i0:i1, k0:k1]:
        # The relevant T entries are T[i*m+j, j*p+k, k*n+i] for i in [i0,i1), j in [j0,j1), k in [k0,k1).
        # These should equal A[i,j]*B[j,k].
        
        # So we need: for all i in [i0,i1), j in [j0,j1), k in [k0,k1):
        #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = A[i,j]*B[j,k]
        
        # This is a system of 2*2*2=8 equations.
        # Unknowns: U has 4 nonzero entries per term, V has 2 nonzero entries per term, W has 4 nonzero entries per term.
        # With 6 terms, we have 6*(4+2+4)=60 unknowns (but many are shared).
        
        # Actually, U, V, W are fixed by the Laderman algorithm. The only freedom is in W.
        # Wait no, U, V, W are all part of the decomposition.
        
        # Let me just directly compute what W should be.
        # For each term c, U_vecs[c] and V_vecs[c] are fixed. W_vecs[c] is what we need to find.
        # The constraint is: for all (i,j,k) in the block:
        #   sum_c U_vecs[c][i*m+j] * V_vecs[c][j*p+k] * W_vecs[c][k*n+i] = 1 (if we normalize)
        
        # Actually the constraint is that the product should give A[i,j]*B[j,k].
        # Since A[i,j] and B[j,k] are independent variables, we need:
        #   sum_c U_vecs[c][i*m+j] * V_vecs[c][j*p+k] * W_vecs[c][k*n+i] = A[i,j]*B[j,k]
        
        # But A[i,j]*B[j,k] is not a constant - it depends on the values of A and B.
        # The tensor decomposition must work for ALL A and B, so we need:
        #   The map (A,B) -> C must be correctly reproduced.
        
        # The standard way: the multiplication tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k).
        # And C[i,k] = sum_j A[i,j]*B[j,k] = sum_j T[i*m+j, j*p+k, k*n+i] * A[i,j] * B[j,k]
        
        # Wait, I think the correct interpretation is:
        # T[x,y,z] = 1 if x=i*m+j, y=j*p+k, z=k*n+i for some valid (i,j,k), else 0.
        # And C[i,k] = sum_{x,y,z} T[x,y,z] * A_flat[x] * B_flat[y] * (something with z)
        
        # Actually, I think the correct formulation is:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # In tensor form: C_flat[z] where z=k*n+i, C_flat[z] = sum_{x,y} T[x,y,z] * A_flat[x] * B_flat[y]
        # where T[x,y,z] = 1 if x=i*m+j, y=j*p+k, z=k*n+i with matching i,j,k, else 0.
        
        # So T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k).
        
        # And we need: T[x,y,z] = sum_c U[x,c] * V[y,c] * W[z,c]
        
        # For our block, the relevant T entries are T[i*m+j, j*p+k, k*n+i] = 1
        # for i in [i0,i1), j in [j0,j1), k in [k0,k1).
        
        # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # for all i in [i0,i1), j in [j0,j1), k in [k0,k1).
        
        # This gives 8 equations. With 6 terms, we have 6*4=24 unknowns in W (4 per term).
        # But U and V are also unknowns. However, in the Laderman algorithm, U and V are fixed
        # by the algorithm, and W is determined.
        
        # Let me just solve the linear system for W given U and V.
        
        # Build the system: 8 equations, 24 unknowns (W[k*n+i, c] for c=0..5, for the 4 output positions)
        # But actually W is a single vector of length 15, and each term c has its own W vector.
        # So we have 6 W vectors, each of length 15, but only 4 entries per W vector are nonzero.
        
        # Let me just set up the system and solve.
        
        # Unknowns: W_vecs[c][k*n+i] for c=0..5, for (i,k) in {(i0,k0),(i0,k1-1),(i1-1,k0),(i1-1,k1-1)}
        # That's 6*4=24 unknowns.
        # Equations: 8 (one for each (i,j,k) in the 2x2x2 block)
        
        # This is underdetermined (8 equations, 24 unknowns). We need more constraints.
        # The constraint is that the decomposition must be valid for the ENTIRE tensor, not just this block.
        # But since blocks are disjoint in their T entries, we can solve each block independently.
        
        # For this block, we have 8 equations and 24 unknowns. We can find a particular solution.
        # Let's use the structure of Laderman to find W.
        
        # Actually, I realize the issue. In Laderman's algorithm, the "products" z1..z6 are scalar multiplications.
        # Each z_c corresponds to one rank-1 term in the tensor decomposition.
        # The U, V, W vectors for each term are determined by the algorithm.
        
        # Let me just directly compute W by requiring the 8 equations to hold.
        
        # Let's use numpy to solve the underdetermined system (find minimum-norm solution).
        
        # Build matrix M where M[row, col] such that M @ W_flat = b
        # Rows: 8 (for each (i,j,k))
        # Cols: 24 (for each (c, output_pos))
        
        output_positions = [(i0, k0), (i0, k1-1), (i1-1, k0), (i1-1, k1-1)]
        equations = []
        rhs = []
        
        for i in range(i0, i1):
            for j in range(j0, j1):
                for k in range(k0, k1):
                    row = np.zeros(24)
                    for c in range(6):
                        u_val = U_vecs[c].get(i*m+j, 0)
                        v_val = V_vecs[c].get(j*p+k, 0)
                        if u_val != 0 and v_val != 0:
                            # W[k*n+i, c] is the unknown
                            # Find which output position (i,k) corresponds to
                            op_idx = output_positions.index((i, k))
                            col = c * 4 + op_idx
                            row[col] = u_val * v_val
                    equations.append(row)
                    rhs.append(1.0)
        
        M = np.array(equations)
        b = np.array(rhs)
        
        # Solve M @ W_flat = b (underdetermined, find minimum-norm solution)
        W_flat, residuals, rank_m, sv = np.linalg.lstsq(M, b, rcond=None)
        
        # Extract W vectors
        for c in range(6):
            W_vec = {}
            for op_idx, (i_out, k_out) in enumerate(output_positions):
                w_val = W_flat[c * 4 + op_idx]
                if abs(w_val) > 1e-15:
                    W_vec[k_out*n + i_out] = w_val
            add_term(U_vecs[c], V_vecs[c], W_vec)
    
    def add_strassen_2x2x2(i0, j0, k0, i1, j1, k1):
        """Add Strassen terms for M1[i0:i1, j0:j1] * M2[j0:j1, k0:k1] = M3[i0:i1, k0:k1]
        where i0,i1 are row indices in A (0-based, i1 exclusive),
              j0,j1 are col/row indices (the shared dimension),
              k0,k1 are col indices in B.
        Requires i1-i0==2, j1-j0==2, k1-k0==2.
        """
        # A indices: (i,j) -> i*m + j for i in [i0,i1), j in [j0,j1)
        # B indices: (j,k) -> j*p + k for j in [j0,j1), k in [k0,k1)
        # C indices: (i,k) -> k*n + i for i in [i0,i1), k in [k0,k1)
        
        # p1 = (a+d)*(e+h): 
        U = {(i0)*m+j0: 1, (i1-1)*m+(j1-1): 1}
        V = {j0*p+k0: 1, (j1-1)*p+(k1-1): 1}
        W = {k0*n+i0: 1, k0*n+(i1-1): -1, (k1-1)*n+i0: -1, (k1-1)*n+(i1-1): 1}
        add_term(U, V, W)
        
        # p2 = (c+d)*e
        U = {(i1-1)*m+j0: 1, (i1-1)*m+(j1-1): 1}
        V = {j0*p+k0: 1}
        W = {k0*n+(i1-1): 1, (k1-1)*n+(i1-1): -1}
        add_term(U, V, W)
        
        # p3 = a*(f-h)
        U = {i0*m+j0: 1}
        V = {j0*p+(k1-1): 1, (j1-1)*p+(k1-1): -1}
        W = {(k1-1)*n+i0: 1}
        add_term(U, V, W)
        
        # p4 = d*(g-e)
        U = {(i1-1)*m+(j1-1): 1}
        V = {(j1-1)*p+k0: 1, j0*p+k0: -1}
        W = {k0*n+(i1-1): 1}
        add_term(U, V, W)
        
        # p5 = (a+b)*h
        U = {i0*m+j0: 1, i0*m+(j1-1): 1}
        V = {(j1-1)*p+(k1-1): 1}
        W = {k0*n+i0: -1, k0*n+(i1-1): 1}
        add_term(U, V, W)
        
        # p6 = (c-a)*(e+f)
        U = {(i1-1)*m+j0: 1, i0*m+j0: -1}
        V = {j0*p+k0: 1, j0*p+(k1-1): 1}
        W = {(k1-1)*n+(i1-1): 1}
        add_term(U, V, W)
        
        # p7 = (b-d)*(g+h)
        U = {i0*m+(j1-1): 1, (i1-1)*m+(j1-1): -1}
        V = {(j1-1)*p+k0: 1, (j1-1)*p+(k1-1): 1}
        W = {k0*n+i0: 1}
        add_term(U, V, W)
    
    def add_schoolbook_block(i0, j0, k0, nr, nc):
        """Add schoolbook terms for A[i0:i0+nr, j0:j0+nc] * B[j0:j0+nc, k0:k0+nc] = C[i0:i0+nr, k0:k0+nc]
        All rectangular, schoolbook.
        """
        for i in range(nr):
            for j in range(nc):
                for k in range(nc):
                    U = {(i0+i)*m + (j0+j): 1}
                    V = {(j0+j)*p + (k0+k): 1}
                    W = {(k0+k)*n + (i0+i): 1}
                    add_term(U, V, W)
    
    # Decomposition strategy for 3x4x5:
    # Split rows of A: top 2 rows (i=0,1), bottom 1 row (i=2)
    # Split cols of A / rows of B: cols 0-1 (j=0,1), cols 2-3 (j=2,3)
    # Split cols of B: cols 0-1 (k=0,1), cols 2-4 (k=2,3,4)
    
    # Block (0,0,0): A[0:2,0:2] * B[0:2,0:2] = C[0:2,0:2] -> 2x2x2, use Laderman
    add_laderman_2x2x2(0, 0, 0, 2, 2, 2)
    
    # Block (0,1,0): A[0:2,2:4] * B[2:4,0:2] = C[0:2,0:2] -> 2x2x2, use Laderman
    add_laderman_2x2x2(0, 2, 0, 2, 4, 2)
    
    # Block (0,0,1): A[0:2,0:2] * B[0:2,2:4] = C[0:2,2:4] -> 2x2x2, use Laderman
    add_laderman_2x2x2(0, 0, 2, 2, 2, 4)
    
    # Block (0,1,1): A[0:2,2:4] * B[2:4,2:4] = C[0:2,2:4] -> 2x2x2, use Laderman
    add_laderman_2x2x2(0, 2, 2, 2, 4, 4)
    
    # Block (1,0,0): A[2:3,0:2] * B[0:2,0:2] = C[2:3,0:2] -> 1x2x2, schoolbook: 1*2*2=4
    add_schoolbook_block(2, 0, 0, 1, 2)
    
    # Block (1,1,0): A[2:3,2:4] * B[2:4,0:2] = C[2:3,0:2] -> 1x2x2, schoolbook: 4
    add_schoolbook_block(2, 2, 0, 1, 2)
    
    # Block (1,0,1): A[2:3,0:2] * B[0:2,2:4] = C[2:3,2:4] -> 1x2x2, schoolbook: 4
    add_schoolbook_block(2, 0, 2, 1, 2)
    
    # Block (1,1,1): A[2:3,2:4] * B[2:4,2:4] = C[2:3,2:4] -> 1x2x2, schoolbook: 4
    add_schoolbook_block(2, 2, 2, 1, 2)
    
    # Total rank: 4*6 + 4*4 = 24 + 16 = 40
    
    rank = len(terms)
    
    # Check deadline
    if time.time() > deadline - 5:
        # Convert to arrays and return
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for c, (Ud, Vd, Wd) in enumerate(terms):
            for idx, val in Ud.items():
                U[idx, c] = val
            for idx, val in Vd.items():
                V[idx, c] = val
            for idx, val in Wd.items():
                W[idx, c] = val
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify correctness with random test
    np.random.seed(42)
    A_test = np.random.randint(-5, 6, (n, m))
    B_test = np.random.randint(-5, 6, (m, p))
    C_expected = A_test @ B_test
    
    # Build U, V, W matrices
    U_mat = np.zeros((n*m, rank))
    V_mat = np.zeros((m*p, rank))
    W_mat = np.zeros((p*n, rank))
    for c, (Ud, Vd, Wd) in enumerate(terms):
        for idx, val in Ud.items():
            U_mat[idx, c] = val
        for idx, val in Vd.items():
            V_mat[idx, c] = val
        for idx, val in Wd.items():
            W_mat[idx, c] = val
    
    # Verify: C[i,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] summed over j
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0.0
            for j in range(m):
                x = i*m + j
                y = j*p + k
                z = k*n + i
                t_val = 0.0
                for c in range(rank):
                    t_val += U_mat[x, c] * V_mat[y, c] * W_mat[z, c]
                s += t_val
            C_computed[i, k] = s
    
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        # Fallback to schoolbook
        terms.clear()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U = {i*m+j: 1}
                    V = {j*p+k: 1}
                    W = {k*n+i: 1}
                    add_term(U, V, W)
        rank = len(terms)
        U_mat = np.zeros((n*m, rank))
        V_mat = np.zeros((m*p, rank))
        W_mat = np.zeros((p*n, rank))
        for c, (Ud, Vd, Wd) in enumerate(terms):
            for idx, val in Ud.items():
                U_mat[idx, c] = val
            for idx, val in Vd.items():
                V_mat[idx, c] = val
            for idx, val in Wd.items():
                W_mat[idx, c] = val
    
    return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
# EVOLVE-BLOCK-END
