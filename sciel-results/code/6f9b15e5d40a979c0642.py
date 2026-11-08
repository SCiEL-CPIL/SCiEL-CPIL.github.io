import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    Uses a randomized linear algebra approach to search for a rank-minimal 
    decomposition with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    deadline = time.time() + 115
    
    # Target ranks to try, starting from lower bounds
    # Lower bound for 4x4x5 is roughly (4*4*5)/2 = 40?
    # Known: 4x4x4 is 49. 4x4x5 might be around 60.
    # We try a range of ranks.
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def block_strassen_fallback():
        """Fallback to the known 65-rank decomposition."""
        # Re-use the logic from the parent if needed, but simplified for fallback
        # For now, just return schoolbook if search fails, or a simple 65-rank construction
        # The parent program has a 65-rank construction. We will implement a simplified version
        # of the 4x4x4 Strassen + 4x4x1 direct.
        
        def strassen_2x2(a_blocks, b_blocks, c_map):
            # Simplified Strassen for 2x2 blocks
            # a_blocks, b_blocks: lists of (pos_dict, sign)
            # pos_dict: {(local_r, local_c): global_idx}
            
            a_coeff = {(0,0):{}, (0,1):{}, (1,0):{}, (1,1):{}}
            for pos, sign in a_blocks:
                for local, gidx in pos.items():
                    a_coeff[local][gidx] = a_coeff[local].get(gidx, 0) + sign
            
            b_coeff = {(0,0):{}, (0,1):{}, (1,0):{}, (1,1):{}}
            for pos, sign in b_blocks:
                for local, gidx in pos.items():
                    b_coeff[local][gidx] = b_coeff[local].get(gidx, 0) + sign
            
            a00, a01, a10, a11 = a_coeff[(0,0)], a_coeff[(0,1)], a_coeff[(1,0)], a_coeff[(1,1)]
            b00, b01, b10, b11 = b_coeff[(0,0)], b_coeff[(0,1)], b_coeff[(1,0)], b_coeff[(1,1)]
            
            terms = []
            
            def make_w(local_pos, sign):
                if local_pos not in c_map:
                    return {}
                c_block, c_sign = c_map[local_pos]
                w = {}
                for gidx in c_block.values():
                    w[gidx] = c_sign * sign
                return w
            
            # Q1
            u1 = {}
            for d in [a00, a11]:
                for k, v in d.items(): u1[k] = u1.get(k, 0) + v
            v1 = {}
            for d in [b00, b11]:
                for k, v in d.items(): v1[k] = v1.get(k, 0) + v
            w1 = {}
            for local, s in [((0,0), 1), ((1,1), 1)]:
                for k, v in make_w(local, s).items(): w1[k] = w1.get(k, 0) + v
            if w1: terms.append((u1, v1, w1))
            
            # Q2
            u2 = {}
            for d in [a10, a11]:
                for k, v in d.items(): u2[k] = u2.get(k, 0) + v
            v2 = dict(b00)
            w2 = {}
            for local, s in [((1,0), 1), ((1,1), -1)]:
                for k, v in make_w(local, s).items(): w2[k] = w2.get(k, 0) + v
            if w2: terms.append((u2, v2, w2))
            
            # Q3
            u3 = dict(a00)
            v3 = {}
            for d, s in [(b01, 1), (b11, -1)]:
                for k, v in d.items(): v3[k] = v3.get(k, 0) + s * v
            w3 = {}
            for local, s in [((0,1), 1), ((1,1), 1)]:
                for k, v in make_w(local, s).items(): w3[k] = w3.get(k, 0) + v
            if w3: terms.append((u3, v3, w3))
            
            # Q4
            u4 = dict(a11)
            v4 = {}
            for d, s in [(b10, 1), (b00, -1)]:
                for k, v in d.items(): v4[k] = v4.get(k, 0) + s * v
            w4 = {}
            for local, s in [((0,0), 1), ((1,0), 1)]:
                for k, v in make_w(local, s).items(): w4[k] = w4.get(k, 0) + v
            if w4: terms.append((u4, v4, w4))
            
            # Q5
            u5 = {}
            for d in [a00, a01]:
                for k, v in d.items(): u5[k] = u5.get(k, 0) + v
            v5 = dict(b11)
            w5 = {}
            for local, s in [((0,0), -1), ((0,1), 1)]:
                for k, v in make_w(local, s).items(): w5[k] = w5.get(k, 0) + v
            if w5: terms.append((u5, v5, w5))
            
            # Q6
            u6 = {}
            for d, s in [(a10, 1), (a00, -1)]:
                for k, v in d.items(): u6[k] = u6.get(k, 0) + s * v
            v6 = {}
            for d in [b00, b01]:
                for k, v in d.items(): v6[k] = v6.get(k, 0) + v
            w6 = {}
            for local, s in [((1,1), 1)]:
                for k, v in make_w(local, s).items(): w6[k] = w6.get(k, 0) + v
            if w6: terms.append((u6, v6, w6))
            
            # Q7
            u7 = {}
            for d, s in [(a01, 1), (a11, -1)]:
                for k, v in d.items(): u7[k] = u7.get(k, 0) + s * v
            v7 = {}
            for d in [b10, b11]:
                for k, v in d.items(): v7[k] = v7.get(k, 0) + v
            w7 = {}
            for local, s in [((0,0), 1)]:
                for k, v in make_w(local, s).items(): w7[k] = w7.get(k, 0) + v
            if w7: terms.append((u7, v7, w7))
            
            return terms
        
        def bpos(base_i, base_j):
            return {(0,0):(base_i,base_j), (0,1):(base_i,base_j+1),
                    (1,0):(base_i+1,base_j), (1,1):(base_i+1,base_j+1)}
        
        A11, A12 = bpos(0,0), bpos(0,2)
        A21, A22 = bpos(2,0), bpos(2,2)
        B11, B12 = bpos(0,0), bpos(0,2)
        B21, B22 = bpos(2,0), bpos(2,2)
        C11, C12 = bpos(0,0), bpos(0,2)
        C21, C22 = bpos(2,0), bpos(2,2)
        
        terms = []
        terms += strassen_2x2([(A11,1),(A22,1)], [(B11,1),(B22,1)], {(0,0):(C11,1), (1,1):(C22,1)})
        terms += strassen_2x2([(A21,1),(A22,1)], [(B11,1)], {(1,0):(C21,1), (1,1):(C22,-1)})
        terms += strassen_2x2([(A11,1)], [(B12,1),(B22,-1)], {(0,1):(C12,1), (1,1):(C22,1)})
        terms += strassen_2x2([(A22,1)], [(B21,1),(B11,-1)], {(0,0):(C11,1), (1,0):(C21,1)})
        terms += strassen_2x2([(A11,1),(A12,1)], [(B22,1)], {(0,0):(C11,-1), (0,1):(C12,1)})
        terms += strassen_2x2([(A21,1),(A11,-1)], [(B11,1),(B12,1)], {(1,1):(C22,1)})
        terms += strassen_2x2([(A22,1),(A21,-1)], [(B21,1),(B22,1)], {(0,0):(C11,1)})
        
        # Direct part for 5th column
        for i in range(4):
            for j in range(4):
                u = {(i,j): 1}
                v = {(j,4): 1}
                w = {(4,i): 1}
                terms.append((u, v, w))
        
        rank = len(terms)
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        for col, (u, v, w) in enumerate(terms):
            for (i,j), coeff in u.items():
                U[i*m+j][col] = coeff
            for (j,k), coeff in v.items():
                V[j*p+k][col] = coeff
            for (k,i), coeff in w.items():
                W[k*n+i][col] = coeff
        
        return {"U": U, "V": V, "W": W}

    def verify(result):
        U = np.array(result["U"], dtype=float)
        V = np.array(result["V"], dtype=float)
        W = np.array(result["W"], dtype=float)
        rank = U.shape[1]
        
        # Check diagonal terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m+j
                    b = j*p+k
                    c = k*n+i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        
        # Check off-diagonal terms (sample)
        for _ in range(1000):
            i1 = np.random.randint(n)
            j1 = np.random.randint(m)
            j2 = np.random.randint(m)
            k2 = np.random.randint(p)
            i3 = np.random.randint(n)
            k3 = np.random.randint(p)
            if not (i1==i3 and j1==j2 and k2==k3):
                a = i1*m+j1
                b = j2*p+k2
                c = k3*n+i3
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val) > 1e-9:
                    return False
        return True

    def search_rank(R, num_trials=200):
        """Search for a rank-R decomposition using random U, V."""
        # U shape: (n*m, R)
        # V shape: (m*p, R)
        # W shape: (p*n, R)
        
        # We want integer/half-integer entries.
        # We will generate U, V with entries in {-1, 0, 1}
        # Then solve for W.
        
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Generate random sparse U, V
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            
            # Density 0.3
            for r in range(R):
                # Pick random non-zero entries for U
                idx_u = np.random.choice(n*m, size=int(n*m*0.3), replace=False)
                U[idx_u, r] = np.random.choice([-1, 1], size=len(idx_u))
                
                idx_v = np.random.choice(m*p, size=int(m*p*0.3), replace=False)
                V[idx_v, r] = np.random.choice([-1, 1], size=len(idx_v))
            
            # Build linear system for W
            # Equation: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            # Let A be the matrix where rows correspond to (i,j,k) and columns to (k,i,r)
            # But W[k,i,r] is the variable.
            
            # We have 80 equations (i,j,k)
            # We have 20*R unknowns (k,i,r)
            
            # Let's form the system M * w_vec = t_vec
            # M[row_idx, col_idx] = U[i,j,r] * V[j,k,r]
            # where row_idx corresponds to (i,j,k) and col_idx to (k,i,r)
            
            M = np.zeros((n*m*p, p*n*R))
            t_vec = np.zeros(n*m*p)
            
            row = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        t_vec[row] = 1.0
                        # For this (i,j,k), the equation is:
                        # 1 = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                        # The variable W[k,i,r] has index: (k*n + i) * R + r
                        for r in range(R):
                            coeff = U[i*m+j, r] * V[j*p+k, r]
                            if coeff != 0:
                                col = (k*n + i) * R + r
                                M[row, col] = coeff
                        row += 1
            
            # Solve M * w_vec = t_vec
            # Use least squares or pseudo-inverse
            try:
                w_vec, residuals, rank_mat, sv = np.linalg.lstsq(M, t_vec, rcond=None)
            except Exception:
                continue
            
            # Check if solution is exact
            if np.linalg.norm(M @ w_vec - t_vec) > 1e-9:
                continue
            
            # Reshape w_vec to W
            W = w_vec.reshape(p*n, R)
            
            # Check if W has integer/half-integer entries
            # We round to nearest 0.5
            W_rounded = np.round(W * 2) / 2
            
            # Verify with rounded W
            if not verify({"U": U.tolist(), "V": V.tolist(), "W": W_rounded.tolist()}):
                continue
            
            # If we get here, we have a valid decomposition
            return {"U": U.tolist(), "V": V.tolist(), "W": W_rounded.tolist()}
        
        return None

    # Try ranks from 50 to 64
    for R in range(50, 65):
        if time.time() > deadline - 10:
            break
        result = search_rank(R, num_trials=50)
        if result:
            return result
    
    # Fallback to known 65-rank
    return block_strassen_fallback()

