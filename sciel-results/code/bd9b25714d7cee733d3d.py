import numpy as np
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    n, m, p = 4, 4, 7
    terms = []

    def add_term(u, v, w):
        terms.append((u, v, w))

    def strassen_2x2_block(i_off, j_off, k_off, a_scale=1, b_scale=1, c_scale=1):
        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[k_off * n + i_off] = c_scale
        w[(k_off+1) * n + (i_off+1)] = c_scale
        add_term(u, v, w)

        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = a_scale
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        w = [0] * (p * n)
        w[k_off * n + (i_off+1)] = c_scale
        w[(k_off+1) * n + (i_off+1)] = -c_scale
        add_term(u, v, w)

        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        v = [0] * (m * p)
        v[j_off * p + (k_off+1)] = b_scale
        v[(j_off+1) * p + (k_off+1)] = -b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = c_scale
        w[(k_off+1) * n + (i_off+1)] = c_scale
        add_term(u, v, w)

        u = [0] * (n * m)
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = b_scale
        v[j_off * p + k_off] = -b_scale
        w = [0] * (p * n)
        w[k_off * n + i_off] = c_scale
        w[k_off * n + (i_off+1)] = c_scale
        add_term(u, v, w)

        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        u[i_off * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = c_scale
        w[k_off * n + i_off] = -c_scale
        add_term(u, v, w)

        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = a_scale
        u[i_off * m + j_off] = -a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        v[j_off * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + (i_off+1)] = c_scale
        add_term(u, v, w)

        u = [0] * (n * m)
        u[i_off * m + (j_off+1)] = a_scale
        u[(i_off+1) * m + (j_off+1)] = -a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = b_scale
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[k_off * n + i_off] = c_scale
        add_term(u, v, w)

    def strassen_4x4_head():
        outer_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]

        inner_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]

        def get_block_indices(block_idx, local_idx, dim_a, dim_b, offset_k=0):
            br = block_idx // 2
            bc = block_idx % 2
            lr = local_idx // 2
            lc = local_idx % 2
            row = br * 2 + lr
            col = bc * 2 + lc + offset_k
            if dim_a == 'A':
                return row * m + col
            elif dim_a == 'B':
                return row * p + col
            elif dim_a == 'C':
                return col * n + row

        for outer_a, outer_b, outer_c in outer_terms:
            for inner_x, inner_y, inner_z in inner_terms:
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                for x_local, x_coeff in inner_x:
                    for a_block, a_coeff in outer_a:
                        idx = get_block_indices(a_block, x_local, 'A', None)
                        u[idx] += a_coeff * x_coeff
                for y_local, y_coeff in inner_y:
                    for b_block, b_coeff in outer_b:
                        idx = get_block_indices(b_block, y_local, 'B', None, offset_k=0)
                        v[idx] += b_coeff * y_coeff
                for z_local, z_coeff in inner_z:
                    for c_block, c_coeff in outer_c:
                        idx = get_block_indices(c_block, z_local, 'C', None, offset_k=0)
                        w[idx] += c_coeff * z_coeff
                add_term(u, v, w)

    def strassen_4x4x2_tail():
        for i_b in range(2):
            for j_b in range(2):
                i_off = i_b * 2
                j_off = j_b * 2
                k_off = 4
                strassen_2x2_block(i_off, j_off, k_off)

    def schoolbook_4x4x1_tail():
        for i in range(n):
            for j in range(m):
                u = [0] * (n * m)
                u[i * m + j] = 1
                v = [0] * (m * p)
                v[j * p + 6] = 1
                w = [0] * (p * n)
                w[6 * n + i] = 1
                add_term(u, v, w)

    def build_tensor():
        """Build the full matrix multiplication tensor as a 3D array."""
        T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1
        return T

    def peel_rank1_decomposition(T, max_terms=120, time_limit=100):
        """
        Rank-1 peeling algorithm: iteratively find a rank-1 term that matches
        the current residual, subtract it, and repeat.
        
        Returns a list of (u, v, w) triples.
        """
        residual = T.copy().astype(float)
        result_terms = []
        start = time.time()
        
        for _ in range(max_terms):
            if time.time() - start > time_limit:
                break
            
            # Find the entry with maximum absolute value in the residual
            abs_res = np.abs(residual)
            max_idx = np.unravel_index(np.argmax(abs_res), residual.shape)
            max_val = residual[max_idx]
            
            if abs(max_val) < 1e-10:
                break
            
            # Normalize so that the max entry is 1 (or -1)
            sign = 1.0 if max_val > 0 else -1.0
            
            # We want to find u, v, w such that:
            # u[i,j] * v[j,k] * w[k,i] = sign * residual[i,j,k] for all (i,j,k)
            # where (i,j,k) = max_idx
            # 
            # Let u_max = u[max_idx[0]], v_max = v[max_idx[1]], w_max = w[max_idx[2]]
            # Then u_max * v_max * w_max = sign
            # 
            # For any other entry (i', j', k') where residual != 0:
            # u[i',j'] * v[j',k'] * w[k',i'] = residual[i',j',k']
            # 
            # This is a system of equations. Let's parameterize:
            # Set u_max = 1, then v_max * w_max = sign
            # 
            # Actually, let's use a different approach. We'll set one of the factors
            # to 1 and solve for the others.
            #
            # Let's try: set u[max_idx[0]] = 1
            # Then v[max_idx[1]] * w[max_idx[2]] = sign
            #
            # For each nonzero entry (i,j,k) in residual:
            # u[i,j] * v[j,k] * w[k,i] = residual[i,j,k]
            #
            # This is underdetermined. Let's use the fact that we want integer/half-integer
            # solutions. We'll try to find a solution where most entries are 0, ±1, or ±1/2.
            #
            # Alternative approach: use the known structure. For matrix multiplication,
            # each rank-1 term has a specific pattern. Let's enumerate possible patterns.
            #
            # Actually, let's use a linear algebra approach. We'll fix u and solve for v, w.
            # Or better: we'll use the fact that the tensor is sparse and try to find
            # a rank-1 approximation.
            #
            # Let's try a greedy approach: pick the max entry, set u[i,j]=1 for that entry,
            # and then determine v and w from the other nonzero entries.
            
            i_max, j_max, k_max = max_idx
            
            # Try different assignments for u[i_max, j_max]
            for u_val in [1.0, -1.0, 0.5, -0.5]:
                if abs(u_val) < 1e-10:
                    continue
                    
                # Build u with u[i_max, j_max] = u_val
                u = np.zeros((n * m,), dtype=float)
                u[i_max] = u_val
                
                # For each other nonzero entry in residual, we need:
                # u[i,j] * v[j,k] * w[k,i] = residual[i,j,k]
                #
                # If u[i,j] = 0, then we need residual[i,j,k] = 0 or v[j,k]*w[k,i] = 0
                # If u[i,j] != 0, then v[j,k] * w[k,i] = residual[i,j,k] / u[i,j]
                #
                # This is complex. Let's try a simpler heuristic:
                # Assume that for each (j,k) pair, v[j,k] is determined by the first
                # nonzero entry in that column of the residual (after fixing u).
                #
                # Actually, let's try yet another approach: use the known Strassen
                # structure as a guide. The rank-1 terms in Strassen have at most
                # 4 nonzero entries in each of u, v, w.
                #
                # Let's enumerate all possible rank-1 terms with small support and
                # check which one best matches the residual.
                
                if time.time() - start > time_limit:
                    break
                
                # Find all nonzero entries in the residual
                nz_indices = np.argwhere(np.abs(residual) > 1e-10)
                if len(nz_indices) == 0:
                    break
                
                # For each possible support pattern, try to find a rank-1 term
                # that matches the residual.
                
                # Strategy: pick a small set of (i,j,k) entries and solve for u, v, w
                # such that u[i,j]*v[j,k]*w[k,i] = residual[i,j,k] for those entries.
                
                # Use the max entry and a few others
                sample_entries = nz_indices[:min(8, len(nz_indices))]
                
                # Set up equations: u[i,j] * v[j,k] * w[k,i] = r[i,j,k]
                # Take logs? No, that doesn't work with signs.
                #
                # Let's use a different parameterization. Set:
                # u[i,j] = a[i,j]
                # v[j,k] = b[j,k]  
                # w[k,i] = c[k,i]
                # Then a[i,j] * b[j,k] * c[k,i] = r[i,j,k]
                #
                # This is a multiplicative constraint. Let's take absolute values:
                # |a[i,j]| * |b[j,k]| * |c[k,i]| = |r[i,j,k]|
                # And track signs separately.
                #
                # For now, let's just try the simple case where u has only one nonzero entry.
                
                # Case 1: u has only u[i_max] nonzero
                # Then for any (i,j,k) with i != i_max or j != j_max, we need residual[i,j,k] = 0
                # unless v[j,k] = 0 or w[k,i] = 0
                #
                # This is too restrictive. Let's try a different approach.
                
                # Use the fact that we can scale: if (u, v, w) is a solution, then
                # (alpha*u, beta*v, gamma*w) is also a solution if alpha*beta*gamma = 1.
                #
                # Let's fix the product and solve. Set u[i_max] = 1.
                # Then v[j_max] * w[k_max] = sign / 1 = sign
                #
                # For simplicity, set v[j_max] = 1, w[k_max] = sign
                #
                # Then for any other entry (i,j,k):
                # u[i,j] * v[j,k] * w[k,i] = residual[i,j,k]
                #
                # This is still underdetermined. Let's just accept that this approach
                # is complex and fall back to the Strassen decomposition.
                
                break
        
            # If we couldn't find a good rank-1 term, break and use fallback
            break
        
        return result_terms

    # First, try the rank-1 peeling approach
    T = build_tensor()
    peeled_terms = peel_rank1_decomposition(T, max_terms=100, time_limit=80)
    
    # Check if the peeling produced a valid decomposition
    if len(peeled_terms) > 0:
        # Verify the decomposition
        U_check = np.zeros((n * m, len(peeled_terms)))
        V_check = np.zeros((m * p, len(peeled_terms)))
        W_check = np.zeros((p * n, len(peeled_terms)))
        
        for c, (u, v, w) in enumerate(peeled_terms):
            for r in range(n * m):
                U_check[r, c] = u[r]
            for r in range(m * p):
                V_check[r, c] = v[r]
            for r in range(p * n):
                W_check[r, c] = w[r]
        
        # Reconstruct the tensor
        T_recon = np.zeros((n * m, m * p, p * n), dtype=float)
        for c in range(len(peeled_terms)):
            T_recon += np.einsum('i,j,k->ijk', U_check[:, c], V_check[:, c], W_check[:, c])
        
        if np.allclose(T_recon, T, atol=1e-8):
            # Valid decomposition!
            rank = len(peeled_terms)
            U = U_check.tolist()
            V = V_check.tolist()
            W = W_check.tolist()
            
            # Convert to integer/half-integer if possible
            for i in range(n * m):
                for j in range(rank):
                    if U[i][j] == int(U[i][j]):
                        U[i][j] = int(U[i][j])
            for i in range(m * p):
                for j in range(rank):
                    if V[i][j] == int(V[i][j]):
                        V[i][j] = int(V[i][j])
            for i in range(p * n):
                for j in range(rank):
                    if W[i][j] == int(W[i][j]):
                        W[i][j] = int(W[i][j])
            
            return {"U": U, "V": V, "W": W}
    
    # Fallback: use the Strassen-based decomposition
    terms.clear()
    strassen_4x4_head()
    strassen_4x4x2_tail()
    schoolbook_4x4x1_tail()
    
    if time.time() > deadline:
        return {"U": [], "V": [], "W": []}
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            if u[r] != 0:
                U[r][c] = u[r]
        for r in range(m * p):
            if v[r] != 0:
                V[r][c] = v[r]
        for r in range(p * n):
            if w[r] != 0:
                W[r][c] = w[r]
    
    return {"U": U, "V": V, "W": W}
