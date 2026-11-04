import numpy as np
import time
import random

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Generate all possible 2x2x2 blocks
    I_sets = [ {0,1}, {1,2}, {0,2} ]
    J_sets = [ {j1, j2} for j1 in range(m) for j2 in range(j1+1, m) ]
    K_sets = [ {k1, k2} for k1 in range(p) for k2 in range(k1+1, p) ]
    
    all_blocks = []
    for I in I_sets:
        for J in J_sets:
            for K in K_sets:
                all_blocks.append((I, J, K))
                
    def are_disjoint(b1, b2):
        I1, J1, K1 = b1
        I2, J2, K2 = b2
        if I1 & I2:
            if (J1 & J2) and (K1 & K2):
                return False
        return True
        
    def compute_rank_for_blocks(blocks):
        """Compute the actual rank by building the decomposition and removing linearly dependent columns."""
        if not blocks:
            return n * m * p, None, None, None
        
        # Build the full decomposition
        U = np.zeros((n * m, 0), dtype=float)
        V = np.zeros((m * p, 0), dtype=float)
        W = np.zeros((p * n, 0), dtype=float)
        
        covered = set()
        
        for I, J, K in blocks:
            i0, i1 = sorted(I)
            j0, j1 = sorted(J)
            k0, k1 = sorted(K)
            
            for i in I:
                for j in J:
                    for k in K:
                        covered.add((i, j, k))
                        
            # Strassen's 7 terms
            terms = [
                # (U_entries, V_entries, W_entries)
                ([(i0*m+j0, 1), (i1*m+j1, 1)], [(j0*p+k0, 1), (j1*p+k1, 1)], [(k0*n+i0, 1), (k1*n+i1, 1)]),
                ([(i1*m+j0, 1), (i1*m+j1, 1)], [(j0*p+k0, 1)], [(k0*n+i1, 1), (k1*n+i1, -1)]),
                ([(i0*m+j0, 1)], [(j0*p+k1, 1), (j1*p+k1, -1)], [(k1*n+i0, 1), (k1*n+i1, 1)]),
                ([(i1*m+j1, 1)], [(j1*p+k0, 1), (j0*p+k0, -1)], [(k0*n+i0, 1), (k0*n+i1, 1)]),
                ([(i0*m+j0, 1), (i0*m+j1, 1)], [(j1*p+k1, 1)], [(k1*n+i0, 1), (k0*n+i0, -1)]),
                ([(i1*m+j0, 1), (i0*m+j0, -1)], [(j0*p+k0, 1), (j0*p+k1, 1)], [(k1*n+i1, 1)]),
                ([(i0*m+j1, 1), (i1*m+j1, -1)], [(j1*p+k0, 1), (j1*p+k1, 1)], [(k0*n+i0, 1)]),
            ]
            
            for u_terms, v_terms, w_terms in terms:
                u_col = np.zeros(n * m)
                v_col = np.zeros(m * p)
                w_col = np.zeros(p * n)
                for idx, val in u_terms:
                    u_col[idx] = val
                for idx, val in v_terms:
                    v_col[idx] = val
                for idx, val in w_terms:
                    w_col[idx] = val
                U = np.column_stack([U, u_col]) if U.shape[1] > 0 else u_col.reshape(-1, 1)
                V = np.column_stack([V, v_col]) if V.shape[1] > 0 else v_col.reshape(-1, 1)
                W = np.column_stack([W, w_col]) if W.shape[1] > 0 else w_col.reshape(-1, 1)
        
        # Add schoolbook terms for uncovered entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        u_col = np.zeros(n * m)
                        v_col = np.zeros(m * p)
                        w_col = np.zeros(p * n)
                        u_col[i * m + j] = 1
                        v_col[j * p + k] = 1
                        w_col[k * n + i] = 1
                        U = np.column_stack([U, u_col]) if U.shape[1] > 0 else u_col.reshape(-1, 1)
                        V = np.column_stack([V, v_col]) if V.shape[1] > 0 else v_col.reshape(-1, 1)
                        W = np.column_stack([W, w_col]) if W.shape[1] > 0 else w_col.reshape(-1, 1)
        
        # Remove linearly dependent columns using SVD
        # A column is redundant if it can be expressed as a linear combination of others
        # We do this by checking the rank of the combined system
        # For each column, check if removing it reduces the "effective rank"
        
        # Build the tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Two columns r1, r2 are linearly dependent if there exist scalars a,b not both zero
        # such that a*U[:,r1]+b*U[:,r2], a*V[:,r1]+b*V[:,r2], a*W[:,r1]+b*W[:,r2] 
        # produce a valid decomposition term (which is always true since they're just vectors)
        # The real question is whether the bilinear form is the same.
        
        # Actually, for tensor decomposition, columns are independent if the trilinear forms are independent.
        # Two terms (u1,v1,w1) and (u2,v2,w2) are linearly dependent if the bilinear forms
        # sum_i,j u[i,j]*v[j,k]*w[k,i] are proportional.
        
        # Simpler approach: check if the 3D tensor formed by the columns has rank less than the number of columns.
        # But that's expensive. Instead, we can check pairwise: two columns are dependent if
        # u1*v1*w1 and u2*v2*w2 are proportional as tensors.
        
        # For now, let's just count the columns as the rank (conservative).
        # A more sophisticated approach would be to build the full tensor and compute its rank,
        # but that's O(n*m*p*rank^3) which might be too slow.
        
        rank = U.shape[1]
        return rank, U, V, W
    
    best_blocks = []
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Greedy search with random restarts
    num_restarts = 100
    for _ in range(num_restarts):
        if time.time() > deadline - 10:
            break
        random.shuffle(all_blocks)
        current_blocks = []
        for b in all_blocks:
            if all(are_disjoint(b, cb) for cb in current_blocks):
                current_blocks.append(b)
        
        rank, U, V, W = compute_rank_for_blocks(current_blocks)
        if rank < best_rank:
            best_rank = rank
            best_blocks = current_blocks
            best_U, best_V, best_W = U, V, W
            if best_rank <= 95:
                break
                
    # If we found a good solution, use it
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback: no Strassen blocks, just schoolbook
    U = np.zeros((n * m, n * m * p), dtype=float)
    V = np.zeros((m * p, n * m * p), dtype=float)
    W = np.zeros((p * n, n * m * p), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
