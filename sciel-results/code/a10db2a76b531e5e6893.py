import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Partitions
    n_parts = [2, 1, 2]
    m_parts = [2, 1, 2]
    p_parts = [2, 1, 2]
    
    n_starts = [0]
    m_starts = [0]
    p_starts = [0]
    for i in range(1, len(n_parts)):
        n_starts.append(n_starts[-1] + n_parts[i-1])
    for i in range(1, len(m_parts)):
        m_starts.append(m_starts[-1] + m_parts[i-1])
    for i in range(1, len(p_parts)):
        p_starts.append(p_starts[-1] + p_parts[i-1])
        
    # Pre-allocate lists for terms to allow for potential merging
    # Each term is (u_vec, v_vec, w_vec) as numpy arrays
    terms = []
    
    # Helper to add a term
    def add_term(u, v, w):
        # u: index in n*m
        # v: index in m*p
        # w: index in p*n
        u_vec = np.zeros(n*m)
        v_vec = np.zeros(m*p)
        w_vec = np.zeros(p*n)
        u_vec[u] = 1
        v_vec[v] = 1
        w_vec[w] = 1
        terms.append((u_vec, v_vec, w_vec))

    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni = n_parts[i]
                mj = m_parts[j]
                pk = p_parts[k]
                
                i_start = n_starts[i]
                j_start = m_starts[j]
                k_start = p_starts[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    # Strassen 2x2x2
                    i0, i1 = i_start, i_start + 1
                    j0, j1 = j_start, j_start + 1
                    k0, k1 = k_start, k_start + 1
                    
                    # M1
                    u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
                    u[i0*m+j0]=1; u[i1*m+j1]=1
                    v[j0*p+k0]=1; v[j1*p+k1]=1
                    w[k0*n+i0]=1; w[k1*n+i1]=1
                    terms.append((u,v,w))
                    
                    # M2
                    u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
                    u[i1*m+j0]=1; u[i1*m+j1]=1
                    v[j0*p+k0]=1
                    w[k0*n+i1]=1; w[k1*n+i1]=-1
                    terms.append((u,v,w))
                    
                    # M3
                    u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
                    u[i0*m+j0]=1
                    v[j0*p+k1]=1; v[j1*p+k1]=-1
                    w[k1*n+i0]=1; w[k1*n+i1]=1
                    terms.append((u,v,w))
                    
                    # M4
                    u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
                    u[i1*m+j1]=1
                    v[j1*p+k0]=1; v[j0*p+k0]=-1
                    w[k0*n+i0]=1; w[k0*n+i1]=1
                    terms.append((u,v,w))
                    
                    # M5
                    u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
                    u[i0*m+j0]=1; u[i0*m+j1]=1
                    v[j1*p+k1]=1
                    w[k0*n+i0]=-1; w[k1*n+i0]=1
                    terms.append((u,v,w))
                    
                    # M6
                    u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
                    u[i1*m+j0]=1; u[i0*m+j0]=-1
                    v[j0*p+k0]=1; v[j0*p+k1]=1
                    w[k1*n+i1]=1
                    terms.append((u,v,w))
                    
                    # M7
                    u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
                    u[i0*m+j1]=1; u[i1*m+j1]=-1
                    v[j1*p+k0]=1; v[j1*p+k1]=1
                    w[k0*n+i0]=1
                    terms.append((u,v,w))
                    
                else:
                    # Schoolbook
                    for u_idx in range(ni):
                        for v_idx in range(mj):
                            for w_idx in range(pk):
                                u_row = (i_start + u_idx) * m + (j_start + v_idx)
                                v_row = (j_start + v_idx) * p + (k_start + w_idx)
                                w_row = (k_start + w_idx) * n + (i_start + u_idx)
                                add_term(u_row, v_row, w_row)
    
    # Attempt Greedy Merge
    # Check if any two terms sum to a rank-1 tensor
    # T = u1 v1^T w1 + u2 v2^T w2
    # Rank 1 iff u1, u2 collinear AND v1, v2 collinear AND w1, w2 collinear
    # We check this by looking at the supports.
    
    merged_terms = []
    used = [False] * len(terms)
    
    # Optimization: Only check pairs where supports overlap significantly or are simple
    # For 71 terms, O(N^2) is 71^2/2 ~ 2500 checks. Fast.
    
    for i in range(len(terms)):
        if used[i]:
            continue
        u1, v1, w1 = terms[i]
        
        best_j = -1
        best_weight = 0
        best_sum = None
        
        for j in range(i+1, len(terms)):
            if used[j]:
                continue
            u2, v2, w2 = terms[j]
            
            # Check collinearity
            # u1 + u2 must be rank 1 (it is a vector, so always rank 1 unless zero)
            # The condition for the SUM T1+T2 to be rank 1 is:
            # The matrix [u1, u2] must have rank 1? No.
            # T = u1 \otimes v1 \otimes w1 + u2 \otimes v2 \otimes w2
            # If u1, u2 are linearly independent, the rank of the unfolding U is 2, so tensor rank >= 2.
            # So u1 and u2 MUST be collinear.
            # Similarly v1, v2 collinear.
            # And w1, w2 collinear.
            
            # Check u
            if not np.allclose(u1, 0) and not np.allclose(u2, 0):
                # Check if u1 is a scalar multiple of u2
                # Since entries are 1, -1, 0, and supports might differ
                # If supports differ, they are not collinear (unless one is zero, which isn't the case for single term)
                # Actually, if support(u1) != support(u2), they are linearly independent.
                if not np.array_equal(np.nonzero(u1)[0], np.nonzero(u2)[0]):
                    continue
                # If supports are same, check if values are proportional
                # Since entries are +-1, they must be equal or negated
                if not np.allclose(u1, u2) and not np.allclose(u1, -u2):
                    continue
            elif np.allclose(u1, 0) and np.allclose(u2, 0):
                pass # Both zero u? Impossible for valid term
            else:
                continue # One zero, one non-zero -> independent
            
            # Check v
            if not np.array_equal(np.nonzero(v1)[0], np.nonzero(v2)[0]):
                continue
            if not np.allclose(v1, v2) and not np.allclose(v1, -v2):
                continue
                
            # Check w
            if not np.array_equal(np.nonzero(w1)[0], np.nonzero(w2)[0]):
                continue
            if not np.allclose(w1, w2) and not np.allclose(w1, -w2):
                continue
            
            # If we get here, they are collinear in all dimensions.
            # The sum is a single rank-1 term.
            # u_sum = u1 + u2
            # v_sum = v1 + v2
            # w_sum = w1 + w2
            # We replace terms[i] and terms[j] with (u_sum, v_sum, w_sum)
            
            u_sum = u1 + u2
            v_sum = v1 + v2
            w_sum = w1 + w2
            
            # Check if sum is zero (cancellation)
            if np.allclose(u_sum, 0) or np.allclose(v_sum, 0) or np.allclose(w_sum, 0):
                # Both terms cancel out. Remove both.
                used[i] = True
                used[j] = True
                # Do not add to merged_terms
                break
            else:
                # Add the combined term
                # We modify terms[i] to be the sum, and mark j as used.
                # Then we continue with the modified term i in the next iteration?
                # Simpler: Add to a temp list and handle carefully.
                # For now, just record the merge.
                merged_terms.append((u_sum, v_sum, w_sum))
                used[i] = True
                used[j] = True
                break # Merge one pair per i, restart loop? 
                # Greedy: pick first valid merge, add it, mark used, restart search for next.
        
        # If no merge found for i, keep term i
        if not used[i]:
            merged_terms.append(terms[i])
            used[i] = True
            
    # Build final U, V, W matrices
    rank = len(merged_terms)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for t, (u, v, w) in enumerate(merged_terms):
        U[:, t] = u
        V[:, t] = v
        W[:, t] = w
        
    return {"U": U, "V": V, "W": W}

