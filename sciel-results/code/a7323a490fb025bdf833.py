import numpy as np

def construct():
    n, m, p = 2, 4, 8
    
    # Generate the 56 Strassen terms
    U_list = []
    V_list = []
    W_list = []
    
    strassen_terms = [
        {'x': {(0,0):1, (1,1):1}, 'y': {(0,0):1, (1,1):1}, 'z': {(0,0):1, (1,1):1}},
        {'x': {(1,0):1, (1,1):1}, 'y': {(0,0):1}, 'z': {(1,0):1, (1,1):-1}},
        {'x': {(0,0):1}, 'y': {(0,1):1, (1,1):-1}, 'z': {(0,1):1, (1,1):1}},
        {'x': {(1,1):1}, 'y': {(1,0):1, (0,0):-1}, 'z': {(0,0):1, (1,0):1}},
        {'x': {(0,0):1, (0,1):1}, 'y': {(1,1):1}, 'z': {(0,0):-1, (0,1):1}},
        {'x': {(1,0):1, (0,0):-1}, 'y': {(0,0):1, (0,1):1}, 'z': {(1,1):1}},
        {'x': {(0,1):1, (1,1):-1}, 'y': {(1,0):1, (1,1):1}, 'z': {(0,0):1}},
    ]
    
    for a in range(2):
        for b in range(4):
            for t in range(7):
                term = strassen_terms[t]
                u_vec = np.zeros(n*m)
                v_vec = np.zeros(m*p)
                w_vec = np.zeros(p*n)
                
                for (i_prime, j_prime), coeff in term['x'].items():
                    u_idx = i_prime * m + (2*a + j_prime)
                    u_vec[u_idx] += coeff
                
                for (j_prime, k_prime), coeff in term['y'].items():
                    v_idx = (2*a + j_prime) * p + (2*b + k_prime)
                    v_vec[v_idx] += coeff
                
                for (i_prime, k_prime), coeff in term['z'].items():
                    w_idx = (2*b + k_prime) * n + i_prime
                    w_vec[w_idx] += coeff
                
                U_list.append(u_vec)
                V_list.append(v_vec)
                W_list.append(w_vec)
    
    U_arr = np.array(U_list).T # (8, 56)
    V_arr = np.array(V_list).T # (32, 56)
    W_arr = np.array(W_list).T # (16, 56)
    
    # Flatten to check for linear dependence
    # Tensor T = sum u_i v_i w_i
    # Vector representation: vec(T) = sum (u_i \otimes v_i \otimes w_i)
    
    # To check rank of the set of rank-1 tensors, we can check if the vectors
    # u_i \otimes v_i \otimes w_i are linearly independent.
    
    # However, checking linear independence of 56 vectors in R^4096 is cheap.
    
    # Construct the matrix of flattened tensors
    # M = [ vec(u_0 \otimes v_0 \otimes w_0), ..., vec(u_55 \otimes v_55 \otimes w_55) ]
    
    # Instead of constructing the huge matrix, we can use the fact that
    # if there is a linear dependence, there exists non-zero c such that
    # sum c_i (u_i \otimes v_i \otimes w_i) = 0.
    
    # This is equivalent to the rank of the 56 vectors being < 56.
    
    # We can check this by constructing the matrix K where K_{k, i} = <e_k, u_i \otimes v_i \otimes w_i>
    # But we can just use the SVD of the matrix formed by the 56 vectors in the tensor space.
    
    # Let's build the matrix of 56 vectors of length 4096.
    # This is 56 * 4096 floats ~ 230k floats, very small.
    
    T_vecs = []
    for i in range(56):
        t_vec = np.outer(np.outer(U_arr[:, i], V_arr[:, i]), W_arr[:, i]).flatten()
        T_vecs.append(t_vec)
    
    T_mat = np.array(T_vecs).T # (4096, 56)
    
    # Compute SVD
    try:
        U_svd, s, Vh = np.linalg.svd(T_mat, full_matrices=False)
        # Rank is the number of singular values > tolerance
        tol = 1e-9
        rank = np.sum(s > tol)
        
        if rank < 56:
            # We found a linear dependence.
            # We can return the first 'rank' columns?
            # No, we need a basis for the span.
            # The columns of U_svd corresponding to non-zero singular values form a basis for the column space.
            # But we need the U, V, W factors for the decomposition.
            
            # If there is a dependence, it means one of the rank-1 terms can be expressed as a sum of others.
            # This doesn't directly give us a lower rank decomposition in the same format (list of rank-1 terms).
            # However, if the rank is 55, we can drop one term and adjust?
            # No, dropping a term changes the sum.
            
            # Actually, if the vectors are linearly dependent, we can find a combination that sums to 0.
            # This doesn't reduce the rank of the decomposition unless we can merge terms.
            
            # For now, if rank < 56, we might be able to do better, but implementing the reduction is complex.
            # Given the disjoint supports, it is unlikely to be dependent.
            pass
    except:
        rank = 56
    
    # Return the 56 decomposition
    # Convert back to lists of lists
    U_out = U_arr.tolist()
    V_out = V_arr.tolist()
    W_out = W_arr.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
