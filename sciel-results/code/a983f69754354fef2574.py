import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is n*m*p
    # We will use Strassen reduction for 2x2x2 sub-problems
    
    # Indices
    I = list(range(n))
    J = list(range(m))
    K = list(range(p))
    
    # Groups for Strassen
    # We can form 2x2x2 sub-problems
    # I groups: {0,1}, {2}
    # J groups: {0,1}, {2,3}
    # K groups: {0,1}, {2,3}, {4,5}, {6,7}
    
    I_groups = [(0, 1)]
    J_groups = [(0, 1), (2, 3)]
    K_groups = [(0, 1), (2, 3), (4, 5), (6, 7)]
    
    # Used indices
    used_I = set()
    used_J = set()
    used_K = set()
    
    # For each group, we need to ensure we don't reuse indices in a way that overlaps terms
    # But since we are partitioning the tensor sum, we just need disjoint sets of (i,j,k) triples.
    # If we use I_group (0,1) with J_group (0,1) and K_group (0,1), the triples are 
    # (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)
    # If we use I_group (0,1) with J_group (0,1) and K_group (2,3), the triples are
    # (0,0,2), (0,0,3), (0,1,2), (0,1,3), (1,0,2), (1,0,3), (1,1,2), (1,1,3)
    # These are disjoint.
    
    # We will iterate over all combinations of I_groups, J_groups, K_groups
    # But we must ensure that for a fixed I_group, we don't use the same J_group and K_group twice?
    # No, each combination defines a unique set of triples.
    # We just need to make sure we don't process the same triple twice.
    # Since the groups are disjoint in J and K, and we only have one I_group of size 2,
    # all triples involving I in {0,1} are covered by the 8 combinations.
    # Triples involving I=2 are not covered.
    
    U = []
    V = []
    W = []
    
    def add_term(u_indices, v_indices, w_indices, coeff=1):
        # u_indices: list of (i, j)
        # v_indices: list of (j, k)
        # w_indices: list of (k, i)
        # coeff: scalar
        U.append([0] * (n * m))
        V.append([0] * (m * p))
        W.append([0] * (p * n))
        for idx in u_indices:
            U[-1][idx[0] * m + idx[1]] = coeff
        for idx in v_indices:
            V[-1][idx[0] * p + idx[1]] = coeff
        for idx in w_indices:
            W[-1][idx[0] * n + idx[1]] = coeff
    
    # Process Strassen sub-problems
    for i_g in I_groups:
        for j_g in J_groups:
            for k_g in K_groups:
                i1, i2 = i_g
                j1, j2 = j_g
                k1, k2 = k_g
                
                # M1 = (A[i1,j1] + A[i2,j2]) * (B[j1,k1] + B[j2,k2])
                # C[i1,k1] += M1, C[i2,k2] += M1
                add_term([(i1,j1), (i2,j2)], [(j1,k1), (j2,k2)], [(k1,i1), (k2,i2)], 1)
                
                # M2 = (A[i2,j1] + A[i2,j2]) * B[j1,k1]
                # C[i2,k1] += M2
                add_term([(i2,j1), (i2,j2)], [(j1,k1)], [(k1,i2)], 1)
                
                # M3 = A[i1,j1] * (B[j1,k2] - B[j2,k2])
                # C[i1,k2] += M3
                add_term([(i1,j1)], [(j1,k2), (j2,k2)], [(k2,i1)], 1)
                # Note: B[j2,k2] has coeff -1
                # So V has 1 at (j1,k2) and -1 at (j2,k2)
                # We need to handle coefficients properly.
                # My add_term assumes same coeff for all indices.
                # I need to modify add_term or create a more flexible one.
                
    # Let's refine add_term to handle different coefficients.
    
    def add_term_flex(u_data, v_data, w_data):
        # u_data: list of (index, coeff)
        # v_data: list of (index, coeff)
        # w_data: list of (index, coeff)
        U.append([0] * (n * m))
        V.append([0] * (m * p))
        W.append([0] * (p * n))
        for idx, c in u_data:
            U[-1][idx] = c
        for idx, c in v_data:
            V[-1][idx] = c
        for idx, c in w_data:
            W[-1][idx] = c
            
    # Clear previous lists
    U = []
    V = []
    W = []
    
    for i_g in I_groups:
        for j_g in J_groups:
            for k_g in K_groups:
                i1, i2 = i_g
                j1, j2 = j_g
                k1, k2 = k_g
                
                # M1
                add_term_flex(
                    [(i1*m+j1, 1), (i2*m+j2, 1)],
                    [(j1*p+k1, 1), (j2*p+k2, 1)],
                    [(k1*n+i1, 1), (k2*n+i2, 1)]
                )
                
                # M2
                add_term_flex(
                    [(i2*m+j1, 1), (i2*m+j2, 1)],
                    [(j1*p+k1, 1)],
                    [(k1*n+i2, 1)]
                )
                
                # M3
                add_term_flex(
                    [(i1*m+j1, 1)],
                    [(j1*p+k2, 1), (j2*p+k2, -1)],
                    [(k2*n+i1, 1)]
                )
                
                # M4
                add_term_flex(
                    [(i2*m+j2, 1)],
                    [(j2*p+k1, 1), (j1*p+k1, -1)],
                    [(k1*n+i2, 1)]
                )
                
                # M5
                add_term_flex(
                    [(i1*m+j1, 1), (i1*m+j2, 1)],
                    [(j2*p+k2, 1)],
                    [(k2*n+i1, 1)]
                )
                
                # M6
                add_term_flex(
                    [(i2*m+j1, 1), (i1*m+j1, -1)],
                    [(j1*p+k1, 1), (j1*p+k2, 1)],
                    [(k1*n+i2, 1)]
                )
                
                # M7
                add_term_flex(
                    [(i1*m+j2, 1), (i2*m+j2, -1)],
                    [(j2*p+k1, 1), (j2*p+k2, 1)],
                    [(k1*n+i1, 1)]
                )
                
    # Add remaining schoolbook terms for i=2
    for i in [2]:
        for j in range(m):
            for k in range(p):
                add_term_flex(
                    [(i*m+j, 1)],
                    [(j*p+k, 1)],
                    [(k*n+i, 1)]
                )
                
    return {"U": U, "V": V, "W": W}
