import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = 3, 5, 7
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Known ranks for small blocks
    # rank[i][j][k] for multiplying i x j by j x k
    # Schoolbook is i*j*k
    # Strassen for 2x2x2 is 7
    # We can extend Strassen to rectangular blocks by padding or splitting.
    # However, for simplicity and correctness, we will use the following logic:
    # If a block is 2x2x2, use Strassen (rank 7).
    # Otherwise, use Schoolbook (rank i*j*k).
    # Note: There are better ranks for some rectangular cases (e.g. 2x3x2 is 14? Schoolbook is 12. No, 2*3*2=12. Strassen on 2x2 part + schoolbook on rest?).
    # Actually, for 2x2x2, rank is 7.
    # For 2x1x2 (or 1x2x2), rank is 4 (schoolbook). Can we do better?
    # C = A B where A is 2x1, B is 1x2. C is 2x2.
    # C00 = A0 B0, C01 = A0 B1, C10 = A1 B0, C11 = A1 B1.
    # This is 4 rank-1 terms.
    # Is rank 3 possible? No, bilinear complexity of 2x1x2 is 4.
    # So for blocks not 2x2x2, we use schoolbook.
    
    def get_block_rank(i, j, k):
        if i == 2 and j == 2 and k == 2:
            return 7
        else:
            return i * j * k

    # We need to partition the indices.
    # Output rows i: 0..2
    # Output cols k: 0..6
    # Input j: 0..4
    
    # We will search for the best partition of I (rows of A) and K (cols of B).
    # For a fixed partition of I and K, we also need to partition J (input) for each block?
    # No, the standard block decomposition sums over J.
    # T = sum_{blocks (I_b, K_b)} T_{I_b, K_b}
    # T_{I_b, K_b} = sum_{j=0}^{m-1} T_{I_b, j, K_b}
    # This doesn't help because J is shared.
    
    # Correct Block Decomposition Strategy:
    # We partition I into sets I_1, ..., I_a
    # We partition K into sets K_1, ..., K_b
    # We partition J into sets J_1, ..., J_c
    # Then T = sum_{x,y,z} T_{I_x, J_z, K_y}
    # Each T_{I_x, J_z, K_y} is a tensor of size |I_x| x |J_z| x |K_y|.
    # The rank of the sum is the sum of the ranks of the components IF the supports are disjoint.
    # The supports are disjoint in the I-K plane for different (x,y).
    # Wait, the tensor lives in I x J x K.
    # The basis vectors are e_i \otimes e_j \otimes e_k.
    # If we partition I, J, K into disjoint sets, the sub-tensors have disjoint supports in the full I x J x K space.
    # Therefore, rank(T) = sum_{x,y,z} rank(T_{I_x, J_z, K_y}).
    
    # So we need to find the partition of I={0,1,2}, J={0,1,2,3,4}, K={0..6} that minimizes the sum of ranks.
    
    # Since the dimensions are small, we can enumerate partitions.
    # A partition of a set of size N into non-empty subsets can be represented by assigning each element a group ID.
    # However, the group IDs must be contiguous or we can just use set partitions.
    # Number of set partitions of 3 is 5.
    # Number of set partitions of 5 is 52.
    # Number of set partitions of 7 is 877.
    # Total combinations: 5 * 52 * 877 ~ 227,000. This is feasible.
    
    # But we can be smarter.
    # The rank function is roughly multiplicative.
    # Let's just enumerate all set partitions of I, J, K.
    
    def get_set_partitions(n_elements):
        # Returns a list of partitions, where each partition is a tuple of tuples of indices.
        # E.g. for n=3: [((0,),(1,),(2,)), ((0,1),(2,)), ...]
        # We use Bell numbers logic or recursive generation.
        if n_elements == 0:
            return [()]
        
        partitions = []
        # Assign element n_elements-1 to an existing set or a new set
        prev_partitions = get_set_partitions(n_elements - 1)
        for part in prev_partitions:
            # Option 1: Put new element in a new set
            new_part = part + ((n_elements - 1,),)
            partitions.append(new_part)
            
            # Option 2: Put new element in one of the existing sets
            for i in range(len(part)):
                new_set = part[i] + (n_elements - 1,)
                new_part = part[:i] + (new_set,) + part[i+1:]
                # Avoid duplicates? The order of sets in the tuple might matter for uniqueness if we don't sort.
                # Let's sort the sets by their minimum element to ensure canonical form.
                # Actually, for the purpose of summing ranks, the order of blocks doesn't matter, just the grouping.
                # So we can just use the raw partitions and accept some redundancy or use a set.
                partitions.append(new_part)
        
        # Remove duplicates by converting to frozenset of frozensets
        unique_parts = set()
        for p in partitions:
            unique_parts.add(frozenset(frozenset(s) for s in p))
        
        return [tuple(sorted(s, reverse=True)) for s in unique_parts] # Sort sets by max element for consistency? Or just keep as is.

    # Precompute partitions for I (size 3), J (size 5), K (size 7)
    # Note: get_set_partitions is recursive and might be slow for 7 if not optimized, but 877 is small.
    # Let's cache them.
    
    parts_I = get_set_partitions(3)
    parts_J = get_set_partitions(5)
    parts_K = get_set_partitions(7)
    
    best_rank = float('inf')
    best_partition = None
    
    # Iterate over all combinations of partitions
    # To speed up, we can prune if the current sum exceeds best_rank.
    
    for part_I in parts_I:
        # part_I is a tuple of tuples of indices. E.g. ((0,1), (2,))
        # Calculate rank contribution for each block in part_I?
        # No, the rank is sum over x,y,z of rank(|I_x|, |J_z|, |K_y|).
        # This is a sum over all combinations of blocks from I, J, K.
        
        # Let's precompute the sizes of blocks in each partition.
        sizes_I = [len(s) for s in part_I]
        
        for part_J in parts_J:
            sizes_J = [len(s) for s in part_J]
            
            # Pruning: even if all K blocks were size 1 (min rank), can we beat best?
            # Min rank for block (i,j,1) is i*j.
            # Sum_{x,y} rank(i_x, j_y, 1) = Sum_{x,y} i_x * j_y = (Sum i_x) * (Sum j_y) = 3 * 5 = 15.
            # This lower bound is constant for fixed I, J partitions.
            # Actually, the K partition affects the number of blocks.
            # Let's just compute the full sum.
            
            for part_K in parts_K:
                sizes_K = [len(s) for s in part_K]
                
                total_rank = 0
                for i_size in sizes_I:
                    for j_size in sizes_J:
                        for k_size in sizes_K:
                            total_rank += get_block_rank(i_size, j_size, k_size)
                            
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_partition = (part_I, part_J, part_K)
                    
                    # Pruning: if best_rank is close to theoretical min, we can stop?
                    # Theoretical min is unknown, but schoolbook is 105.
                    # If we find 77, that's good.
                    
    # Now construct the decomposition using best_partition
    part_I, part_J, part_K = best_partition
    
    rank_terms = []
    
    # Helper to add schoolbook term
    def add_schoolbook(i, j, k):
        rank_terms.append((i, j, k, 1.0))
        
    # Helper to add Strassen block
    # Strassen is only used if i_size=2, j_size=2, k_size=2
    def add_strassen_block(i0, i1, k0, k1, j0, j1):
        # Indices:
        # A: (i0, j0), (i0, j1), (i1, j0), (i1, j1)
        # B: (j0, k0), (j0, k1), (j1, k0), (j1, k1)
        # C: (i0, k0), (i0, k1), (i1, k0), (i1, k1)
        
        def make_vec_A(i, j, val=1.0):
            vec = np.zeros(nm)
            vec[i*m + j] = val
            return vec
            
        def make_vec_B(j, k, val=1.0):
            vec = np.zeros(mp)
            vec[j*p + k] = val
            return vec
            
        def make_vec_C(i, k, val=1.0):
            vec = np.zeros(pn)
            vec[k*n + i] = val
            return vec
            
        # M1 = (A00+A11)(B00+B11)
        U1 = make_vec_A(i0, j0) + make_vec_A(i1, j1)
        V1 = make_vec_B(j0, k0) + make_vec_B(j1, k1)
        W1 = make_vec_C(i0, k0, 1.0) + make_vec_C(i1, k1, 1.0)
        
        # M2 = (A10+A11)B00
        U2 = make_vec_A(i1, j0) + make_vec_A(i1, j1)
        V2 = make_vec_B(j0, k0)
        W2 = make_vec_C(i1, k0, 1.0) + make_vec_C(i1, k1, -1.0)
        
        # M3 = A00(B10-B11)
        U3 = make_vec_A(i0, j0)
        V3 = make_vec_B(j1, k0) - make_vec_B(j1, k1)
        W3 = make_vec_C(i0, k1, 1.0) + make_vec_C(i1, k1, 1.0)
        
        # M4 = A11(B01-B00)
        U4 = make_vec_A(i1, j1)
        V4 = make_vec_B(j0, k1) - make_vec_B(j0, k0)
        W4 = make_vec_C(i0, k0, 1.0) + make_vec_C(i1, k0, 1.0)
        
        # M5 = (A00+A01)B11
        U5 = make_vec_A(i0, j0) + make_vec_A(i0, j1)
        V5 = make_vec_B(j1, k1)
        W5 = make_vec_C(i0, k1, 1.0) - make_vec_C(i0, k0, 1.0)
        
        # M6 = (A10-A00)(B00+B01)
        U6 = make_vec_A(i1, j0) - make_vec_A(i0, j0)
        V6 = make_vec_B(j0, k0) + make_vec_B(j0, k1)
        W6 = make_vec_C(i1, k1, 1.0)
        
        # M7 = (A01-A11)(B10+B11)
        U7 = make_vec_A(i0, j1) - make_vec_A(i1, j1)
        V7 = make_vec_B(j1, k0) + make_vec_B(j1, k1)
        W7 = make_vec_C(i0, k0, 1.0)
        
        rank_terms.append((U1, V1, W1))
        rank_terms.append((U2, V2, W2))
        rank_terms.append((U3, V3, W3))
        rank_terms.append((U4, V4, W4))
        rank_terms.append((U5, V5, W5))
        rank_terms.append((U6, V6, W6))
        rank_terms.append((U7, V7, W7))
        
    # Iterate through the blocks defined by the partition
    for set_I in part_I:
        for set_J in part_J:
            for set_K in part_K:
                i_list = list(set_I)
                j_list = list(set_J)
                k_list = list(set_K)
                
                i_size = len(i_list)
                j_size = len(j_list)
                k_size = len(k_list)
                
                # Check if this block is 2x2x2
                if i_size == 2 and j_size == 2 and k_size == 2:
                    # Use Strassen
                    # Map the sorted lists to i0, i1 etc.
                    # The order matters for the Strassen formulas?
                    # Strassen formulas assume A00, A01, A10, A11.
                    # We need to map the actual indices to these positions.
                    # Let i0, i1 be the two rows.
                    # Let j0, j1 be the two cols.
                    # Let k0, k1 be the two cols of B.
                    # The output is C[i0, k0], C[i0, k1], C[i1, k0], C[i1, k1].
                    
                    # We just pass the indices. The function add_strassen_block handles the vector construction.
                    # It assumes the "00" position is (i0, j0) in A and (j0, k0) in B.
                    # This corresponds to the top-left element of the submatrix.
                    # Since we are using specific indices, this is fine.
                    
                    add_strassen_block(i_list[0], i_list[1], k_list[0], k_list[1], j_list[0], j_list[1])
                else:
                    # Use Schoolbook
                    for i in i_list:
                        for j in j_list:
                            for k in k_list:
                                add_schoolbook(i, j, k)
                                
    # Construct matrices
    U_mat = np.zeros((nm, len(rank_terms)))
    V_mat = np.zeros((mp, len(rank_terms)))
    W_mat = np.zeros((pn, len(rank_terms)))
    
    for r, term in enumerate(rank_terms):
        if len(term) == 3:
            U_mat[:, r] = term[0]
            V_mat[:, r] = term[1]
            W_mat[:, r] = term[2]
        else:
            i, j, k, val = term
            U_mat[i*m+j, r] = val
            V_mat[j*p+k, r] = val
            W_mat[k*n+i, r] = val
            
    # Verify
    def verify(U, V, W, R):
        U_3d = np.array(U).reshape(n, m, R)
        V_3d = np.array(V).reshape(m, p, R)
        W_3d = np.array(W).reshape(p, n, R)
        try:
            T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
            return np.allclose(T, 1.0, atol=1e-9)
        except:
            return False
            
    if verify(U_mat, V_mat, W_mat, len(rank_terms)):
        return {
            "U": U_mat.tolist(),
            "V": V_mat.tolist(),
            "W": W_mat.tolist()
        }
    else:
        # Fallback to schoolbook
        U_school = np.zeros((nm, 105))
        V_school = np.zeros((mp, 105))
        W_school = np.zeros((pn, 105))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, col] = 1.0
                    V_school[j * p + k, col] = 1.0
                    W_school[k * n + i, col] = 1.0
                    col += 1
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
