import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # We will build the decomposition recursively.
    # Each term is a tuple (u_row, v_row, w_row, val)
    # But we need to accumulate into matrices U, V, W.
    
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # We don't know the rank ahead of time, so we'll collect terms in a list.
    # Each term is a dict or tuple: (u_index, v_index, w_index, value)
    # u_index is in [0, nm), v_index in [0, mp), w_index in [0, pn).
    
    terms = []
    
    def add_term(u_idx, v_idx, w_idx, val):
        if val != 0:
            terms.append((u_idx, v_idx, w_idx, val))
    
    def strassen_222(base_i, base_j, base_k):
        # Adds 7 terms for 2x2x2 block starting at (base_i, base_j, base_k)
        # Indices:
        # i: base_i, base_i+1
        # j: base_j, base_j+1
        # k: base_k, base_k+1
        
        # Strassen formulas:
        # M1 = (A11 + A22)(B21 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # In terms of tensor decomposition:
        # Each M corresponds to a product of sums of A and B entries.
        # A[i,j] and B[j,k] are basis elements.
        
        # We need to expand these into rank-1 terms of the form:
        # U[i,j,c] * V[j,k,c] * W[k,i,c]
        
        # Let's map the Strassen intermediates to the decomposition.
        # This is complex to do symbolically here.
        # Instead, we can use the known explicit rank-7 decomposition for 2x2x2.
        
        # Standard rank-7 decomposition of 2x2x2 matmul tensor:
        # There are 7 terms.
        # Let's define them explicitly.
        
        # Term 1: (A11+A22)(B21+B22) -> contributes to C11, C22
        # This term is not rank 1 in the A,B space?
        # Wait, the tensor is T[i,j,k].
        # The decomposition is T = sum_c U_c V_c W_c.
        # For 2x2x2, there is a known set of 7 tensors.
        
        # Let's use the explicit coordinates.
        # i in {0,1}, j in {0,1}, k in {0,1}
        
        # We will hardcode the 7 terms for 2x2x2.
        # Reference: "Rank 7 decomposition of 2x2x2"
        
        # The 7 rank-1 tensors (U, V, W) are:
        # 1. U: [[1,0],[0,1]] (i.e. U[0,0]=1, U[1,1]=1? No, U is 2x1 vector for each j? No, U is 2x2 matrix? No.)
        # U is indexed by (i,j). So U is a 2x2 matrix for each term c.
        # V is a 2x2 matrix for each term c.
        # W is a 2x2 matrix for each term c.
        
        # Let's look up the specific rank-7 decomposition.
        # One common one:
        # c=0: U=[[1,0],[0,0]], V=[[0,0],[1,0]], W=[[0,1],[0,0]] -> A00 B10 C01?
        # This is tedious to hardcode correctly from memory.
        
        # Alternative: Use the recursive split for 2x2x2 as well?
        # No, 2x2x2 is the base case for Strassen.
        
        # Let's try to find a simpler valid decomposition for 2x2x2 with rank 7.
        # Or just use schoolbook rank 8 if we can't get 7?
        # If we use schoolbook for 2x2x2, rank is 8.
        # Total rank would be:
        # 2x2x6: 3 * 8 = 24
        # 2x1x6: 3 * 1 = 3
        # 2x3x6: 24 + 3 = 27
        # 2x2x6: 24
        # 2x5x6: 24 + 27 = 51
        # 1x5x6: 5
        # Total: 56.
        # Score: 90/56 = 1.607.
        # This is still better than 1.0714.
        
        # Let's try to implement Strassen 2x2x2 correctly.
        # I will use a known explicit rank-7 decomposition.
        
        # The 7 terms (U_c, V_c, W_c) for 2x2x2:
        # Indices 0,1 for i,j,k.
        
        # Term 0:
        # U: [[1, 1], [0, 0]]  (i=0: j=0,1; i=1: j=0,1) -> U[0,0]=1, U[0,1]=1
        # V: [[0, 0], [1, 1]]  -> V[1,0]=1, V[1,1]=1
        # W: [[1, 0], [1, 0]]  -> W[0,0]=1, W[1,0]=1
        # Check:
        # C00 = U00 V00 W00 + U01 V10 W10 = 1*0*1 + 1*1*1 = 1. (Should be 1)
        # C01 = U00 V01 W01 + U01 V11 W11 = 1*0*0 + 1*1*0 = 0. (Should be 0? No, C01 = A00 B01 + A01 B11)
        # This hardcoding is error-prone.
        
        # I will stick to the Schoolbook for 2x2x2 to ensure correctness,
        # but I will try to optimize the 1xNxN and Nx1xN cases.
        
        # Rank 8 for 2x2x2.
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    ui = (base_i + i) * 2 + (base_j + j) # Wait, m is 2 here?
                    # In the 2x2x2 block, the local dims are 2,2,2.
                    # Global indices:
                    gi = base_i + i
                    gj = base_j + j
                    gk = base_k + k
                    # U index: gi * m_global + gj ? No.
                    # U is indexed by (i, j) in the global n, m.
                    # So u_idx = gi * m + gj
                    u_idx = gi * m + gj
                    v_idx = gj * p + gk
                    w_idx = gk * n + gi
                    add_term(u_idx, v_idx, w_idx, 1)
    
    def decompose(n_local, m_local, p_local, base_i, base_j, base_k):
        # n_local, m_local, p_local are the sizes of the block
        # base_i, base_j, base_k are the starting indices in the global matrix
        
        if time.time() > deadline - 5:
            return
        
        if n_local == 2 and m_local == 2 and p_local == 2:
            strassen_222(base_i, base_j, base_k)
            return
        
        if n_local == 1:
            # Rank is m_local
            # C_k = sum_j A_j B_jk
            # Terms: for each j, k?
            # T[0, j, k] = 1
            # Rank m_local?
            # We can use m_local terms.
            # Term c (0..m_local-1):
            # U[0, j, c] = 1 if j==c
            # V[j, k, c] = 1 if j==c
            # W[k, 0, c] = 1
            # Sum_c U V W = sum_c (delta_jc * delta_jc * 1) = delta_jj = 1 if j is in range.
            # This works for all k.
            for j in range(m_local):
                u_idx = (base_i) * m + (base_j + j)
                # V and W depend on k?
                # No, the term is rank 1.
                # For a fixed j, we have a term that covers all k?
                # U[0, j, c] = 1.
                # V[j, k, c] = 1 for all k?
                # W[k, 0, c] = 1 for all k?
                # Then sum_c ... = 1 * 1 * 1 = 1.
                # But we need to sum over j?
                # The decomposition is sum_c.
                # If we set c=j, then U[0,j,j]=1.
                # V[j,k,j]=1 for all k.
                # W[k,0,j]=1 for all k.
                # Then for a specific (0, j', k'),
                # Sum_c U[0,j',c] V[j',k',c] W[k',0,c]
                # Only c=j' is non-zero in U.
                # Term = 1 * 1 * 1 = 1.
                # So yes, m_local terms.
                
                # Add term
                u_idx = base_i * m + (base_j + j)
                # We need to add this term for all k?
                # No, the term is defined by its U, V, W vectors.
                # U vector has 1 at (0, j).
                # V vector has 1 at (j, k) for all k?
                # W vector has 1 at (k, 0) for all k?
                
                # We will store this as a single term with sparse vectors.
                # But our `add_term` only adds one (u_idx, v_idx, w_idx).
                # This structure requires V and W to be non-zero at multiple indices.
                # My `terms` list stores (u_idx, v_idx, w_idx, val).
                # This implies U, V, W are sparse with one non-zero per term?
                # No, rank-1 tensor is outer product of 3 vectors.
                # U_c is a vector, V_c is a vector, W_c is a vector.
                # My `add_term` is insufficient for dense vectors.
                
                # I need to change the data structure.
                # I will use a dictionary: key (u_idx, v_idx, w_idx) -> val.
                # But for rank 1, U_c[i,j] * V_c[j,k] * W_c[k,i].
                # If U_c is non-zero at (i0, j0), V_c at (j0, k0), W_c at (k0, i0),
                # then the tensor is non-zero only at (i0, j0, k0).
                # This is the "diagonal" or "sparse" rank-1 term.
                
                # For 1xNxN, the tensor T[0,j,k]=1 is dense in k.
                # So we need V_c and W_c to be dense.
                # This means my simple (u,v,w) term representation is not enough.
                
                # I will switch to storing U, V, W as numpy arrays directly.
                # But I don't know the rank.
                # I will collect terms as (U_vec, V_vec, W_vec, val).
                # U_vec is a numpy array of size nm.
                
                pass # Logic needs restructuring
                
        # ... rest of decompose ...
    
    # Given the complexity of mixing sparse/dense terms in a simple list,
    # and the time limit, I will revert to a pure Schoolbook approach 
    # but with the specific optimization that 1xNxN and Nx1xN 
    # can be done with rank min(N,M) using dense vectors.
    
    # However, implementing dense vector storage is easy.
    # I will initialize U, V, W as lists of zeros of size (nm, max_rank).
    # Max rank is 90.
    
    U = np.zeros((nm, 90))
    V = np.zeros((mp, 90))
    W = np.zeros((pn, 90))
    current_rank = 0
    
    def add_rank1_term(u_idx, v_idx, w_idx, val):
        nonlocal current_rank
        U[u_idx, current_rank] += val
        V[v_idx, current_rank] += val
        W[w_idx, current_rank] += val
        current_rank += 1
        
    # This only works for sparse rank-1 terms.
    # For 1xNxN, we need:
    # U[0, j, c] = 1
    # V[j, k, c] = 1 for all k
    # W[k, 0, c] = 1 for all k
    # This is NOT a sparse rank-1 term in the (u,v,w) sense.
    # It is a rank-1 tensor where V and W are dense.
    
    # I will handle 1xNxN and Nx1xN separately by filling the arrays directly.
    
    def decompose(n_local, m_local, p_local, base_i, base_j, base_k):
        nonlocal current_rank
        
        if n_local == 1 and m_local == 1 and p_local == 1:
            u_idx = base_i * m + base_j
            v_idx = base_j * p + base_k
            w_idx = base_k * n + base_i
            add_rank1_term(u_idx, v_idx, w_idx, 1)
            return
        
        if n_local == 1:
            # Rank m_local
            for j in range(m_local):
                u_idx = base_i * m + (base_j + j)
                U[u_idx, current_rank] = 1
                # V is dense in k for this j
                for k in range(p_local):
                    v_idx = (base_j + j) * p + (base_k + k)
                    V[v_idx, current_rank] = 1
                # W is dense in k for this i
                for k in range(p_local):
                    w_idx = (base_k + k) * n + base_i
                    W[w_idx, current_rank] = 1
                current_rank += 1
            return
            
        if m_local == 1:
            # Rank p_local? No, Rank n_local?
            # C_i = sum_k A_ik B_k
            # T[i, 0, k] = 1
            # Rank p_local?
            # Term c (0..p_local-1):
            # U[i, 0, c] = 1 if i==c? No.
            # Let's use p_local terms.
            for k in range(p_local):
                v_idx = base_j * p + (base_k + k)
                V[v_idx, current_rank] = 1
                # U dense in i
                for i in range(n_local):
                    u_idx = (base_i + i) * m + base_j
                    U[u_idx, current_rank] = 1
                # W dense in i
                for i in range(n_local):
                    w_idx = (base_k + k) * n + (base_i + i)
                    W[w_idx, current_rank] = 1
                current_rank += 1
            return
            
        if n_local == 2 and m_local == 2 and p_local == 2:
            # Strassen
            # I will use schoolbook for now to ensure correctness, 
            # as implementing Strassen with dense vectors is complex.
            # Schoolbook rank 8.
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        u_idx = (base_i + i) * m + (base_j + j)
                        v_idx = (base_j + j) * p + (base_k + k)
                        w_idx = (base_k + k) * n + (base_i + i)
                        add_rank1_term(u_idx, v_idx, w_idx, 1)
            return
        
        # Split largest dimension
        if n_local >= m_local and n_local >= p_local:
            if n_local > 2:
                split = 2
            else:
                split = 1
            decompose(split, m_local, p_local, base_i, base_j, base_k)
            decompose(n_local - split, m_local, p_local, base_i + split, base_j, base_k)
        elif m_local >= n_local and m_local >= p_local:
            if m_local > 2:
                split = 2
            else:
                split = 1
            decompose(n_local, split, p_local, base_i, base_j, base_k)
            decompose(n_local, m_local - split, p_local, base_i, base_j + split, base_k)
        else:
            if p_local > 2:
                split = 2
            else:
                split = 1
            decompose(n_local, m_local, split, base_i, base_j, base_k)
            decompose(n_local, m_local, p_local - split, base_i, base_j, base_k + split)
            
    decompose(n, m, p, 0, 0, 0)
    
    # Trim arrays to current_rank
    U = U[:, :current_rank]
    V = V[:, :current_rank]
    W = W[:, :current_rank]
    
    # Convert to lists of lists for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
