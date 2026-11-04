"""Matrix multiplication 3x5x5 - rank-minimal tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, atol=1e-9):
    """Check if the decomposition exactly reconstructs the matrix multiplication tensor."""
    rank = U.shape[1]
    rng = np.random.RandomState(42)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    AB_direct = A @ B

    U_reshaped = U.T.reshape(rank, n, m)
    V_reshaped = V.T.reshape(rank, m, p)
    W_reshaped_temp = W.T.reshape(rank, p, n)
    W_reshaped = W_reshaped_temp.transpose(0, 2, 1)

    U_A = np.einsum('rij,ij->r', U_reshaped, A)
    V_B = np.einsum('rjk,jk->r', V_reshaped, B)
    AB_decomp = np.einsum('r,r,rik->ik', U_A, V_B, W_reshaped)

    return np.allclose(AB_direct, AB_decomp, atol=atol)


def _verify_strict(U, V, W, n, m, p, atol=1e-12):
    """Stricter verification with multiple random matrices."""
    for seed in [42, 123, 999]:
        rng = np.random.RandomState(seed)
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        AB_direct = A @ B

        U_reshaped = U.T.reshape(U.shape[1], n, m)
        V_reshaped = V.T.reshape(V.shape[1], m, p)
        W_reshaped_temp = W.T.reshape(W.shape[1], p, n)
        W_reshaped = W_reshaped_temp.transpose(0, 2, 1)

        U_A = np.einsum('rij,ij->r', U_reshaped, A)
        V_B = np.einsum('rjk,jk->r', V_reshaped, B)
        AB_decomp = np.einsum('r,r,rik->ik', U_A, V_B, W_reshaped)

        if not np.allclose(AB_direct, AB_decomp, atol=atol):
            return False
    return True


def _snap_to_half_integer(arr, tol=1e-6):
    """Snap values close to half-integers to exact half-integers."""
    snapped = np.round(arr * 2) / 2
    if np.allclose(arr, snapped, atol=tol):
        return snapped
    # Try integer snapping
    snapped_int = np.round(arr)
    if np.allclose(arr, snapped_int, atol=tol):
        return snapped_int
    return arr


def _strassen_2x2x2():
    """
    Strassen decomposition of 2x2x2 matrix multiplication (rank 7).
    Returns U, V, W of shapes (4,7), (4,7), (4,7).
    Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    For 2x2x2: n=m=p=2, so U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen's 7 products for 2x2 matrix multiplication
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+b)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+d)(e+f)
    # M6 = (b-c)(g+h)
    # M7 = (a-c)(f+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M5
    # C22 = M1 - M2 + M3 + M6
    
    # U factors (for A): rows indexed by (i,j) where i=row, j=col of A
    # V factors (for B): rows indexed by (j,k) where j=row, k=col of B
    # W factors (for C): rows indexed by (k,i) where k=col of C, i=row of C
    # C[i,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] ... wait
    # Actually: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor: T[i,j,k] = delta (it's the multiplication tensor)
    # Decomposition: T = sum_r U_r ⊗ V_r ⊗ W_r
    # where U_r[i,j], V_r[j,k], W_r[k,i]... 
    # Actually the convention is:
    # U has shape (n*m, rank): U[i*m + j, r]
    # V has shape (m*p, rank): V[j*p + k, r]
    # W has shape (p*n, rank): W[k*n + i, r]
    # And C[i,k] = sum_{j,r} U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]
    # Wait, that's not right either. Let me re-derive.
    # 
    # The matrix multiplication tensor T[i,j,k] for C[i,k] = sum_j A[i,j]*B[j,k]
    # is T[i,j,k] = 1 if we think of it as T[i,k,j] = delta... 
    # 
    # Actually, the standard tensor decomposition for matmul:
    # C[i,k] = sum_{j,r} A[i,j] B[j,k] T[i,j,k]
    # where T[i,j,k] = 1 (it's just the identity in the summation index)
    # 
    # Wait no. The matmul tensor is: T[i,j,k] such that
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This is NOT a tensor decomposition in the usual sense.
    #
    # The correct formulation: We want to compute C = AB where C[i,k] = sum_j A[i,j] B[j,k].
    # We introduce intermediate variables: for each r, compute
    #   U_r = U_r^T * A  (shape n)  -- wait, U has shape (n*m, rank)
    #   V_r = V_r^T * B  (shape p)  -- V has shape (m*p, rank)
    #   Then C = sum_r U_r * V_r^T * W_r? No...
    #
    # Let me re-read the verification code:
    # U_reshaped = U.T.reshape(rank, n, m)  # U_reshaped[r, i, j] = U[i*m+j, r]
    # V_reshaped = V.T.reshape(rank, m, p)  # V_reshaped[r, j, k] = V[j*p+k, r]
    # W_reshaped[r, i, k] = W[k*n+i, r]
    #
    # U_A[r] = sum_{i,j} U_reshaped[r,i,j] * A[i,j]
    # V_B[r] = sum_{j,k} V_reshaped[r,j,k] * B[j,k]
    # AB_decomp[i,k] = sum_r U_A[r] * V_B[r] * W_reshaped[r,i,k]
    #
    # So: AB_decomp[i,k] = sum_r (sum_{i',j'} U[r,i',j'] A[i',j']) (sum_{j'',k''} V[r,j'',k''] B[j'',k'']) W[r,i,k]
    #
    # We need this to equal sum_j A[i,j] B[j,k] for all A, B.
    #
    # This means: sum_r U[r,i',j'] V[r,j'',k''] W[r,i,k] = delta_{i',i} delta_{j'',k} * ... 
    # Actually: sum_r U[r,i',j'] V[r,j'',k''] W[r,i,k] must equal delta_{i',i} delta_{j',j''} delta_{k'',k}
    # Wait, that's not right either. Let me think more carefully.
    #
    # AB_decomp[i,k] = sum_r (sum_{a,b} U[r,a,b] A[a,b]) (sum_{c,d} V[r,c,d] B[c,d]) W[r,i,k]
    # = sum_r sum_{a,b,c,d} U[r,a,b] V[r,c,d] W[r,i,k] A[a,b] B[c,d]
    #
    # We need: sum_r sum_{a,b,c,d} U[r,a,b] V[r,c,d] W[r,i,k] A[a,b] B[c,d] = sum_j A[i,j] B[j,k]
    #
    # For this to hold for all A, B:
    # sum_r sum_{a,b,c,d} U[r,a,b] V[r,c,d] W[r,i,k] A[a,b] B[c,d] = sum_j A[i,j] B[j,k]
    #
    # This requires: sum_r U[r,a,b] V[r,c,d] W[r,i,k] = delta_{a,i} delta_{b,c} delta_{d,k}
    #
    # So the condition is: for all (a,b,c,d,i,k):
    # sum_r U[r,a,b] V[r,c,d] W[r,i,k] = delta_{a,i} delta_{b,c} delta_{d,k}
    #
    # This is the tensor decomposition of the matmul tensor.
    
    # Strassen for 2x2x2:
    # The 7 products are:
    # P1 = (A11+A12)(B11+B22) -> U1: A11+A12, V1: B11+B22, W1: C11+=1, C12+=0, C21+=0, C22-=1
    # P2 = (A21+A22)B11        -> U2: A21+A22, V2: B11, W2: C21+=1, C22-=1
    # P3 = A11(B12-B22)        -> U3: A11, V3: B12-B22, W3: C12+=1
    # P4 = A22(B21-B11)        -> U4: A22, V4: B21-B11, W4: C11+=1, C21+=0... 
    
    # Let me use the standard Strassen formulas:
    # M1 = (a+b)(e+h)
    # M2 = (c+d)e  
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+d)(e+f)
    # M6 = (b-c)(g+h)
    # M7 = (a-c)(f+h)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M5
    # C22 = M1 - M2 + M3 + M6
    #
    # Where A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[C11,C12],[C21,C22]]
    #
    # In our indexing: i=0,1 (rows of A), j=0,1 (cols of A / rows of B), k=0,1 (cols of B)
    # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    # C[0,0]=C11, C[0,1]=C12, C[1,0]=C21, C[1,1]=C22
    #
    # U[r, i, j]: coefficient of A[i,j] in product r
    # V[r, j, k]: coefficient of B[j,k] in product r
    # W[r, i, k]: coefficient in C[i,k]
    
    # M1 = (A[0,0]+A[0,1])(B[0,0]+B[1,1])
    U[0*2+0, 0] = 1; U[0*2+1, 0] = 1  # A[0,0]+A[0,1]
    V[0*2+0, 0] = 1; V[1*2+1, 0] = 1  # B[0,0]+B[1,1]
    W[0*2+0, 0] = 1; W[1*2+1, 0] = -1  # C[0,0]+=1, C[1,1]-=1
    
    # M2 = (A[1,0]+A[1,1])B[0,0]
    U[1*2+0, 1] = 1; U[1*2+1, 1] = 1  # A[1,0]+A[1,1]
    V[0*2+0, 1] = 1  # B[0,0]
    W[1*2+0, 1] = 1; W[1*2+1, 1] = -1  # C[1,0]+=1, C[1,1]-=1
    
    # M3 = A[0,0](B[0,1]-B[1,1])
    U[0*2+0, 2] = 1  # A[0,0]
    V[0*2+1, 2] = 1; V[1*2+1, 2] = -1  # B[0,1]-B[1,1]
    W[0*2+1, 2] = 1  # C[0,1]+=1
    
    # M4 = A[1,1](B[1,0]-B[0,0])
    U[1*2+1, 3] = 1  # A[1,1]
    V[1*2+0, 3] = 1; V[0*2+0, 3] = -1  # B[1,0]-B[0,0]
    W[0*2+0, 3] = 1  # C[0,0]+=1
    
    # M5 = (A[0,0]+A[1,1])(B[0,0]+B[0,1])
    U[0*2+0, 4] = 1; U[1*2+1, 4] = 1  # A[0,0]+A[1,1]
    V[0*2+0, 4] = 1; V[0*2+1, 4] = 1  # B[0,0]+B[0,1]
    W[0*2+0, 4] = -1; W[0*2+1, 4] = 1; W[1*2+0, 4] = 1  # C[0,0]-=1, C[0,1]+=1, C[1,0]+=1
    
    # M6 = (A[0,1]-A[1,0])(B[1,0]+B[1,1])
    U[0*2+1, 5] = 1; U[1*2+0, 5] = -1  # A[0,1]-A[1,0]
    V[1*2+0, 5] = 1; V[1*2+1, 5] = 1  # B[1,0]+B[1,1]
    W[1*2+1, 5] = 1  # C[1,1]+=1
    
    # M7 = (A[0,0]-A[1,0])(B[0,1]+B[1,1])
    U[0*2+0, 6] = 1; U[1*2+0, 6] = -1  # A[0,0]-A[1,0]
    V[0*2+1, 6] = 1; V[1*2+1, 6] = 1  # B[0,1]+B[1,1]
    W[0*2+0, 6] = 1  # C[0,0]+=1
    
    return U, V, W


def _schoolbook_local(n, m, p):
    """Standard schoolbook decomposition for n x m @ m x p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _embed_block(U_block, V_block, W_block, n, m, p, i_off, j_off, k_off, i_size, j_size, k_size):
    """
    Embed a block decomposition into the full matrix.
    The block computes C_block[i_off:i_off+i_size, k_off:k_off+k_size] = 
    A_block[i_off:i_off+i_size, j_off:j_off+j_size] @ B_block[j_off:j_off+j_size, k_off:k_off+k_size]
    
    U_block has shape (i_size*j_size, rank_block)
    V_block has shape (j_size*k_size, rank_block)
    W_block has shape (k_size*i_size, rank_block)
    
    Returns full U, V, W contributions (only for this block's rows).
    """
    rank_block = U_block.shape[1]
    U_full = np.zeros((n * m, rank_block))
    V_full = np.zeros((m * p, rank_block))
    W_full = np.zeros((p * n, rank_block))
    
    for r in range(rank_block):
        for local_i in range(i_size):
            for local_j in range(j_size):
                full_i = i_off + local_i
                full_j = j_off + local_j
                U_full[full_i * m + full_j, r] = U_block[local_i * j_size + local_j, r]
        
        for local_j in range(j_size):
            for local_k in range(k_size):
                full_j = j_off + local_j
                full_k = k_off + local_k
                V_full[full_j * p + full_k, r] = V_block[local_j * k_size + local_k, r]
        
        for local_k in range(k_size):
            for local_i in range(i_size):
                full_k = k_off + local_k
                full_i = i_off + local_i
                W_full[full_k * n + full_i, r] = W_block[local_k * i_size + local_i, r]
    
    return U_full, V_full, W_full


def _block_decomposition(n, m, p, i_splits, j_splits, k_splits, use_strassen_for_2x2x2=True):
    """
    Block decomposition with given splits.
    i_splits: list of row sizes for A (sum = n)
    j_splits: list of inner sizes (sum = m)
    k_splits: list of col sizes for B (sum = p)
    
    For each (i_block, j_block, k_block) triple:
    - If all sizes are 2, use Strassen (rank 7)
    - Otherwise use schoolbook
    """
    n_blocks = len(i_splits)
    m_blocks = len(j_splits)
    p_blocks = len(k_splits)
    
    # Compute offsets
    i_offs = np.cumsum([0] + i_splits)
    j_offs = np.cumsum([0] + j_splits)
    k_offs = np.cumsum([0] + k_splits)
    
    total_rank = 0
    U_parts = []
    V_parts = []
    W_parts = []
    
    for bi in range(n_blocks):
        for bj in range(m_blocks):
            for bk in range(p_blocks):
                i_size = i_splits[bi]
                j_size = j_splits[bj]
                k_size = k_splits[bk]
                
                if use_strassen_for_2x2x2 and i_size == 2 and j_size == 2 and k_size == 2:
                    U_b, V_b, W_b = _strassen_2x2x2()
                else:
                    U_b, V_b, W_b = _schoolbook_local(i_size, j_size, k_size)
                
                U_full, V_full, W_full = _embed_block(
                    U_b, V_b, W_b, n, m, p,
                    i_offs[bi], j_offs[bj], k_offs[bk],
                    i_size, j_size, k_size
                )
                U_parts.append(U_full)
                V_parts.append(V_full)
                W_parts.append(W_full)
                total_rank += U_b.shape[1]
    
    U = np.hstack(U_parts)
    V = np.hstack(V_parts)
    W = np.hstack(W_parts)
    
    return U, V, W


def _try_block_splits(n, m, p, deadline):
    """Try various block splits and find the best rank."""
    best_rank = n * m * p
    best_U, best_V, best_W = _schoolbook_local(n, m, p)
    
    # Generate possible splits
    def gen_splits(total, max_parts=3):
        """Generate all compositions of 'total' into at most max_parts positive integers."""
        results = []
        if max_parts == 1:
            results.append([total])
            return results
        for first in range(1, total + 1):
            for rest in gen_splits(total - first, max_parts - 1):
                results.append([first] + rest)
        return results
    
    i_splits_options = gen_splits(n, 3)
    j_splits_options = gen_splits(m, 3)
    k_splits_options = gen_splits(p, 3)
    
    # Try combinations, prioritizing those with 2x2x2 blocks
    candidates = []
    for i_s in i_splits_options:
        for j_s in j_splits_options:
            for k_s in k_splits_options:
                # Estimate rank
                rank_est = 0
                for i_size in i_s:
                    for j_size in j_s:
                        for k_size in k_s:
                            if i_size == 2 and j_size == 2 and k_size == 2:
                                rank_est += 7
                            else:
                                rank_est += i_size * j_size * k_size
                candidates.append((rank_est, i_s, j_s, k_s))
    
    # Sort by estimated rank
    candidates.sort(key=lambda x: x[0])
    
    for rank_est, i_s, j_s, k_s in candidates:
        if time.time() > deadline:
            break
        if rank_est >= best_rank:
            break
        
        U, V, W = _block_decomposition(n, m, p, i_s, j_s, k_s)
        
        if _verify(U, V, W, n, m, p):
            actual_rank = U.shape[1]
            if actual_rank < best_rank:
                best_rank = actual_rank
                best_U, best_V, best_W = U, V, W
    
    return best_rank, best_U, best_V, best_W


def _try_polish(U, V, W, n, m, p, deadline, max_iters=50):
    """
    Try to reduce rank by removing dependent columns.
    Check if any column is a linear combination of others.
    """
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W
    
    # Build the tensor and check for linear dependencies
    # The condition is: sum_r U[r,a,b] V[r,c,d] W[r,i,k] = delta_{a,i} delta_{b,c} delta_{d,k}
    # If column r is a linear combination of others, we can remove it.
    
    # Check each column for redundancy
    for r in range(rank):
        if time.time() > deadline:
            break
        
        # Try removing column r
        if r == 0:
            U_test = U[:, 1:]
            V_test = V[:, 1:]
            W_test = W[:, 1:]
        elif r == rank - 1:
            U_test = U[:, :-1]
            V_test = V[:, :-1]
            W_test = W[:, :-1]
        else:
            U_test = np.concatenate([U[:, :r], U[:, r+1:]], axis=1)
            V_test = np.concatenate([V[:, :r], V[:, r+1:]], axis=1)
            W_test = np.concatenate([W[:, :r], W[:, r+1:]], axis=1)
        
        if _verify(U_test, V_test, W_test, n, m, p):
            U, V, W = U_test, V_test, W_test
            rank = U.shape[1]
            # Restart from beginning since indices changed
            r = 0
    
    return U, V, W


def _try_rank_reduction_search(U, V, W, n, m, p, deadline, target_rank=None):
    """
    Try to find a lower-rank decomposition by random perturbation and optimization.
    This is a heuristic approach.
    """
    if target_rank is None:
        target_rank = U.shape[1] - 1
    
    if target_rank < 1:
        return U, V, W
    
    best_U, best_V, best_W = U, V, W
    best_rank = U.shape[1]
    
    # Try random low-rank initializations and see if they can be completed
    # This is a complex optimization problem, so we'll use a simpler heuristic:
    # Try to express the last few columns as combinations of earlier ones.
    
    for start_rank in range(min(best_rank, target_rank + 5), max(1, target_rank - 5), -1):
        if time.time() > deadline:
            break
        if start_rank >= best_rank:
            continue
        
        # Try to find coefficients such that columns [start_rank:] are combinations of [0:start_rank]
        # This is a nonlinear problem, so we'll use a simple approach:
        # Check if the tensor defined by columns [start_rank:] is in the span of [0:start_rank]
        
        U_base = U[:, :start_rank]
        V_base = V[:, :start_rank]
        W_base = W[:, :start_rank]
        
        U_extra = U[:, start_rank:]
        V_extra = V[:, start_rank:]
        W_extra = W[:, start_rank:]
        
        # The tensor from base: T_base[a,b,c,d,i,k] = sum_{r<start_rank} U_base[a,b,r] V_base[c,d,r] W_base[i,k,r]
        # The tensor from extra: T_extra[a,b,c,d,i,k] = sum_{r>=start_rank} U_extra[a,b,r] V_extra[c,d,r] W_extra[i,k,r]
        # We need T_base + T_extra = delta tensor
        # So T_extra = delta - T_base
        # And T_extra must be in the "span" of the base factors (in a tensor sense)
        
        # This is hard to check directly. Instead, let's just try random perturbations.
        pass
    
    return best_U, best_V, best_W


def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition (rank 75)
    U, V, W = _schoolbook_local(n, m, p)
    
    if not _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = n * m * p
    best_U, best_V, best_W = U, V, W
    
    # Try block decomposition with various splits
    if time.time() < deadline:
        rank_bs, U_bs, V_bs, W_bs = _try_block_splits(n, m, p, deadline)
        if rank_bs < best_rank:
            best_rank = rank_bs
            best_U, best_V, best_W = U_bs, V_bs, W_bs
    
    # Try to polish (remove redundant columns)
    if time.time() < deadline:
        best_U, best_V, best_W = _try_polish(best_U, best_V, best_W, n, m, p, deadline)
        best_rank = best_U.shape[1]
    
    # Final verification
    if not _verify(best_U, best_V, best_W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook_local(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
