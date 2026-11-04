import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 150.0 # 150s wall clock limit, keep margin
    
    n, m, p = DIMENSIONS
    
    # Splits
    n_splits = [1, 2]
    m_splits = [1, 2, 2]
    p_splits = [1, 2, 2, 2]
    
    n_offsets = np.cumsum([0] + n_splits)[:-1]
    m_offsets = np.cumsum([0] + m_splits)[:-1]
    p_offsets = np.cumsum([0] + p_splits)[:-1]
    
    # Pre-calculate total rank
    total_rank = 0
    blocks = []
    for i_idx, n_sz in enumerate(n_splits):
        for j_idx, m_sz in enumerate(m_splits):
            for k_idx, p_sz in enumerate(p_splits):
                if n_sz == 2 and m_sz == 2 and p_sz == 2:
                    r = 7
                else:
                    r = n_sz * m_sz * p_sz
                blocks.append((i_idx, j_idx, k_idx, n_sz, m_sz, p_sz, r))
                total_rank += r
                
    # Use numpy arrays for U, V, W
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    for i_idx, j_idx, k_idx, n_sz, m_sz, p_sz, r in blocks:
        if time.time() > deadline - 5:
            break # Safety margin
            
        i_off = n_offsets[i_idx]
        j_off = m_offsets[j_idx]
        k_off = p_offsets[k_idx]
        
        if n_sz == 2 and m_sz == 2 and p_sz == 2:
            # Strassen 2x2x2
            # We need to map the 7 Strassen products to the U, V, W factors.
            # U has shape (2*2, 7) local -> (4, 7)
            # V has shape (2*2, 7) local -> (4, 7)
            # W has shape (2*2, 7) local -> (4, 7)
            
            # Local indices:
            # U rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # V rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # W rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -- Note: W is indexed [k*n+i, r] in global, 
            # but locally for a block, W corresponds to C[k, i]. 
            # In the global W matrix, row index is k*p + i? 
            # Wait, the task says: "output factor follows the official trace convention W[k*n+i,r]"
            # So W row index is `k * n + i`.
            # In the block, local k is 0..1, local i is 0..1.
            # Global W row offset is `k_off * n + i_off`.
            # Local W row index is `lk * n_sz + li`? No, `lk * n + li`?
            # The block is a sub-matrix of the global tensor.
            # The W matrix for the whole tensor has rows corresponding to (k, i).
            # In the block, k ranges from k_off to k_off+p_sz-1.
            # i ranges from i_off to i_off+n_sz-1.
            # So the global row index for local (lk, li) is `(k_off + lk) * n + (i_off + li)`.
            
            # Let's define the Strassen terms explicitly.
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M6 - M7
            
            # We need to populate U, V, W such that sum_r U[:,r] V[:,r] reconstructs the products.
            # Actually, U and V are factors for A and B.
            # U is (n*m, rank). A is (n, m).
            # V is (m*p, rank). B is (m, p).
            # W is (p*n, rank). C is (p, n)?? 
            # Wait, "W[k*n+i,r]". C is usually (n, p) for A(n,m) B(m,p).
            # But here W is (p, n). This implies C is transposed or the convention is different.
            # Standard: C = A B. C is (n, p).
            # If W is (p, n), then W[r, k*n+i] corresponds to C[i, k]?
            # Let's check the tensor definition.
            # Tensor T[i, j, k] = delta. C[i, k] = sum_j A[i, j] B[j, k].
            # Decomposition: C[i, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]?
            # No, the standard tensor decomposition for matrix multiplication is:
            # C[i, k] = sum_{j, r} A[i, j] B[j, k] ... this is not a rank-1 decomposition of the tensor T.
            # The tensor T is defined by T[i, j, k] = 1 if the multiplication involves that term?
            # No, the tensor for matrix multiplication is $T = \sum_{i,j,k} E_{ij} \otimes F_{jk} \otimes G_{ki}$.
            # Rank-r decomposition: $T = \sum_{r=1}^R U_r \otimes V_r \otimes W_r$.
            # Here $U_r$ is a matrix of size $n \times m$, $V_r$ is $m \times p$, $W_r$ is $p \times n$.
            # And $C = \sum_{r=1}^R (U_r \odot A) (V_r \odot B) W_r^T$?
            # No, the standard form is:
            # $C = \sum_{r=1}^R (A \odot U_r) (B \odot V_r) \odot W_r$?
            # Actually, the interface returns U, V, W as factor matrices.
            # U: (n*m, rank). This is the flattened U_r matrices.
            # V: (m*p, rank). Flattened V_r matrices.
            # W: (p*n, rank). Flattened W_r matrices.
            
            # The multiplication is:
            # $C = \sum_{r=1}^R \text{reshape}(U[:, r], (n, m)) \odot A \times \text{reshape}(V[:, r], (m, p)) \odot B \times \text{reshape}(W[:, r], (p, n))^T$?
            # No, usually it is:
            # $C[i, k] = \sum_{r} (\sum_{j} A[i, j] U_r[i, j]) (\sum_{j} B[j, k] V_r[j, k]) W_r[k, i]$
            # Let's assume this standard bilinear form.
            
            # For Strassen:
            # We have 7 products $M_r$.
            # $M_r = (\sum_{i,j} A_{ij} U_{r,ij}) (\sum_{j,k} B_{jk} V_{r,jk})$
            # Then $C_{ik} = \sum_{r} M_r W_{r,ki}$
            
            # Let's fill U, V, W for the 2x2x2 block.
            
            # Term 1: M1 = (A00+A11)(B00+B11)
            # U1: A00=1, A11=1. Others 0.
            # V1: B00=1, B11=1. Others 0.
            # W1: C00 += 1, C11 += 1. So W1[0,0]=1, W1[1,1]=1.
            
            # Term 2: M2 = (A10+A11)B00
            # U2: A10=1, A11=1.
            # V2: B00=1.
            # W2: C10 += 1. W2[0,1]=1? 
            # W index is [k, i]. C10 is i=1, k=0. So W2[0, 1] = 1.
            
            # Term 3: M3 = A00(B01-B11)
            # U3: A00=1.
            # V3: B01=1, B11=-1.
            # W3: C01 += 1. i=0, k=1. W3[1, 0] = 1.
            
            # Term 4: M4 = A11(B10-B00)
            # U4: A11=1.
            # V4: B10=1, B00=-1.
            # W4: C10 += 1, C00 += 1.
            # C10 (i=1, k=0) -> W4[0, 1] = 1.
            # C00 (i=0, k=0) -> W4[0, 0] = 1.
            
            # Term 5: M5 = (A00+A01)B11
            # U5: A00=1, A01=1.
            # V5: B11=1.
            # W5: C01 += 1, C00 -= 1.
            # C01 (i=0, k=1) -> W5[1, 0] = 1.
            # C00 (i=0, k=0) -> W5[0, 0] = -1.
            
            # Term 6: M6 = (A10-A00)(B00+B01)
            # U6: A10=1, A00=-1.
            # V6: B00=1, B01=1.
            # W6: C11 += 1. i=1, k=1. W6[1, 1] = 1.
            
            # Term 7: M7 = (A01-A11)(B10+B11)
            # U7: A01=1, A11=-1.
            # V7: B10=1, B11=1.
            # W7: C00 += 1, C11 -= 1.
            # C00 (i=0, k=0) -> W7[0, 0] = 1.
            # C11 (i=1, k=1) -> W7[1, 1] = -1.

            # Now map to global indices.
            # U global row: i_off + li, j_off + lj. Index: (i_off+li)*m + (j_off+lj).
            # V global row: j_off + lj, k_off + lk. Index: (j_off+lj)*p + (k_off+lk).
            # W global row: k_off + lk, i_off + li. Index: (k_off+lk)*n + (i_off+li).
            
            def add_strassen_term(u_vals, v_vals, w_vals, u_idx, v_idx, w_idx):
                nonlocal col
                for (ui, uj), val in zip(u_idx, u_vals):
                    U[(i_off + ui) * m + (j_off + uj), col] += val
                for (vj, vk), val in zip(v_idx, v_vals):
                    V[(j_off + vj) * p + (k_off + vk), col] += val
                for (wk, wi), val in zip(w_idx, w_vals):
                    W[(k_off + wk) * n + (i_off + wi), col] += val
                col += 1

            # T1
            add_strassen_term([1, 1], [1, 1], [1, 1], 
                              [(0,0), (1,1)], [(0,0), (1,1)], [(0,0), (1,1)])
            # T2
            add_strassen_term([1, 1], [1], [1], 
                              [(1,0), (1,1)], [(0,0)], [(0,1)])
            # T3
            add_strassen_term([1], [1, -1], [1], 
                              [(0,0)], [(0,1), (1,1)], [(1,0)])
            # T4
            add_strassen_term([1], [1, -1], [1, 1], 
                              [(1,1)], [(1,0), (0,0)], [(0,1), (0,0)])
            # T5
            add_strassen_term([1, 1], [1], [1, -1], 
                              [(0,0), (0,1)], [(1,1)], [(1,0), (0,0)])
            # T6
            add_strassen_term([1, -1], [1, 1], [1], 
                              [(1,0), (0,0)], [(0,0), (0,1)], [(1,1)])
            # T7
            add_strassen_term([1, -1], [1, 1], [1, -1], 
                              [(0,1), (1,1)], [(1,0), (1,1)], [(0,0), (1,1)])

        else:
            # Schoolbook
            for ui in range(n_sz):
                for uj in range(m_sz):
                    for uk in range(p_sz):
                        U[(i_off + ui) * m + (j_off + uj), col] = 1
                        V[(j_off + uj) * p + (k_off + uk), col] = 1
                        W[(k_off + uk) * n + (i_off + ui), col] = 1
                        col += 1

    return {"U": U, "V": V, "W": W}
