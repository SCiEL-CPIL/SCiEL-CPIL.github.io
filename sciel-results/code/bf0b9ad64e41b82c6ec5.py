"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Identify non-zero entries of the multiplication tensor
    # Indices: a = i*m + j, b = j*p + k, c = k*n + i
    # T[a, b, c] = 1
    nonzero_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero_indices.append((a, b, c))
    
    N_nonzero = len(nonzero_indices)
    
    # Generate schoolbook as fallback
    U_sb = np.zeros((nm, schoolbook_rank))
    V_sb = np.zeros((mp, schoolbook_rank))
    W_sb = np.zeros((pn, schoolbook_rank))
    for idx, (a, b, c) in enumerate(nonzero_indices):
        U_sb[a, idx] = 1
        V_sb[b, idx] = 1
        W_sb[c, idx] = 1
        
    best_result = {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
    best_rank = schoolbook_rank
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Search for lower rank
    # Lower bound: max(nm, mp, pn) is not a valid lower bound for tensor rank, 
    # but a heuristic starting point. Actually, rank >= max(nm, mp, pn) is not true.
    # Rank >= n*m*p / max(n,m,p)? No.
    # Let's try ranks from a reasonable estimate.
    # For 4x4x9, known bounds might be around 100-120?
    # We will scan down from schoolbook.
    
    for target_rank in range(schoolbook_rank - 1, 30, -1):
        if time.time() > deadline:
            break
            
        # Try a few random initializations for this rank
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + target_rank * 100 + attempt)
            
            # Initialize with small random values
            U = np.random.randn(nm, target_rank) * 0.1
            V = np.random.randn(mp, target_rank) * 0.1
            W = np.random.randn(pn, target_rank) * 0.1
            
            # Optimization loop
            lr = 0.01
            prev_err = float('inf')
            stuck_count = 0
            
            for step in range(1000):
                if time.time() > deadline:
                    break
                    
                # Compute Y = sum_c U_c \otimes V_c \otimes W_c
                # Y is a 3D tensor of shape (nm, mp, pn)
                # We only need the error at non-zero indices and the norm of the rest?
                # Actually, E = ||Y - T||_F^2.
                # T is sparse.
                
                # Efficiently compute Y at non-zero indices and the full norm?
                # Computing full Y is expensive if we do it naively.
                # But nm*mp*pn = 16*36*36 = 20736.
                # rank r.
                # Y = sum_c outer(U[:,c], outer(V[:,c], W[:,c]))
                
                # Let's compute the error at non-zero indices first
                # Y_nonzero = np.zeros(N_nonzero)
                # for c in range(target_rank):
                #     uc = U[:, c]
                #     vc = V[:, c]
                #     wc = W[:, c]
                #     # Extract values at non-zero indices
                #     # a_idx, b_idx, c_idx
                #     a_idx = [x[0] for x in nonzero_indices]
                #     b_idx = [x[1] for x in nonzero_indices]
                #     c_idx = [x[2] for x in nonzero_indices]
                #     Y_nonzero += uc[a_idx] * vc[b_idx] * wc[c_idx]
                
                # This loop is slow. Let's vectorize.
                a_idx = np.array([x[0] for x in nonzero_indices])
                b_idx = np.array([x[1] for x in nonzero_indices])
                c_idx = np.array([x[2] for x in nonzero_indices])
                
                # U_sel: (N_nonzero, r)
                U_sel = U[a_idx, :]
                V_sel = V[b_idx, :]
                W_sel = W[c_idx, :]
                
                # Y_nonzero: (N_nonzero,)
                # Y_nonzero = sum_c U_sel[:,c] * V_sel[:,c] * W_sel[:,c]
                Y_nonzero = np.einsum('ic,jc,kc->c', U_sel, V_sel, W_sel) # Wait, this sums over c? No.
                # We want sum over c of product.
                # U_sel is (N, r). V_sel is (N, r). W_sel is (N, r).
                # Y_nonzero = np.sum(U_sel * V_sel * W_sel, axis=1)
                
                diff_nonzero = Y_nonzero - 1.0
                
                # Now we need the gradient.
                # grad_U[a, c] = 2 * sum_{b,c'} (Y_{a,b,c'} - T_{a,b,c'}) V[b, c] W[c', c]
                # This requires summing over all b, c'.
                # This is effectively a tensor contraction.
                
                # Let's compute the full error tensor E_{a,b,c} = Y_{a,b,c} - T_{a,b,c}
                # Y_{a,b,c} = sum_c' U[a, c'] V[b, c'] W[c, c']
                
                # We can compute the gradient using matrix multiplications.
                # grad_U = 2 * (E \cdot (V \otimes W)^T) ?
                # E is (nm, mp, pn).
                # V is (mp, r). W is (pn, r).
                # We want grad_U[a, c] = sum_{b, c'} E[a, b, c'] V[b, c] W[c', c]
                # Reshape E to (nm, mp*pn).
                # Reshape V and W?
                # Let M_VW be a matrix of size (mp*pn, r) where M_VW[b*pn+c, c'] = V[b, c'] W[c, c']?
                # No, the sum is over b and c'.
                # grad_U[a, c] = sum_{b} sum_{c'} E[a, b, c'] V[b, c] W[c', c]
                # This is not a simple matrix product because c' in W is the same index as the summation index for E's last dim?
                # Wait, E[a, b, c'] depends on c'.
                # V[b, c] depends on c (target gradient index).
                # W[c', c] depends on c (target gradient index) and c' (summation).
                
                # Let's swap indices.
                # grad_U[a, c] = sum_{b, c'} E[a, b, c'] V[b, c] W[c', c]
                # This looks like a batched matrix multiplication or einsum.
                # np.einsum('abc,bd,cd->ad', E, V, W)
                
                # Computing E is the bottleneck.
                # E = Y - T.
                # Y = sum_c' U[:, c'] \otimes V[:, c'] \otimes W[:, c']
                
                # If we compute Y explicitly as a 3D array:
                # Y = np.einsum('ac,bc,dc->abd', U, V, W)
                # This creates an array of size nm * mp * pn.
                # 16 * 36 * 36 = 20736 floats. Very small.
                
                Y_full = np.einsum('ac,bc,dc->abd', U, V, W)
                
                # Create T_full (sparse)
                # Instead of creating a full T_full, we can compute E_full = Y_full and then subtract 1 at non-zero indices.
                E_full = Y_full.copy()
                E_full[a_idx, b_idx, c_idx] -= 1.0
                
                # Compute gradients
                grad_U = 2.0 * np.einsum('abc,bd,cd->ad', E_full, V, W)
                grad_V = 2.0 * np.einsum('abc,ad,cd->bd', E_full, U, W)
                grad_W = 2.0 * np.einsum('abc,ac,bd->cd', E_full, U, V)
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Compute error
                err = np.sum(E_full ** 2)
                
                if err < 1e-6:
                    # Found a solution
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    best_rank = target_rank
                    break
                
                if err > prev_err * 1.01:
                    stuck_count += 1
                    if stuck_count > 10:
                        break
                else:
                    stuck_count = 0
                prev_err = err
            
            if best_rank == target_rank:
                break
        
        if best_rank == target_rank:
            break
        
    return best_result
# EVOLVE-BLOCK-END
