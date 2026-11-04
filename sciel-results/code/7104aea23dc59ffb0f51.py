import time
import numpy as np
from fractions import Fraction

N, M, P = 3, 4, 4
NM, MP, PN = N * M, M * P, P * N


def _verify(U, V, W, tol=1e-10):
    """Verify using einsum."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float).reshape(N, M, rank)
    V_arr = np.array(V, dtype=float).reshape(M, P, rank)
    W_arr = np.array(W, dtype=float).reshape(P, N, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol, max_err


def _schoolbook():
    rank = NM * MP * PN
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    c = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][c] = 1
                V[j * P + k][c] = 1
                W[k * N + i][c] = 1
                c += 1
    return {'U': U, 'V': V, 'W': W}


def _strassen_44():
    rank = 44
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    
    s_steps = [
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1, (1, 1): -1}),
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 1): 1, (0, 0): -1}),
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1})
    ]
    
    blocks = [(0, 0, 0, 0, 0, 0), (0, 2, 2, 0, 0, 0), (0, 0, 0, 2, 0, 2), (0, 2, 2, 2, 0, 2)]
    
    col = 0
    for (xi, xj, yj, yk, zi, zk) in blocks:
        for (u_c, v_c, w_c) in s_steps:
            for (a, b), val in u_c.items():
                U[(a + xi) * M + (b + xj)][col] += val
            for (b, c), val in v_c.items():
                V[(b + yj) * P + (c + yk)][col] += val
            for (a, c), val in w_c.items():
                W[(c + zk) * N + (a + zi)][col] += val
            col += 1
    
    for j in range(M):
        for k in range(P):
            U[2 * M + j][col] = 1
            V[j * P + k][col] = 1
            W[k * N + 2][col] = 1
            col += 1
    
    assert col == rank
    return {'U': U, 'V': V, 'W': W}


def _lax_reduce(U, V, W, deadline, max_iterations=100):
    """
    Proper Lax subspace method: iteratively check if any column's tensor
    is in the span of all other columns' tensors, and if so, remove it.
    """
    rank = len(U[0])
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    best_U = [[float(x) for x in row] for row in U_frac]
    best_V = [[float(x) for x in row] for row in V_frac]
    best_W = [[float(x) for x in row] for row in W_frac]
    best_rank = rank
    
    for iteration in range(max_iterations):
        if time.time() > deadline - 10:
            break
        
        found_reduction = False
        current_rank = len(U_frac[0])
        
        for target in range(current_rank - 1, -1, -1):
            if time.time() > deadline - 10:
                break
            
            # Check if column target is in the span of all other columns
            # Build the tensor for column target
            target_tensor = []
            for i in range(NM):
                for j in range(MP):
                    for k in range(PN):
                        target_tensor.append(U_frac[i][target] * V_frac[j][target] * W_frac[k][target])
            
            # Build tensors for all other columns
            other_tensors = []
            for c in range(current_rank):
                if c == target:
                    continue
                t = []
                for i in range(NM):
                    for j in range(MP):
                        for k in range(PN):
                            t.append(U_frac[i][c] * V_frac[j][c] * W_frac[k][c])
                other_tensors.append(t)
            
            # Check if target_tensor is in the span of other_tensors
            # Use Gaussian elimination over the field of fractions
            # Build augmented matrix: [other_tensors | target_tensor]
            # Each row is a tensor entry, each column is a coefficient
            
            if not other_tensors:
                # Only one column left, can't reduce
                continue
            
            num_dims = NM * MP * PN
            num_cols = len(other_tensors)
            
            # Build matrix where each column is a tensor
            # M[i][j] = other_tensors[j][i]
            # Augment with target_tensor as the last column
            
            # Use row-reduced echelon form to check if target is in span
            # Create augmented matrix
            aug = []
            for i in range(num_dims):
                row = [other_tensors[c][i] for c in range(num_cols)]
                row.append(target_tensor[i])
                aug.append(row)
            
            # Gaussian elimination
            rank_aug = 0
            pivot_cols = []
            for col in range(num_cols + 1):
                if col >= len(aug[0]):
                    break
                # Find pivot row
                pivot_row = None
                for r in range(rank_aug, num_dims):
                    if aug[r][col] != 0:
                        pivot_row = r
                        break
                if pivot_row is None:
                    continue
                
                # Swap rows
                aug[rank_aug], aug[pivot_row] = aug[pivot_row], aug[rank_aug]
                
                # Normalize
                pivot_val = aug[rank_aug][col]
                for c2 in range(col, num_cols + 1):
                    aug[rank_aug][c2] = aug[rank_aug][c2] / pivot_val
                
                # Eliminate
                for r in range(num_dims):
                    if r != rank_aug and aug[r][col] != 0:
                        factor = aug[r][col]
                        for c2 in range(col, num_cols + 1):
                            aug[r][c2] = aug[r][c2] - factor * aug[rank_aug][c2]
                
                pivot_cols.append(col)
                rank_aug += 1
            
            # If the target column (last column, index num_cols) has no pivot,
            # it means it's in the span of the other columns
            if num_cols not in pivot_cols:
                # Column target is redundant, remove it
                U_frac = [row[:target] + row[target+1:] for row in U_frac]
                V_frac = [row[:target] + row[target+1:] for row in V_frac]
                W_frac = [row[:target] + row[target+1:] for row in W_frac]
                
                best_rank = len(U_frac[0])
                best_U = [[float(x) for x in row] for row in U_frac]
                best_V = [[float(x) for x in row] for row in V_frac]
                best_W = [[float(x) for x in row] for row in W_frac]
                
                found_reduction = True
                break
        
        if not found_reduction:
            break
    
    return {'U': best_U, 'V': best_V, 'W': best_W}, best_rank


def _random_transform_reduce(U, V, W, deadline, num_trials=50):
    """
    Apply random invertible linear transformations to U, V, W to potentially
    expose hidden linear dependencies, then apply Lax reduction.
    """
    rank = len(U[0])
    best_result = {'U': U[:], 'V': V[:], 'W': W[:]}
    best_rank = rank
    
    for trial in range(num_trials):
        if time.time() > deadline - 15:
            break
        
        # Generate random invertible matrices
        # For U: (NM x rank) -> apply random (rank x rank) invertible matrix on the right
        # For V: (MP x rank) -> apply random (rank x rank) invertible matrix on the right
        # For W: (PN x rank) -> apply random (rank x rank) invertible matrix on the right
        # But we need the product to be preserved: U' = U @ A, V' = V @ B, W' = W @ C
        # where A, B, C are invertible and A^{-1} B^{-1} C^{-1} = I (i.e., A B C = I)
        # Actually, for tensor decomposition, if we apply:
        # U' = U @ A, V' = V @ B, W' = W @ C
        # Then the tensor becomes sum_c U'_i,c V'_j,c W'_k,c
        # = sum_c sum_{c1,c2,c3} U_i,c1 A_{c1,c} V_j,c2 B_{c2,c} W_k,c3 C_{c3,c}
        # This is not generally equal to the original unless A=B=C=I
        # 
        # The correct transformation: if we have a change of basis in the rank space,
        # we need to apply the same transformation to all three:
        # U' = U @ A, V' = V @ A^{-1}, W' = W @ A^{-1}
        # Then sum_c U'_i,c V'_j,c W'_k,c = sum_c sum_{c1,c2,c3} U_i,c1 A_{c1,c} V_j,c2 (A^{-1})_{c2,c} W_k,c3 (A^{-1})_{c3,c}
        # This is still not simply the original.
        #
        # Actually, the proper way: the tensor T = sum_c u_c ⊗ v_c ⊗ w_c
        # If we apply a linear transformation A to the rank space:
        # T' = sum_c (A u_c) ⊗ v_c ⊗ w_c = sum_c sum_{c'} A_{c',c} u_{c'} ⊗ v_c ⊗ w_c
        # This changes the decomposition.
        #
        # The key insight: if there's a linear dependency among the u_c vectors,
        # it might be hidden. By applying random transformations, we might expose it.
        # But we need to maintain the tensor equality.
        #
        # Simpler approach: just try random combinations of columns
        # Pick two random columns c1, c2 and try to replace them with
        # c1 + alpha*c2 and c2 + beta*c1 for random alpha, beta
        # This preserves the span but might change the tensor.
        
        # Actually, let's just try the Lax method with different orderings
        # and see if we get stuck differently
        
        # Shuffle column order
        U_shuffled = [row[:] for row in U]
        V_shuffled = [row[:] for row in V]
        W_shuffled = [row[:] for row in W]
        
        # Random permutation of columns
        perm = list(range(rank))
        np.random.shuffle(perm)
        
        U_shuffled = [row[c] for row in U for c in perm]  # This is wrong
        # Let me fix this
        U_shuffled = [[U[i][perm[j]] for j in range(rank)] for i in range(NM)]
        V_shuffled = [[V[i][perm[j]] for j in range(rank)] for i in range(MP)]
        W_shuffled = [[W[i][perm[j]] for j in range(rank)] for i in range(PN)]
        
        # Apply Lax reduction
        lax_result, lax_rank = _lax_reduce(U_shuffled, V_shuffled, W_shuffled, deadline - 10, max_iterations=30)
        
        if lax_rank < best_rank:
            best_rank = lax_rank
            best_result = lax_result
    
    return best_result, best_rank


def construct():
    start = time.time()
    
    # Start with the known Strassen-based decomposition with rank 44
    result = _strassen_44()
    
    # Verify correctness
    ok, err = _verify(result['U'], result['V'], result['W'])
    if not ok:
        # Fallback to schoolbook (should never happen)
        result = _schoolbook()
        return result
    
    best_result = result
    best_rank = len(result['U'][0])
    
    # Deadline: 140 seconds (10s safety margin from 150s limit)
    deadline = start + 140.0
    
    # Apply Lax subspace reduction
    if time.time() < deadline - 20:
        lax_result, lax_rank = _lax_reduce(result['U'], result['V'], result['W'], deadline)
        if lax_rank < best_rank:
            ok, err = _verify(lax_result['U'], lax_result['V'], lax_result['W'])
            if ok:
                best_result = lax_result
                best_rank = lax_rank
    
    # Try random transformations
    if time.time() < deadline - 10:
        random_result, random_rank = _random_transform_reduce(best_result['U'], best_result['V'], best_result['W'], deadline)
        if random_rank < best_rank:
            ok, err = _verify(random_result['U'], random_result['V'], random_result['W'])
            if ok:
                best_result = random_result
                best_rank = random_rank
    
    # Final verification
    ok, err = _verify(best_result['U'], best_result['V'], best_result['W'])
    if not ok:
        best_result = result
    
    return best_result
