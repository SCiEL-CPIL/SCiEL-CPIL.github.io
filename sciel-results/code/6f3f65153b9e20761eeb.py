"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    # Indices:
    # A: n x m (rows i, cols j)
    # B: m x p (rows j, cols k)
    # C: n x p (rows i, cols k)
    # Tensor T[i, j, k] = 1 if output C[i, k] depends on A[i, j] * B[j, k]
    # Factorization: sum_r U[i, j, r] * V[j, k, r] * W[i, k, r] = T[i, j, k]
    # Note: W[k*n+i, r] in contract means W index is k*n+i.
    # Contract: W[k*n+i, r] -> W_{i,k,r}
    # U[i*m+j, r] -> U_{i,j,r}
    # V[j*p+k, r] -> V_{j,k,r}
    
    # 1. Initialize with Schoolbook Decomposition (Rank = n*m*p)
    rank = n * m * p
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Term r corresponds to (i, j, k)
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
                r += 1
    
    # Helper to check if a set of columns exactly decomposes the tensor
    def is_valid(U_mat, V_mat, W_mat):
        # Compute T_approx[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need T_approx[i, j, k] == 1 for all i, j, k.
        # Since entries are integers, we can check with integers.
        
        # Efficient check:
        # For each r, we have vectors u_r (n*m), v_r (m*p), w_r (p*n)
        # Contribution to T is sum_r outer(u_r reshaped, v_r reshaped, w_r reshaped)
        
        # Let's reconstruct the tensor T_approx
        # T_approx has shape (n, m, p)
        T_approx = np.zeros((n, m, p), dtype=int)
        
        # Iterate over columns
        for r in range(U_mat.shape[1]):
            u_col = U_mat[:, r]
            v_col = V_mat[:, r]
            w_col = W_mat[:, r]
            
            # Skip if any is zero
            if not (np.any(u_col) and np.any(v_col) and np.any(w_col)):
                continue
                
            # u_col is flattened (n, m)
            # v_col is flattened (m, p)
            # w_col is flattened (p, n) -> needs to be (n, p) for matching indices?
            # W index is k*n + i. So w_col[k*n + i] corresponds to W[i, k].
            # Let's reshape w_col to (p, n) then transpose to (n, p)
            
            u_2d = u_col.reshape((n, m))
            v_2d = v_col.reshape((m, p))
            w_2d = w_col.reshape((p, n)).T # Now (n, p)
            
            # The term for T[i, j, k] is u_2d[i, j] * v_2d[j, k] * w_2d[i, k]
            
            # We can compute this via loops or broadcasting
            # u_2d[i, j] * v_2d[j, k] -> (n, p) matrix M where M[i, k] = sum_j u_2d[i, j] v_2d[j, k]
            # Then T_contrib[i, k] = M[i, k] * w_2d[i, k]
            
            M = u_2d @ v_2d # (n, p)
            T_contrib = M * w_2d # (n, p)
            
            # But T_approx is (n, m, p).
            # Wait, the tensor T[i, j, k] is 1 only if i, j, k are linked.
            # In matrix mult, C[i, k] = sum_j A[i, j] B[j, k].
            # The tensor of the multiplication operation is defined such that
            # T[i, j, k] = 1.
            # The decomposition is: sum_r U[i, j, r] V[j, k, r] W[i, k, r] = 1 for ALL i, j, k.
            # Is that right?
            # Yes, the "multiplication tensor" $T$ has entries $T_{ijk} = 1$.
            # Any decomposition satisfying this computes the matrix product.
            
            # So we just need sum_r ... to equal 1 for every (i, j, k).
            
            # My previous logic added contributions to a (n, p) matrix, which loses the 'j' dimension.
            # I need to accumulate into (n, m, p).
            
            # u_2d is (n, m), v_2d is (m, p), w_2d is (n, p).
            # Term for (i, j, k) is u_2d[i, j] * v_2d[j, k] * w_2d[i, k].
            
            # Let's build a (n, m, p) array for this column.
            # u_2d[:, :, None] * v_2d[None, :, :] * w_2d[:, None, :]
            # Dimensions:
            # u: (n, m, 1)
            # v: (1, m, p)
            # w: (n, 1, p)
            # Result: (n, m, p)
            
            T_col = u_2d[:, :, None] * v_2d[None, :, :] * w_2d[:, None, :]
            T_approx += T_col
            
        # Check if T_approx is all ones
        return np.all(T_approx == 1)

    # Local Search Strategy:
    # 1. Start with Schoolbook.
    # 2. Try to remove a column (set to 0) if the remaining columns still sum to 1.
    # 3. If removal fails, try to replace a column with a "better" one that allows removal of another?
    #    Simpler: Greedy Removal.
    #    Try removing column r. If valid, keep removed. If not, restore.
    #    Repeat until no single removal works.
    #    Then try "Swap": Replace column r with a new random column, then try to remove another column s.
    
    # To speed up is_valid, we can maintain the current sum T_sum.
    # T_sum[i, j, k] = sum_{active r} U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    # Let's maintain T_sum explicitly.
    T_sum = np.zeros((n, m, p), dtype=int)
    active = [True] * rank
    
    # Initialize T_sum
    for r in range(rank):
        if active[r]:
            u_col = U[:, r].reshape((n, m))
            v_col = V[:, r].reshape((m, p))
            w_col = W[:, r].reshape((p, n)).T
            T_sum += u_col[:, :, None] * v_col[None, :, :] * w_col[:, None, :]
    
    # Verify initial
    if not np.all(T_sum == 1):
        # Should not happen
        pass

    current_rank = rank
    
    # Optimization: Pre-compute column contributions
    # contrib[r] is the (n, m, p) tensor for column r.
    # This is expensive to store (rank * n * m * p) = 120 * 4 * 5 * 6 = 14400 ints. Very small.
    contribs = []
    for r in range(rank):
        u_col = U[:, r].reshape((n, m))
        v_col = V[:, r].reshape((m, p))
        w_col = W[:, r].reshape((p, n)).T
        c = u_col[:, :, None] * v_col[None, :, :] * w_col[:, None, :]
        contribs.append(c)
    
    # Greedy Removal Loop
    improved = True
    while improved:
        improved = False
        # Shuffle order to avoid bias
        indices = list(range(rank))
        random.shuffle(indices)
        
        for r in indices:
            if not active[r]:
                continue
            
            # Try removing r
            T_sum -= contribs[r]
            if np.all(T_sum == 1):
                active[r] = False
                current_rank -= 1
                improved = True
            else:
                T_sum += contribs[r]
                
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # If greedy removal stagnates, try "Perturb and Remove"
    # Pick an active column r, modify it slightly (or replace with random small ints), 
    # then try to remove another column s.
    # This allows escaping local minima where no single column is redundant.
    
    # A more powerful move: 
    # 1. Pick two active columns r, s.
    # 2. Set s to 0.
    # 3. Try to adjust r to compensate? 
    #    This is complex.
    
    # Let's try a simpler heuristic:
    # Randomly set a column to 0. If invalid, try to fix it by modifying another column?
    # Or: "2-Swap".
    # 1. Pick r, s.
    # 2. Zero out s.
    # 3. Compute deficit D = T_sum - contribs[r] - contribs[s] - 1? No.
    #    We want sum_{active \ {r,s}} + U'_r V'_r W'_r = 1.
    #    Let R = 1 - sum_{active \ {r,s}}.
    #    We need U'_r V'_r W'_r = R.
    #    If R is rank 1, we can find U', V', W'.
    #    Rank of R (as a tensor) is hard to check.
    
    # Let's stick to "Greedy Removal" + "Random Restart with Pruning".
    # Given the time limit, Greedy Removal on Schoolbook is a strong baseline.
    # For 4x5x6, the rank is likely around 100-110?
    # Schoolbook is 120.
    
    # Let's implement a quick "2-Exchange" search if time permits.
    # But first, let's see how much we prune.
    
    # To make it faster, we can check validity of removal by looking at the support.
    # But full check is O(nmp) = 120 ops. Very fast.
    # 120 columns * 120 checks * 120 ops = ~1.7M ops per pass.
    # We can do thousands of passes in 110s.
    
    # Let's do multiple passes until no improvement.
    
    # Finalize matrices
    # We need to return U, V, W with only the active columns.
    active_indices = [i for i in range(rank) if active[i]]
    
    U_final = U[:, active_indices]
    V_final = V[:, active_indices]
    W_final = W[:, active_indices]
    
    # Convert to lists of lists as per interface?
    # Interface: {'U', 'V', 'W'}: factor matrices of shapes ...
    # The example uses lists. NumPy arrays might be accepted?
    # "factor matrices of shapes (n*m, rank)"
    # Usually, these evaluators accept numpy arrays or lists.
    # The seed program returns lists of lists.
    # Let's return lists of lists to be safe, or numpy arrays if faster.
    # Numpy arrays are usually fine for "matrices".
    # But let's check the seed: `return {"U": U, "V": V, "W": W}` where U is list of lists.
    # I will return numpy arrays. If the evaluator strictly checks type, it might fail.
    # However, "integer/half-integer entries" suggests numerical types.
    # Let's convert to lists of lists to match seed exactly.
    
    U_out = U_final.tolist()
    V_out = V_final.tolist()
    W_out = W_final.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
