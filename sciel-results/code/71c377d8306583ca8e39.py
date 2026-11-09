"""
Strategy: Exhaustive tiling search over 2x2x2 sub-blocks.

Instead of a fixed block partition, we consider all 64 possible 2x2x2 sub-blocks
positioned at (i0, j0, k0) for i0 in {0,1,2,3}, j0 in {0,1,2,3}, k0 in {0,1,2,3}.
Each sub-block covers the tensor entries T[i0:i0+2, j0:j0+2, k0:k0+2] (clipped to [0,5)).

We need to select a subset of these sub-blocks such that:
1. Every (i,j,k) in [0,5)^3 is covered by exactly one selected sub-block.
2. The total rank is minimized (Strassen=7 per block, schoolbook=8 per block).

This is a set cover / exact cover problem. We use backtracking with pruning.
The key insight: we want to maximize the number of Strassen blocks (each saves 1 rank
vs schoolbook), so we want as many Strassen blocks as possible in a valid tiling.

Since 5 = 2+2+1, the natural tilings use blocks of size 2x2x2 and 1x1x1 (or mixed).
But we can also have overlapping-free tilings with various 2x2x2 blocks.

Actually, let's think more carefully. The 5x5x5 tensor has 125 entries.
A 2x2x2 block covers 8 entries. A 1x1x1 block covers 1 entry.
We need to tile 125 entries with non-overlapping blocks.

If we use x blocks of 2x2x2 and y blocks of 1x1x1:
8x + y = 125, x + y = number of blocks, rank = 7x + 8y... wait no.

Actually each 2x2x2 sub-block, when decomposed, contributes either 7 (Strassen) or 8 (schoolbook) rank terms.
Each 1x1x1 sub-block contributes 1 rank term.

But the constraint is that the blocks must tile the space exactly (no overlap, full coverage).

Let's use a different approach: since 5 is small, we can do a DFS over the 125 cells,
assigning each cell to a block. But that's 64^125 which is infeasible.

Better approach: The problem is equivalent to tiling a 5x5x5 grid with 2x2x2 and 1x1x1 blocks.
We want to maximize the number of 2x2x2 blocks (since each saves 1 rank vs using 8 individual 1x1x1 blocks... wait).

Actually: one 2x2x2 block with Strassen = rank 7. Eight 1x1x1 blocks = rank 8.
So each 2x2x2 Strassen block saves 1 rank compared to 8 individual terms.

So we want to maximize the number of 2x2x2 Strassen blocks in a valid tiling.

The maximum number of 2x2x2 blocks that can tile a 5x5x5 grid:
In each dimension, we can fit at most 2 non-overlapping 2-blocks (positions 0-1 and 2-3), leaving position 4 uncovered.
So at most 2*2*2 = 8 blocks of size 2x2x2, covering 8*8=64 cells, leaving 125-64=61 cells for 1x1x1 blocks.

But we can do better with mixed sizes. Actually, the question is: what's the maximum number of 2x2x2 blocks in a tiling of 5x5x5?

In 1D: tiling [0,5) with intervals of length 2 and 1. Max number of length-2 intervals: 2 (at positions 0-1 and 2-3, or 1-2 and 3-4).
In 3D: the max number of 2x2x2 blocks is at most 2*2*2=8, but they must be non-overlapping.

With 8 blocks: e.g., at (0,0,0), (0,0,2), (0,2,0), (0,2,2), (2,0,0), (2,0,2), (2,2,0), (2,2,2).
This covers [0,2)x[0,2)x[0,2) etc., leaving the "slabs" at index 4 in each dimension uncovered.

The uncovered region is: {4}x[0,5)x[0,5) union [0,5)x{4}x[0,5) union [0,5)x[0,5)x{4} minus overlaps.
Actually the uncovered cells are those where at least one coordinate is 4 AND not covered by any block.

With the 8 blocks at even positions, uncovered cells are those with at least one coordinate = 4.
Count: total 125 - 64 = 61 cells.

So rank = 8*7 + 61*1 = 56 + 61 = 117.

Can we do better? What if we use a different tiling?

Alternative: use 2x2x2 blocks at odd positions: (1,1,1) covers [1,3)x[1,3)x[1,3).
Then we can also place blocks at (0,0,0)? No, they'd overlap at (1,1,1).

Let's think about this differently. The maximum number of non-overlapping 2x2x2 blocks in a 5x5x5 grid:
This is a 3D packing problem. In each dimension, we can have at most 2 blocks (since 2*2=4 < 5).
The blocks must be at positions 0 or 1 or 2 or 3 in each dimension (start index), but they must not overlap.

In 1D: max 2 non-overlapping intervals of length 2 in [0,5). Positions: (0,1) and (2,3) or (0,1) and (3,4) or (1,2) and (3,4).
So 2 per dimension, giving 8 total in 3D.

Can we get more than 8? No, because in each dimension we're limited to 2.

So max 8 Strassen blocks, rank = 8*7 + 61 = 117.

But wait - can we mix in some 2x2x1 or 2x1x2 blocks? The task says we need integer/half-integer entries.
Actually, the decomposition doesn't require the blocks to be cubic. We could have rectangular blocks.

Hmm, but Strassen's algorithm is specifically for 2x2x2. For rectangular blocks, we'd need different algorithms.

Let me reconsider. The known result from the cell findings is rank 117 with the 2+2+1 partition using Strassen on all eight 2x2x2 blocks.

So the structural change should be: instead of hardcoding the 2+2+1 partition, we should SEARCH over all possible tilings and decompositions to find the minimum rank.

But given the time constraint (120s), a full search is infeasible. Instead, let's try a different structural approach:

Use a GLOBAL algebraic identity. The 5x5x5 multiplication tensor can be embedded in a larger tensor where a known low-rank decomposition exists, and then projected back.

Actually, let me try a completely different approach: use the fact that 5x5 matrix multiplication can be related to 7x7 or other sizes via embedding.

Wait - the directive says: "Apply a global algebraic transformation by embedding the 5x5x5 multiplication tensor into a 7x7x7 tensor, decomposing the latter using the known rank-21 Strassen-like decomposition, and projecting back to the 5x5 subspace to exploit cross-term cancellations that reduce the effective rank below the blockwise 117."

But I don't know a rank-21 decomposition of 7x7x7. The best known for 7x7 is much higher.

Let me try yet another approach: use bilinear algorithms from the literature. The Coppersmith-Winograd algorithm or its variants might give lower rank for 5x5x5.

Actually, for small sizes, the best known bounds are:
- 5x5x5: upper bound 117 (from blockwise Strassen), lower bound around 100.

Let me try a different structural approach: use a GRADIENT-BASED optimization to find a low-rank decomposition.

We parameterize U, V, W as dense matrices and use gradient descent to minimize the error ||T - sum_r u_r ⊗ v_r ⊗ w_r||^2, starting from a random initialization. If we can find a rank-R decomposition that exactly matches T, we're done.

This is a different optimization paradigm: continuous optimization vs. combinatorial search.

Let me implement this with a deadline guard.
"""

import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    n, m, p = 5, 5, 5
    
    # Build the target tensor T[i,j,k] = delta (i.e., T[i,j,k] = 1 if this is a valid matmul index)
    # The matmul tensor: T[i,j,k] = 1 for all (i,j,k) in [0,n)x[0,m)x[0,p)
    # Actually, the matmul tensor is: T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # So T is just all ones of shape (n,m,p) = (5,5,5)
    
    T = np.ones((n, m, p))
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    # First, try the known blockwise Strassen approach (rank 117) as a baseline
    def build_blockwise_strassen():
        """Build the 2+2+1 partition with Strassen on 2x2x2 blocks."""
        U_cols = []
        V_cols = []
        W_cols = []
        
        def add_term(u_idx, v_idx, w_idx):
            u_col = np.zeros(n * m)
            v_col = np.zeros(m * p)
            w_col = np.zeros(p * n)
            for (i, j), c in u_idx.items():
                u_col[i * m + j] = c
            for (j, k), c in v_idx.items():
                v_col[j * p + k] = c
            for (k, i), c in w_idx.items():
                w_col[k * n + i] = c
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
        
        # Strassen terms for 2x2x2
        strassen_terms = [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            ({(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}, {(1,1):1}),
            ({(0,1):1, (1,1):-1}, {(0,0):1, (1,0):1}, {(0,0):1}),
        ]
        
        # 2+2+1 partition: blocks at (0,0,0), (0,0,2), (0,2,0), (0,2,2), (2,0,0), (2,0,2), (2,2,0), (2,2,2)
        # Each is 2x2x2, use Strassen
        block_starts = [(0,0,0), (0,0,2), (0,2,0), (0,2,2), (2,0,0), (2,0,2), (2,2,0), (2,2,2)]
        
        for (i0, j0, k0) in block_starts:
            for u_loc, v_loc, w_loc in strassen_terms:
                u_g = {}
                v_g = {}
                w_g = {}
                for (di, dj), c in u_loc.items():
                    u_g[(i0+di, j0+dj)] = c
                for (dj, dk), c in v_loc.items():
                    v_g[(j0+dj, k0+dk)] = c
                for (dk, di), c in w_loc.items():
                    w_g[(k0+dk, i0+di)] = c
                add_term(u_g, v_g, w_g)
        
        # Remaining cells: those with at least one coordinate = 4
        # For each remaining (i,j,k), add a schoolbook term
        covered = set()
        for (i0, j0, k0) in block_starts:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered.add((i0+di, j0+dj, k0+dk))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        add_term({(i,j):1}, {(j,k):1}, {(k,i):1})
        
        rank = len(U_cols)
        return U_cols, V_cols, W_cols, rank
    
    # Build baseline
    U_base, V_base, W_base, rank_base = build_blockwise_strassen()
    print(f"Blockwise Strassen baseline rank: {rank_base}")
    
    # Verify baseline
    def verify(U_cols, V_cols, W_cols, rank, num_tests=3):
        np.random.seed(123)
        for _ in range(num_tests):
            A = np.random.randint(-3, 3, (n, m)).astype(float)
            B = np.random.randint(-3, 3, (m, p)).astype(float)
            C_expected = A @ B
            C_calc = np.zeros((n, p))
            for r in range(rank):
                u_col = np.array(U_cols[r])
                v_col = np.array(V_cols[r])
                w_col = np.array(W_cols[r])
                alpha = np.zeros(n)
                for i in range(n):
                    for j in range(m):
                        alpha[i] += u_col[i*m+j] * A[i,j]
                beta = np.zeros(p)
                for k in range(p):
                    for j in range(m):
                        beta[k] += v_col[j*p+k] * B[j,k]
                for i in range(n):
                    for k in range(p):
                        C_calc[i,k] += alpha[i] * beta[k] * w_col[k*n+i]
            if not np.allclose(C_calc, C_expected, atol=1e-8):
                return False
        return True
    
    valid_base = verify(U_base, V_base, W_base, rank_base)
    print(f"Baseline valid: {valid_base}")
    
    if not valid_base:
        # Fallback to schoolbook
        rank = n * m * p
        U_fb = np.zeros((n*m, rank))
        V_fb = np.zeros((m*p, rank))
        W_fb = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    best_rank = rank_base
    best_result = {"U": U_base, "V": V_base, "W": W_base}
    
    # Now try gradient-based optimization to find a lower rank decomposition
    # We'll try ranks from 116 down to some minimum, using random restarts
    
    def try_rank_optimization(target_rank, num_restarts=5, max_iters=5000):
        """Try to find a rank-target_rank decomposition using gradient descent."""
        for restart in range(num_restarts):
            if time_left() < 10:
                return None
            
            np.random.seed(restart * 1000 + 42)
            
            # Initialize U, V, W randomly
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            # Compute the decomposed tensor
            def compute_tensor(U, V, W):
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                T_calc = np.zeros((n, m, p))
                for r in range(target_rank):
                    u_col = U[:, r].reshape(n, m)
                    v_col = V[:, r].reshape(m, p)
                    w_col = W[:, r].reshape(p, n)
                    # T[i,j,k] += u_col[i,j] * v_col[j,k] * w_col[k,i]
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T_calc[i,j,k] += u_col[i,j] * v_col[j,k] * w_col[k,i]
                return T_calc
            
            # Initial error
            T_calc = compute_tensor(U, V, W)
            error = np.sum((T_calc - T)**2)
            
            if error < 1e-10:
                return U, V, W
            
            # Gradient descent
            lr = 0.01
            for it in range(max_iters):
                if time_left() < 5:
                    break
                
                # Compute gradients
                T_calc = compute_tensor(U, V, W)
                diff = T_calc - T  # shape (n,m,p)
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for r in range(target_rank):
                    u_col = U[:, r].reshape(n, m)
                    v_col = V[:, r].reshape(m, p)
                    w_col = W[:, r].reshape(p, n)
                    
                    for i in range(n):
                        for j in range(m):
                            # dU[i*m+j, r] = sum_k diff[i,j,k] * v_col[j,k] * w_col[k,i]
                            for k in range(p):
                                dU[i*m+j, r] += diff[i,j,k] * v_col[j,k] * w_col[k,i]
                    
                    for j in range(m):
                        for k in range(p):
                            # dV[j*p+k, r] = sum_i diff[i,j,k] * u_col[i,j] * w_col[k,i]
                            for i in range(n):
                                dV[j*p+k, r] += diff[i,j,k] * u_col[i,j] * w_col[k,i]
                    
                    for k in range(p):
                        for i in range(n):
                            # dW[k*n+i, r] = sum_j diff[i,j,k] * u_col[i,j] * v_col[j,k]
                            for j in range(m):
                                dW[k*n+i, r] += diff[i,j,k] * u_col[i,j] * v_col[j,k]
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                if it % 500 == 0:
                    T_calc = compute_tensor(U, V, W)
                    error = np.sum((T_calc - T)**2)
                    if error < 1e-8:
                        break
                
                # Adaptive learning rate
                if it % 1000 == 0 and it > 0:
                    lr *= 0.5
            
            T_calc = compute_tensor(U, V, W)
            error = np.sum((T_calc - T)**2)
            
            if error < 1e-6:
                # Try to round to nearest integers/half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                T_calc_r = compute_tensor(U_rounded, V_rounded, W_rounded)
                error_r = np.sum((T_calc_r - T)**2)
                
                if error_r < 1e-4:
                    return U_rounded, V_rounded, W_rounded
                else:
                    return U, V, W
        
        return None
    
    # Try to find a decomposition with rank < 117
    for target_rank in range(116, 100, -1):
        if time_left() < 20:
            break
        
        print(f"Trying rank {target_rank}...")
        result = try_rank_optimization(target_rank, num_restarts=3, max_iters=2000)
        
        if result is not None:
            U_opt, V_opt, W_opt = result
            rank = target_rank
            
            # Verify
            valid = verify(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), rank, num_tests=3)
            if valid:
                print(f"Found valid decomposition with rank {rank}!")
                best_rank = rank
                best_result = {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
                break
            else:
                print(f"Rank {target_rank} result failed verification.")
    
    print(f"Best rank found: {best_rank}")
    
    # Convert to lists if needed
    U_out = best_result["U"]
    V_out = best_result["V"]
    W_out = best_result["W"]
    
    if isinstance(U_out, np.ndarray):
        U_out = U_out.tolist()
    if isinstance(V_out, np.ndarray):
        V_out = V_out.tolist()
    if isinstance(W_out, np.ndarray):
        W_out = W_out.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
