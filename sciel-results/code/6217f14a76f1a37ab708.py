"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS

    def check_time():
        return time.time() > deadline

    def schoolbook():
        """Guaranteed valid schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify_fast(U, V, W, n, m, p):
        """Fast verification using numpy einsum."""
        rank = len(U[0])
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        # on-entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.einsum('r,r,r->', U_np[i*m+j], V_np[j*p+k], W_np[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        # full tensor check
        T = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = 1.0
        if not np.allclose(T, target, atol=1e-9):
            return False
        return True

    def build_block_terms():
        """Return list of 11 terms for M_{2,3,2}."""
        strassen_terms = [
            {'U': [(0,0,1), (1,1,1)], 'V': [(0,0,1), (1,1,1)], 'W': [(0,0,1), (1,1,1)]},
            {'U': [(1,0,1), (1,1,1)], 'V': [(0,0,1)], 'W': [(0,1,1), (1,1,-1)]},
            {'U': [(0,0,1)], 'V': [(0,1,1), (1,1,-1)], 'W': [(1,0,1), (1,1,1)]},
            {'U': [(1,1,1)], 'V': [(1,0,1), (0,0,-1)], 'W': [(0,0,1), (0,1,1)]},
            {'U': [(0,0,1), (0,1,1)], 'V': [(1,1,1)], 'W': [(0,0,-1), (1,0,1)]},
            {'U': [(1,0,1), (0,0,-1)], 'V': [(0,0,1), (0,1,1)], 'W': [(1,1,1)]},
            {'U': [(0,1,1), (1,1,-1)], 'V': [(1,0,1), (1,1,1)], 'W': [(0,0,1)]}
        ]
        outer_terms = [
            {'U': [(0,2,1)], 'V': [(2,0,1)], 'W': [(0,0,1)]},
            {'U': [(0,2,1)], 'V': [(2,1,1)], 'W': [(1,0,1)]},
            {'U': [(1,2,1)], 'V': [(2,0,1)], 'W': [(0,1,1)]},
            {'U': [(1,2,1)], 'V': [(2,1,1)], 'W': [(1,1,1)]}
        ]
        return strassen_terms + outer_terms

    def build_block_decomposition(rank_per_block):
        """Build full U,V,W for p=10 as 5 blocks of size 2, each with rank_per_block terms."""
        num_blocks = p // 2
        block_size = 2
        total_rank = num_blocks * rank_per_block
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        terms = build_block_terms()
        col_offset = 0
        for b in range(num_blocks):
            for term in terms:
                col = col_offset
                for i, j, val in term['U']:
                    U[i * m + j][col] += val
                for j, k_local, val in term['V']:
                    k_global = b * block_size + k_local
                    V[j * p + k_global][col] += val
                for k_local, i, val in term['W']:
                    k_global = b * block_size + k_local
                    W[k_global * n + i][col] += val
                col_offset += 1
        return U, V, W

    # Build and verify the rank-55 block decomposition
    best_result = None
    try:
        U55, V55, W55 = build_block_decomposition(11)
        if verify_fast(U55, V55, W55, n, m, p):
            best_result = {"U": U55, "V": V55, "W": W55}
        else:
            best_result = schoolbook()
    except Exception:
        best_result = schoolbook()

    if check_time():
        return best_result

    # Try to find a lower-rank decomposition via direct search on M_{2,3,10}
    # Strategy: start from rank-55, try to remove one term at a time and solve
    # for the remaining entries to maintain exactness.
    def try_reduce_rank(U_init, V_init, W_init, n, m, p, target_rank, max_attempts=200):
        """Try to reduce rank by removing terms and solving for corrections."""
        rank_init = len(U_init[0])
        U_np = np.array(U_init, dtype=float)
        V_np = np.array(V_init, dtype=float)
        W_np = np.array(W_init, dtype=float)
        
        # Target tensor
        target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Try removing one column at a time
        for col_to_remove in range(rank_init):
            if check_time():
                break
            if rank_init - 1 < target_rank:
                continue
            
            # Keep all columns except col_to_remove
            keep_cols = [c for c in range(rank_init) if c != col_to_remove]
            U_sub = U_np[:, keep_cols]
            V_sub = V_np[:, keep_cols]
            W_sub = W_np[:, keep_cols]
            
            # Compute current tensor with remaining terms
            T_sub = np.einsum('ar,br,cr->abc', U_sub, V_sub, W_sub)
            residual = target - T_sub
            
            # We need to add one more rank-1 term that matches the residual
            # This is generally not possible, but let's check if residual is rank-1
            # A tensor is rank-1 if it can be written as u⊗v⊗w
            # Check by flattening and checking rank
            flat = residual.reshape(-1)
            # For rank-1 check, we need the tensor to be an outer product
            # Reshape to (n*m, m*p) for each fixed c, or similar
            # Simpler: try to find u,v,w such that u[i*m+j]*v[j*p+k]*w[k*n+i] = residual[i*m+j, j*p+k, k*n+i]
            # This is hard in general. Instead, try a few random rank-1 candidates.
            
            # Alternative: try to solve for a single new term that corrects the residual
            # This requires the residual to be exactly rank-1, which is unlikely.
            # So this approach is not very promising. Skip.
            pass
        
        return None

    # Better strategy: use a Kronecker product structure
    # M_{2,3,10} = M_{2,1,1} ⊗ M_{1,3,10}
    # Rank of M_{2,1,1} = 2, Rank of M_{1,3,10} = 30
    # Kronecker rank = 2 * 30 = 60 (worse than 55)
    # But we can use Strassen on the 2x3 part to get rank 7 for M_{2,3,1}
    # Then M_{2,3,10} via Kronecker: rank = 7 * 10 = 70 (worse)
    # So the block-wise Strassen approach (rank 55) is better.
    
    # Try a different block partition: 4 blocks of size 2 + 1 block of size 2
    # Same as before. Try mixing: 3 blocks of size 2 (rank 11 each) + 1 block of size 4
    # M_{2,3,4} rank: use Strassen twice: rank 7*2 = 14? No, that's not right.
    # M_{2,3,4} can be done with rank 11*2 = 22 (two blocks of size 2)
    # Or directly: the rank of M_{2,3,4} is at most 22, but could be lower.
    # Actually, M_{2,3,4} = M_{2,3,2} ⊗ M_{1,1,2}, rank = 11 * 2 = 22
    # So 3 blocks of size 2 (rank 11) + 1 block of size 4 (rank 22) = 33 + 22 = 55. Same.
    
    # The block-wise Strassen approach with rank 55 appears to be optimal for this family.
    # Let's just return the best result we have.
    
    return best_result
# EVOLVE-BLOCK-END
