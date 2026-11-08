"""Matrix multiplication 2x4x5 rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    _start = time.time()
    _deadline = _start + 118

    n, m, p = DIMENSIONS  # n=2, m=4, p=5

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    def build_strassen_36():
        rank = 36
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        def strassen_block(y, z, w):
            nonlocal col
            y00, y01, y10, y11 = y
            z00, z01, z10, z11 = z

            U[u_idx(*y00)][col] = 1; U[u_idx(*y11)][col] = 1
            V[v_idx(*z00)][col] = 1; V[v_idx(*z11)][col] = 1
            W[w_idx(*w[0])][col] += 1; W[w_idx(*w[3])][col] += 1
            col += 1

            U[u_idx(*y10)][col] = 1; U[u_idx(*y11)][col] = 1
            V[v_idx(*z00)][col] = 1
            W[w_idx(*w[2])][col] += 1; W[w_idx(*w[3])][col] -= 1
            col += 1

            U[u_idx(*y00)][col] = 1
            V[v_idx(*z01)][col] = 1; V[v_idx(*z11)][col] = -1
            W[w_idx(*w[1])][col] += 1; W[w_idx(*w[3])][col] += 1
            col += 1

            U[u_idx(*y11)][col] = 1
            V[v_idx(*z10)][col] = 1; V[v_idx(*z00)][col] = -1
            W[w_idx(*w[0])][col] += 1; W[w_idx(*w[2])][col] += 1
            col += 1

            U[u_idx(*y00)][col] = 1; U[u_idx(*y01)][col] = 1
            V[v_idx(*z11)][col] = 1
            W[w_idx(*w[0])][col] -= 1; W[w_idx(*w[1])][col] += 1
            col += 1

            U[u_idx(*y10)][col] = 1; U[u_idx(*y00)][col] = -1
            V[v_idx(*z00)][col] = 1; V[v_idx(*z01)][col] = 1
            W[w_idx(*w[3])][col] += 1
            col += 1

            U[u_idx(*y01)][col] = 1; U[u_idx(*y11)][col] = -1
            V[v_idx(*z10)][col] = 1; V[v_idx(*z11)][col] = 1
            W[w_idx(*w[0])][col] += 1
            col += 1

        strassen_block(
            [(0,0),(0,1),(1,0),(1,1)],
            [(0,0),(0,1),(1,0),(1,1)],
            [(0,0),(1,0),(0,1),(1,1)]
        )
        strassen_block(
            [(0,2),(0,3),(1,2),(1,3)],
            [(2,0),(2,1),(3,0),(3,1)],
            [(0,0),(1,0),(0,1),(1,1)]
        )
        strassen_block(
            [(0,0),(0,1),(1,0),(1,1)],
            [(0,2),(0,3),(1,2),(1,3)],
            [(2,0),(3,0),(2,1),(3,1)]
        )
        strassen_block(
            [(0,2),(0,3),(1,2),(1,3)],
            [(2,2),(2,3),(3,2),(3,3)],
            [(2,0),(3,0),(2,1),(3,1)]
        )

        # 2x2 * 2x1 blocks (rank 4 each)
        U[u_idx(0,0)][col] = 1; V[v_idx(0,4)][col] = 1; W[w_idx(4,0)][col] += 1; col += 1
        U[u_idx(0,1)][col] = 1; V[v_idx(1,4)][col] = 1; W[w_idx(4,0)][col] += 1; col += 1
        U[u_idx(1,0)][col] = 1; V[v_idx(0,4)][col] = 1; W[w_idx(4,1)][col] += 1; col += 1
        U[u_idx(1,1)][col] = 1; V[v_idx(1,4)][col] = 1; W[w_idx(4,1)][col] += 1; col += 1

        U[u_idx(0,2)][col] = 1; V[v_idx(2,4)][col] = 1; W[w_idx(4,0)][col] += 1; col += 1
        U[u_idx(0,3)][col] = 1; V[v_idx(3,4)][col] = 1; W[w_idx(4,0)][col] += 1; col += 1
        U[u_idx(1,2)][col] = 1; V[v_idx(2,4)][col] = 1; W[w_idx(4,1)][col] += 1; col += 1
        U[u_idx(1,3)][col] = 1; V[v_idx(3,4)][col] = 1; W[w_idx(4,1)][col] += 1; col += 1

        return U, V, W, rank

    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        num_U = len(U)
        num_V = len(V)
        num_W = len(W)
        rank = len(U[0])
        
        for a in range(num_U):
            i, j = divmod(a, m)
            for b in range(num_V):
                j2, k = divmod(b, p)
                if j != j2:
                    continue
                c = k * n + i
                target = 1
                computed = 0
                for r in range(rank):
                    computed += U[a][r] * V[b][r] * W[c][r]
                if abs(computed - target) > 1e-9:
                    return False
        return True

    # Get baseline
    U0, V0, W0, rank0 = build_strassen_36()
    assert verify(U0, V0, W0), "Baseline verification failed"

    best_U, best_V, best_W, best_rank = U0, V0, W0, rank0

    # Try to reduce rank by merging columns
    # Key insight: if two columns have the same U and V patterns (up to scaling),
    # we can merge them if the W patterns are compatible
    
    # First, let's try a more aggressive approach:
    # Use the fact that the last 8 columns (the 2x2x1 blocks) might be mergeable
    # with some of the Strassen columns
    
    # Actually, let's try a different strategy:
    # The 2x4x5 tensor can be viewed as:
    # C = A * B where A is 2x4, B is 4x5
    # Split B into B1 (4x4) and B2 (4x1)
    # C1 = A * B1 (2x4) and C2 = A * B2 (2x1)
    # C1 uses rank 28 (4 Strassen blocks of 7)
    # C2 uses rank 8 (2 blocks of 4)
    # Total: 36
    
    # Can we share computation between C1 and C2?
    # The last column of B (B[:,4]) is used in both C1 (as part of B1[:,4] if we extended)
    # Wait, no. B1 is B[:,0:4] and B2 is B[:,4:5]
    # So there's no overlap in the B matrix indices.
    
    # However, we can try to find a lower rank decomposition of 2x4x4.
    # Known: rank(M_{2,4,4}) = 28? Or can we do better?
    # Actually, I recall that rank(M_{2,4,4}) might be 24 or 26.
    # Let's try to find a rank-24 decomposition for 2x4x4.
    
    # Alternative: try to reduce the rank of the 2x4x1 part.
    # rank(M_{2,4,1}) = 8 (since it's just 8 independent products)
    # But wait, C[i,4] = sum_j A[i,j] * B[j,4]
    # This is 4 products for each of 2 outputs, so 8 independent terms.
    # Rank is exactly 8.
    
    # So the question is: can we reduce rank(M_{2,4,4}) below 28?
    # Let's try a randomized search for a lower rank decomposition.
    
    def try_rank_reduction(target_rank):
        """Try to find a decomposition of rank target_rank."""
        if time.time() > _deadline - 30:
            return None
        
        # Use the structure: split into 2x2x4 and 2x2x4 blocks
        # Actually, let's try a direct approach with random initialization
        
        num_U = n * m
        num_V = m * p
        num_W = p * n
        
        # Build the target tensor as a 3D array
        # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
        T = np.zeros((num_U, num_V, num_W))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[u_idx(i,j), v_idx(j,k), w_idx(k,i)] = 1
        
        # Try random initialization and ALS
        for attempt in range(10):
            if time.time() > _deadline - 30:
                return None
            
            np.random.seed(attempt * 12345 + 7)
            
            # Initialize with random values
            U_rand = np.random.randn(num_U, target_rank) * 0.1
            V_rand = np.random.randn(num_V, target_rank) * 0.1
            W_rand = np.random.randn(num_W, target_rank) * 0.1
            
            # ALS iterations
            for it in range(50):
                if time.time() > _deadline - 30:
                    return None
                
                # Update U
                for a in range(num_U):
                    # T[a,:,:] = sum_r U[a,r] * V[:,r] * W[:,r]
                    # For fixed a: T[a,b,c] = sum_r U[a,r] * (V[b,r] * W[c,r])
                    # Let M_r = outer(V[:,r], W[:,r])
                    # Then T[a,:,:] = sum_r U[a,r] * M_r
                    # This is a linear system: U[a,:] * (M_r reshaped) = T[a,:,:]
                    # Actually, for each r, we have a matrix M_r of shape (num_V, num_W)
                    # We want: sum_r U[a,r] * M_r = T[a,:,:]
                    # Flatten: sum_r U[a,r] * M_r.flatten() = T[a,:,:].flatten()
                    # This is: U[a,:] @ M_flat = T_flat where M_flat is (target_rank, num_V*num_W)
                    
                    M_flat = np.zeros((target_rank, num_V * num_W))
                    for r in range(target_rank):
                        M_flat[r, :] = (V_rand[:, r] * W_rand[:, r]).flatten()
                    
                    # Solve: U[a,:] @ M_flat = T[a,:,:].flatten()
                    # This is an underdetermined system if num_V*num_W > target_rank
                    # Use least squares
                    try:
                        U_rand[a, :] = np.linalg.lstsq(M_flat, T[a, :, :].flatten(), rcond=None)[0]
                    except:
                        continue
                
                # Update V
                for b in range(num_V):
                    M_flat = np.zeros((target_rank, num_U * num_W))
                    for r in range(target_rank):
                        M_flat[r, :] = (U_rand[:, r] * W_rand[:, r]).flatten()
                    try:
                        V_rand[b, :] = np.linalg.lstsq(M_flat, T[:, b, :].flatten(), rcond=None)[0]
                    except:
                        continue
                
                # Update W
                for c in range(num_W):
                    M_flat = np.zeros((target_rank, num_U * num_V))
                    for r in range(target_rank):
                        M_flat[r, :] = (U_rand[:, r] * V_rand[:, r]).flatten()
                    try:
                        W_rand[c, :] = np.linalg.lstsq(M_flat, T[:, :, c].flatten(), rcond=None)[0]
                    except:
                        continue
            
            # Check if we got an exact (or near-exact) decomposition
            # And if the entries are close to integers or half-integers
            U_list = [[round(U_rand[i, r], 6) for r in range(target_rank)] for i in range(num_U)]
            V_list = [[round(V_rand[i, r], 6) for r in range(target_rank)] for i in range(num_V)]
            W_list = [[round(W_rand[i, r], 6) for r in range(target_rank)] for i in range(num_W)]
            
            if verify(U_list, V_list, W_list):
                return U_list, V_list, W_list, target_rank
        
        return None
    
    # Try ranks from 35 down to 28
    for r in range(35, 27, -1):
        if time.time() > _deadline - 40:
            break
        result = try_rank_reduction(r)
        if result:
            best_U, best_V, best_W, best_rank = result
            break
    
    # If we found a better rank, return it; otherwise return the baseline
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
