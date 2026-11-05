import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 5, 5, 5

    # Partitioning strategy: 5 -> 2 + 2 + 1
    # This allows using Strassen (rank 7) for 2x2x2 subproblems.
    # Total rank: 8 blocks of 2x2x2 (rank 7 each) + 19 blocks of other sizes (schoolbook rank).
    # Schoolbook total is 125.
    # 8 blocks of 2x2x2 have schoolbook rank 8 each. Using Strassen saves 1 rank per block.
    # Total saving = 8.
    # Final Rank = 125 - 8 = 117.

    partition = [2, 2, 1]
    
    # Precompute offsets
    offsets = [0]
    for s in partition:
        offsets.append(offsets[-1] + s)

    # Define Strassen factors for 2x2x2
    # Each factor is a tuple of three 4-element vectors (flattened 2x2 matrices)
    # u, v, w correspond to U, V, W matrices in the tensor decomposition
    # Convention: 
    # U is n*m x rank. Index: row = i*m + j.
    # V is m*p x rank. Index: row = j*p + k.
    # W is p*n x rank. Index: row = k*n + i.
    # For a local 2x2x2 block at global offsets (oi, oj, ok):
    # Local indices li, lj, lk in {0,1}.
    # Global U index: (oi+li)*m + (oj+lj)
    # Global V index: (oj+lj)*p + (ok+lk)
    # Global W index: (ok+lk)*n + (oi+li)
    
    strassen_factors = [
        ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)),
        ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, -1)),
        ((1, 0, 0, 0), (0, 1, 0, -1), (0, 1, 0, 1)),
        ((0, 0, 0, 1), (-1, 0, 1, 0), (1, 0, 1, 0)),
        ((1, 1, 0, 0), (0, 0, 0, 1), (-1, 1, 0, 0)),
        ((-1, 0, 1, 0), (1, 1, 0, 0), (0, 0, 0, 1)),
        ((0, 1, 0, -1), (0, 0, 1, 1), (1, 0, 0, 0)),
    ]

    # Calculate total rank
    total_rank = 0
    block_specs = []
    
    for i, ni in enumerate(partition):
        for j, mj in enumerate(partition):
            for k, pk in enumerate(partition):
                if ni == 2 and mj == 2 and pk == 2:
                    rank_block = 7
                    is_strassen = True
                else:
                    rank_block = ni * mj * pk
                    is_strassen = False
                total_rank += rank_block
                block_specs.append((i, j, k, ni, mj, pk, rank_block, is_strassen))

    # Initialize matrices as lists of lists for easy manipulation, or use numpy arrays
    # Using numpy arrays for speed and cleanliness
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)

    col = 0
    for spec in block_specs:
        i, j, k, ni, mj, pk, rank_block, is_strassen = spec
        
        if time.time() > deadline:
            # If we hit deadline, we still need to return a valid result.
            # Since this is deterministic and fast, we should not hit deadline.
            # But to be safe, we break. The partial result might be invalid if not complete.
            # However, the task requires exact decomposition.
            # Given the small size, this loop is very fast.
            pass

        if is_strassen:
            oi, oj, ok = offsets[i], offsets[j], offsets[k]
            
            for factor_idx in range(7):
                u_flat, v_flat, w_flat = strassen_factors[factor_idx]
                
                # Update U
                for li in range(2):
                    for lj in range(2):
                        val = u_flat[li * 2 + lj]
                        if val != 0:
                            u_idx = (oi + li) * m + (oj + lj)
                            U[u_idx, col] += val
                            
                # Update V
                for lj in range(2):
                    for lk in range(2):
                        val = v_flat[lj * 2 + lk]
                        if val != 0:
                            v_idx = (oj + lj) * p + (ok + lk)
                            V[v_idx, col] += val

                # Update W
                for lk in range(2):
                    for li in range(2):
                        val = w_flat[lk * 2 + li]
                        if val != 0:
                            w_idx = (ok + lk) * n + (oi + li)
                            W[w_idx, col] += val

                col += 1
        else:
            oi, oj, ok = offsets[i], offsets[j], offsets[k]
            for li in range(ni):
                for lj in range(mj):
                    for lk in range(pk):
                        u_idx = (oi + li) * m + (oj + lj)
                        v_idx = (oj + lj) * p + (ok + lk)
                        w_idx = (ok + lk) * n + (oi + li)
                        
                        U[u_idx, col] = 1.0
                        V[v_idx, col] = 1.0
                        W[w_idx, col] = 1.0
                        col += 1

    # Convert to lists to match expected interface if necessary, 
    # though numpy arrays are usually accepted or convertible.
    # The prompt implies 'factor matrices of shapes ...'. 
    # Returning lists of lists is safer if strict type checking is done against list.
    # However, standard practice in these tasks often accepts arrays.
    # Let's return lists to be safe with the "integer/half-integer entries" description 
    # which might imply simple numeric types in a list structure.
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
