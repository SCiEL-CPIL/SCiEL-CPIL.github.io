import time
import numpy as np
import random

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    nm, mp, pn = n * m, m * p, p * n  # 15, 35, 21
    
    def verify(U, V, W):
        """Verify that the decomposition is correct using einsum."""
        try:
            # Target tensor T[a,b,c] is 1 if a=i*m+j, b=j*p+k, c=k*n+i
            # We check sum_r U[a,r] V[b,r] W[c,r] == T[a,b,c]
            computed = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            
            # Construct target tensor
            T = np.zeros((nm, mp, pn))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        T[a, b, c] = 1.0
                        
            if not np.allclose(computed, T, atol=1e-9):
                return False
            return True
        except Exception:
            return False

    def build_decomposition(blocks):
        """
        Build U, V, W given a list of disjoint 2x2x2 blocks.
        Each block is a tuple (i0, i1, j0, j1, k0, k1).
        """
        # Determine total rank: 7 per block + 1 per remaining term
        # Total terms = n*m*p = 105
        # Each block covers 8 terms, replaces with 7 rank-1 terms.
        # Savings = 1 per block.
        num_blocks = len(blocks)
        remaining_terms = n * m * p - 8 * num_blocks
        rank = 7 * num_blocks + remaining_terms
        
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        
        col = 0
        
        # Helper to add a rank-1 term
        def add_term(u_indices, v_indices, w_indices, coeffs):
            nonlocal col
            if col >= rank:
                return False
            for idx, coeff in zip(u_indices, coeffs[0]):
                U[idx, col] += coeff
            for idx, coeff in zip(v_indices, coeffs[1]):
                V[idx, col] += coeff
            for idx, coeff in zip(w_indices, coeffs[2]):
                W[idx, col] += coeff
            col += 1
            return True
        
        # Mark covered indices
        covered = set()
        
        # Process Strassen blocks
        for i0, i1, j0, j1, k0, k1 in blocks:
            # Indices in global space
            # U index: i*m + j
            # V index: j*p + k
            # W index: k*n + i
            
            # M1: (A[i0,j0]+A[i1,j1])(B[j0,k0]+B[j1,k1]) -> C[k0,i0]+=M1, C[k1,i1]+=M1
            u_idx = [i0*m+j0, i1*m+j1]
            v_idx = [j0*p+k0, j1*p+k1]
            w_idx = [k0*n+i0, k1*n+i1]
            add_term(u_idx, v_idx, w_idx, ([1.0, 1.0], [1.0, 1.0], [1.0, 1.0]))
            
            # M2: (A[i1,j0]+A[i1,j1])B[j0,k0] -> C[k0,i1]+=M2, C[k1,i1]-=M2
            u_idx = [i1*m+j0, i1*m+j1]
            v_idx = [j0*p+k0]
            w_idx = [k0*n+i1, k1*n+i1]
            add_term(u_idx, v_idx, w_idx, ([1.0, 1.0], [1.0], [1.0, -1.0]))
            
            # M3: A[i0,j0](B[j0,k1]-B[j1,k1]) -> C[k1,i0]+=M3, C[k1,i1]+=M3
            u_idx = [i0*m+j0]
            v_idx = [j0*p+k1, j1*p+k1]
            w_idx = [k1*n+i0, k1*n+i1]
            add_term(u_idx, v_idx, w_idx, ([1.0], [1.0, -1.0], [1.0, 1.0]))
            
            # M4: A[i1,j1](B[j1,k0]-B[j0,k0]) -> C[k0,i0]+=M4, C[k0,i1]+=M4
            u_idx = [i1*m+j1]
            v_idx = [j1*p+k0, j0*p+k0]
            w_idx = [k0*n+i0, k0*n+i1]
            add_term(u_idx, v_idx, w_idx, ([1.0], [1.0, -1.0], [1.0, 1.0]))
            
            # M5: (A[i0,j0]+A[i0,j1])B[j1,k1] -> C[k0,i0]-=M5, C[k1,i0]+=M5
            u_idx = [i0*m+j0, i0*m+j1]
            v_idx = [j1*p+k1]
            w_idx = [k0*n+i0, k1*n+i0]
            add_term(u_idx, v_idx, w_idx, ([1.0, 1.0], [1.0], [-1.0, 1.0]))
            
            # M6: (A[i1,j0]-A[i0,j0])(B[j0,k0]+B[j0,k1]) -> C[k1,i1]+=M6
            u_idx = [i1*m+j0, i0*m+j0]
            v_idx = [j0*p+k0, j0*p+k1]
            w_idx = [k1*n+i1]
            add_term(u_idx, v_idx, w_idx, ([1.0, -1.0], [1.0, 1.0], [1.0]))
            
            # M7: (A[i0,j1]-A[i1,j1])(B[j1,k0]+B[j1,k1]) -> C[k0,i0]+=M7
            u_idx = [i0*m+j1, i1*m+j1]
            v_idx = [j1*p+k0, j1*p+k1]
            w_idx = [k0*n+i0]
            add_term(u_idx, v_idx, w_idx, ([1.0, -1.0], [1.0, 1.0], [1.0]))
            
            # Mark all 8 (i,j,k) tuples as covered
            for i in [i0, i1]:
                for j in [j0, j1]:
                    for k in [k0, k1]:
                        covered.add((i, j, k))
                        
        # Process remaining schoolbook terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        u_idx = [i*m+j]
                        v_idx = [j*p+k]
                        w_idx = [k*n+i]
                        add_term(u_idx, v_idx, w_idx, ([1.0], [1.0], [1.0]))
                        
        if col != rank:
             # This should not happen if logic is correct, but resize if it does
             U = U[:, :col]
             V = V[:, :col]
             W = W[:, :col]
             rank = col
             
        return U, V, W, rank

    # Generate all possible 2x2x2 blocks
    all_blocks = []
    for i0 in range(n):
        for i1 in range(i0+1, n):
            for j0 in range(m):
                for j1 in range(j0+1, m):
                    for k0 in range(p):
                        for k1 in range(k0+1, p):
                            all_blocks.append((i0, i1, j0, j1, k0, k1))
                            
    # Greedy search for maximum disjoint blocks
    best_blocks = []
    best_count = 0
    
    # Try multiple random seeds and heuristics
    random.seed(42)
    
    # Heuristic 1: Random greedy
    for trial in range(100):
        if time.time() > deadline - 20:
            break
        shuffled = all_blocks[:]
        random.shuffle(shuffled)
        chosen = []
        used = set()
        for block in shuffled:
            i0, i1, j0, j1, k0, k1 = block
            overlap = False
            for i in [i0, i1]:
                for j in [j0, j1]:
                    for k in [k0, k1]:
                        if (i, j, k) in used:
                            overlap = True
                            break
                    if overlap: break
                if overlap: break
            if not overlap:
                chosen.append(block)
                for i in [i0, i1]:
                    for j in [j0, j1]:
                        for k in [k0, k1]:
                            used.add((i, j, k))
        if len(chosen) > best_count:
            best_count = len(chosen)
            best_blocks = chosen[:]
            
    # Heuristic 2: Sort by "center-ness" to leave edges for others? Or just deterministic order
    # Let's try a deterministic sort by index sum to see if it finds a different packing
    sorted_blocks = sorted(all_blocks, key=lambda b: b[0]+b[1]+b[2]+b[3]+b[4]+b[5])
    chosen = []
    used = set()
    for block in sorted_blocks:
        i0, i1, j0, j1, k0, k1 = block
        overlap = False
        for i in [i0, i1]:
            for j in [j0, j1]:
                for k in [k0, k1]:
                    if (i, j, k) in used:
                        overlap = True
                        break
                if overlap: break
            if overlap: break
        if not overlap:
            chosen.append(block)
            for i in [i0, i1]:
                for j in [j0, j1]:
                    for k in [k0, k1]:
                        used.add((i, j, k))
    if len(chosen) > best_count:
        best_count = len(chosen)
        best_blocks = chosen[:]
        
    # Heuristic 3: Local search / Swap? 
    # If we have a set of blocks, can we swap one out for two others? 
    # This is complex. Given the small size, random greedy usually finds the max independent set in this specific hypergraph structure quickly.
    # The theoretical max number of 2x2x2 blocks in 3x5x7:
    # Volume 105. Each block takes 8. 105/8 = 13.125.
    # But they must be disjoint in indices.
    # i-dim: 3. Max disjoint pairs in 1D: 1 pair (0,1) or (1,2). So at most 1 layer of i-pairs can be active for a specific j,k pair? No, blocks are 3D.
    # Actually, since n=3, we can have blocks using i-pairs (0,1) and (1,2). They overlap at i=1.
    # So for a fixed (j,k) pair, we can't have both (0,1) and (1,2) blocks.
    # This constraint is tight.
    
    # Let's run a slightly more intensive search if time permits
    if time.time() < deadline - 40:
        for trial in range(500):
            if time.time() > deadline - 40:
                break
            # Perturb the previous best or random
            # Random restart is simpler
            shuffled = all_blocks[:]
            random.shuffle(shuffled)
            chosen = []
            used = set()
            for block in shuffled:
                i0, i1, j0, j1, k0, k1 = block
                overlap = False
                for i in [i0, i1]:
                    for j in [j0, j1]:
                        for k in [k0, k1]:
                            if (i, j, k) in used:
                                overlap = True
                                break
                    if overlap: break
                if overlap: break
            if not overlap:
                chosen.append(block)
                for i in [i0, i1]:
                    for j in [j0, j1]:
                        for k in [k0, k1]:
                            used.add((i, j, k))
            if len(chosen) > best_count:
                best_count = len(chosen)
                best_blocks = chosen[:]
                
    # Build the best found decomposition
    try:
        U, V, W, rank = build_decomposition(best_blocks)
        if verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
        
    # Fallback: Known good Rank 99 solution (6 blocks)
    # This corresponds to the tiling in the original prompt.
    # Blocks:
    # (0,1, 0,1, 0,1), (0,1, 0,1, 2,3), (0,1, 0,1, 4,5)
    # (0,1, 2,3, 0,1), (0,1, 2,3, 2,3), (0,1, 2,3, 4,5)
    fallback_blocks = [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 0, 1, 2, 3),
        (0, 1, 0, 1, 4, 5),
        (0, 1, 2, 3, 0, 1),
        (0, 1, 2, 3, 2, 3),
        (0, 1, 2, 3, 4, 5)
    ]
    
    try:
        U, V, W, rank = build_decomposition(fallback_blocks)
        if verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
        
    # Ultimate Fallback: Schoolbook
    U = np.zeros((nm, n*m*p), dtype=float)
    V = np.zeros((mp, n*m*p), dtype=float)
    W = np.zeros((pn, n*m*p), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
