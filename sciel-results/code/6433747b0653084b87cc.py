import itertools
import time
import random
import numpy as np
from scipy import sparse

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit

    # Precompute all t-subsets
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}

    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map block tuple to index for O(1) lookup
    block_to_idx = {block: i for i, block in enumerate(all_blocks)}

    # Build incidence matrix A: (n_blocks, n_triplets)
    # A[b, t] = 1 if block b covers triplet t
    rows = []
    cols = []
    for b_idx, block in enumerate(all_blocks):
        for triplet in itertools.combinations(block, t):
            rows.append(b_idx)
            cols.append(triplet_to_id[triplet])
    
    A = sparse.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(n_blocks, n_triplets))
    
    # Precompute list of triplets for each block for quick access if needed
    # But we will mostly use the matrix

    def greedy_set_cover(seed):
        rng = random.Random(seed)
        # gain[b] = number of currently uncovered triplets covered by block b
        # Initially all triplets are uncovered, so gain is just the number of triplets in the block
        gain = np.array(A.getnnz(axis=1), dtype=np.int32)
        is_uncovered = np.ones(n_triplets, dtype=np.bool_)

        selected_blocks = []
        # We will maintain a count of how many times each triplet is covered
        # But for greedy, we just need to know if it's covered.
        # Efficient update:
        # When we pick block b, for each triplet t in b:
        #   if is_uncovered[t]:
        #       is_uncovered[t] = False
        #       for all blocks b' covering t: gain[b'] -= 1
        
        # To speed up, we can precompute the list of blocks covering each triplet
        # This is the transpose of A
        A_T = A.T.tocsr()
        
        while is_uncovered.any():
            if time.time() > deadline - 30:
                break

            max_gain = np.max(gain)
            if max_gain == 0:
                break

            # Pick random among max-gain blocks
            candidates = np.where(gain == max_gain)[0]
            b_idx = rng.choice(candidates.tolist())

            selected_blocks.append(b_idx)

            # Update gains
            # Get triplets covered by b_idx
            t_indices = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
            
            for t_idx in t_indices:
                if is_uncovered[t_idx]:
                    is_uncovered[t_idx] = False
                    # Decrement gain for all blocks covering this triplet
                    covering_blocks = A_T.indices[A_T.indptr[t_idx]:A_T.indptr[t_idx+1]]
                    gain[covering_blocks] -= 1

        return selected_blocks

    def reduce_cover(selected_blocks):
        """Remove redundant blocks."""
        if not selected_blocks:
            return selected_blocks
        
        # Build sub-matrix for current cover
        M = A[selected_blocks, :].toarray()
        n_cur = len(selected_blocks)
        
        active = np.ones(n_cur, dtype=bool)
        changed = True
        while changed:
            changed = False
            for i in range(n_cur):
                if not active[i]:
                    continue
                # Check if block i is redundant
                # Sum of M excluding row i must be >= 1 everywhere M[i] is 1
                # Or simply: if M[i] has a 1, is there another 1 in that column?
                # Vectorized check:
                # col_sums = M.sum(axis=0)
                # redundant if for all t where M[i,t]==1, col_sums[t] > 1
                
                # To be faster, just check the columns where M[i] is 1
                ones_in_i = np.where(M[i] == 1)[0]
                if len(ones_in_i) == 0:
                    active[i] = False
                    changed = True
                    continue
                
                # Check if any of these columns have sum == 1 (which implies only block i covers it)
                # Since we are removing blocks one by one, we need to be careful with the count.
                # Actually, `reduce_cover` logic:
                # If we remove block i, does any triplet become uncovered?
                # A triplet t is covered by block i if M[i,t]=1.
                # It remains covered if sum(M[j,t] for j != i, active[j]) > 0.
                
                # Let's just recompute the count for the active blocks
                # This is O(N^2 * T) in worst case if done naively, but N is small (~80)
                # T is 2024.
                # 80 * 2024 * 80 ~ 12M ops. Acceptable.
                
                # However, doing this in a loop is slow in Python.
                # Let's use matrix multiplication or simple numpy ops.
                
                # Sum of active blocks excluding i
                # M_active = M[active]
                # This changes size.
                
                # Alternative: Maintain a count matrix
                pass
            
            # Let's implement a simpler, robust reduction using counts
            # Reset and do a full pass
            counts = np.zeros(n_triplets, dtype=np.int32)
            for i in range(n_cur):
                if active[i]:
                    counts += M[i]
            
            for i in range(n_cur):
                if not active[i]:
                    continue
                # Check redundancy
                ones_in_i = np.where(M[i] == 1)[0]
                if len(ones_in_i) > 0:
                    # If any triplet in this block has count == 1, it's not redundant
                    if np.any(counts[ones_in_i] == 1):
                        continue
                # It is redundant
                active[i] = False
                counts -= M[i]
                changed = True
            
        return [selected_blocks[i] for i in range(n_cur) if active[i]]

    # Phase 1: Greedy Restart
    best_cover = None
    best_count = float('inf')
    seed = 0
    greedy_time_budget = 40.0
    
    while time.time() < start_time + greedy_time_budget:
        cover = greedy_set_cover(seed)
        cover = reduce_cover(cover)
        if len(cover) < best_count:
            best_count = len(cover)
            best_cover = cover
        seed += 1
        if seed > 200: break # Limit restarts if fast

    if best_cover is None:
        best_cover = greedy_set_cover(0)
        best_cover = reduce_cover(best_cover)

    # Phase 2: Local Search
    # Current cover is a list of block indices
    current_blocks = list(best_cover)
    
    # We will work with the matrix M (n_cur, n_triplets)
    # And the global matrix A (n_blocks, n_triplets)
    
    # Helper to get M from current_blocks
    def get_M(blocks):
        if not blocks:
            return np.zeros((0, n_triplets), dtype=np.int32)
        return A[blocks, :].toarray()
    
    M = get_M(current_blocks)
    
    # Precompute A_T for faster column access if needed, but we have A
    # For swap, we need to evaluate gain of any block b against deficits of removing i, j.
    
    # Deficit D_{ij} = max(0, 1 - (M_sum - M_i - M_j))
    # Where M_sum is column sums of M.
    
    # We iterate over pairs (i, j).
    # For each pair, calculate D.
    # Then for all blocks b, calculate Gain(b) = sum(A[b, t] * D[t])
    # If Gain(b) == sum(D) and b not in current cover (or we allow duplicates? No, distinct blocks required),
    # then it's a valid swap.
    
    # Optimization:
    # Precompute A as CSR.
    # For a given D (vector of length n_triplets), Gain = A @ D.
    # This is a sparse-dense matrix-vector product. Very fast in SciPy.
    
    # We need to check if b is already in current_blocks.
    # current_set = set(current_blocks)
    
    # Local Search Loop
    improved = True
    while improved:
        improved = False
        
        # 1. Remove Redundants
        # Use the reduce_cover logic but in-place on M and current_blocks
        if len(current_blocks) > 0:
            # Recompute M if changed
            counts = M.sum(axis=0)
            to_remove = []
            for i, b_idx in enumerate(current_blocks):
                ones_in_i = np.where(M[i] == 1)[0]
                if len(ones_in_i) > 0 and np.any(counts[ones_in_i] == 1):
                    continue
                to_remove.append(i)
            
            if to_remove:
                # Remove them
                keep_indices = [i for i in range(len(current_blocks)) if i not in to_remove]
                current_blocks = [current_blocks[i] for i in keep_indices]
                M = M[keep_indices, :]
                improved = True
                continue # Restart loop to check for more redundants or new swaps
        
        n_cur = len(current_blocks)
        if n_cur < 2:
            break
        
        # 2. Try 2-for-1 Swap
        # To speed up, we can randomize the order of pairs or limit checks
        # But we want the BEST swap? Or ANY valid swap?
        # Usually, finding ANY valid swap that reduces size is enough for progress.
        # However, a "bad" swap might lead to a dead end.
        # Given time limits, we should look for swaps quickly.
        
        # Strategy: Iterate through pairs.
        # To avoid O(N^2) full scan every time, we can keep a list of "promising" pairs?
        # Or just scan and stop at first valid swap?
        # Let's scan and stop at first valid swap to keep it fast.
        
        # Precompute column sums of M for quick deficit calculation
        col_sums = M.sum(axis=0)
        
        # We need to check pairs.
        # Randomize pair order to escape local minima?
        # Let's shuffle the indices and try pairs.
        
        indices = list(range(n_cur))
        random.shuffle(indices)
        
        found_swap = False
        for idx_i in indices:
            if time.time() > deadline - 10:
                break
            if found_swap: break
            for idx_j in indices:
                if idx_j <= idx_i:
                    continue
                if time.time() > deadline - 10:
                    break
                
                # Calculate Deficit
                # M_without_ij = col_sums - M[idx_i] - M[idx_j]
                # D = max(0, 1 - M_without_ij)
                # D = max(0, 1 - (col_sums - M_i - M_j))
                # If col_sums[t] - M_i[t] - M_j[t] == 0, D[t] = 1
                # If col_sums[t] - M_i[t] - M_j[t] >= 1, D[t] = 0
                
                # Vectorized:
                # mask = (col_sums - M[idx_i] - M[idx_j]) == 0
                # D = mask.astype(np.int32)
                
                # Check if D is all zeros (both redundant, handled above, but maybe one redundant?)
                # If D is all zeros, we can just remove one? No, if both are redundant, we remove both?
                # If D is all zeros, it means removing i and j leaves no triplet uncovered.
                # This implies both are redundant? Not necessarily.
                # If triplet t is covered by i and j, and no one else, then col_sums=2, M_i=1, M_j=1 -> 2-1-1=0 -> D=1.
                # So D=0 means every triplet covered by i or j is covered by someone else.
                # In that case, we can remove BOTH i and j?
                # Yes, if D is all 0, we can remove both.
                
                D = (col_sums - M[idx_i] - M[idx_j] == 0).astype(np.int32)
                
                if not np.any(D):
                    # Both redundant
                    # Remove both
                    keep = [x for x in range(n_cur) if x != idx_i and x != idx_j]
                    current_blocks = [current_blocks[x] for x in keep]
                    M = M[keep, :]
                    improved = True
                    found_swap = True
                    break
                
                # We need to find a block b such that A[b] covers all 1s in D.
                # Gain(b) = A[b] @ D
                # Target = sum(D)
                
                # Calculate gains for all blocks
                # Gains = A @ D
                # This is a sparse matrix dense vector product.
                gains = A @ D
                target = np.sum(D)
                
                # Find blocks with gain == target
                # And b not in current_blocks
                # current_set = set(current_blocks)
                
                # To avoid set lookup overhead in loop, we can use a boolean mask
                # in_current = np.zeros(n_blocks, dtype=bool)
                # for b in current_blocks: in_current[b] = True
                
                # But building this mask every pair is O(N).
                # N is ~80. 80*80 = 6400 pairs. 6400 * 80 = 512k ops. Trivial.
                
                in_current = np.zeros(n_blocks, dtype=bool)
                for b in current_blocks:
                    in_current[b] = True
                
                valid_blocks = np.where((gains == target) & (~in_current))[0]
                
                if len(valid_blocks) > 0:
                    # Pick one (random or first)
                    # Let's pick the one that maximizes some secondary metric?
                    # Or just the first one.
                    # Random selection might help diversity
                    new_b = valid_blocks[0] # or random.choice(valid_blocks)
                    
                    # Apply swap
                    # Remove i and j
                    keep = [x for x in range(n_cur) if x != idx_i and x != idx_j]
                    current_blocks = [current_blocks[x] for x in keep]
                    M = M[keep, :]
                    
                    # Add new_b
                    current_blocks.append(new_b)
                    # Append row to M
                    new_row = A[new_b, :].toarray()
                    M = np.vstack([M, new_row])
                    
                    improved = True
                    found_swap = True
                    break
                    
        if not improved:
            break

    # Final Redundancy Check
    if len(current_blocks) > 0:
        counts = M.sum(axis=0)
        to_remove = []
        for i, b_idx in enumerate(current_blocks):
            ones_in_i = np.where(M[i] == 1)[0]
            if len(ones_in_i) > 0 and np.any(counts[ones_in_i] == 1):
                continue
            to_remove.append(i)
        
        if to_remove:
            keep_indices = [i for i in range(len(current_blocks)) if i not in to_remove]
            current_blocks = [current_blocks[i] for i in keep_indices]

    # Convert back to list of blocks
    result = [list(all_blocks[b_idx]) for b_idx in current_blocks]

    # Verify validity
    covered = set()
    for block in result:
        for triplet in itertools.combinations(block, t):
            covered.add(triplet)

    if len(covered) < n_triplets:
        # Fallback
        return [list(all_blocks[b_idx]) for b_idx in best_cover]

    return result
