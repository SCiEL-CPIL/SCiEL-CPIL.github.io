import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.monotonic()
    deadline = start_time + 115.0

    points = list(range(v))
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)

    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)

    # Build coverage matrix M (num_kblocks x num_tsubs)
    M = np.zeros((num_kblocks, num_tsubs), dtype=np.uint8)
    for i, block in enumerate(all_kblocks):
        for sub in itertools.combinations(block, t):
            idx = tsub_to_idx[sub]
            M[i, idx] = 1

    # Build reverse index
    tsub_blocks = [[] for _ in range(num_tsubs)]
    for i, block in enumerate(all_kblocks):
        for sub in itertools.combinations(block, t):
            idx = tsub_to_idx[sub]
            tsub_blocks[idx].append(i)
    tsub_blocks = [np.array(lst, dtype=np.int64) if lst else np.array([], dtype=np.int64) for lst in tsub_blocks]

    rng = random.Random(42)

    def get_blocks(indices):
        return [list(all_kblocks[i]) for i in indices]

    # Precompute covered_indices for all blocks to avoid repeated np.where
    all_covered = [np.where(M[i] == 1)[0] for i in range(num_kblocks)]

    # Phase 1: Vectorized Greedy Construction
    uncovered = np.ones(num_tsubs, dtype=np.uint8)
    counts = np.zeros(num_tsubs, dtype=np.int32)
    selected_mask = np.zeros(num_kblocks, dtype=bool)
    selected_indices = []

    while np.any(uncovered):
        if time.monotonic() > deadline - 10.0:
            break
        gains = M @ uncovered
        best_idx = np.argmax(gains)
        if gains[best_idx] == 0:
            break
        selected_indices.append(best_idx)
        selected_mask[best_idx] = True
        covered_indices = all_covered[best_idx]
        uncovered[covered_indices] = 0
        counts[covered_indices] += 1

    if not selected_indices:
        return []

    # Phase 2: Local Search Polish with ILS
    current_indices = selected_indices[:]
    current_counts = counts.copy()
    current_mask = selected_mask.copy()

    # Precompute covered_indices for all blocks to avoid repeated np.where
    all_covered = [np.where(M[i] == 1)[0] for i in range(num_kblocks)]

    def try_remove_redundant():
        removed = False
        changed = True
        while changed:
            changed = False
            for b_idx in current_indices[:]:
                covered_indices = all_covered[b_idx]
                if np.all(current_counts[covered_indices] > 1):
                    current_mask[b_idx] = False
                    current_indices.remove(b_idx)
                    current_counts[covered_indices] -= 1
                    removed = True
                    changed = True
        return removed

    def try_swaps():
        improved = False
        idx_list = current_indices[:]
        rng.shuffle(idx_list)

        for b_idx in idx_list:
            if time.monotonic() > deadline - 2.0:
                break
            covered_indices = all_covered[b_idx]
            critical_mask = current_counts == 1
            critical_indices = covered_indices[critical_mask[covered_indices]]

            if len(critical_indices) == 0:
                current_mask[b_idx] = False
                current_indices.remove(b_idx)
                current_counts[covered_indices] -= 1
                improved = True
                break

            if len(critical_indices) == 1:
                crit = critical_indices[0]
                candidates = tsub_blocks[crit]
                candidates = candidates[~current_mask[candidates]]
                if len(candidates) == 0:
                    continue
                best_cand = None
                best_score = -1
                for cand in candidates[:50]:
                    cand_row = M[cand]
                    score = np.sum(cand_row * (current_counts == 1))
                    if score > best_score:
                        best_score = score
                        best_cand = cand
                if best_cand is not None:
                    current_mask[b_idx] = False
                    current_indices.remove(b_idx)
                    current_counts[covered_indices] -= 1
                    current_mask[best_cand] = True
                    current_indices.append(best_cand)
                    cand_covered = all_covered[best_cand]
                    current_counts[cand_covered] += 1
                    improved = True
                    break
            elif len(critical_indices) <= 3:
                common = None
                for crit in critical_indices:
                    blocks_c = tsub_blocks[crit]
                    if common is None:
                        common = set(blocks_c.tolist())
                    else:
                        common &= set(blocks_c.tolist())
                if common:
                    candidates = np.array(list(common), dtype=np.int64)
                    candidates = candidates[~current_mask[candidates]]
                    if len(candidates) > 0:
                        best_cand = None
                        best_score = -1
                        for cand in candidates[:30]:
                            cand_row = M[cand]
                            score = np.sum(cand_row * (current_counts == 1))
                            if score > best_score:
                                best_score = score
                                best_cand = cand
                        if best_cand is not None:
                            current_mask[b_idx] = False
                            current_indices.remove(b_idx)
                            current_counts[covered_indices] -= 1
                            current_mask[best_cand] = True
                            current_indices.append(best_cand)
                            cand_covered = all_covered[best_cand]
                            current_counts[cand_covered] += 1
                            improved = True
                            break

        return improved

    # Precompute overlap graph for spectral embedding (done once)
    # For blocks in the current solution, compute pairwise shared 4-subsets
    # We'll use this to identify clusters of highly-overlapping blocks
    # that can be perturbed together
    
    # Build a sparse-ish overlap structure: for each block, which other blocks share 4-subsets
    # Use the tsub_blocks index: for each t-subset, list of blocks containing it
    # Two blocks i,j share a 4-subset iff they appear together in some tsub_blocks entry
    
    # Efficient approach: compute overlap counts via matrix multiplication on subset incidence
    # overlap[i,j] = number of 4-subsets shared by blocks i and j
    # This equals sum over t-subsets s of M[i,s]*M[j,s] = (M @ M.T)[i,j]
    # But M is 50388 x 3876, so M@M.T is 50388x50388 = too big
    # Instead, we work with the subset of blocks in the current solution
    
    # For spectral embedding on the current solution's blocks:
    # Build adjacency from shared 4-subsets among selected blocks
    # Use the fact that overlap[i,j] = |C(block_i,4) ∩ C(block_j,4)|
    
    def compute_overlap_matrix(block_indices):
        """Compute pairwise overlap (shared 4-subsets) for a list of block indices."""
        n = len(block_indices)
        if n == 0:
            return np.array([])
        # For each block, get its 4-subsets as indices into all_tsubs
        # Then overlap[i,j] = |set_i ∩ set_j|
        # Efficient: build binary matrix and multiply
        submat = M[block_indices, :]  # n x num_tsubs
        overlap = (submat @ submat.T).astype(np.int32)
        # Diagonal is C(7,4)=35
        np.fill_diagonal(overlap, 0)
        return overlap
    
    def spectral_perturb():
        """Use spectral embedding to identify high-overlap clusters for coordinated removal."""
        current_list = current_indices[:]
        n_cur = len(current_list)
        if n_cur < 10:
            return False
        
        # Compute overlap matrix for current blocks
        overlap = compute_overlap_matrix(current_list)
        
        # Build graph Laplacian: L = D - A where A = overlap, D = degree
        degrees = overlap.sum(axis=1)
        # Normalize for stability
        max_deg = degrees.max() if degrees.max() > 0 else 1
        A_norm = overlap.astype(np.float64) / max_deg
        
        # Laplacian
        D_mat = np.diag(degrees.astype(np.float64))
        L = D_mat - A_norm * max_deg  # scale back to keep eigenvalues meaningful
        # Actually use normalized Laplacian for better conditioning
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degrees, 1).astype(np.float64)))
        L_norm = np.eye(n_cur) - D_inv_sqrt @ A_norm @ D_inv_sqrt
        
        # Compute eigenvectors (lowest few)
        try:
            evals, evecs = np.linalg.eigh(L_norm)
        except np.linalg.LinAlgError:
            return False
        
        # Take the 3 lowest non-trivial eigenvectors (skip trivial constant mode)
        num_modes = min(3, n_cur - 1)
        modes = evecs[:, 1:1+num_modes]  # n_cur x num_modes
        
        # Project blocks into low-dimensional embedding
        # Use the embedding to find clusters: blocks close in embedding space
        # are in high-overlap clusters
        
        # Simple clustering: k-means with k=2-3 on the embedding
        # Or: identify the mode with largest variance and split there
        
        # Strategy: use the first non-trivial eigenvector (Fiedler vector) to split
        # into two groups; remove blocks from the smaller group that have low marginal gain
        fiedler = evecs[:, 1] if n_cur > 1 else np.zeros(n_cur)
        
        # Center the Fiedler vector
        fiedler_centered = fiedler - fiedler.mean()
        
        # Split into two groups based on sign
        group_a_mask = fiedler_centered > 0
        group_b_mask = ~group_a_mask
        
        # For each group, compute average overlap within group
        def group_avg_overlap(mask):
            sub = overlap[np.ix_(mask, mask)]
            n_g = mask.sum()
            if n_g <= 1:
                return 0
            return sub.sum() / (n_g * (n_g - 1))
        
        avg_a = group_avg_overlap(group_a_mask)
        avg_b = group_avg_overlap(group_b_mask)
        
        # Remove blocks from the group with lower internal overlap (more "peripheral")
        # and also low marginal gain
        unique_mask = (current_counts == 1).astype(np.uint8)
        gains = M @ unique_mask
        
        if avg_a <= avg_b:
            # Group A is more peripheral; remove from it
            candidates = [current_list[i] for i in range(n_cur) if group_a_mask[i]]
        else:
            candidates = [current_list[i] for i in range(n_cur) if group_b_mask[i]]
        
        if not candidates:
            return False
        
        # Sort candidates by gain (ascending) and remove the lowest-gain ones
        candidates.sort(key=lambda b: gains[b])
        n_remove = min(3 + rng.randint(0, 3), max(1, len(candidates) // 2))
        to_remove = candidates[:n_remove]
        
        for b_idx in to_remove:
            if b_idx in current_indices:
                current_mask[b_idx] = False
                current_indices.remove(b_idx)
                covered_indices = all_covered[b_idx]
                current_counts[covered_indices] -= 1
        
        # Repair: greedily add blocks to cover uncovered 4-subsets
        uncovered_mask = (current_counts == 0).astype(np.uint8)
        while np.any(uncovered_mask):
            if time.monotonic() > deadline - 5.0:
                break
            gains_greedy = M @ uncovered_mask
            best_idx = np.argmax(gains_greedy)
            if gains_greedy[best_idx] == 0:
                break
            current_mask[best_idx] = True
            current_indices.append(best_idx)
            covered_indices = all_covered[best_idx]
            current_counts[covered_indices] += 1
            uncovered_mask = (current_counts == 0).astype(np.uint8)
        
        return True
    
    def perturb():
        """Perturbation: mix of spectral cluster removal and random/low-gain removal."""
        # With some probability, use spectral perturbation
        if rng.random() < 0.5:
            if spectral_perturb():
                return
        
        # Fallback: original perturbation strategy
        unique_mask = (current_counts == 1).astype(np.uint8)
        gains = M @ unique_mask
        
        n_remove = min(5 + rng.randint(0, 5), max(1, len(current_indices) // 3))
        
        sorted_indices = np.argsort(gains)
        low_gain_candidates = sorted_indices[:max(1, n_remove * 2)]
        
        valid_low_gain = [i for i in low_gain_candidates if current_mask[i]]
        
        to_remove = set()
        
        if valid_low_gain:
            pick_low = rng.sample(valid_low_gain, min(len(valid_low_gain), n_remove // 2 + 1))
            to_remove.update(pick_low)
            
        remaining = [i for i in current_indices if i not in to_remove]
        if remaining:
            pick_rand = rng.sample(remaining, min(len(remaining), n_remove - len(to_remove)))
            to_remove.update(pick_rand)
            
        if len(to_remove) < n_remove:
            remaining = [i for i in current_indices if i not in to_remove]
            if remaining:
                pick_extra = rng.sample(remaining, n_remove - len(to_remove))
                to_remove.update(pick_extra)

        to_remove_list = list(to_remove)
        
        for b_idx in to_remove_list:
            if b_idx in current_indices:
                current_mask[b_idx] = False
                current_indices.remove(b_idx)
                covered_indices = all_covered[b_idx]
                current_counts[covered_indices] -= 1

        uncovered_mask = (current_counts == 0).astype(np.uint8)
        while np.any(uncovered_mask):
            if time.monotonic() > deadline - 5.0:
                break
            gains_greedy = M @ uncovered_mask
            best_idx = np.argmax(gains_greedy)
            if gains_greedy[best_idx] == 0:
                break
            current_mask[best_idx] = True
            current_indices.append(best_idx)
            covered_indices = all_covered[best_idx]
            current_counts[covered_indices] += 1
            uncovered_mask = (current_counts == 0).astype(np.uint8)

    # Main local search loop with Simulated Annealing
    best_len = len(current_indices)
    best_solution = current_indices[:]
    best_counts = current_counts.copy()
    best_mask = current_mask.copy()

    # SA Parameters
    T = 5.0
    cooling = 0.95
    min_T = 0.1
    stagnation = 0
    max_stagnation = 20 # Trigger kick more often

    # Track multiple restarts with different seeds for diversity
    seed_idx = 0
    while time.monotonic() < deadline - 3.0:
        improved = False
        old_len = len(current_indices)

        if try_remove_redundant():
            improved = True

        if not improved and try_swaps():
            improved = True

        if improved:
            stagnation = 0
            T = max(T * cooling, min_T)
            if len(current_indices) < best_len:
                best_len = len(current_indices)
                best_solution = current_indices[:]
                best_counts = current_counts.copy()
                best_mask = current_mask.copy()
        else:
            stagnation += 1
            # Kick condition: stagnation or random chance if T is high
            if stagnation >= max_stagnation or (T > min_T and rng.random() < 0.2):
                # Save state for potential revert
                save_indices = current_indices[:]
                save_counts = current_counts.copy()
                save_mask = current_mask.copy()
                save_len = len(current_indices)

                perturb()
                try_remove_redundant()
                
                new_len = len(current_indices)
                delta = new_len - save_len
                
                accept = False
                if delta < 0:
                    accept = True
                elif delta == 0:
                    # Accept to explore landscape
                    accept = True 
                else:
                    # Worse, accept with prob exp(-delta/T)
                    p_accept = np.exp(-delta / T)
                    if rng.random() < p_accept:
                        accept = True

                if accept:
                    T = max(T * cooling, min_T)
                    stagnation = 0
                    if new_len < best_len:
                        best_len = new_len
                        best_solution = current_indices[:]
                        best_counts = current_counts.copy()
                        best_mask = current_mask.copy()
                else:
                    # Revert
                    current_indices = save_indices
                    current_counts = save_counts
                    current_mask = save_mask
                    # Reheat slightly to encourage more exploration
                    T = min(T * 1.1, 10.0)
                    stagnation = 0
            
            # Decay temperature
            T = max(T * cooling, min_T)

            # Check if we should restart with a fresh greedy solution
            if stagnation > 50 and seed_idx < 3:
                seed_idx += 1
                # Reset to a fresh greedy solution with different seed
                rng2 = random.Random(42 + seed_idx * 1000)
                
                # Quick greedy restart
                unc2 = np.ones(num_tsubs, dtype=np.uint8)
                cnt2 = np.zeros(num_tsubs, dtype=np.int32)
                sel_mask2 = np.zeros(num_kblocks, dtype=bool)
                sel_idx2 = []
                
                while np.any(unc2) and time.monotonic() < deadline - 15.0:
                    gains2 = M @ unc2
                    best2 = np.argmax(gains2)
                    if gains2[best2] == 0:
                        break
                    sel_idx2.append(best2)
                    sel_mask2[best2] = True
                    cov2 = all_covered[best2]
                    unc2[cov2] = 0
                    cnt2[cov2] += 1
                
                if len(sel_idx2) < best_len:
                    # Accept this fresh solution
                    current_indices = sel_idx2[:]
                    current_counts = cnt2.copy()
                    current_mask = sel_mask2.copy()
                    T = 5.0
                    stagnation = 0
                else:
                    # Continue with current
                    T = max(T * cooling, min_T)

    final_indices = best_solution
    final_counts = best_counts.copy()
    final_mask = best_mask.copy()

    for b_idx in final_indices[:]:
        if not final_mask[b_idx]:
            continue
        covered_indices = all_covered[b_idx]
        if np.all(final_counts[covered_indices] > 1):
            final_mask[b_idx] = False
            final_indices.remove(b_idx)
            final_counts[covered_indices] -= 1

    verify_counts = np.zeros(num_tsubs, dtype=np.int32)
    for b_idx in final_indices:
        covered_indices = all_covered[b_idx]
        verify_counts[covered_indices] += 1

    if not np.all(verify_counts >= 1):
        final_indices = selected_indices

    return get_blocks(final_indices)
