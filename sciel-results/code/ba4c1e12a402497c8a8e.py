import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    # Precompute all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)

    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # Map each block to the list of t-subset indices it covers
    block_to_tsubs = []
    for b in all_blocks:
        indices = [t_sub_to_idx[ts] for ts in itertools.combinations(b, t)]
        block_to_tsubs.append(indices)

    # Reverse map: t_sub_idx -> list of block indices
    t_sub_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, t_indices in enumerate(block_to_tsubs):
        for t_idx in t_indices:
            t_sub_to_blocks[t_idx].append(b_idx)

    # Build a numpy array: block_to_tsubs_np[i] = array of t-subset indices for block i
    block_to_tsubs_np = np.array(block_to_tsubs, dtype=np.int32)

    # Bitset representation for fast coverage checks
    # Each block gets a bitset of the t-subsets it covers
    # Use Python integers as bitsets (arbitrary precision)
    block_bitsets = []
    for t_indices in block_to_tsubs:
        bits = 0
        for idx in t_indices:
            bits |= (1 << idx)
        block_bitsets.append(bits)

    cover_count = np.zeros(num_t_subsets, dtype=np.int32)
    blocks_in_cover = []
    in_cover_set = set()

    def add_block(b_idx):
        cover_count[block_to_tsubs_np[b_idx]] += 1
        blocks_in_cover.append(b_idx)
        in_cover_set.add(b_idx)

    def remove_block(b_idx):
        cover_count[block_to_tsubs_np[b_idx]] -= 1
        in_cover_set.discard(b_idx)
        if b_idx in blocks_in_cover:
            blocks_in_cover.remove(b_idx)

    def is_redundant(b_idx):
        """Check if removing b_idx still covers all its t-subsets."""
        t_indices = block_to_tsubs_np[b_idx]
        return np.all(cover_count[t_indices] > 1)

    def get_unique_tsubs(b_idx):
        """Get t-subsets only covered by b_idx."""
        t_indices = block_to_tsubs_np[b_idx]
        return t_indices[cover_count[t_indices] == 1]

    # 1. Hypergraph Peeling Greedy Initialization
    # Track active (uncovered) t-subsets
    uncovered_mask = np.ones(num_t_subsets, dtype=bool)
    num_uncovered = num_t_subsets

    while num_uncovered > 0:
        if time.time() > deadline - 20:
            break

        # Find the t-subset with fewest candidate blocks (most constrained)
        # This is the "peeling" aspect - focus on hardest constraints first
        uncovered_indices = np.where(uncovered_mask)[0]
        
        # Score each uncovered t-subset by how many blocks cover it
        # Pick the one with fewest candidates (most critical)
        best_target = None
        best_candidate_count = float('inf')
        
        # Sample a subset of uncovered t-subsets for speed
        sample_size = min(len(uncovered_indices), 50)
        if len(uncovered_indices) > sample_size:
            sample = uncovered_indices[random.sample(range(len(uncovered_indices)), sample_size)]
        else:
            sample = uncovered_indices
        
        for target_idx in sample:
            cand_count = len(t_sub_to_blocks[target_idx])
            if cand_count < best_candidate_count:
                best_candidate_count = cand_count
                best_target = target_idx
        
        if best_target is None:
            break

        candidates = t_sub_to_blocks[best_target]
        cand_arr = np.array(candidates, dtype=np.int32)
        
        # Vectorized gain computation: count newly covered t-subsets
        gains = np.sum(cover_count[block_to_tsubs_np[cand_arr]] == 0, axis=1)
        best_local = np.argmax(gains)
        best_block_idx = cand_arr[best_local]
        max_gain = gains[best_local]

        if max_gain > 0:
            add_block(best_block_idx)
            newly_covered = block_to_tsubs_np[best_block_idx]
            newly_uncovered = (cover_count[newly_covered] == 0)
            was_uncovered = uncovered_mask[newly_covered]
            num_uncovered -= np.sum(was_uncovered & ~newly_uncovered)
            uncovered_mask[newly_covered] = newly_uncovered

    # 2. Aggressive Local Search with ILS-style restarts
    best_size = len(blocks_in_cover)
    best_blocks_snapshot = list(blocks_in_cover)
    
    no_improvement_count = 0
    max_no_improvement = 50
    restart_count = 0
    max_restarts = 5

    while time.time() < deadline - 5:
        improved = False

        # Phase 1: Delete all redundant blocks at once
        to_remove = []
        for b_idx in blocks_in_cover:
            if is_redundant(b_idx):
                to_remove.append(b_idx)

        for b_idx in to_remove:
            remove_block(b_idx)
            improved = True

        if improved:
            no_improvement_count = 0
            if len(blocks_in_cover) < best_size:
                best_size = len(blocks_in_cover)
                best_blocks_snapshot = list(blocks_in_cover)
            continue

        # Phase 2: Try 1-for-1 swaps for critical blocks
        critical_blocks = []
        for b_idx in blocks_in_cover:
            if np.any(cover_count[block_to_tsubs_np[b_idx]] == 1):
                critical_blocks.append(b_idx)

        if critical_blocks:
            random.shuffle(critical_blocks)

            swapped = False
            for b_old_idx in critical_blocks:
                if time.time() > deadline - 5:
                    break

                unique_tsubs = get_unique_tsubs(b_old_idx)
                if len(unique_tsubs) == 0:
                    continue

                # Find blocks that cover all unique t-subsets
                candidate_set = set(t_sub_to_blocks[unique_tsubs[0]])
                for t_idx in unique_tsubs[1:]:
                    candidate_set.intersection_update(t_sub_to_blocks[t_idx])
                    if not candidate_set:
                        break

                if candidate_set:
                    candidate_set.discard(b_old_idx)
                    if candidate_set:
                        # Pick candidate that maximizes additional coverage
                        best_new = None
                        best_score = -1
                        cand_list = list(candidate_set)
                        random.shuffle(cand_list)
                        for b_new_idx in cand_list[:30]:
                            score = np.sum(cover_count[block_to_tsubs_np[b_new_idx]] == 0)
                            if score > best_score:
                                best_score = score
                                best_new = b_new_idx

                        if best_new is not None:
                            remove_block(b_old_idx)
                            add_block(best_new)
                            improved = True
                            swapped = True
                            break

            if swapped:
                no_improvement_count = 0
                if len(blocks_in_cover) < best_size:
                    best_size = len(blocks_in_cover)
                    best_blocks_snapshot = list(blocks_in_cover)
                continue

        # Phase 3: Try 2-for-1 swaps (remove 2, add 1) - only if size reduction possible
        if len(blocks_in_cover) > 10 and time.time() < deadline - 10:
            two_swap_done = False
            for i in range(min(20, len(critical_blocks) if critical_blocks else 0)):
                if time.time() > deadline - 10:
                    break
                if not critical_blocks:
                    break
                    
                b1 = critical_blocks[i % len(critical_blocks)]
                if b1 not in in_cover_set:
                    continue
                    
                u1 = get_unique_tsubs(b1)
                if len(u1) == 0:
                    continue
                
                # Find another critical block
                for b2 in critical_blocks:
                    if b2 == b1 or b2 not in in_cover_set:
                        continue
                    u2 = get_unique_tsubs(b2)
                    if len(u2) == 0:
                        continue
                    
                    # Combined unique t-subsets
                    combined_unique = np.union1d(u1, u2)
                    
                    # Find blocks covering all combined unique t-subsets
                    candidate_set = set(t_sub_to_blocks[combined_unique[0]])
                    for t_idx in combined_unique[1:]:
                        candidate_set.intersection_update(t_sub_to_blocks[t_idx])
                        if not candidate_set:
                            break
                    
                    if candidate_set:
                        candidate_set.discard(b1)
                        candidate_set.discard(b2)
                        if candidate_set:
                            best_new = None
                            best_score = -1
                            for b_new_idx in list(candidate_set)[:15]:
                                score = np.sum(cover_count[block_to_tsubs_np[b_new_idx]] == 0)
                                if score > best_score:
                                    best_score = score
                                    best_new = b_new_idx
                            
                            if best_new is not None:
                                remove_block(b1)
                                remove_block(b2)
                                add_block(best_new)
                                improved = True
                                two_swap_done = True
                                break
                if two_swap_done:
                    break
            
            if two_swap_done:
                no_improvement_count = 0
                if len(blocks_in_cover) < best_size:
                    best_size = len(blocks_in_cover)
                    best_blocks_snapshot = list(blocks_in_cover)
                continue

        # No improvement - consider restart
        no_improvement_count += 1
        if no_improvement_count > max_no_improvement:
            # ILS restart: perturb current solution
            restart_count += 1
            if restart_count > max_restarts:
                break
            
            # Restore best snapshot
            blocks_in_cover.clear()
            in_cover_set.clear()
            cover_count.fill(0)
            for b_idx in best_blocks_snapshot:
                add_block(b_idx)
            
            # Perturb: remove some random blocks and re-greedy
            if len(blocks_in_cover) > 5:
                num_remove = max(1, len(blocks_in_cover) // 10)
                to_perturb = random.sample(blocks_in_cover, min(num_remove, len(blocks_in_cover)))
                for b_idx in to_perturb:
                    remove_block(b_idx)
                
                # Re-greedy to fill gaps
                uncovered_mask = (cover_count == 0)
                while np.any(uncovered_mask) and time.time() < deadline - 5:
                    uncovered_indices = np.where(uncovered_mask)[0]
                    if len(uncovered_indices) == 0:
                        break
                    target_idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
                    candidates = t_sub_to_blocks[target_idx]
                    cand_arr = np.array(candidates, dtype=np.int32)
                    gains = np.sum(cover_count[block_to_tsubs_np[cand_arr]] == 0, axis=1)
                    best_local = np.argmax(gains)
                    best_block_idx = cand_arr[best_local]
                    if gains[best_local] > 0:
                        add_block(best_block_idx)
                        newly_covered = block_to_tsubs_np[best_block_idx]
                        uncovered_mask[newly_covered] = (cover_count[newly_covered] == 0)
            
            no_improvement_count = 0

    # Final cleanup: remove any remaining redundant blocks
    changed = True
    while changed:
        changed = False
        to_remove = []
        for b_idx in blocks_in_cover:
            if is_redundant(b_idx):
                to_remove.append(b_idx)
        for b_idx in to_remove:
            remove_block(b_idx)
            changed = True

    # Convert block indices back to lists of points
    final_blocks = [list(all_blocks[i]) for i in blocks_in_cover]
    final_blocks.sort()

    return final_blocks
