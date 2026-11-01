<<<<<<< SEARCH
def _try_block_replacement(block_indices, Cover, n_tsubs, cover_counts, deadline):
    """Try to replace blocks with alternatives that maintain coverage."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        new_block_indices = []
        for idx in block_indices:
            # Find critical t-subsets covered by this block (count == 1)
            critical_mask = (Cover[idx] == 1) & (cover_counts == 1)
            critical_tsubs = np.where(critical_mask)[0]
            
            if len(critical_tsubs) == 0:
                # No critical t-subsets; this block is redundant
                cover_counts -= Cover[idx]
                changed = True
                continue
            
            # Find alternative blocks that cover ALL critical t-subsets
            candidate_scores = np.sum(Cover[:, critical_tsubs], axis=1)
            valid_candidates = candidate_scores == len(critical_tsubs)
            valid_candidates[idx] = False
            
            if not np.any(valid_candidates):
                new_block_indices.append(idx)
                continue
            
            # Among valid candidates, pick one that covers the most low-count t-subsets
            low_count_mask = (cover_counts <= 2).astype(np.float64)
            candidate_low_score = np.sum(Cover * low_count_mask, axis=1)
            
            valid_indices = np.where(valid_candidates)[0]
            best_alt = valid_indices[np.argmax(candidate_low_score[valid_indices])]
            
            # Replace
            cover_counts -= Cover[idx]
            cover_counts += Cover[best_alt]
            new_block_indices.append(best_alt)
            changed = True
        
        block_indices = new_block_indices
    
    return block_indices
=======
def _try_block_replacement(block_indices, Cover, n_tsubs, cover_counts, deadline):
    """Try to replace blocks with alternatives that maintain coverage."""
    changed = True
    while changed and time.time() < deadline - 1.0:
        changed = False
        new_block_indices = []
        for idx in block_indices:
            # Find critical t-subsets covered by this block (count == 1)
            critical_mask = (Cover[idx] == 1) & (cover_counts == 1)
            critical_tsubs = np.where(critical_mask)[0]
            
            if len(critical_tsubs) == 0:
                # No critical t-subsets; this block is redundant
                cover_counts -= Cover[idx]
                changed = True
                continue
            
            # Find alternative blocks that cover ALL critical t-subsets
            candidate_scores = np.sum(Cover[:, critical_tsubs], axis=1)
            valid_candidates = candidate_scores == len(critical_tsubs)
            valid_candidates[idx] = False
            
            if not np.any(valid_candidates):
                new_block_indices.append(idx)
                continue
            
            # Among valid candidates, pick one that covers the most low-count t-subsets
            low_count_mask = (cover_counts <= 2).astype(np.float64)
            candidate_low_score = np.sum(Cover * low_count_mask, axis=1)
            
            valid_indices = np.where(valid_candidates)[0]
            best_alt = valid_indices[np.argmax(candidate_low_score[valid_indices])]
            
            # Replace
            cover_counts -= Cover[idx]
            cover_counts += Cover[best_alt]
            new_block_indices.append(best_alt)
            changed = True
        
        block_indices = new_block_indices
    
    return block_indices


def _final_polish(block_indices, Cover, n_tsubs, deadline):
    """Final high-precision polish: try removing one block and re-adding optimal replacement."""
    if len(block_indices) < 2:
        return block_indices
    
    # Compute coverage counts
    cover_counts = np.zeros(n_tsubs, dtype=np.int32)
    for idx in block_indices:
        cover_counts += Cover[idx]
    
    # Try removing each block and see if we can cover the newly uncovered t-subsets
    # with a single other block
    for i, idx in enumerate(block_indices):
        if time.time() >= deadline - 0.5:
            break
        
        # Temporarily remove this block
        cover_counts -= Cover[idx]
        
        # Find t-subsets that are now uncovered (count == 0)
        uncovered = np.where(cover_counts == 0)[0]
        
        if len(uncovered) == 0:
            # Block was redundant
            block_indices.pop(i)
            changed = True
            # Recompute cover_counts
            cover_counts = np.zeros(n_tsubs, dtype=np.int32)
            for j in block_indices:
                cover_counts += Cover[j]
            break
        
        # Find a single block that covers all uncovered t-subsets
        if len(uncovered) <= 3:  # Only try if few uncovered
            candidate_scores = np.sum(Cover[:, uncovered], axis=1)
            valid = candidate_scores == len(uncovered)
            valid[idx] = False
            
            if np.any(valid):
                # Check if this replacement doesn't create new issues
                valid_indices = np.where(valid)[0]
                # Pick the one that covers the most other low-count t-subsets
                low_mask = (cover_counts <= 1).astype(np.float64)
                scores = np.sum(Cover[valid_indices] * low_mask, axis=1)
                best_alt = valid_indices[np.argmax(scores)]
                
                # Verify: after adding best_alt, no t-subset has count 0
                test_counts = cover_counts + Cover[best_alt]
                if np.all(test_counts > 0):
                    block_indices[i] = best_alt
                    cover_counts = test_counts
                    changed = True
                    # Recompute to be safe
                    cover_counts = np.zeros(n_tsubs, dtype=np.int32)
                    for j in block_indices:
                        cover_counts += Cover[j]
                    break
        
        # Restore if not removed
        cover_counts += Cover[idx]
    
    return block_indices
>>>>>>> REPLACE
