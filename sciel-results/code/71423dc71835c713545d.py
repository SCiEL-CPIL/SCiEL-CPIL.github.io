import itertools
import random
import math
import time
import numpy as np
import scipy.sparse as sp

def construct(v, k, t):
    points = list(range(v))
    
    # Precompute all subsets of size t
    all_subsets = list(itertools.combinations(points, t))
    n_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # Precompute all blocks of size k
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # Build sparse coverage matrix
    # Rows: blocks, Cols: subsets
    # M[i, j] = 1 if block i covers subset j
    
    row_indices = []
    col_indices = []
    data = []
    
    # For each block, find which subsets it covers
    for b_idx, block in enumerate(all_blocks):
        # Get all t-subsets of this block
        for sub in itertools.combinations(block, t):
            s_idx = subset_to_idx[sub]
            row_indices.append(b_idx)
            col_indices.append(s_idx)
            data.append(1)
            
    # Create CSR matrix for fast row access and dot product
    # dtype=bool or uint8. uint8 is safer for dot product with int vectors.
    cover_matrix = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_blocks, n_subsets), dtype=np.uint8)
    
    # Greedy Set Cover
    # uncovered_vector: 1 if subset is not yet covered, 0 otherwise
    uncovered_vector = np.ones(n_subsets, dtype=np.uint8)
    
    selected_block_indices = []
    
    while np.any(uncovered_vector):
        # Calculate gain for each block: number of currently uncovered subsets it covers
        # gains[i] = sum(cover_matrix[i, :] * uncovered_vector)
        gains = cover_matrix.dot(uncovered_vector)
        
        # Find the block with maximum gain
        best_gain = np.max(gains)
        
        if best_gain == 0:
            # Should not happen if uncovered_vector has 1s and matrix is valid
            break
            
        # Find the index of the best block
        # There might be ties, pick the first one (or randomize for diversity? Stick to first for determinism)
        best_block_idx = np.argmax(gains)
        
        selected_block_indices.append(best_block_idx)
        
        # Update uncovered_vector
        # Get the row for the selected block
        # cover_matrix[best_block_idx] is a sparse matrix
        # We need to set the columns where this block has 1s to 0 in uncovered_vector
        # .indices gives the column indices of non-zero elements in that row
        covered_cols = cover_matrix[best_block_idx].indices
        uncovered_vector[covered_cols] = 0
        
    blocks = [list(all_blocks[i]) for i in selected_block_indices]
    
    # Local Search: Remove redundant blocks
    # We need to check if removing a block leaves any subset uncovered.
    # This is equivalent to checking if any subset has coverage count 1.
    
    # Rebuild coverage counts for the selected blocks
    # We can use the sparse matrix again
    # Create a sparse matrix of just the selected blocks
    if selected_block_indices:
        selected_matrix = cover_matrix[selected_block_indices, :]
        # coverage_counts[j] = number of selected blocks covering subset j
        coverage_counts = selected_matrix.sum(axis=0)
        # coverage_counts is a sparse matrix or array, convert to dense for easy access
        coverage_counts = np.asarray(coverage_counts.todense()).flatten()
        
        changed = True
        while changed:
            changed = False
            # Iterate through selected blocks in reverse to avoid index shifting issues
            for i in range(len(selected_block_indices) - 1, -1, -1):
                b_idx = selected_block_indices[i]
                # Get subsets covered by this block
                covered_cols = cover_matrix[b_idx].indices
                
                # Check if any of these subsets have coverage count 1
                # If yes, removing this block would uncover them.
                # If no, it's redundant.
                is_redundant = True
                for c in covered_cols:
                    if coverage_counts[c] == 1:
                        is_redundant = False
                        break
                        
                if is_redundant:
                    # Remove block
                    selected_block_indices.pop(i)
                    # Update coverage counts
                    for c in covered_cols:
                        coverage_counts[c] -= 1
                    changed = True
                    break # Restart loop as list changed
        
        blocks = [list(all_blocks[i]) for i in selected_block_indices]
        
    return blocks
