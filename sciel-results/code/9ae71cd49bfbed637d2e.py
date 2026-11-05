        is_uncovered = np.ones(n_t, dtype=bool)
        blocks = []
        
        while is_uncovered.any():
            # Convert bool to int for dot product
            # Or use sparse matrix multiplication with bool?
            # sp.csr_matrix @ np.array(bool) -> works?
            # Let's try. If not, cast.
            counts = block_matrix.dot(is_uncovered)
            
            max_covered = np.max(counts)
            if max_covered == 0:
                break
                
            candidates = np.where(counts == max_covered)[0]
            best_block_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            blocks.append(list(k_subsets[best_block_idx]))
            
            row = block_matrix[best_block_idx]
            is_uncovered[row.indices] = False
