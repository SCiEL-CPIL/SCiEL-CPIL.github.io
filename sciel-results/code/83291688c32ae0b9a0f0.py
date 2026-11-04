    for out_block_idx in range(4):
        for r in range(7):
            coeff = outer_W_coeffs[out_block_idx].get(r, 0)
            if coeff == 0: continue
            
            block_indices = get_block_indices(out_block_idx // 2, out_block_idx % 2)
            
            for local_out_idx, global_out_idx in enumerate(block_indices):
                for t in range(7):
                    col = r * 7 + t
                    W[global_out_idx, col] += coeff * W_inner[local_out_idx, t]
