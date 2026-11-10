        for w_idx in uncovered_ids[0]:
            val = int(uncovered[w_idx])
            if val == 0: continue
            while val:
                # Get index of lowest set bit
                # Python 3.11+ has int.bit_count, but we need index.
                # val & -val gives lowest set bit.
                lsb = val & -val
                # Index is log2(lsb). Since lsb is power of 2, we can use bit_length - 1
                b_idx = lsb.bit_length() - 1
                tid = w_idx * 64 + b_idx
                if tid < num_triples:
                    if triple_to_blocks[tid]:
                        b_idx_block = triple_to_blocks[tid][0]
                        if b_idx_block not in selected_set:
                            selected_indices.append(b_idx_block)
                            selected_set.add(b_idx_block)
                val &= val - 1 # Clear lowest set bit
