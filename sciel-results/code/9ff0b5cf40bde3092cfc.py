                block_i_without_xi = [p for p in blocks[i] if p != x_i]
                block_j_without_xj = [p for p in blocks[j] if p != x_j]
                
                if x_j in block_i_without_xi:
                    continue
                if x_i in block_j_without_xj:
                    continue
