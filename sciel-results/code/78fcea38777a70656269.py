            clique_mask = _find_clique_size_gt(comp_adj, n, 10)
            if clique_mask is None:
                return _adj_to_edges(adj, n)
            
            S = []
            temp = clique_mask
            while temp:
                v_bit = temp & -temp
                v = v_bit.bit_length() - 1
                S.append(v)
                temp ^= v_bit
