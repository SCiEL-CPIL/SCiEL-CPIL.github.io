        # Update candidates
        new_cand = candidates_mask & ~e_low
        
        # Only check candidates that share u or v
        relevant_cand = new_cand & (vertex_edge_mask[u] | vertex_edge_mask[v])
        
        while relevant_cand:
            f_low = relevant_cand & -relevant_cand
            f_idx = f_low.bit_length() - 1
            relevant_cand ^= f_low
            fu, fv = endpoints[f_idx]
            
            if fu == u or fv == u:
                w = fv if fu == u else fu
                if (new_adj[v] >> w) & 1:
                    new_cand ^= f_low
            elif fu == v or fv == v:
                w = fv if fu == v else fu
                if (new_adj[u] >> w) & 1:
                    new_cand ^= f_low
