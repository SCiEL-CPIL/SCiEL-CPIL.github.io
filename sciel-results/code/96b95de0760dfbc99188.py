    def local_search(mask):
        improved = True
        while improved:
            improved = False
            # Try add_back
            m2 = add_back(mask)
            if m2 != mask:
                mask = m2
                improved = True
                continue
            
            # Try swap 1-for-1
            excluded = [i for i in range(1, N + 1) if not (mask & (1 << i))]
            random.shuffle(excluded)
            for v in excluded:
                if is_valid_add(mask, v):
                    mask |= (1 << v)
                    improved = True
                    break
                
                # Find blockers
                blocking_nodes = set()
                for t_idx in node_triples[v]:
                    t = bad_triple_masks[t_idx]
                    others = t & ~(1 << v)
                    if (mask & others) == others:
                        x = others
                        while x:
                            lsb = x & -x
                            idx = lsb.bit_length() - 1
                            blocking_nodes.add(idx)
                            x ^= lsb
                
                for u in blocking_nodes:
                    if mask & (1 << u):
                        new_mask = (mask & ~(1 << u)) | (1 << v)
                        if is_valid(new_mask):
                            mask = new_mask
                            improved = True
                            break
                if improved:
                    break
        return mask
