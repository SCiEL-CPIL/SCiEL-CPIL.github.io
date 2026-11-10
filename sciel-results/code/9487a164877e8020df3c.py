        def _search(cand_mask, cur_val, chosen):
            global _best_set, _best_val
            if not _time_left():
                return
            if cand_mask == 0:
                if cur_val > _best_val:
                    _best_val = cur_val
                    _best_set = list(chosen)
                return
            
            # Upper bound
            ub = _color_ub(cand_mask)
            if cur_val + ub <= _best_val + 1e-15:
                return
            
            # Select branching vertex
            # Heuristic: Maximize WEIGHT[v] * (1 + degree(v) in cand_mask)
            verts = [i for i in range(N) if (cand_mask >> i) & 1]
            if not verts:
                return
            
            # Calculate degree in cand_mask for each vert
            # To save time, we can just pick max weight if we are slow, but let's try the heuristic
            best_v = -1
            best_score = -1.0
            for v in verts:
                deg = bin(CONF[v] & cand_mask).count('1')
                score = WEIGHT[v] * (1 + deg)
                if score > best_score:
                    best_score = score
                    best_v = v
            
            v = best_v
            
            # Branch include
            new_cand = cand_mask & ~CONF[v] & ~(1 << v)
            _search(new_cand, cur_val + WEIGHT[v], chosen + [v])
            if not _time_left():
                return
            
            # Branch exclude
            _search(cand_mask & ~(1 << v), cur_val, chosen)
        