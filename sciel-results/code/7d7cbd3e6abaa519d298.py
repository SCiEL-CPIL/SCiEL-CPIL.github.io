    # Final exact polish pass on the best set (add any non-conflicting node)
    if time.time() < deadline - 5.0:
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            sel_list = list(best_iset)
            if sel_list:
                blocked = C[sel_list].any(axis=0)
            else:
                blocked = np.zeros(N_sup, dtype=bool)
            
            sel_mask = np.zeros(N_sup, dtype=bool)
            for v in best_iset:
                sel_mask[v] = True
                
            cands = np.flatnonzero(~sel_mask & ~blocked)
            if len(cands) > 0:
                for c in cands:
                    best_iset.add(c)
                changed = True
