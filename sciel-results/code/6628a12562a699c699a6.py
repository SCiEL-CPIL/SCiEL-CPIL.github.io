def _local_search(chosen, rng, deadline):
    chosen_set = set(chosen)
    remaining = [i for i in range(_N_SUBSETS) if i not in chosen_set]
    rng.shuffle(remaining)
    
    t_start = time.time()
    stuck_count = 0
    
    while time.time() - t_start < deadline:
        improved = False
        added = False
        
        for idx, c in enumerate(remaining):
            if not _is_conflict(c, chosen, _INTER):
                chosen.append(c)
                chosen_set.add(c)
                remaining.pop(idx)
                improved = True
                added = True
                stuck_count = 0
                break
        
        if not added:
            stuck_count += 1
            if stuck_count > 200:
                break
                
            if len(chosen) > 50:
                remove_idx = rng.randrange(len(chosen))
                removed_set = chosen[remove_idx]
                chosen.pop(remove_idx)
                chosen_set.remove(removed_set)
                
                for idx, c in enumerate(remaining):
                    if not _is_conflict(c, chosen, _INTER):
                        chosen.append(c)
                        chosen_set.add(c)
                        remaining.pop(idx)
                        improved = True
                        stuck_count = 0
                        break
                        
    return chosen
