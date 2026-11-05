<<<<<<< SEARCH
            # Local Search
            # 1. Removal: if a vertex is redundant, remove it.
            improved = True
            while improved:
                improved = False
                vs = list(current_set)
                rng_solver.shuffle(vs)
                for v in vs:
                    if time.time() > deadline:
                        break
                    can_remove = True
                    for ci in vertex_cliques[v]:
                        # Check if clique ci is hit by any other vertex in current_set
                        # ci is a frozenset
                        c = valid_cliques[ci]
                        # If c is subset of current_set - {v}, then v is redundant for this clique
                        # Actually, we just need to know if there exists u in c \ {v} such that u in current_set
                        found_other = False
                        for u in c:
                            if u != v and u in current_set:
                                found_other = True
                                break
                        if not found_other:
                            can_remove = False
                            break
                    if can_remove:
                        current_set.remove(v)
                        improved = True
                        
            # 2. Swap: try to replace one vertex with another to reduce size? 
            # Actually, if we just remove, we get a minimal set. 
            # To find smaller sets, we might need 2-move or different greedy.
            # However, random restarts of greedy usually find good minimal transversals.
            # Let's also try a "reverse delete" from a large set? No, that's hard to initialize.
            
            # Let's try a simple 1-swap: 
            # Pick v in S, u not in S. If S - {v} + {u} is a valid cover, and maybe "better"?
            # Since we minimize size, swaps don't reduce size. They only help escape local optima for future removals?
            # Or maybe we just stick to removal. The greedy usually gets close to optimal.
            
            if len(current_set) < best_size:
                best_size = len(current_set)
                best_set = set(current_set)
                
                # Early stopping if we hit a theoretical lower bound?
                # Lower bound: Max over cliques of 1? No.
                # LP relaxation lower bound? Too expensive.
                # Just keep searching.
                
        return sorted(best_set) if best_set else []
=======
            # Local Search
            # 1. Removal: if a vertex is redundant, remove it.
            improved = True
            while improved:
                improved = False
                vs = list(current_set)
                rng_solver.shuffle(vs)
                for v in vs:
                    if time.time() > deadline:
                        break
                    can_remove = True
                    for ci in vertex_cliques[v]:
                        # Check if clique ci is hit by any other vertex in current_set
                        # ci is a frozenset
                        c = valid_cliques[ci]
                        # If c is subset of current_set - {v}, then v is redundant for this clique
                        # Actually, we just need to know if there exists u in c \ {v} such that u in current_set
                        found_other = False
                        for u in c:
                            if u != v and u in current_set:
                                found_other = True
                                break
                        if not found_other:
                            can_remove = False
                            break
                    if can_remove:
                        current_set.remove(v)
                        improved = True
                        
            # 2. 1-swap: replace one vertex with another to potentially enable further removals
            swap_improved = True
            while swap_improved:
                swap_improved = False
                vs = list(current_set)
                rng_solver.shuffle(vs)
                for v in vs:
                    if time.time() > deadline:
                        break
                    # Try replacing v with each u not in current_set
                    for u in range(n):
                        if u in current_set:
                            continue
                        temp_set = set(current_set) - {v} | {u}
                        # Check if temp_set is a valid hitting set
                        valid = True
                        for ci in range(C):
                            c = valid_cliques[ci]
                            if not any(x in temp_set for x in c):
                                valid = False
                                break
                        if valid:
                            # Check if this enables removal of another vertex
                            for w in list(temp_set):
                                can_remove_w = True
                                for ci in vertex_cliques[w]:
                                    c = valid_cliques[ci]
                                    found_other = False
                                    for x in c:
                                        if x != w and x in temp_set:
                                            found_other = True
                                            break
                                    if not found_other:
                                        can_remove_w = False
                                        break
                                if can_remove_w:
                                    current_set = temp_set - {w}
                                    swap_improved = True
                                    break
                            if swap_improved:
                                break
                    if swap_improved:
                        break
                
            # 3. 2-swap local search
            current_set, hit_cliques = local_search_2swap(current_set, hit_cliques)
            
            # Final removal pass after 2-swap
            improved = True
            while improved:
                improved = False
                vs = list(current_set)
                rng_solver.shuffle(vs)
                for v in vs:
                    if time.time() > deadline:
                        break
                    can_remove = True
                    for ci in vertex_cliques[v]:
                        c = valid_cliques[ci]
                        found_other = False
                        for u in c:
                            if u != v and u in current_set:
                                found_other = True
                                break
                        if not found_other:
                            can_remove = False
                            break
                    if can_remove:
                        current_set.remove(v)
                        improved = True
                        
            if len(current_set) < best_size:
                best_size = len(current_set)
                best_set = set(current_set)
                
        return sorted(best_set) if best_set else []
>>>>>>> REPLACE
