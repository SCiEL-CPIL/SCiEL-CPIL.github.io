    def local_search():
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            
            # 1. Try to add points
            candidates = [i for i in range(M) if i not in S_set]
            random.shuffle(candidates)
            
            for p in candidates:
                if time.time() > deadline - 10:
                    break
                if try_add(p):
                    add_point(p)
                    improved = True
                    break # Restart the loop to re-evaluate
            
            if improved:
                continue
            
            # 2. Try to remove a point and refill
            if not S_list:
                break
                
            # Try removing each point in random order
            S_copy = S_list[:]
            random.shuffle(S_copy)
            
            for q in S_copy:
                if time.time() > deadline - 10:
                    break
                    
                # Save state
                old_S_list = S_list[:]
                old_S_set = S_set.copy()
                old_dist_to_S = {k: set(v) for k, v in dist_to_S.items()}
                
                # Remove q
                remove_point(q)
                
                # Try to add as many points as possible
                new_size = len(S_list)
                candidates = [i for i in range(M) if i not in S_set]
                random.shuffle(candidates)
                
                for p in candidates:
                    if time.time() > deadline - 10:
                        break
                    if try_add(p):
                        add_point(p)
                
                if len(S_list) > new_size:
                    improved = True
                    # Keep the new state
                    break
                else:
                    # Restore state
                    S_list[:] = old_S_list
                    S_set.clear()
                    S_set.update(old_S_set)
                    dist_to_S.clear()
                    for k, v in old_dist_to_S.items():
                        dist_to_S[k] = v

        return len(S_list)
