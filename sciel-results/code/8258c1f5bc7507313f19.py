        def find_largest_sidon(n):
            best_set = []
            
            def backtrack(start, current_set, used_sums):
                nonlocal best_set
                if len(current_set) > len(best_set):
                    best_set = current_set.copy()
                
                # Pruning: if remaining elements cannot beat best_size, return
                if len(current_set) + (n - start) <= len(best_set):
                    return
                
                for i in range(start, n):
                    # Check if i can be added
                    ok = True
                    new_sums = []
                    for j in current_set:
                        s_val = i + j
                        if s_val in used_sums:
                            ok = False
                            break
                        new_sums.append(s_val)
                    if i + i in used_sums:
                        ok = False
                    
                    if ok:
                        current_set.append(i)
                        used_sums.update(new_sums)
                        used_sums.add(i + i)
                        
                        backtrack(i + 1, current_set, used_sums)
                        
                        current_set.pop()
                        used_sums.difference_update(new_sums)
                        used_sums.discard(i + i)
            
            backtrack(0, [], set())
            return best_set
        