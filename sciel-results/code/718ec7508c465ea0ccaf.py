<<<<<<< SEARCH
    def local_search(selected, covers, num_elements, time_limit=20):
        """Local search with add-then-remove heuristic."""
        t0 = time.time()
        best = list(selected)
        best_len = len(best)
        
        # Compute coverage for best
        def compute_cov_count(sel):
            cc = np.zeros(num_elements, dtype=np.int32)
            for si in sel:
                for e in covers[si]:
                    cc[e] += 1
            return cc
        
        while time.time() - t0 < time_limit:
            # Try adding a random codeword, then removing
            if len(best) >= num_elements:
                break
            cc = compute_cov_count(best)
            
            # Find uncovered elements
            uncovered = np.where(cc == 0)[0]
            if len(uncovered) > 0:
                # Need to add codewords to cover uncovered
                # Pick a codeword that covers many uncovered
                best_add = -1
                best_add_count = -1
                for si in range(num_elements):
                    if si in best:
                        continue
                    count = sum(1 for e in covers[si] if cc[e] == 0)
                    if count > best_add_count:
                        best_add_count = count
                        best_add = si
                if best_add >= 0 and best_add_count > 0:
                    candidate = best + [best_add]
                    candidate = try_remove(candidate, covers, num_elements)
                    if len(candidate) < best_len:
                        best = candidate
                        best_len = len(candidate)
                        continue
            
            # Try swap: replace one codeword with another
            if len(best) > 0:
                # Try removing one and see if we can cover with a different one
                for i in range(len(best)):
                    if time.time() - t0 > time_limit:
                        break
                    si = best[i]
                    temp = best[:i] + best[i+1:]
                    cc = compute_cov_count(temp)
                    uncovered = np.where(cc == 0)[0]
                    if len(uncovered) == 0:
                        # Can remove
                        best = temp
                        best_len = len(best)
                    elif len(uncovered) <= 3:
                        # Try to find a single codeword that covers all uncovered
                        for sj in range(num_elements):
                            if sj in best:
                                continue
                            if all(cc[e] == 0 and e in covers[sj] for e in uncovered):
                                candidate = temp + [sj]
                                candidate = try_remove(candidate, covers, num_elements)
                                if len(candidate) < best_len:
                                    best = candidate
                                    best_len = len(candidate)
                                break
        
        return best
=======
    def local_search(selected, covers, num_elements, time_limit=20):
        """Local search with add-then-remove heuristic, robust against list resizing."""
        t0 = time.time()
        best = list(selected)
        best_len = len(best)
        
        # Compute coverage for best
        def compute_cov_count(sel):
            cc = np.zeros(num_elements, dtype=np.int32)
            for si in sel:
                for e in covers[si]:
                    cc[e] += 1
            return cc
        
        while time.time() - t0 < time_limit:
            if len(best) >= num_elements:
                break
            
            cc = compute_cov_count(best)
            uncovered = np.where(cc == 0)[0]
            
            if len(uncovered) > 0:
                # Need to add codewords to cover uncovered
                # Pick a codeword that covers many uncovered
                best_add = -1
                best_add_count = -1
                for si in range(num_elements):
                    if si in best:
                        continue
                    count = sum(1 for e in covers[si] if cc[e] == 0)
                    if count > best_add_count:
                        best_add_count = count
                        best_add = si
                if best_add >= 0 and best_add_count > 0:
                    candidate = best + [best_add]
                    candidate = try_remove(candidate, covers, num_elements)
                    if len(candidate) < best_len:
                        best = candidate
                        best_len = len(candidate)
                        continue
            
            # Try swap: replace one codeword with another
            # We iterate over indices, but if best changes, we restart the loop
            # to avoid IndexError from stale indices.
            improved = False
            for i in range(len(best)):
                if time.time() - t0 > time_limit:
                    break
                si = best[i]
                temp = best[:i] + best[i+1:]
                cc_temp = compute_cov_count(temp)
                uncovered_temp = np.where(cc_temp == 0)[0]
                
                if len(uncovered_temp) == 0:
                    # Can remove
                    best = temp
                    best_len = len(best)
                    improved = True
                    break
                elif len(uncovered_temp) <= 3:
                    # Try to find a single codeword that covers all uncovered
                    for sj in range(num_elements):
                        if sj in best:
                            continue
                        if all(cc_temp[e] == 0 and e in covers[sj] for e in uncovered_temp):
                            candidate = temp + [sj]
                            candidate = try_remove(candidate, covers, num_elements)
                            if len(candidate) < best_len:
                                best = candidate
                                best_len = len(candidate)
                                improved = True
                            break
                    if improved:
                        break
                        
            if not improved:
                # If no improvement in this pass, we might loop forever if we don't change state
                # To prevent infinite loops without progress, we can add a random perturbation or break
                # But typically local search will find a move if one exists.
                # If no move found, we can try a random restart of the local search logic or break.
                # For now, we just continue the while loop, which will re-evaluate.
                # To avoid stuck loops, we could add a counter.
                pass
        
        return best
>>>>>>> REPLACE

Additionally, I will optimize the greedy completion in the main loop to use a set for `selected` to speed up `if si in selected` checks, and ensure the `best_sets` logic is robust.

