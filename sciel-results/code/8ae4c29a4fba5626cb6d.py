<<<<<<< SEARCH
    def try_swap(c):
        if np.min(counts[neighbors_list[c]]) > 1:
            return False

        crit_points = [i for i in neighbors_list[c] if counts[i] == 1]
        if not crit_points:
            return False

        found_swap = False
        shuffled_crit = list(crit_points)
        random.shuffle(shuffled_crit)

        for p in shuffled_crit:
            if found_swap:
                break

            cands = [x for x in neighbor_sets[p] if x not in code_set]
            
            best_w = -1
            best_score = -1
            # Score: minimize redundancy. 
            # Current redundancy in c's neighborhood: sum(counts[nb] - 1 for nb in nb_c if counts[nb]>1)
            # New redundancy: sum(counts[nb] - 1 for nb in nb_w if counts[nb]>1) 
            # Actually, we want to maximize the number of points that become "safe" (count > 1) 
            # or minimize the total sum of counts.
            
            # Simple score: maximize points covered by w that are currently count 1 (critical for c)
            # This is already handled by the validity check.
            # Secondary score: maximize points in w's neighborhood that are count 1 (making them count 2, increasing redundancy but safe)
            # Wait, we want to REMOVE c. So we want w to cover the critical points.
            # Any additional coverage by w is "free" if it doesn't increase size.
            # So we prefer w that covers *more* critical points? 
            # The validity check requires covering *all* critical points of c.
            # So all valid w cover all crit_points.
            # Then we prefer w that has high "potential" to cover other things?
            # Or low "cost"?
            # Let's stick to: prefer w that has the most neighbors with count == 1.
            # This creates redundancy, which helps future removals.
            
            for w in cands:
                valid = True
                for k in crit_points:
                    if k not in neighbor_sets[w]:
                        valid = False
                        break

                if not valid:
                    continue

                score = int(np.sum(counts[neighbors_list[w]] == 1))
                if score > best_score:
                    best_score = score
                    best_w = w

            if best_w != -1:
                code.remove(c)
                code_set.remove(c)
                counts[neighbors_list[c]] -= 1

                code.append(best_w)
                code_set.add(best_w)
                counts[neighbors_list[best_w]] += 1

                found_swap = True
                return True

        # Strategy 2: Relocation
        temp_counts = counts.copy()
        temp_counts[neighbors_list[c]] -= 1
        would_be_uncovered = [i for i in neighbors_list[c] if temp_counts[i] == 0]

        if not would_be_uncovered:
            return False

        all_needed = set(crit_points) | set(would_be_uncovered)
        
        best_w = -1
        best_score = -1
        first_point = list(all_needed)[0]
        cands = [x for x in neighbor_sets[first_point] if x not in code_set]

        for w in cands:
            valid = True
            for k in all_needed:
                if k not in neighbor_sets[w]:
                    valid = False
                    break
            if not valid:
                continue

            score = int(np.sum(counts[neighbors_list[w]] == 1))
            if score > best_score:
                best_score = score
                best_w = w

        if best_w != -1:
            code.remove(c)
            code_set.remove(c)
            counts[neighbors_list[c]] -= 1

            code.append(best_w)
            code_set.add(best_w)
            counts[neighbors_list[best_w]] += 1
            return True

        return False
=======
    def try_swap(c):
        if np.min(counts[neighbors_list[c]]) > 1:
            return False

        crit_points = [i for i in neighbors_list[c] if counts[i] == 1]
        if not crit_points:
            return False

        # Strategy 1: Find a single replacement that covers all critical points
        # Precompute candidate set: intersection of neighbor sets of all crit_points
        if len(crit_points) > 1:
            candidate_set = set(neighbor_sets[crit_points[0]])
            for p in crit_points[1:]:
                candidate_set &= neighbor_sets[p]
        else:
            candidate_set = set(neighbor_sets[crit_points[0]])
        
        cands = [x for x in candidate_set if x not in code_set]
        
        if cands:
            best_w = -1
            best_score = -1
            for w in cands:
                score = int(np.sum(counts[neighbors_list[w]] == 1))
                if score > best_score:
                    best_score = score
                    best_w = w

            if best_w != -1:
                code.remove(c)
                code_set.remove(c)
                counts[neighbors_list[c]] -= 1

                code.append(best_w)
                code_set.add(best_w)
                counts[neighbors_list[best_w]] += 1
                return True

        # Strategy 2: Relocation - find a word that covers all points that would become uncovered
        temp_counts = counts.copy()
        temp_counts[neighbors_list[c]] -= 1
        would_be_uncovered = [i for i in neighbors_list[c] if temp_counts[i] == 0]

        if not would_be_uncovered:
            return False

        all_needed = set(crit_points) | set(would_be_uncovered)
        
        # Find candidates that cover all needed points
        if len(all_needed) > 1:
            candidate_set = set(neighbor_sets[list(all_needed)[0]])
            for p in list(all_needed)[1:]:
                candidate_set &= neighbor_sets[p]
        else:
            candidate_set = set(neighbor_sets[list(all_needed)[0]])
        
        cands = [x for x in candidate_set if x not in code_set]

        best_w = -1
        best_score = -1
        for w in cands:
            score = int(np.sum(counts[neighbors_list[w]] == 1))
            if score > best_score:
                best_score = score
                best_w = w

        if best_w != -1:
            code.remove(c)
            code_set.remove(c)
            counts[neighbors_list[c]] -= 1

            code.append(best_w)
            code_set.add(best_w)
            counts[neighbors_list[best_w]] += 1
            return True

        return False
>>>>>>> REPLACE

