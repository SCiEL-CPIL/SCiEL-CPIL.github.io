<<<<<<< SEARCH
    def local_search(present, k222_count, count, rng):
        # Phase 1: Exhaustive Add
        count = exhaustive_add(present, k222_count, count)
        
        # Phase 2: WalkSAT-style stochastic local search
        # Maintain a list of present indices for fast random sampling
        present_indices = [i for i in range(num_total) if present[i]]
        
        max_steps = 5000
        no_improve_limit = 100
        
        for step in range(max_steps):
            if time_check():
                break
            
            # 1. Try to add any missing triple (climb phase)
            # Instead of scanning all 560, we can pick a random subset of missing ones
            # Or just scan a few random missing ones
            # Let's try a few random missing ones first
            added_any = False
            for _ in range(10):
                if time_check():
                    break
                # Pick a random index
                ti = rng.randint(0, num_total - 1)
                if not present[ti] and can_add(ti, k222_count):
                    do_add(ti, present, k222_count)
                    count += 1
                    present_indices.append(ti)
                    added_any = True
                    break
            
            if added_any:
                continue
            
            # 2. If no add, try a swap
            # Pick a random missing triple
            # To ensure we pick a missing one, we might need a list of missing ones
            # But maintaining a list of missing is expensive.
            # Let's just pick random indices until we find a missing one
            t_add = -1
            for _ in range(20):
                ti = rng.randint(0, num_total - 1)
                if not present[ti]:
                    t_add = ti
                    break
            
            if t_add == -1:
                # All present? Unlikely. Or we are stuck.
                break
                
            # Find blocking critical K222s for t_add
            blocking_crits = []
            for ci in triple_to_k222[t_add]:
                if k222_count[ci] == 7:
                    blocking_crits.append(ci)
                    
            if not blocking_crits:
                # Should have been added. Maybe we missed it?
                # If can_add was false, there must be a blocker.
                # If can_add was true, we should have added it.
                # So this case implies can_add was false but no count==7?
                # That's impossible if can_add checks count==7.
                # Unless there's a bug.
                # Let's just try to add it again?
                if can_add(t_add, k222_count):
                    do_add(t_add, present, k222_count)
                    count += 1
                    present_indices.append(t_add)
                    continue
                else:
                    # Still blocked? Weird.
                    continue

            # Pick a random blocker
            blocker_ci = rng.choice(blocking_crits)
            
            # Pick a random present triple in this blocker to remove
            # We need to find present triples in k222_config_lists[blocker_ci]
            # This is O(8)
            candidates = []
            for ti in k222_config_lists[blocker_ci]:
                if present[ti]:
                    candidates.append(ti)
                    
            if not candidates:
                continue
                
            t_rem = rng.choice(candidates)
            
            # Perform swap
            do_remove(t_rem, present, k222_count)
            count -= 1
            present_indices.remove(t_rem) # O(N), but N is smallish
            
            # Try to add t_add
            if can_add(t_add, k222_count):
                do_add(t_add, present, k222_count)
                count += 1
                present_indices.append(t_add)
            else:
                # Failed swap, revert
                do_add(t_rem, present, k222_count)
                count += 1
                present_indices.append(t_rem)
                
        # Final exhaustive add
        count = exhaustive_add(present, k222_count, count)
        
        return count
=======
    def find_targeted_swap(t_add, present, k222_count, rng):
        """Find a present triple to remove that would allow adding t_add."""
        # Find all blocking configs
        blocking_crits = [ci for ci in triple_to_k222[t_add] if k222_count[ci] == 7]
        if not blocking_crits:
            return -1
        
        # For each blocker, find candidates to remove
        # Prefer candidates that appear in the fewest other critical configs
        # This minimizes the chance of breaking other constraints
        best_candidate = -1
        best_score = float('inf')
        
        for ci in blocking_crits:
            for ti in k222_config_lists[ci]:
                if present[ti]:
                    # Count how many other critical configs this triple is in
                    critical_count = sum(1 for cj in triple_to_k222[ti] if k222_count[cj] == 7)
                    if critical_count < best_score:
                        best_score = critical_count
                        best_candidate = ti
        
        return best_candidate

    def try_2_for_1(present, k222_count, count, rng):
        """Try to add two triples by removing one, if possible."""
        # Find a missing triple that is blocked
        t_add1 = -1
        for _ in range(30):
            ti = rng.randint(0, num_total - 1)
            if not present[ti]:
                t_add1 = ti
                break
        if t_add1 == -1:
            return False, count
        
        # Find a second missing triple that shares a blocker with t_add1
        t_add2 = -1
        blockers1 = [ci for ci in triple_to_k222[t_add1] if k222_count[ci] == 7]
        if not blockers1:
            return False, count
        
        for _ in range(30):
            ti = rng.randint(0, num_total - 1)
            if not present[ti] and ti != t_add1:
                # Check if it shares a blocker
                for ci in triple_to_k222[ti]:
                    if ci in blockers1 and k222_count[ci] == 7:
                        t_add2 = ti
                        break
                if t_add2 != -1:
                    break
        
        if t_add2 == -1:
            return False, count
        
        # Find a single triple to remove that unblocks both
        # It must be in a critical config for both t_add1 and t_add2
        candidate_removals = set()
        for ci in blockers1:
            for ti in k222_config_lists[ci]:
                if present[ti]:
                    # Check if removing ti would help t_add2
                    blockers2 = [cj for cj in triple_to_k222[t_add2] if k222_count[cj] == 7]
                    # ti must be in at least one of t_add2's blockers
                    for cj in blockers2:
                        if ti in k222_config_lists[cj]:
                            candidate_removals.add(ti)
                            break
        
        if not candidate_removals:
            return False, count
        
        t_rem = rng.choice(list(candidate_removals))
        
        # Perform the 2-for-1 move
        do_remove(t_rem, present, k222_count)
        count -= 1
        
        added1 = False
        added2 = False
        if can_add(t_add1, k222_count):
            do_add(t_add1, present, k222_count)
            count += 1
            added1 = True
        if can_add(t_add2, k222_count):
            do_add(t_add2, present, k222_count)
            count += 1
            added2 = True
        
        if added1 and added2:
            return True, count
        elif added1 or added2:
            # Net zero or positive, keep it
            return True, count
        else:
            # Revert
            do_add(t_rem, present, k222_count)
            count += 1
            return False, count

    def local_search(present, k222_count, count, rng):
        # Phase 1: Exhaustive Add
        count = exhaustive_add(present, k222_count, count)
        
        # Phase 2: Enhanced stochastic local search
        max_steps = 8000
        no_improve_limit = 200
        no_improve = 0
        best_in_run = count
        
        for step in range(max_steps):
            if time_check():
                break
            
            # 1. Try to add any missing triple (climb phase)
            added_any = False
            for _ in range(15):
                if time_check():
                    break
                ti = rng.randint(0, num_total - 1)
                if not present[ti] and can_add(ti, k222_count):
                    do_add(ti, present, k222_count)
                    count += 1
                    added_any = True
                    no_improve = 0
                    break
            
            if added_any:
                if count > best_in_run:
                    best_in_run = count
                continue
            
            # 2. If no add, try targeted swap
            t_add = -1
            for _ in range(25):
                ti = rng.randint(0, num_total - 1)
                if not present[ti]:
                    t_add = ti
                    break
            
            if t_add == -1:
                break
                
            # Find blocking critical K222s for t_add
            blocking_crits = [ci for ci in triple_to_k222[t_add] if k222_count[ci] == 7]
            
            if not blocking_crits:
                if can_add(t_add, k222_count):
                    do_add(t_add, present, k222_count)
                    count += 1
                    no_improve = 0
                    if count > best_in_run:
                        best_in_run = count
                continue

            # Try targeted swap: find best candidate to remove
            t_rem = find_targeted_swap(t_add, present, k222_count, rng)
            
            if t_rem != -1:
                # Perform swap
                do_remove(t_rem, present, k222_count)
                count -= 1
                
                if can_add(t_add, k222_count):
                    do_add(t_add, present, k222_count)
                    count += 1
                    no_improve = 0
                    if count > best_in_run:
                        best_in_run = count
                else:
                    # Failed swap, revert
                    do_add(t_rem, present, k222_count)
                    count += 1
                    no_improve += 1
            else:
                no_improve += 1
            
            # 3. Occasionally try 2-for-1 moves to escape local optima
            if step % 50 == 0:
                success, count = try_2_for_1(present, k222_count, count, rng)
                if success and count > best_in_run:
                    best_in_run = count
                    no_improve = 0
                elif success:
                    no_improve = 0
                else:
                    no_improve += 1
            
            # 4. If stuck for too long, do a random perturbation
            if no_improve > no_improve_limit:
                # Remove 3-5 random present triples to escape
                num_remove = rng.randint(3, 5)
                present_indices = [i for i in range(num_total) if present[i]]
                rng.shuffle(present_indices)
                for ti in present_indices[:num_remove]:
                    do_remove(ti, present, k222_count)
                    count -= 1
                no_improve = 0
                # Try to rebuild
                count = exhaustive_add(present, k222_count, count)
                if count > best_in_run:
                    best_in_run = count
        
        # Final exhaustive add
        count = exhaustive_add(present, k222_count, count)
        
        return count
>>>>>>> REPLACE

