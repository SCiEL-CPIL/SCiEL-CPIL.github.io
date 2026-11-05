<<<<<<< SEARCH
    def solve_heuristic(S, time_limit):
        n = len(S)
        S_set = set(S)
        
        def greedy_sorted(order):
            result = set()
            for x in order:
                if can_add(x, result):
                    result.add(x)
            return result

        # Multiple initializations
        inits = []
        inits.append(greedy_sorted(S))
        inits.append(greedy_sorted(list(reversed(S))))
        odds = [x for x in S if x % 2 != 0]
        inits.append(set(odds))
        
        # Suffix approach
        suffix = S
        for k in range(1, n+1):
            if 2 * S[n-k] > S[n-1]:
                suffix = S[n-k:]
            else:
                break
        inits.append(set(suffix))
        
        for _ in range(10):
            shuffled = list(S)
            random.shuffle(shuffled)
            inits.append(greedy_sorted(shuffled))
        
        best = max(inits, key=len)
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            improved = False
            best_copy = set(best)
            
            # Try removing each element
            for x in list(best_copy):
                trial = set(best_copy)
                trial.discard(x)
                
                # Try to add elements from S \ trial
                for y in S:
                    if y in trial:
                        continue
                    if can_add(y, trial):
                        trial.add(y)
                
                if len(trial) > len(best):
                    best = trial
                    improved = True
                    break
            
            if not improved:
                # Try random restarts with perturbation
                found = False
                for _ in range(20):
                    if time.time() - start_time > time_limit:
                        break
                    perturbed = set(best)
                    for _ in range(random.randint(1, 3)):
                        if perturbed:
                            perturbed.discard(random.choice(list(perturbed)))
                    remaining = [x for x in S if x not in perturbed]
                    random.shuffle(remaining)
                    for y in remaining:
                        if can_add(y, perturbed):
                            perturbed.add(y)
                    if len(perturbed) > len(best):
                        best = perturbed
                        found = True
                        break
                if found:
                    continue
                else:
                    # Try pair removals if still stuck and time permits
                    elements = list(best)
                    if len(elements) > 10:
                        for i in range(len(elements)):
                            if time.time() - start_time > time_limit:
                                break
                            for j in range(i+1, len(elements)):
                                trial = set(best)
                                trial.discard(elements[i])
                                trial.discard(elements[j])
                                for y in S:
                                    if y in trial:
                                        continue
                                    if can_add(y, trial):
                                        trial.add(y)
                                if len(trial) > len(best):
                                    best = trial
                                    improved = True
                                    break
                            if improved:
                                break
                    if not improved:
                        break
        
        return list(best)
=======
    def solve_heuristic(S, time_limit):
        n = len(S)
        S_set = set(S)
        
        def greedy_sorted(order):
            result = set()
            for x in order:
                if can_add(x, result):
                    result.add(x)
            return result

        def greedy_by_conflict(order):
            # Sort by number of conflicts with current set
            result = set()
            remaining = list(order)
            while remaining:
                # Pick element with fewest conflicts
                best_x = None
                best_conflicts = float('inf')
                for x in remaining:
                    conflicts = 0
                    for a in result:
                        if (x - a) in result or (a + x) in result or (2*x) in result:
                            conflicts += 1
                    if conflicts < best_conflicts:
                        best_conflicts = conflicts
                        best_x = x
                    if conflicts == 0:
                        break
                if best_x is not None:
                    result.add(best_x)
                    remaining.remove(best_x)
            return result

        # Multiple diverse initializations
        inits = []
        inits.append(greedy_sorted(S))
        inits.append(greedy_sorted(list(reversed(S))))
        
        # Odds (sum-free in many cases)
        odds = [x for x in S if x % 2 != 0]
        if odds:
            inits.append(set(odds))
        
        # Evens divided by 2 recursively might help, but let's try other structures
        
        # Suffix approach: largest elements
        sorted_S = sorted(S)
        for k in range(1, min(n+1, 30)):
            suffix = sorted_S[n-k:]
            if 2 * suffix[0] > suffix[-1]:
                inits.append(set(suffix))
        
        # Conflict-based greedy
        inits.append(greedy_by_conflict(S))
        inits.append(greedy_by_conflict(list(reversed(S))))
        
        # Random restarts
        for _ in range(20):
            shuffled = list(S)
            random.shuffle(shuffled)
            inits.append(greedy_sorted(shuffled))
        
        best = max(inits, key=len)
        best_size = len(best)
        
        start_time = time.time()
        no_improve_count = 0
        
        while time.time() - start_time < time_limit:
            improved = False
            best_copy = set(best)
            elements = list(best_copy)
            random.shuffle(elements)
            
            # Try removing each element and greedily adding
            for x in elements:
                if time.time() - start_time > time_limit:
                    break
                trial = set(best_copy)
                trial.discard(x)
                
                # Try to add elements from S \ trial, sorted by potential
                candidates = [y for y in S if y not in trial]
                # Sort by how many conflicts they'd have
                def conflict_count(y):
                    count = 0
                    for a in trial:
                        if (y - a) in trial or (a + y) in trial or (2*y) in trial:
                            count += 1
                    return count
                candidates.sort(key=conflict_count)
                
                for y in candidates:
                    if can_add(y, trial):
                        trial.add(y)
                
                if len(trial) > best_size:
                    best = trial
                    best_size = len(best)
                    improved = True
                    no_improve_count = 0
                    break
            
            if not improved:
                no_improve_count += 1
                
                # Try pair removals if stuck
                if no_improve_count >= 2 and len(elements) > 8:
                    found = False
                    elements_list = list(best)
                    random.shuffle(elements_list)
                    # Try random pairs
                    for _ in range(min(50, len(elements_list) * (len(elements_list)-1) // 2)):
                        if time.time() - start_time > time_limit:
                            break
                        i = random.randint(0, len(elements_list)-1)
                        j = random.randint(0, len(elements_list)-1)
                        if i == j:
                            continue
                        trial = set(best)
                        trial.discard(elements_list[i])
                        trial.discard(elements_list[j])
                        
                        candidates = [y for y in S if y not in trial]
                        candidates.sort(key=lambda y: sum(1 for a in trial if (y-a) in trial or (a+y) in trial or (2*y) in trial))
                        for y in candidates:
                            if can_add(y, trial):
                                trial.add(y)
                        
                        if len(trial) > best_size:
                            best = trial
                            best_size = len(best)
                            found = True
                            no_improve_count = 0
                            break
                    if found:
                        continue
                
                # Try random perturbation restarts
                found = False
                for _ in range(30):
                    if time.time() - start_time > time_limit:
                        break
                    perturbed = set(best)
                    # Remove 1-3 random elements
                    num_remove = random.randint(1, 3)
                    for _ in range(num_remove):
                        if perturbed:
                            perturbed.discard(random.choice(list(perturbed)))
                    remaining = [x for x in S if x not in perturbed]
                    random.shuffle(remaining)
                    for y in remaining:
                        if can_add(y, perturbed):
                            perturbed.add(y)
                    if len(perturbed) > best_size:
                        best = perturbed
                        best_size = len(best)
                        found = True
                        no_improve_count = 0
                        break
                
                if not found and no_improve_count >= 5:
                    # Try triple removals
                    if len(elements) > 12:
                        found = False
                        for _ in range(30):
                            if time.time() - start_time > time_limit:
                                break
                            trial = set(best)
                            elems = list(trial)
                            random.shuffle(elems)
                            to_remove = elems[:3]
                            for x in to_remove:
                                trial.discard(x)
                            candidates = [y for y in S if y not in trial]
                            candidates.sort(key=lambda y: sum(1 for a in trial if (y-a) in trial or (a+y) in trial or (2*y) in trial))
                            for y in candidates:
                                if can_add(y, trial):
                                    trial.add(y)
                            if len(trial) > best_size:
                                best = trial
                                best_size = len(best)
                                found = True
                                no_improve_count = 0
                                break
                        if not found:
                            break
                    else:
                        break
                elif not found:
                    continue
        
        return list(best)
>>>>>>> REPLACE

