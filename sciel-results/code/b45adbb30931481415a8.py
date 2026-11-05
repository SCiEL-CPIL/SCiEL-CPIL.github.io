    def local_search(initial_set, time_limit):
        current_set = set(initial_set)
        current_score = get_score(current_set)
        best_set = set(current_set)
        best_score = current_score
        start = time.time()
        no_improvement_count = 0
        
        while time.time() - start < time_limit:
            # Pick a random element to replace
            elem = random.choice(list(current_set))
            new_elem = random.randint(1, 10000)
            
            if new_elem in current_set:
                continue
                
            new_set = set(current_set)
            new_set.remove(elem)
            new_set.add(new_elem)
            
            new_score = get_score(new_set)
            
            if new_score < current_score:
                current_set = new_set
                current_score = new_score
                no_improvement_count = 0
                if current_score < best_score:
                    best_score = current_score
                    best_set = set(current_set)
            else:
                no_improvement_count += 1
                
                if no_improvement_count > 5000:
                    # Perturb current set: change 5 random elements
                    new_set = set(current_set)
                    elems_to_change = random.sample(list(new_set), 5)
                    for e in elems_to_change:
                        new_set.remove(e)
                    # Add 5 new random elements
                    available = [x for x in range(1, 10001) if x not in new_set]
                    new_set.update(random.sample(available, 5))
                    new_score = get_score(new_set)
                    if new_score < current_score:
                        current_set = new_set
                        current_score = new_score
                        no_improvement_count = 0
                        if current_score < best_score:
                            best_score = current_score
                            best_set = set(current_set)
                    else:
                        # Reset counter to try again
                        no_improvement_count = 0

        return best_set, best_score
