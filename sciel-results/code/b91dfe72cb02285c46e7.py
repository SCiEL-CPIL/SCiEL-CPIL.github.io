        found_swap = False
        # Randomize the order of a
        selected_copy = selected[:]
        random.shuffle(selected_copy)
        
        for a in selected_copy:
            if time.time() > deadline:
                return selected
            
            sums_involving_a = get_sums_involving(a, selected)
            
            # Sample x
            # We want x not in selected and x != a
            # We can just pick random x in 1..N and check
            # Or generate candidates.
            
            # Let's try 50 random x's
            for _ in range(50):
                if time.time() > deadline:
                    return selected
                x = random.randint(1, n)
                if x == a or x in selected:
                    continue
                
                if is_valid_swap(x, a, selected, sums_used, sums_involving_a):
                    remove_element(a, selected, sums_used)
                    commit_addition(x, selected, sums_used)
                    found_swap = True
                    break
            
            if found_swap:
                break
