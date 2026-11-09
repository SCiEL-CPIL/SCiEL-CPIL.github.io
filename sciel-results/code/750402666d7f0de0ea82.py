<<<<<<< SEARCH
    # Local search with add, delete, and swap moves
    def local_search(present, config_count):
        improved = True
        stall_count = 0
        
        while improved and time.time() < deadline - 10:
            improved = False
            
            # Try to add a triple
            added = False
            for _ in range(50):  # Try up to 50 random candidates
                t = random.randint(0, num_triples - 1)
                if present[t]:
                    continue
                # Check if we can add t
                can_add = True
                for ci in triple_to_configs[t]:
                    if config_count[ci] >= 7:
                        can_add = False
                        break
                if can_add:
                    present[t] = True
                    for ci in triple_to_configs[t]:
                        config_count[ci] += 1
                    improved = True
                    added = True
                    break
            
            if added:
                continue
            
            # Try a swap: remove one triple, add another
            swapped = False
            for _ in range(50):  # Try up to 50 random swap candidates
                t_remove = random.randint(0, num_triples - 1)
                if not present[t_remove]:
                    continue
                
                # Find which configs are satisfied by removing t_remove
                # After removing t_remove, some configs go from 7 to 6, becoming "slack"
                # We need to find a triple t_add that fits in all its configs after removal
                
                # First, check which configs become available
                slack_configs = set()
                for ci in triple_to_configs[t_remove]:
                    if config_count[ci] == 7:
                        slack_configs.add(ci)
                
                if not slack_configs:
                    continue
                
                t_add = random.randint(0, num_triples - 1)
                if present[t_add]:
                    continue
                
                # Check if t_add can be added after removing t_remove
                can_add = True
                for ci in triple_to_configs[t_add]:
                    if ci in slack_configs:
                        # After removing t_remove, this config has 6, so adding t_add makes 7 - OK
                        continue
                    if config_count[ci] >= 7:
                        can_add = False
                        break
                
                if can_add:
                    # Perform the swap
                    present[t_remove] = False
                    for ci in triple_to_configs[t_remove]:
                        config_count[ci] -= 1
                    present[t_add] = True
                    for ci in triple_to_configs[t_add]:
                        config_count[ci] += 1
                    improved = True
                    swapped = True
                    break
            
            if swapped:
                stall_count = 0
                continue
            
            # No improvement found
            stall_count += 1
            if stall_count > 20:
                break
        
        return present, config_count
=======
    # Local search with add, 1-swap, and 2-swap moves
    def local_search(present, config_count):
        improved = True
        stall_count = 0
        
        while improved and time.time() < deadline - 10:
            improved = False
            
            # Try to add a triple
            added = False
            for _ in range(50):  # Try up to 50 random candidates
                t = random.randint(0, num_triples - 1)
                if present[t]:
                    continue
                # Check if we can add t
                can_add = True
                for ci in triple_to_configs[t]:
                    if config_count[ci] >= 7:
                        can_add = False
                        break
                if can_add:
                    present[t] = True
                    for ci in triple_to_configs[t]:
                        config_count[ci] += 1
                    improved = True
                    added = True
                    break
            
            if added:
                continue
            
            # Try a swap: remove one triple, add another
            swapped = False
            for _ in range(50):  # Try up to 50 random swap candidates
                t_remove = random.randint(0, num_triples - 1)
                if not present[t_remove]:
                    continue
                
                # Find which configs become available after removing t_remove
                slack_configs = set()
                for ci in triple_to_configs[t_remove]:
                    if config_count[ci] == 7:
                        slack_configs.add(ci)
                
                if not slack_configs:
                    continue
                
                t_add = random.randint(0, num_triples - 1)
                if present[t_add]:
                    continue
                
                # Check if t_add can be added after removing t_remove
                can_add = True
                for ci in triple_to_configs[t_add]:
                    if ci in slack_configs:
                        continue
                    if config_count[ci] >= 7:
                        can_add = False
                        break
                
                if can_add:
                    # Perform the swap
                    present[t_remove] = False
                    for ci in triple_to_configs[t_remove]:
                        config_count[ci] -= 1
                    present[t_add] = True
                    for ci in triple_to_configs[t_add]:
                        config_count[ci] += 1
                    improved = True
                    swapped = True
                    break
            
            if swapped:
                stall_count = 0
                continue
            
            # Try a 2-swap: remove two triples, add two different ones
            two_swapped = False
            for _ in range(30):  # Try up to 30 random 2-swap candidates
                t_remove1 = random.randint(0, num_triples - 1)
                if not present[t_remove1]:
                    continue
                t_remove2 = random.randint(0, num_triples - 1)
                if not present[t_remove2] or t_remove2 == t_remove1:
                    continue
                
                # Compute slack configs after removing both
                slack_configs = set()
                for ci in triple_to_configs[t_remove1]:
                    if config_count[ci] == 7:
                        slack_configs.add(ci)
                    elif config_count[ci] == 6 and ci in triple_to_configs[t_remove2]:
                        # Will drop to 5 after removing t_remove2, so it's slack
                        slack_configs.add(ci)
                for ci in triple_to_configs[t_remove2]:
                    if config_count[ci] == 7:
                        slack_configs.add(ci)
                    elif config_count[ci] == 6 and ci in triple_to_configs[t_remove1]:
                        slack_configs.add(ci)
                
                if not slack_configs:
                    continue
                
                # Try to find two triples to add
                t_add1 = random.randint(0, num_triples - 1)
                if present[t_add1] or t_add1 == t_remove1 or t_add1 == t_remove2:
                    continue
                
                # Check if t_add1 can be added after removing both
                can_add1 = True
                for ci in triple_to_configs[t_add1]:
                    if ci in slack_configs:
                        continue
                    if config_count[ci] >= 7:
                        can_add1 = False
                        break
                
                if not can_add1:
                    continue
                
                t_add2 = random.randint(0, num_triples - 1)
                if present[t_add2] or t_add2 == t_remove1 or t_add2 == t_remove2 or t_add2 == t_add1:
                    continue
                
                # Check if t_add2 can be added after removing both (accounting for t_add1)
                can_add2 = True
                for ci in triple_to_configs[t_add2]:
                    if ci in slack_configs:
                        # Check if t_add1 also uses this config
                        if ci in triple_to_configs[t_add1]:
                            # Both add to same config; need count to be <= 5 before adds
                            if config_count[ci] >= 6:
                                can_add2 = False
                                break
                        continue
                    if config_count[ci] >= 7:
                        can_add2 = False
                        break
                
                if can_add2:
                    # Perform the 2-swap
                    present[t_remove1] = False
                    for ci in triple_to_configs[t_remove1]:
                        config_count[ci] -= 1
                    present[t_remove2] = False
                    for ci in triple_to_configs[t_remove2]:
                        config_count[ci] -= 1
                    present[t_add1] = True
                    for ci in triple_to_configs[t_add1]:
                        config_count[ci] += 1
                    present[t_add2] = True
                    for ci in triple_to_configs[t_add2]:
                        config_count[ci] += 1
                    improved = True
                    two_swapped = True
                    break
            
            if two_swapped:
                stall_count = 0
                continue
            
            # No improvement found
            stall_count += 1
            if stall_count > 20:
                break
        
        return present, config_count
>>>>>>> REPLACE

