<<<<<<< SEARCH
    # Phase 1: Aggressive greedy removal
    # Try removing comparators, checking from both ends
    changed = True
    while changed:
        if time.time() > deadline - 10:
            break
        changed = False
        
        # Try removing from end to start (most redundant are often at the end)
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline - 10:
                return best_net
            
            # Create new network without idx
            new_net = net[:idx] + net[idx+1:]
            
            # Verify
            if _is_sorted_network(new_net):
                net = new_net
                best_net = net[:]
                best_count = len(net)
                changed = True
                break  # Restart the loop with the new network
        
        # If no change from end-to-start, try start-to-end
        if not changed:
            for idx in range(len(net)):
                if time.time() > deadline - 10:
                    return best_net
                
                new_net = net[:idx] + net[idx+1:]
                
                if _is_sorted_network(new_net):
                    net = new_net
                    best_net = net[:]
                    best_count = len(net)
                    changed = True
                    break
    
    # Phase 2: Random single removal with more attempts
    if time.time() < deadline - 5:
        attempts = 0
        max_attempts = 500
        while time.time() < deadline - 3 and attempts < max_attempts:
            if len(net) < 1:
                break
            idx = random.randrange(len(net))
            new_net = net[:idx] + net[idx+1:]
            
            if _is_sorted_network(new_net):
                net = new_net
                best_net = net[:]
                best_count = len(net)
                attempts = 0  # Reset attempts on success
            else:
                attempts += 1
    
    # Phase 3: Random pair removal
    if time.time() < deadline - 3:
        attempts = 0
        max_attempts = 300
        while time.time() < deadline - 2 and attempts < max_attempts:
            if len(net) < 2:
                break
            idx1, idx2 = random.sample(range(len(net)), 2)
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1
            
            new_net = net[:idx1] + net[idx1+1:idx2] + net[idx2+1:]
            
            if _is_sorted_network(new_net):
                net = new_net
                best_net = net[:]
                best_count = len(net)
                attempts = 0  # Reset attempts on success
            else:
                attempts += 1
    
    return best_net
=======
    # Phase 1: Aggressive greedy removal with fast filtering
    changed = True
    while changed:
        if time.time() > deadline - 10:
            break
        changed = False
        
        # Try removing from end to start (most redundant are often at the end)
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline - 10:
                return best_net
            
            # Create new network without idx
            new_net = net[:idx] + net[idx+1:]
            
            # Fast filter first
            if _is_sorted_network(new_net, full_check=False):
                # Full check
                if _is_sorted_network(new_net, full_check=True):
                    net = new_net
                    best_net = net[:]
                    best_count = len(net)
                    changed = True
                    break  # Restart the loop with the new network
        
        # If no change from end-to-start, try start-to-end
        if not changed:
            for idx in range(len(net)):
                if time.time() > deadline - 10:
                    return best_net
                
                new_net = net[:idx] + net[idx+1:]
                
                # Fast filter first
                if _is_sorted_network(new_net, full_check=False):
                    # Full check
                    if _is_sorted_network(new_net, full_check=True):
                        net = new_net
                        best_net = net[:]
                        best_count = len(net)
                        changed = True
                        break
    
    # Phase 2: Random single removal with more attempts
    if time.time() < deadline - 5:
        attempts = 0
        max_attempts = 1000
        while time.time() < deadline - 3 and attempts < max_attempts:
            if len(net) < 1:
                break
            idx = random.randrange(len(net))
            new_net = net[:idx] + net[idx+1:]
            
            # Fast filter first
            if _is_sorted_network(new_net, full_check=False):
                # Full check
                if _is_sorted_network(new_net, full_check=True):
                    net = new_net
                    best_net = net[:]
                    best_count = len(net)
                    attempts = 0  # Reset attempts on success
            else:
                attempts += 1
    
    # Phase 3: Random pair removal
    if time.time() < deadline - 3:
        attempts = 0
        max_attempts = 500
        while time.time() < deadline - 2 and attempts < max_attempts:
            if len(net) < 2:
                break
            idx1, idx2 = random.sample(range(len(net)), 2)
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1
            
            new_net = net[:idx1] + net[idx1+1:idx2] + net[idx2+1:]
            
            # Fast filter first
            if _is_sorted_network(new_net, full_check=False):
                # Full check
                if _is_sorted_network(new_net, full_check=True):
                    net = new_net
                    best_net = net[:]
                    best_count = len(net)
                    attempts = 0  # Reset attempts on success
            else:
                attempts += 1
    
    return best_net
>>>>>>> REPLACE
