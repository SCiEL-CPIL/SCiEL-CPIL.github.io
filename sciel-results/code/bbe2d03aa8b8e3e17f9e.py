        # Local Search: Try swaps
        if len(S) > 0:
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                # Try to add a random point
                # We maintain a list of "rejected" points? No, just sample.
                for _ in range(50): # Try 50 random additions
                    p = random.randint(0, M-1)
                    if p in S:
                        continue
                    if can_add(p, S, dist_counter):
                        add_point(p, S, dist_counter)
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = S.copy()
                        improved = True
                        break
                
                # If no addition found, try a swap
                if not improved:
                    # Pick a point to remove
                    s_remove = random.choice(S)
                    remove_point(s_remove, S, dist_counter)
                    
                    # Try to add a random point
                    added = False
                    for _ in range(50):
                        p_add = random.randint(0, M-1)
                        if p_add in S or p_add == s_remove:
                            continue
                        if can_add(p_add, S, dist_counter):
                            add_point(p_add, S, dist_counter)
                            added = True
                            # Size is same, so no improvement in size, but might enable future additions
                            break
                    
                    if not added:
                        # Revert swap
                        add_point(s_remove, S, dist_counter)
                    else:
                        # We have a new configuration of same size.
                        # We can continue the while loop to try to add more.
                        improved = True # Force another iteration of the while loop
