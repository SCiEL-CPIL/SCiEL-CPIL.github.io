def greedy_remove(net, deadline):
    """
    Try to remove each comparator one by one. If removal keeps the network valid,
    remove it and continue. Returns the reduced network.
    """
    changed = True
    while changed and time.time() < deadline:
        changed = False
        remove_indices = []
        # We iterate backwards to handle index shifts if we were removing on the fly,
        # but since we collect indices, we can iterate forwards.
        # However, removing multiple at once might make a previously non-removable one removable?
        # No, removing comparators makes the network "less sorted". 
        # Wait. If network N is valid, and we remove C to get N', N' might be invalid.
        # If N' is valid, then removing another C2 from N' to get N'' might be valid.
        # Can we remove C and C2 simultaneously from N?
        # If N - {C, C2} is valid, then both are "removable" in the context of the full set.
        # So yes, we can find all C such that N - {C} is valid, remove them all, and repeat.
        
        for idx in range(len(net)):
            if time.time() > deadline:
                break
            candidate = net[:idx] + net[idx+1:]
            if fast_verify(candidate):
                remove_indices.append(idx)
                changed = True
        
        if remove_indices:
            # Remove from end to start to preserve indices
            for idx in reversed(remove_indices):
                net.pop(idx)
    return net
