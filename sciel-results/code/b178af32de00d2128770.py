    # Robust repair: while any config is violated, break it
    # We track violations by checking config_counts == 8
    # To avoid O(K) scan every time, we can maintain a list of violated configs, 
    # but given K is small (approx 16*15*14/6 * 15? No, K is number of K(2,2,2)s), 
    # a simple while loop with a set of violated configs is fine.
    
    violated_configs = set(ci for ci in range(K) if config_counts[ci] == 8)
    
    while violated_configs:
        if time.time() > deadline:
            break # If we run out of time, we might return invalid? 
                  # The task says "invalid result ... scores 0". 
                  # So we MUST return valid. 
                  # If we run out of time, we should just stop removing? 
                  # No, if we stop removing, we return invalid. 
                  # We should have a tight deadline that ensures we finish repair.
                  # Repair is fast (removing 1 edge per violation).
                  # Let's assume repair finishes quickly.
        
        # Pick a violated config
        ci = next(iter(violated_configs))
        # Remove one edge from it
        # Choose an edge that is in the most other violated configs? 
        # Or just the first one? 
        # To be safe and fast, pick the edge in this config that has the highest 
        # "impact" (participates in most violated configs).
        
        best_e = -1
        best_impact = -1
        for e in config_to_edges[ci]:
            if not present[e]:
                continue
            impact = 0
            for ci2 in edge_to_configs[e]:
                if ci2 in violated_configs:
                    impact += 1
            if impact > best_impact:
                best_impact = impact
                best_e = e
        
        if best_e == -1:
            # Should not happen if config_counts[ci] == 8 implies all edges present
            break
            
        present[best_e] = False
        for ci2 in edge_to_configs[best_e]:
            config_counts[ci2] -= 1
            if config_counts[ci2] < 8 and ci2 in violated_configs:
                violated_configs.discard(ci2)
            # Note: Removing an edge cannot *create* a new violation (count goes down).
            # So we don't need to add to violated_configs here.
