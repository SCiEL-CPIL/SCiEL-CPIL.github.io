# Post-processing: try to remove edges
changed = True
while changed and time.time() < deadline:
    changed = False
    for i in range(len(selected_edges)):
        if time.time() > deadline:
            break
        # Try removing selected_edges[i]
        temp_edges = selected_edges[:i] + selected_edges[i+1:]
        # Check if temp_edges is still 3-chromatic
        # i.e., does it cover all colorings?
        mask = 0
        for e in temp_edges:
            idx = all_edges.index(e) # This is slow
            # Better to store indices
            pass
