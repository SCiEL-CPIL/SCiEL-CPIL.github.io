    for n in target_ns:
        if time_left() < 10:
            break
        # ... search for n ...
        if found:
            best_n = n
            best_colors = colors[:]
            break # Stop searching lower n
