        new_size = len(active)
        if new_size < best_size:
            best_size = new_size
            best_active = set(active)
        elif new_size == best_size and random.random() < 0.05:
            # Accept to explore
            best_size = new_size
            best_active = set(active)
        else:
            # Revert
            active = old_active
            counts = old_counts
