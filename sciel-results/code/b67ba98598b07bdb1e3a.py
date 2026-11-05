    uncovered = list(range(num_triples))
    # ...
    while uncovered:
        target_idx = random.choice(uncovered)
        # ... find best_i
        # ... update
        # Remove covered from uncovered
        # This is tricky because we cover multiple triples.
        # We can just use a set for uncovered for simplicity, 
        # but random.choice on a set is O(1) if we convert to list? No, list(set) is O(N).
        # For 2024 elements, list(set) is very fast.
