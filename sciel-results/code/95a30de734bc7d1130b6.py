<<<<<<< SEARCH
        # Swap-based local search: try replacing one codeword with another to reduce size
        # or improve coverage
        swap_improved = True
        while swap_improved:
            swap_improved = False
            if time.time() > deadline - 3:
                break
            # Try removing each codeword and see if we can add a better one
            for i in range(len(selected)):
                if time.time() > deadline - 3:
                    break
                c = selected[i]
                # What does c uniquely cover?
                others_mask = 0
                for j in range(len(selected)):
                    if j != i:
                        others_mask |= ball_masks[selected[j]]
                unique_to_c = ball_masks[c] & ~others_mask
                if unique_to_c == 0:
                    # c is redundant, remove it
                    selected.pop(i)
                    swap_improved = True
                    break
                # Try to find a codeword d not in selected that covers unique_to_c
                # and whose addition allows removal of c
                # This is complex; for now just do removal
                # Try adding a random codeword and removing two
                # For simplicity, skip swap for now
                pass
=======
        # Swap-based local search: try 1-for-1 exchanges to escape local optima
        # A swap is beneficial if we can remove codeword i and add codeword d (not in selected)
        # such that the union of balls still covers everything, AND the new codeword d
        # covers some words that were only covered by i, allowing potential future removals.
        # For size reduction, we need d's ball to be a subset of the union of all other balls
        # plus i's ball, but that's not directly helpful.
        
        # Better approach: try to find if any codeword can be replaced by another that
        # covers the same unique region but allows the overall set to be smaller.
        # This is hard. Instead, do a "move" local search:
        # For each codeword i, try replacing it with a random candidate d.
        # Accept if the new set still covers everything and has same size (stochastic).
        # Or accept if it allows removing another codeword (size reduction).
        
        # Simplest effective swap: if removing i leaves some words uncovered,
        # try to find a single d that covers ALL those uncovered words.
        # If such d exists, replace i with d (size stays same, but may enable future removals).
        # More aggressively: try to find d that covers uncovered words AND whose ball
        # is mostly redundant, enabling a subsequent removal.
        
        # For now, implement: try 1-for-1 swaps that maintain coverage
        # and randomly accept to explore. This is a simulated annealing lite.
        
        # Actually, let's do something more targeted:
        # After removal pass, try to find pairs (i, j) where removing both and
        # adding one codeword d that covers both i's and j's unique regions.
        # This is a 2-for-1 exchange.
        
        # Given time constraints, let's just do more removal passes with different orderings
        # and rely on more restarts.
        
        # Additional removal pass with shuffled order
        for _ in range(3):
            if time.time() > deadline - 3:
                break
            random.shuffle(selected)
            changed = True
            while changed:
                changed = False
                for i in range(len(selected) - 1, -1, -1):
                    if time.time() > deadline - 3:
                        break
                    c = selected[i]
                    others_mask = 0
                    for j in range(len(selected)):
                        if j != i:
                            others_mask |= ball_masks[selected[j]]
                    unique = ball_masks[c] & ~others_mask
                    if unique == 0:
                        selected.pop(i)
                        changed = True
                        break
>>>>>>> REPLACE

