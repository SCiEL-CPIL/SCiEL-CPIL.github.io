"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 12
    selected = []
    current_sums = {0}
    
    for _ in range(n):
        candidate = 1
        while True:
            if candidate in selected:
                candidate += 1
                continue
            
            # Efficiently check if adding candidate creates distinct sums
            # new_sums = current_sums U {s + candidate for s in current_sums}
            # We need len(new_sums) == 2 * len(current_sums)
            
            # Optimization: 
            # If candidate is small, many collisions are likely.
            # We can break early if we detect a collision.
            
            # Generate new sums
            new_sums = set()
            valid = True
            for s in current_sums:
                new_val = s + candidate
                if new_val in current_sums or new_val in new_sums:
                    valid = False
                    break
                new_sums.add(new_val)
            
            if valid:
                # All new sums are distinct from each other and from old sums
                selected.append(candidate)
                current_sums.update(new_sums)
                break
            else:
                candidate += 1
                
    return selected
# EVOLVE-BLOCK-END
