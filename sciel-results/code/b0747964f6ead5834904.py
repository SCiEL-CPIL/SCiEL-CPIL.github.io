<<<<<<< SEARCH
    # Step 5: Final polish - 2-opt style improvement
    # Try reversing segments to improve overlap
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(num_perms - 1):
            for j in range(i + 2, num_perms):
                # Try reversing segment [i, j]
                old_score = 0
                new_score = 0
                
                # Edges affected: (i-1,i), (i,j+1 or end), (j,j+1 or end)
                if i > 0:
                    old_score += M[best_order[i-1]][best_order[i]]
                    new_score += M[best_order[i-1]][best_order[j]]
                if j + 1 < num_perms:
                    old_score += M[best_order[j]][best_order[j+1]]
                    new_score += M[best_order[i]][best_order[j+1]]
                # Internal edges within [i,j] reverse direction
                for k in range(i, j):
                    old_score += M[best_order[k]][best_order[k+1]]
                    new_score += M[best_order[k+1]][best_order[k]]
                
                delta = new_score - old_score
                if delta > 0:
                    # Apply reversal
                    best_order[i:j+1] = best_order[i:j+1][::-1]
                    best_score += delta
                    improved = True
=======
    # Step 5: Final polish - Fast random 2-opt style improvement
    # We use the M matrix to evaluate deltas.
    # This is much faster than the exhaustive search which caused timeout.
    random.seed(123)
    attempts = 0
    while time.time() < deadline - 2 and attempts < 10000:
        attempts += 1
        i = random.randint(0, num_perms - 2)
        j = random.randint(i + 1, num_perms - 1)
        
        # Calculate delta
        delta = 0
        
        # Boundary i-1 -> i becomes i-1 -> j
        if i > 0:
            delta += M[best_order[i-1]][best_order[j]] - M[best_order[i-1]][best_order[i]]
            
        # Boundary j -> j+1 becomes i -> j+1
        if j + 1 < num_perms:
            delta += M[best_order[i]][best_order[j+1]] - M[best_order[j]][best_order[j+1]]
            
        # Internal edges: k -> k+1 becomes k+1 -> k
        # Sum the difference in overlap for the reversed segment
        for k in range(i, j):
            delta += M[best_order[k+1]][best_order[k]] - M[best_order[k]][best_order[k+1]]
            
        if delta > 0:
            best_order[i:j+1] = best_order[i:j+1][::-1]
            best_score += delta
>>>>>>> REPLACE
