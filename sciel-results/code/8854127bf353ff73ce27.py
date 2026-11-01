<<<<<<< SEARCH
            # Pick a random pair of columns to try to merge
            idx = rng.choice(current_rank, size=2, replace=False)
            
            # Remove one column and optimize the remaining
            keep = [i for i in range(current_rank) if i not in idx[1]]
=======
            # Pick a random pair of columns to try to merge
            idx = rng.choice(current_rank, size=2, replace=False)
            
            # Remove one column and optimize the remaining
            # Fix: idx[1] is a scalar, cannot use 'in'. Use '!='
            col_to_remove = int(idx[1])
            keep = [i for i in range(int(current_rank)) if i != col_to_remove]
>>>>>>> REPLACE
