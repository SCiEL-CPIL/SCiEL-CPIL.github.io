        new_score = score_cell(cell_arr)
        
        if new_score > best_score:
            best_score = new_score
            best_cell = cell_arr.copy()
        else:
            # Revert
            for (i, j) in orbit:
                cell_arr[i, j] = 1 - cell_arr[i, j]
