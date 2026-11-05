    def prune(selected_indices):
        if not selected_indices:
            return selected_indices
            
        selected_list = list(selected_indices)
        coverage_count = np.zeros(total_points, dtype=np.int32)
        
        for idx in selected_list:
            coverage_count[balls[idx]] += 1
            
        changed = True
        while changed:
            changed = False
            for i in range(len(selected_list)):
                idx = selected_list[i]
                ball = balls[idx]
                # If all points in this ball have count > 1, it's redundant
                if np.all(coverage_count[ball] > 1):
                    # Remove
                    rem_idx = selected_list.pop(i)
                    coverage_count[balls[rem_idx]] -= 1
                    changed = True
                    break # Restart scan after change
        return selected_list
