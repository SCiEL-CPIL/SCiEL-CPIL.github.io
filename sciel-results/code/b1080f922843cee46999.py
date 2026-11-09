            # Revert
            colors[e_idx] = old_color
            color_counts[old_color] += 1
            color_counts[new_color] -= 1
            if color_counts[new_color] == 0:
                del color_counts[new_color]
            # Recompute distinct to be safe
            distinct_colors = sum(1 for c in color_counts.values() if c > 0)
            
            for c_idx in edge_to_cycles[e_idx]:
                cycle_violations[c_idx] = old_statuses[c_idx]
            total_violations = old_total_viol
