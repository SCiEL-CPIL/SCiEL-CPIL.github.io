            # Calculate new covered mask
            new_covered_mask = covered_mask
            for d in new_diffs_set:
                if 1 <= d <= 30:
                    new_covered_mask |= (1 << (d - 1))
                    
            # Pruning: If no new target diff is covered, skip
            if (new_covered_mask & target_mask) == (covered_mask & target_mask):
                continue
