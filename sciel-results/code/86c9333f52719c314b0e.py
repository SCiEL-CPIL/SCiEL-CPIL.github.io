                for t_idx in block_coverage[b2]:
                    c = coverage_count[t_idx]
                    if c == 1:
                        essential_points.update(all_t_subsets[t_idx])
