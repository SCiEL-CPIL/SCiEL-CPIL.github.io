                old_violations = []
                for aid in ap_index_map[idx]:
                    old_violations.append(violations[aid])
                    violations[aid] = is_violation(aps[aid], colors)
                
                # Calculate delta
                delta = 0
                for i_aid in range(len(ap_index_map[idx])):
                    aid = ap_index_map[idx][i_aid]
                    if old_violations[i_aid] and not violations[aid]:
                        delta -= 1
                    elif not old_violations[i_aid] and violations[aid]:
                        delta += 1
                        
                new_count = cur_violation_count + delta
                
                if new_count < cur_violation_count:
                    cur_violation_count = new_count
                    stuck_counter = 0
                else:
                    # Revert
                    colors[idx] = old_color
                    for i_aid in range(len(ap_index_map[idx])):
                        aid = ap_index_map[idx][i_aid]
                        violations[aid] = old_violations[i_aid]
                    stuck_counter += 1
                    if stuck_counter > 1000:
                        break
