                if len(candidates) > 1:
                    # Add greedily
                    added_count = 0
                    for v in candidates:
                        if not in_set[v] and blocked_count[v] == 0:
                            add_vertex(v)
                            added_count += 1
                    if added_count > 1:
                        improved = True
                        swap_done = True
                        break
                    else:
                        # Restore u
                        add_vertex(u)
                else:
                    # Restore u
                    add_vertex(u)
