                current_set = set(map(tuple, pts))
                # ...
                for new_pt in neighbors:
                    if tuple(new_pt) in current_set:
                        continue
