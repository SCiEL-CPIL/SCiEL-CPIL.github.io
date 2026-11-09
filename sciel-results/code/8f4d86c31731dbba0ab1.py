            if not added:
                # Re-add w
                add_word(w)
                continue

            # We added v. Try to add v2.
            # Re-check valid neighbors
            # Note: valid_indices was calculated BEFORE adding v.
            # We need to check current counts.
            
            for idx in valid_indices:
                v2 = neighbors[idx]
                if v2 == w or v2 == v:
                    continue
                if counts[v2] == 0:
                    add_word(v2)
                    # Size increased by 1 (net +1: -1 w, +1 v, +1 v2 => +1)
                    if len(code) > best_size:
                        best_size = len(code)
                        best_code = list(code)
                    break
            else:
                # No v2 found. Size is same (-1 w, +1 v).
                # Update best code to current state to explore new region
                best_code = list(code)
