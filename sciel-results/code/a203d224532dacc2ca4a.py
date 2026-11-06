            best_a_in = None
            best_gain = 0
            
            for a_in in unselected_set:
                if scores[a_in] <= score_out:
                    continue
                if is_coprime_to_set(a_in, rest_selected):
                    gain = scores[a_in] - score_out
                    if gain > best_gain:
                        best_gain = gain
                        best_a_in = a_in
                        
            if best_a_in is not None:
                # ... swap logic
