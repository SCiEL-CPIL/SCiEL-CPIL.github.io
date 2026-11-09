                    if score < best_score:
                        best_score = score
                        best_pos = (cx, cy, cz, rw, rh, rd)
                        if best_score == 0:
                            return best_pos
