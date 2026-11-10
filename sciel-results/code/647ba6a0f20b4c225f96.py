<<<<<<< SEARCH
                if delta < 0:
                    # Apply move
                    del occupied[curr_pos]
                    chain[i] = new_pos
                    occupied[new_pos] = i
                    energy += delta
                    improved = True
=======
                if delta < 0:
                    # Apply move
                    if curr_pos in occupied:
                        del occupied[curr_pos]
                    chain[i] = new_pos
                    occupied[new_pos] = i
                    energy += delta
                    improved = True
                    break
>>>>>>> REPLACE
