<<<<<<< SEARCH
                if M_new.shape[0] == 0:
                    if num_rank == 0:
                        # Remove the term
                        current_U.pop(rem_idx)
                        current_V.pop(rem_idx)
                        current_W.pop(rem_idx)
                        current_rank -= 1
                        M = np.array(vectors[:rem_idx] + vectors[rem_idx+1:])
                        removed_any = True
                        break
                    else:
                        continue
=======
                if M_new.shape[0] == 0:
                    if num_rank == 0:
                        # Remove the term
                        current_U = np.delete(current_U, rem_idx, axis=1)
                        current_V = np.delete(current_V, rem_idx, axis=1)
                        current_W = np.delete(current_W, rem_idx, axis=1)
                        current_rank -= 1
                        M = M_new
                        removed_any = True
                        break
                    else:
                        continue
>>>>>>> REPLACE

