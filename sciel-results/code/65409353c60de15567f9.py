                if score < best_score or (score == best_score and 
                    sum(b[3]*b[4]*b[5] for b in bins[best_bin_idx] if best_bin_idx != -1) < 
                    sum(b[3]*b[4]*b[5] for b in bins[b_idx])):
