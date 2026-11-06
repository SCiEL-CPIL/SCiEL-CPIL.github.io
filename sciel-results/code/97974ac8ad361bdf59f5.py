            for i, p in enumerate(primes):
                new_r = x_fail % p
                if new_r == residues[i]:
                    continue
                    
                if is_valid_move(residues, counts, i, new_r, x_fail):
                    # Tentatively apply
                    old_r = residues[i]
                    residues[i] = new_r
                    new_score = calculate_score(residues)
                    
                    if new_score > current_score:
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_residues = residues[:]
                        move_made = True
                        break
                    else:
                        # Revert
                        residues[i] = old_r
