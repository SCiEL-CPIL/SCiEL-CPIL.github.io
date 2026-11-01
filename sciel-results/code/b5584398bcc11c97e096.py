            # Phase 2: Swap-Remove
            for i in range(len(codewords)):
                cw_i = codewords[i]
                U = []
                for x in balls[cw_i]:
                    if cov[x] == 1:
                        U.append(x)
                
                if not U:
                    continue
                
                for w in range(total):
                    if w in codewords:
                        continue
                    if not all(x in balls[w] for x in U):
                        continue
                    
                    # Check if swap allows removal
                    # Simulate cov change
                    # We need to check if any cw_j can be removed in the new state.
                    # New cov[x] = cov[x] - 1 (if x in balls[cw_i]) + 1 (if x in balls[w])
                    
                    # To avoid modifying cov, we check condition directly:
                    # A codeword cw_j can be removed if for all y in balls[cw_j], new_cov[y] > 1.
                    # new_cov[y] = cov[y] - (1 if y in balls[cw_i] else 0) + (1 if y in balls[w] else 0)
                    
                    valid_swap = True
                    for j in range(len(codewords)):
                        if j == i:
                            continue
                        cw_j = codewords[j]
                        for y in balls[cw_j]:
                            new_cov_y = cov[y]
                            if y in balls[cw_i]:
                                new_cov_y -= 1
                            if y in balls[w]:
                                new_cov_y += 1
                            if new_cov_y <= 1:
                                valid_swap = False
                                break
                        if not valid_swap:
                            break
                    
                    if valid_swap:
                        # Perform swap
                        for x in balls[cw_i]:
                            cov[x] -= 1
                        for x in balls[w]:
                            cov[x] += 1
                        codewords[i] = w
                        changed = True
                        break # Restart main loop to perform removal
                if changed:
                    break
