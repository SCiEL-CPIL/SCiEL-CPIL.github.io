                    I = all_sets[a] & all_sets[idx]
                    Forbidden = all_sets[a] | all_sets[idx]
                    Free = [x for x in range(1, 17) if x not in Forbidden]
                    Need = 4 - len(I)
                    
                    if Need < 0 or Need > len(Free):
                        continue
                        
                    for comb in combinations(Free, Need):
                        X = I.union(comb)
                        if X in set_to_idx:
                            k_idx = set_to_idx[frozenset(X)]
                            conflict_count[k_idx] += 1
