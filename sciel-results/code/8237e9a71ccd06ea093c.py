                        for x5 in range(x4 + 1, M + 1):
                            d15 = x5 - x1
                            d25 = x5 - x2
                            d35 = x5 - x3
                            d45 = x5 - x4
                            
                            new_diffs = [d12, d13, d23, d14, d24, d34, d15, d25, d35, d45]
                            
                            # Check internal duplicates
                            if len(set(new_diffs)) < 10:
                                continue
                                
                            # Check conflict with used
                            if any(d in used_diffs for d in new_diffs):
                                continue
