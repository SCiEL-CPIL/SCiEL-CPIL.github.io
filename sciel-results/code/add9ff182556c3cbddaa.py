            # Phase 2: Try swaps
            # ...
            S_list = sorted(S, reverse=True)
            for x in S_list:
                # ...
                S_temp = S - {x}
                # Find smallest y not in S such that is_valid_addition(S_temp, y)
                # ...
                for y in range(1, x): # y must be < x for gain
                    if y in S:
                        continue
                    if is_valid_addition(S_temp, y):
                        # Perform swap
                        S = S_temp
                        S.add(y)
                        # ...
                        found = True
                        break
