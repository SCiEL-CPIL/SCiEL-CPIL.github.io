            if x in sums:
                continue
                
            new_suffix_sums = [x]
            valid = True
            
            if suffix_sums:
                for old_s in suffix_sums:
                    new_s = x + old_s
                    new_suffix_sums.append(new_s)
                    if new_s in sums:
                        valid = False
                        break
            # else: valid remains True
            
            if not valid:
                continue
