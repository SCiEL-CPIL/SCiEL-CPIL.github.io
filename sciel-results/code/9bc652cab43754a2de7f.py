            valid_ps = []
            for p_idx in candidates:
                if p_idx in S:
                    continue
                if W[p_idx] == 0:
                    continue
                
                safe = True
                for x in C:
                    if (W[x] & (1 << d[x][p_idx])) != 0:
                        safe = False
                        break
                        
                if safe:
                    valid_ps.append(p_idx)
                    
            if not valid_ps:
                break
                
            valid_p = random.choice(valid_ps)
