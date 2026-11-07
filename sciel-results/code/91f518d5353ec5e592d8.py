        def backtrack(doms):
            if time.time() > deadline:
                return None
                
            # Check if all assigned
            all_assigned = True
            for key in doms:
                if len(doms[key]) != 1:
                    all_assigned = False
                    break
            if all_assigned:
                return doms
                
            # Find MRV
            min_var = None
            min_size = n + 1
            for key in doms:
                size = len(doms[key])
                if 0 < size < min_size:
                    min_size = size
                    min_var = key
                    if size == 1:
                        break
                        
            if min_var is None:
                return None # No unassigned variable with valid domain?
                
            if min_size == 0:
                return None
                
            for val in list(doms[min_var]):
                new_doms = {k: set(v) for k, v in doms.items()}
                new_doms[min_var] = {val}
                
                if forward_check(min_var, val, new_doms):
                    res = backtrack(new_doms)
                    if res is not None:
                        return res
            return None
