<<<<<<< SEARCH
    def try_extend(base_set, remaining, order):
        """Try to extend a base set by adding elements from remaining in given order."""
        current_set = set(base_set)
        for elem in order:
            if elem in current_set or elem not in remaining:
                continue
            candidate_set = current_set | {elem}
            if is_valid(candidate_set):
                current_set.add(elem)
        return current_set
=======
    def try_extend(base_set, remaining, order):
        """Try to extend a base set by adding elements from remaining in given order."""
        current_set = set(base_set)
        for elem in order:
            if elem in current_set or elem not in remaining:
                continue
            candidate_set = current_set | {elem}
            if is_valid(candidate_set):
                current_set.add(elem)
        return current_set
    
    def swap_improve(s):
        """Try to improve set s by removing one element and adding one or two others."""
        s = set(s)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try 1-for-1 swap
            for remove_elem in list(s):
                if time.time() > deadline - 2:
                    return s
                reduced = s - {remove_elem}
                for add_elem in universe:
                    if add_elem in reduced:
                        continue
                    candidate = reduced | {add_elem}
                    if is_valid(candidate):
                        # Only accept if it doesn't reduce size and we made progress
                        # Actually, 1-for-1 keeps same size, so only accept if it enables further growth
                        # For now, skip 1-for-1 as it doesn't increase cardinality
                        pass
            
            # Try 1-for-2 swap (remove 1, add 2)
            for remove_elem in list(s):
                if time.time() > deadline - 2:
                    return s
                reduced = s - {remove_elem}
                remaining = [x for x in universe if x not in reduced]
                for add1 in remaining:
                    if time.time() > deadline - 2:
                        return s
                    for add2 in remaining:
                        if add2 <= add1:
                            continue
                        candidate = reduced | {add1, add2}
                        if len(candidate) > len(s):
                            if is_valid(candidate):
                                s = candidate
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        return s
>>>>>>> REPLACE

