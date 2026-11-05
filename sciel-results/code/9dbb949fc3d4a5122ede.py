    def backtrack(a, bits, candidates, target_size, deadline, node_limit=100000):
        if len(a) == target_size:
            return a
        if time.time() > deadline:
            return None
        if len(a) + len(candidates) < target_size:
            return None
        if node_limit <= 0:
            return None
        
        node_limit -= 1
        
        shuffled_candidates = candidates[:]
        random.shuffle(shuffled_candidates)
        
        for x in shuffled_candidates:
            if x in a:
                continue
            if can_add(x, a, bits):
                new_a, new_bits = add_element(x, a, bits)
                new_candidates = [c for c in candidates if c != x]
                
                res = backtrack(new_a, new_bits, new_candidates, target_size, deadline, node_limit)
                if res:
                    return res
        return None
