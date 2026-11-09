    def assign_residues(indices):
        if not indices:
            return []
        
        sorted_indices = sorted(indices, key=lambda i: moduli[i])
        
        node_count = 0
        MAX_NODES = 50000
        
        def backtrack(pos, assigned):
            nonlocal node_count
            node_count += 1
            if node_count > MAX_NODES:
                return False
            
            if pos == len(sorted_indices):
                return True
            
            idx = sorted_indices[pos]
            
            valid_residues = []
            for r in range(moduli[idx]):
                valid = True
                for prev_idx, prev_r in assigned.items():
                    g = gcds[idx][prev_idx]
                    if (r % g) == (prev_r % g):
                        valid = False
                        break
                if valid:
                    valid_residues.append(r)
            
            if not valid_residues:
                return False
            
            random.shuffle(valid_residues)
            
            for r in valid_residues:
                assigned[idx] = r
                
                consistent = True
                for next_pos in range(pos + 1, len(sorted_indices)):
                    next_idx = sorted_indices[next_pos]
                    has_valid = False
                    for nr in range(moduli[next_idx]):
                        ok = True
                        for a_idx, a_r in assigned.items():
                            g = gcds[next_idx][a_idx]
                            if (nr % g) == (a_r % g):
                                ok = False
                                break
                        if ok:
                            has_valid = True
                            break
                    if not has_valid:
                        consistent = False
                        break
                
                if consistent:
                    if backtrack(pos + 1, assigned):
                        return True
                
                del assigned[idx]
                
            return False

        for _ in range(100):
            node_count = 0
            if backtrack(0, {}):
                result = []
                for idx in indices:
                    result.append([assigned[idx], moduli[idx]])
                return result
        return None
